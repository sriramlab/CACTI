import torch
import torch.nn as nn
import numpy as np
from functools import partial
import warnings
import sys

from timm.models.vision_transformer import Block as timmBlock
from src.modules.embed import TabEmbed, get_1d_sincos_pos_embed
from src.modules.masker import CopySegmentModule
from src.modules.losses import LossFunctions
from src.modules.scheduler import WarmupCosineAnnealingScheduler

class CACTImodel(nn.Module):
    """
    Implementation of core CACTI model
    """
    def __init__(
        self,
        table_len=372,
        mask_ratio=0.5,
        in_chans=1,
        embed_dim=32,
        depth=6,
        num_heads=4,
        decoder_embed_dim=None, 
        decoder_depth=4,
        decoder_num_heads=4,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        embed_scale=10000,
        context_embed_dim=1024,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    ):
        super().__init__()
        # self.save_hyperparameters()
        self.lr = kwargs['lr'] if 'lr' in kwargs else 0.001

        # Key params
        self.table_len = table_len
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.embed_scale = embed_scale
        self.decoder_embed_dim = decoder_embed_dim if decoder_embed_dim else self.embed_dim
        self.mask_ratio = mask_ratio
        self.context_embed_dim=context_embed_dim
        
        # Global tokens
        self.pad_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))
        self.min_scale = None
        self.max_scale = None

        ### Encoder
        self.info_embed_enc = nn.Linear(self.context_embed_dim,embed_dim//4)
        self.tab_embed = TabEmbed(in_chans, 3*(embed_dim//4))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.table_len + 1, embed_dim), requires_grad=False)  # fixed embeddings

        self.encode_blocks = nn.ModuleList([
            timmBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        ###

        ### Decoder
        self.decoder_embed = nn.Linear(embed_dim, self.decoder_embed_dim, bias=True)
        self.info_embed_dec = nn.Linear(self.context_embed_dim,embed_dim//4)
        self.project_decoder = nn.Linear(self.decoder_embed_dim,3*(self.decoder_embed_dim//4))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.table_len + 1, self.decoder_embed_dim), requires_grad=False)  # fixed decoder embedding

        # Initialize the ModuleList using AttentionBlock
        assert decoder_depth % 2 == 0, "decoder_depth has to be a multiple of 2!"
        self.decode_blocks = nn.ModuleList([
            timmBlock(self.decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        
        self.decoder_norm = norm_layer(self.decoder_embed_dim)

        self.decoder_pred = nn.Linear(self.decoder_embed_dim, 1, bias=True)
        
        self.initialize_weights()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # use xavier_uniform following official JAX ViT:
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.table_len), cls_token=True, scale=self.embed_scale)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.table_len), cls_token=True, scale=self.embed_scale)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # # initialize globale tokens
        nn.init.normal_(self.mask_token, std=.02)
        nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def forward_encoder(self, x, mask, info_embed):
        batch_size = x.shape[0]
        x = self.tab_embed(x)   
        ctxembed_encode = self.info_embed_enc(info_embed.to(torch.float32))

        x = torch.concat((x,ctxembed_encode),dim=2)
        x = x+self.pos_embed[:,1:,:]


        # get context only
        # shuffle and apply copymask only during training
        x, restore_map, obs_mask = CopySegmentModule.contextify(
            x, mask.type(torch.int32), self.pad_token,
            shuffle = self.training
            ) # [B, L, D] -> [B, obs_len, D]


        # concat CLS token
        _clse = self.cls_token + self.pos_embed[:, :1, :]
        clsetk = _clse.expand(x.shape[0], -1, -1)
        x = torch.cat((clsetk, x), dim=1)

        # apply Encoder Transformer blocks
        for blk in self.encode_blocks:
            x = blk(x)
        x = self.norm(x)

        return x, restore_map, obs_mask

    def forward_decoder(self, latent, restore_map, obs_mask, info_embed):
        # embed tokens
        latent = self.decoder_embed(latent)
        ctxembed_decode = self.info_embed_dec(info_embed.to(torch.float32))

        # Pull out CLS
        _clsdtk = latent[:,:1,:]

        # decontextify and restore data structure
        # apply mask tokens to sequence

        _x = CopySegmentModule.decontextify(latent[:,1:,:], restore_map, obs_mask, self.mask_token)
        _x = self.project_decoder(_x)
        _x = torch.concat((_x, ctxembed_decode), dim=2)
       
        # concat CLS token to org-ed data
        x = torch.cat((_clsdtk, _x), dim=1)

        # add pos embed
        x = x + self.decoder_pos_embed
        
        # apply Decoder Transformer blocks
        for blk in self.decode_blocks:
            x = blk(x)
        
        return self.decoder_norm(x)


    def decode_predictor(self, x):
        # Pass the concatenated tensor to the decoder and convert from logits to probs
        x = torch.tanh(self.decoder_pred(x))/2. + 0.5
        
        # remove CLS 
        x = x[:, 1:, :]
        
        x = x.transpose(1, 2) # [B, 1, L]

        return x

    def forward_loss(self, data, pred, miss_mask, recon_mask):
        """
        data: [N, 1, L]
        pred: [N, L, 1]
        miss_mask: [N, L] ; 1 is obs, 0 is miss
        obs_mask: [N, L] ; 1 is obs, 0 is miss
        """
        B, _, L = data.shape

        # make mask of cell that were masked but observed
        imp_mask = 1 - recon_mask  - miss_mask.type(recon_mask.dtype) # (Obs' ^ Miss')

        # calculate mean loss
        loss = (pred.squeeze(1) - data.squeeze(1)) ** 2
        loss = (loss * recon_mask).sum() / recon_mask.sum()  + (loss * imp_mask).sum() / imp_mask.sum()        

        return loss

    def forward(self, x, miss_map, copy_mask, info_embed):
	
        # Encode latent
        if self.training:
            latent, restore_map, obs_mask = self.forward_encoder(x, copy_mask, info_embed)
        else: 
            latent, restore_map, obs_mask = self.forward_encoder(x, miss_map, info_embed)

        # Get Decoder latent 
        xdecode = self.forward_decoder(latent, restore_map, obs_mask, info_embed)

        # Predict all values
        xpred = self.decode_predictor(xdecode)

        # Estimate loss
        loss = self.forward_loss(x, xpred, miss_map, obs_mask)

        return loss, xpred.squeeze(1)