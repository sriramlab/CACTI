import torch.nn as nn
import numpy as np

class LinearEmbed1D(nn.Module):
    """
    1D Data to 1:1 Embedding
    """
    def __init__(self, in_features=1, embed_dim=192):
        super().__init__()
        # Linear layer to transform each input feature to the embedding dimension
        self.dim = embed_dim
        self.proj = nn.Linear(in_features,  self.dim)

    def forward(self, x):
        B, C, L = x.shape
        # Reshape x to [B*L, C] to apply embedding to each element independently
        x = x.view(B * L, C)
        # Apply linear transformation
        x = self.proj(x)
        # Reshape back to [B, L, embed_dim] to maintain original sequence structure with embeddings
        x = x.view(B, L, self.dim)
        return x

class TabEmbed(nn.Module):
    """
    Table scaler to vector Embedding
    """
    def __init__(self, in_features=1, embed_dim=192, norm_layer=None):
        super().__init__()
        # Linear layer to transform each input feature to the embedding dimension
        self.dim = embed_dim
        self.proj = nn.Conv1d(in_features, self.dim, kernel_size=1, stride=1)
        self.norm = norm_layer(self.dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x

# Fixed sin-cos Positional embeddings
# Adapted from FAIR/ijepa
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos, scale=10000):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / scale**omega   # (D/2,)

    pos = pos.reshape(-1)   # (M,)
    out = np.einsum('m,d->md', pos, omega)   # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

# Adapted from FAIR/ijepa
def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, scale=10000):
    """
    grid_size: int of the grid length
    return:
    pos_embed: [grid_size, embed_dim] or [1+grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid, scale=scale)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed
