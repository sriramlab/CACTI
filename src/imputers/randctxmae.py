import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import warnings
import sys

from src.models.rctxmae import RCTXMAE
from src.loaders.baseloader import ContextBaseDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from src.modules.losses import LossFunctions, AMPBackpropOptimizer
from src.modules.scheduler import StepWiseWarmupCosineAnnealingScheduler
from src.utils.checkpoint import CheckpointHandler
from src.utils.summary import ModelSummary
from src.utils.training_helpers import _worker_init_fn




__version__ = "1.0.2"

class RandomContextMAE:
    def __init__(self, args, **kwargs):
        print(f"RCTXMAE v{__version__}")
        self.batch_size = args.batch_size
        self.min_lr = args.min_lr if args.min_lr else 5e-6
        self.lr = args.lr if args.lr else 0.001
        self.grad_clip = args.grad_clip if args.grad_clip else None
        self.warmup_epochs = args.warmup_epochs if args.warmup_epochs else 40
        self.epochs = args.epochs
        self.min_scale = None # kwargs['min_scale']
        self.max_scale = None # kwargs['max_scale']
        self.feats = kwargs['feats']
        self.mask_ratio = args.mask_ratio
        self.device = args.device
        self.num_workers = args.num_workers
        self.eps = 1e-6
        self.weight_decay = args.weight_decay if args.weight_decay else 0.001
        self.embeddings = args.embeddings
        self.context_embed_dim = args.cembed_size
        self.args = args

        # model spec tune
        self.embed_dim = args.embed_dim
        self.nencoder = args.nencoder
        self.ndecoder = args.ndecoder
        self.nhead = self.embed_dim//8 if self.embed_dim >= 64 else 4

        # init model
        self.model = RCTXMAE(
            table_len=len(self.feats),
            mask_ratio=self.mask_ratio,
            embed_dim=self.embed_dim,
            depth=self.nencoder,
            num_heads=self.nhead,
            decoder_depth=self.ndecoder,
            decoder_num_heads=self.nhead,
            context_embed_dim=self.context_embed_dim
        )

        # Directory for saving checkpoints
        self.checkpoint_handler = None
        if args.checkpoint_path:
            self.checkpoint_path = args.checkpoint_path

            # Initialize checkpoint handler
            self.checkpoint_handler = CheckpointHandler(self.model, 
                self.checkpoint_path, 
                self.get_hyperparameters())
        
    def get_hyperparameters(self):
        """
        Return a dictionary of the model and training hyperparameters
        """
        model_hyperparameters = {k: v for k, v in vars(self.model).items() if not isinstance(v, torch.nn.Module) and not isinstance(v, torch.Tensor)}
        return {
            'version': __version__,
            'batch_size': self.batch_size,
            'min_lr': self.min_lr,
            'lr': self.lr,
            'grad_clip': self.grad_clip,
            'warmup_epochs': self.warmup_epochs,
            'epochs': self.epochs,
            'min_scale': self.min_scale,
            'max_scale': self.max_scale,
            'feats': self.feats,
            'mask_ratio': self.mask_ratio,
            'device': self.device,
            'num_workers': self.num_workers,
            'eps': self.eps,
            'args': self.args,
            'model_hyperparameters': model_hyperparameters
        }

    def fit(self, obs_data: torch.Tensor, resume_from_checkpoint=False):
        # Load data
        if isinstance(obs_data, np.ndarray):
            obs_data = torch.tensor(obs_data)
        X = obs_data.clone()
         # scale feats to 0-1 space
        for i, col in enumerate(self.feats):
            X[:, i] = (X[:, i] - self.min_scale[i]) / (self.max_scale[i] - self.min_scale[i] + self.eps)


        dataset = ContextBaseDataset(X, self.embeddings)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=_worker_init_fn
        )
        
        # Move models to device
        self.model.to(self.device)

        # init Optimizers       
        optimizer = AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.95), weight_decay=self.weight_decay)
        scheduler = StepWiseWarmupCosineAnnealingScheduler(
            optimizer=optimizer,
            warmup_epochs=self.warmup_epochs,  # Number of warmup epochs
            max_epochs=self.epochs,    # Total number of epochs
            min_lr=self.min_lr,      # Minimum learning rate
            steps_per_epoch=len(dataloader),
            warmup_start_lr=self.min_lr  # Learning rate to start warmup
        )
        backprop_optimizer = AMPBackpropOptimizer()

        # Load checkpoint if resuming
        start_epoch = 0
        if resume_from_checkpoint:
            checkpoint_path = os.path.join(self.checkpoint_path, "best.pth")
            if os.path.exists(checkpoint_path):
                start_epoch = self.checkpoint_handler.load_checkpoint(checkpoint_path, self.optimizer)

        # Log Model summary
        mstats = ModelSummary(self.model)
        mstats.summarize()
        del mstats
        
        # Warmup sanity check
        if self.warmup_epochs >= self.epochs * 0.5:
            warnings.warn(f"warmup_epochs ({self.warmup_epochs}) is >= 50% of total epochs ({self.epochs}). Best loss may occur during warmup and not be checkpointed.")

        # Start training
        self.model.train()
        best_loss = float('inf')
        for epoch in range(start_epoch, self.epochs):
            optimizer.zero_grad()
            total_loss = 0
            itr = 0
            # dataloader with tqdm for better logging
            progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}", unit=" iters")

            # (re-)Start epoch
            for itr, batch in progress_bar:
                if isinstance(batch, dict):
                    batch = {key: value.to(self.device, non_blocking=True) for key, value in batch.items()}
                else:
                    batch = batch.to(self.device, non_blocking=True)

                # single forward pass
                with torch.cuda.amp.autocast():
                    loss, _ = self.model(batch['table'], batch['miss_map'], batch['info_embed'])
                    loss_value = loss.item()
                    total_loss += loss_value
                
                # safe check
                if not torch.all(torch.isfinite(loss)) :
                    warnings.warn("Found NaN or Inf Loss!")
                    print(f"Loss is {loss_value}, stopping training")
                    sys.exit(1)

                # Update the progress bar description with the loss value
                progress_bar.set_postfix(MSE_loss=loss_value, RMSE_loss=loss_value ** 0.5)

                # backwards
                backprop_optimizer(loss, 
                    optimizer, 
                    clip_grad = self.grad_clip,
                    parameters=self.model.parameters()
                )

                optimizer.zero_grad()

                # Step the scheduler per iteration
                scheduler.step()

            # Log average loss per epoch
            avg_loss = (total_loss / (itr + 1)) ** 0.5
            print(f"Epoch {epoch} :: Average RMSE loss: {avg_loss}")

            # Log the updated learning rate
            current_lr = scheduler.get_last_lr()
            print(f"Epoch {epoch} :: Learning Rate: {current_lr}")

            # Save checkpoint if avg_loss is the best so far
            if self.checkpoint_handler is not None and avg_loss < best_loss and epoch >= self.warmup_epochs:
                best_loss = avg_loss
                self.checkpoint_handler.save_checkpoint(epoch, optimizer)

        # Warn if no checkpoint was saved
        if self.checkpoint_handler is not None:
            symlink_path = os.path.join(self.checkpoint_handler.checkpoint_dir, "best.pth")
            if not os.path.exists(symlink_path):
                warnings.warn(f"No checkpoint was saved during training. Model never improved after warmup epoch {self.warmup_epochs}. Consider reducing --warmup_epochs.")

        return self


    def transform(self, obs_data: torch.Tensor):
        # load data
        if isinstance(obs_data, np.ndarray):
            obs_data = torch.tensor(obs_data)
        
        X = obs_data.clone()
         # scale feats to 0-1 space
        for i, col in enumerate(self.feats):
            X[:, i] = (X[:, i] - self.min_scale[i]) / (self.max_scale[i] - self.min_scale[i] + self.eps)

        dataset = ContextBaseDataset(X, self.embeddings)
        dataloader = DataLoader(
            dataset, batch_size=1,
            shuffle=False,
            pin_memory=True,
            worker_init_fn=_worker_init_fn
        )

        # set to eval 
        self.model.to(self.device)
        self.model.eval()
        imputed_data_list = []
        # run Imputation
        with torch.no_grad():
            # Wrap the dataloader with tqdm for a progress bar
            progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Inference", unit=" samples")
            
            for itr, batch in progress_bar:
                if isinstance(batch, dict):
                    batch = {key: value.to(self.device, non_blocking=True) for key, value in batch.items()}
                else:
                    batch = batch.to(self.device, non_blocking=True)

                _, xest = self.model(batch['table'], batch['miss_map'], batch['info_embed'])
                imputed_data_list.append(xest)

        imputed_data = torch.cat(imputed_data_list, dim=0)

        # re-scale feats back to orginal space
        for i, col in enumerate(self.feats):
            imputed_data[:, i] = imputed_data[:, i] * (self.max_scale[i] - self.min_scale[i] + self.eps) + self.min_scale[i]

        imputed_data = imputed_data.detach().cpu().numpy()

        # safety check
        if np.all(np.isnan(imputed_data)):
            raise RuntimeError("NaNs found in imputed data matrix!")

        obs_mask = 1 - (1 * (np.isnan(obs_data)))
        obs_mask = obs_mask.cpu().numpy()

        return obs_mask * np.nan_to_num(obs_data.detach().cpu().numpy()) + (1 - obs_mask) * imputed_data

    def fit_transform(self, obs_dict):
        obs_data = obs_dict['data']

        if isinstance(obs_data, np.ndarray):
            obs_data = torch.tensor(obs_data, dtype=torch.float32)
        else:
            obs_data = torch.tensor(obs_data.values, dtype=torch.float32)

        # Set min-max scale for training
        self.min_scale = obs_dict['minscale']
        self.max_scale = obs_dict['maxscale']

        if self.checkpoint_handler is not None:
            self.checkpoint_handler.hyperparameters = self.get_hyperparameters()

        # Fit
        _ = self.fit(obs_data)

        # Infer with best loss model; if exists
        if self.checkpoint_handler is not None:
            best_ckpt = torch.load(f"{self.checkpoint_handler.checkpoint_dir}/best.pth")
            self.model.load_state_dict(best_ckpt['model_state_dict'])
            print(f"Loaded best checkpoint model from Epoch {best_ckpt['epoch']}")
            del best_ckpt

        return self.transform(obs_data)

    def train_model(self, obs_dict):
        obs_data = obs_dict['data']

        if isinstance(obs_data, np.ndarray):
            obs_data = torch.tensor(obs_data, dtype=torch.float32)
        else:
            obs_data = torch.tensor(obs_data.values, dtype=torch.float32)

        # Set min-max scale for training
        self.min_scale = obs_dict['minscale']
        self.max_scale = obs_dict['maxscale']

        if self.checkpoint_handler:
            self.checkpoint_handler.hyperparameters = self.get_hyperparameters()

        self.fit(obs_data)

        # Infer with best loss model; if exists
        if self.checkpoint_handler is not None:
            best_ckpt = torch.load(f"{self.checkpoint_handler.checkpoint_dir}/best.pth")
            self.model.load_state_dict(best_ckpt['model_state_dict'])
            print(f"[train_model] Loaded best checkpoint from Epoch {best_ckpt['epoch']}")
            del best_ckpt

        return self

    def load_transform(self, checkpoint_path, obs_data):
        if isinstance(obs_data, np.ndarray):
            obs_data = torch.tensor(obs_data, dtype=torch.float32)
        else:
            obs_data = torch.tensor(obs_data.values, dtype=torch.float32)

        # Load checkpoint and restore scales
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.min_scale = ckpt['hyperparameters']['min_scale']
        self.max_scale = ckpt['hyperparameters']['max_scale']
        del ckpt

        # Apply transform
        return self.transform(obs_data)


