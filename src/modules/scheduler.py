import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineAnnealingScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr, warmup_start_lr=0.0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        super(WarmupCosineAnnealingScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        print(self.last_epoch)
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = (self.last_epoch + 1) / self.warmup_epochs
            return [self.warmup_start_lr + warmup_factor * (base_lr - self.warmup_start_lr) for base_lr in self.base_lrs]
        else:
            # Cosine annealing after warmup
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + cosine_factor * (base_lr - self.min_lr) for base_lr in self.base_lrs]


class StepWiseWarmupCosineAnnealingScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr, steps_per_epoch, warmup_start_lr=0.0, last_epoch=-1):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Number of epochs for the warm-up phase.
            max_epochs (int): Total number of training epochs.
            min_lr (float): Minimum learning rate after cosine annealing.
            steps_per_epoch (int): Number of steps per epoch (batches per epoch).
            warmup_start_lr (float, optional): Starting learning rate for warm-up. Defaults to 0.0.
            last_epoch (int, optional): Index of the last epoch. Defaults to -1.
        """
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = max_epochs * steps_per_epoch
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        super(StepWiseWarmupCosineAnnealingScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup over steps
            warmup_factor = (self.last_epoch + 1) / self.warmup_steps
            return [self.warmup_start_lr + warmup_factor * (base_lr - self.warmup_start_lr) for base_lr in self.base_lrs]
        else:
            # Cosine annealing after warmup
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + cosine_factor * (base_lr - self.min_lr) for base_lr in self.base_lrs]

