import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
import torch

class LossFunctions:
    @staticmethod
    def reconstruction_loss(x_recon, x, reduction='sum', dist = 'gaussian'):
        """
        Computes the binary cross-entropy loss between the reconstructed image and the original image.
        """
        
        bsize = x_recon.shape[0]
        
        if dist == 'gaussian':
            loss = F.mse_loss(x_recon, x, reduction=reduction) / bsize
        elif dist == 'bernoulli':
            loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction=reduction) / bsize
        else:
            raise AttributeError(f"Invalid Distribution provided: {dist}")
        
        return loss

    @staticmethod
    def reconstruction_wmask_loss(x_recon, x, valid_cnt, reduction='sum', dist = 'gaussian'):
        """
        Computes the binary cross-entropy loss between the reconstructed tensor and the 
        original tensor while taking into account the number of unmasked tokens.
        """
        if valid_cnt < 2:
            return 0.0
        
        if dist == 'gaussian':
            loss = F.mse_loss(x_recon, x, reduction=reduction) / valid_cnt
        elif dist == 'bernoulli':
            # print(F.binary_cross_entropy_with_logits(x_recon, x, reduction=reduction))
            loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction=reduction) / valid_cnt
        else:
            raise AttributeError(f"Invalid Distribution provided: {dist}")
        
        return loss

class AMPBackpropOptimizer:
    state_dict_key = "amp_scaler"
    def __init__(self):
        if torch.cuda.is_available():
            self._scaler = GradScaler()
        else:
            # Fallback behavior when CUDA is not available.
            self._scaler = None
    
    def scale_and_backprop(self, loss, create_graph):
        # If no scaler, directly call backward on loss
        if self._scaler is not None:
            self._scaler.scale(loss).backward(create_graph=create_graph)
        else:
            loss.backward(create_graph=create_graph)
    
    def unscale_and_clip(self, optimizer, clip_grad, parameters, clip_fn=None):
        if self._scaler is not None:
            self._scaler.unscale_(optimizer)
            if clip_grad and parameters is not None:
                return clip_fn(parameters, clip_grad) if clip_fn else clip_grad_norm_(parameters, clip_grad)
            return _get_grad_norm(parameters)
        else:
            # If no scaler, just clip gradients without unscaling
            if clip_grad and parameters is not None:
                return clip_fn(parameters, clip_grad) if clip_fn else torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            return _get_grad_norm(parameters)

    def step_optimizer(self, optimizer):
        if self._scaler is not None:
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            # Perform normal optimizer step if no scaler
            optimizer.step()
        
    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True, clip_fn=None):
        self.scale_and_backprop(loss, create_graph)
        if update_grad:
            norm = self.unscale_and_clip(optimizer, clip_grad, parameters, clip_fn)
            self.step_optimizer(optimizer)
        else:
            norm = None
        return norm

    def state_dict(self):
        # Return empty dict if no scaler
        return self._scaler.state_dict() if self._scaler is not None else {}

    def load_state_dict(self, state_dict):
        if self._scaler is not None:
            self._scaler.load_state_dict(state_dict)

    @staticmethod
    def _get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        norm_type = float(norm_type)
        if len(parameters) == 0:
            return torch.tensor(0.)
        device = parameters[0].grad.device
        if norm_type == np.inf:
            total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
        else:
            total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
        return total_norm