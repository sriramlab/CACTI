import os
import torch

class CheckpointHandler:
    def __init__(self, model, checkpoint_dir, hyperparameters):
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.hyperparameters = hyperparameters
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, epoch, optimizer):
        """
        Save the model checkpoint and create a symlink to the latest checkpoint
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.model.__class__.__name__}-epoch{epoch}.pth")

        # Delete older checkpoints except the current one
        for file in os.listdir(self.checkpoint_dir):
            if file.startswith(self.model.__class__.__name__) and file.endswith('.pth') and file != os.path.basename(checkpoint_path):
                os.remove(os.path.join(self.checkpoint_dir, file))

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'hyperparameters' : self.hyperparameters
        }, checkpoint_path)

        # Create a symlink called "last" pointing to the latest checkpoint
        symlink_path = os.path.join(self.checkpoint_dir, "last.pth")
        if os.path.islink(symlink_path):
            os.remove(symlink_path)
        os.symlink(checkpoint_path, symlink_path)

    def load_checkpoint(self, checkpoint_path, optimizer):
        """
        Load the model checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.model.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        return start_epoch