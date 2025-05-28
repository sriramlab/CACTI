import torch
import torch.nn as nn

class ModelSummary:
    """
    A class to summarize a PyTorch model, including layer names, parameter counts, and more.

    Attributes:
        model (nn.Module): The model to summarize.
    """

    def __init__(self, model: nn.Module):
        """
        Initializes the ModelSummary with a model instance.

        Args:
            model (nn.Module): The model to summarize.
        """
        self.model = model
        self.hooks = []

    def summarize(self):
        """
        Prints a summary of the model, including layer names, parameter counts, and hardware information.
        """
        # Report the precision mode used during training
        dtype = self.model.parameters().__next__().dtype
        precision = "bfloat16" if dtype == torch.bfloat16 else "float32" if dtype == torch.float32 else "float16" if dtype == torch.float16 else str(dtype)
        print(f"Using precision: {precision.upper()}")

        # Report hardware availability
        self._report_hardware()

        print(f"\n{'| Name':<15} | {'Type':<20} | {'Params':<10}")
        print('-' * 60)

        # Iterate over model layers and print their parameter counts
        self._print_layer_summaries()

        # Print total parameters information
        self._print_total_params()

    def _report_hardware(self):
        """
        Reports hardware availability and the device used by the model.
        """
        if torch.cuda.is_available():
            gpu_idx = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(gpu_idx)
            gpu_rank_list = [i for i in range(torch.cuda.device_count())]
            print(f"CUDA_VISIBLE_DEVICES: {str(gpu_rank_list)}")
            if next(self.model.parameters()).is_cuda:
                print(f"GPU available: True (cuda), used: True - GPU: {gpu_name} (index {gpu_idx})")
            else:
                print("GPU available: True (cuda), used: False")
        else:
            print("GPU available: False (cuda), used: False")

    def _print_layer_summaries(self):
        """
        Prints the name, type, and number of parameters of each layer in the model.
        """
        for name, module in self.model.named_children():
            num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            formatted_params = self._format_number(num_params)
            print(f"{name:<15} | {module.__class__.__name__:<20} | {formatted_params}")

    def _print_total_params(self):
        """
        Prints the total number of parameters in the model, including trainable and non-trainable parameters.
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params

        print('-' * 60)
        print(f"{self._format_number(trainable_params)}     Trainable params")
        print(f"{self._format_number(non_trainable_params)}     Non-trainable params")
        print(f"{self._format_number(total_params)}     Total params")

        # Calculate total size in MB based on the dtype of each parameter
        total_bits = 0
        for param in self.model.parameters():
            if param.data.is_floating_point():
                total_bits += param.numel() * torch.finfo(param.data.dtype).bits
            else:
                total_bits += param.numel() * torch.iinfo(param.data.dtype).bits

        total_size_mb = total_bits / 8 / 1e6  # Convert bits to bytes and then to MB
        print(f"{total_size_mb:.3f}     Total estimated model params size (MB)\n")

    def _format_number(self, value):
        """
        Formats the number based on the value:
        - Less than 1,000: plain number
        - 1,000 - 999,999: formatted as 'X.XK'
        - 1,000,000 or greater: formatted as 'X.XM'

        Args:
            value (int): The value to format.

        Returns:
            str: The formatted value as a string.
        """
        if value < 1_000:
            return f"{value}"
        elif value < 1_000_000:
            return f"{value / 1_000:.1f}K"
        else:
            return f"{value / 1_000_000:.1f}M"

    def __del__(self):
        """
        Destructor that ensures cleanup of any overhead used during model summary.
        """
        # Remove hooks if any were registered (currently not needed, but for future extensibility)
        for hook in self.hooks:
            hook.remove()

        # Print message indicating that resources are being cleaned up
        # print("Cleaning up resources used by ModelSummary...")

