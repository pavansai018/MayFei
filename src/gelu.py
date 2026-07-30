import torch.nn as nn
import torch


class GELU(nn.Module):
    """Gaussian Error Linear Unit (GELU) activation module.

    Implements the tanh-based approximation of GELU, as used in the
    original GPT-2/BERT implementations, instead of the exact erf-based
    formulation.
    """

    def __init__(self):
        """Initialize the GELU module (no learnable parameters)."""
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the GELU activation function element-wise.

        Args:
            x: Input tensor of any shape.

        Returns:
            torch.Tensor: Tensor of the same shape as ``x`` with the GELU
            activation applied element-wise.
        """
        # tanh approximation of GELU:
        # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        gelu_x: torch.Tensor = 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))
        return gelu_x