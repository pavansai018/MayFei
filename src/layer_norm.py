import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Apply layer normalization over the last dimension of the input.

    Normalizes the input tensor to zero mean and unit variance along its
    final dimension, then applies a learnable affine transformation
    (scale and shift), following Ba et al., 2016 (https://arxiv.org/abs/1607.06450).

    Attributes:
        eps: Small constant added to the variance for numerical stability.
        shift: Learnable per-feature bias, initialized to zeros.
        scale: Learnable per-feature gain, initialized to ones.
    """

    def __init__(self, cfg: dict):
        """Initialize the layer normalization parameters.

        Args:
            cfg: Configuration dictionary containing the key ``emb_dim``,
                the embedding dimension over which normalization is applied.
        """
        super().__init__()
        self.eps: float = 1e-5
        self.shift: nn.Parameter = nn.Parameter(data=torch.zeros(cfg['emb_dim']))
        self.scale: nn.Parameter = nn.Parameter(data=torch.ones(cfg['emb_dim']))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the input and apply the learned affine transform.

        Args:
            x: Input tensor of shape ``(..., emb_dim)``.

        Returns:
            Tensor of the same shape as ``x``, normalized along the last
            dimension and rescaled/shifted by ``scale`` and ``shift``.
        """
        mean: torch.Tensor = x.mean(dim=-1, keepdim=True)
        var: torch.Tensor = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x: torch.Tensor = (x - mean) / (torch.sqrt(var + self.eps))
        return self.scale * norm_x + self.shift
    