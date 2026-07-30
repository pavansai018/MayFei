import torch
import torch.nn as nn
from gelu import GELU

class FeedForward(nn.Module):
    """Position-wise feed-forward network used in transformer blocks.

    Projects the input up to a higher-dimensional space, applies a GELU
    nonlinearity, then projects back down to the original embedding
    dimension, as in Vaswani et al., 2017 (https://arxiv.org/abs/1706.03762).

    Attributes:
        layers: Sequential stack of linear -> GELU -> linear layers.
    """

    def __init__(self, cfg: dict):
        """Initialize the feed-forward layers.

        Args:
            cfg: Configuration dictionary containing the key ``emb_dim``,
                the embedding dimension of the input and output.
        """
        super().__init__()

        self.layers: nn.Sequential = nn.Sequential(
            nn.Linear(in_features=cfg['emb_dim'], out_features=4*cfg['emb_dim']),
            GELU(),
            nn.Linear(in_features=4*cfg['emb_dim'], out_features=cfg['emb_dim'])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward transformation.

        Args:
            x: Input tensor of shape ``(..., emb_dim)``.

        Returns:
            Tensor of the same shape as ``x``.
        """
        return self.layers(x)
    