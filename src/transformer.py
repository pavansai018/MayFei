import torch
import torch.nn as nn
from mha import MultiHeadAttention
from feed_forward import FeedForward
from layer_norm import LayerNorm

class Transformer(nn.Module):
    """Single transformer block with pre-norm attention and feed-forward sublayers.

    Applies layer normalization, multi-head self-attention, and a
    feed-forward network, each wrapped in a residual (skip) connection with
    dropout, as in Vaswani et al., 2017 (https://arxiv.org/abs/1706.03762).

    Attributes:
        token_embeddings: Linear projection from vocabulary space to
            embedding space.
        position_embeddings: Linear projection from context-length space to
            embedding space.
        dropout: Dropout applied after each sublayer.
        norm1: Layer normalization applied before the attention sublayer.
        mha: Multi-head self-attention sublayer.
        norm2: Layer normalization applied before the feed-forward sublayer.
        feed_forward: Position-wise feed-forward sublayer.
    """

    def __init__(self, cfg: dict):
        """Initialize the embeddings, normalization, attention, and feed-forward layers.

        Args:
            cfg: Configuration dictionary containing the keys ``vocab_size``,
                ``emb_dim``, ``context_length``, and ``drop_rate``, along
                with any keys required by ``MultiHeadAttention``,
                ``LayerNorm``, and ``FeedForward``.
        """
        super().__init__()

        self.token_embeddings: nn.Linear = nn.Linear(in_features=cfg['vocab_size'], out_features=cfg['emb_dim'])
        self.position_embeddings: nn.Linear = nn.Linear(in_features=cfg['context_length'], out_features=cfg['emb_dim'])

        self.dropout: nn.Dropout = nn.Dropout(p=cfg['drop_rate'])
        self.norm1: LayerNorm = LayerNorm(cfg=cfg)

        self.mha: MultiHeadAttention = MultiHeadAttention(cfg=cfg)
        self.norm2: LayerNorm = LayerNorm(cfg=cfg)
        self.feed_forward: FeedForward = FeedForward(cfg=cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the pre-norm attention and feed-forward sublayers with residual connections.

        Args:
            x: Input tensor of shape ``(batch, num_tokens, emb_dim)``.

        Returns:
            Tensor of the same shape as ``x``.
        """
        skip_cell: torch.Tensor = x
        x = self.norm1(x)
        x = self.mha(x)
        x = self.dropout(x)

        x = x + skip_cell

        skip_cell = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)

        x = x + skip_cell
        return x
    