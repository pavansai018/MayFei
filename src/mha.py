import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """Causal multi-head self-attention layer.

    Projects the input into query, key, and value tensors, splits them
    across multiple attention heads, computes scaled dot-product attention
    with a causal mask, then recombines the heads and applies an output
    projection, as in Vaswani et al., 2017 (https://arxiv.org/abs/1706.03762).

    Attributes:
        num_heads: Number of attention heads.
        head_dim: Dimensionality of each attention head.
        context_length: Maximum sequence length supported by the causal mask.
        W_Query: Linear projection producing query vectors.
        W_Key: Linear projection producing key vectors.
        W_Value: Linear projection producing value vectors.
        dropout: Dropout applied to attention weights.
        mask: Upper-triangular boolean buffer used for causal masking.
        out_proj: Linear projection applied to the combined attention output.
    """

    def __init__(self, cfg: dict):
        """Initialize the projections, dropout, and causal mask.

        Args:
            cfg: Configuration dictionary containing the keys ``emb_dim``,
                ``num_heads``, ``context_length``, ``qkv_bias``, and
                ``drop_rate``.
        """
        super().__init__()

        assert (cfg['emb_dim'] % cfg['num_heads'] == 0), 'emb_dim must be divisible by num_heads'
        self.num_heads: int = cfg['num_heads']
        self.head_dim: int = cfg['emb_dim'] // self.num_heads 
        self.context_length: int = cfg['context_length']


        self.W_Query: nn.Linear = nn.Linear(in_features=cfg['emb_dim'], out_features=cfg['emb_dim'], bias=cfg['qkv_bias'])
        self.W_Key: nn.Linear = nn.Linear(in_features=cfg['emb_dim'], out_features=cfg['emb_dim'], bias=cfg['qkv_bias'])
        self.W_Value: nn.Linear = nn.Linear(in_features=cfg['emb_dim'], out_features=cfg['emb_dim'], bias=cfg['qkv_bias'])

        self.dropout: nn.Dropout = nn.Dropout(p=cfg['drop_rate'])
        self.register_buffer('mask', torch.triu(torch.ones((self.context_length, self.context_length), ), diagonal=1), persistent=False)
        self.out_proj: nn.Linear = nn.Linear(in_features=cfg['emb_dim'], out_features=cfg['emb_dim'], bias=False)


    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        """Apply causal multi-head self-attention.

        Args:
            input_embeddings: Input tensor of shape
                ``(batch, num_tokens, emb_dim)``.

        Returns:
            Tensor of shape ``(batch, num_tokens, emb_dim)`` containing the
            attention output after the output projection.
        """
        batch, num_tokens, emb_dim = input_embeddings.shape

        # [batch, num_tokens, emb_dim]
        queries: torch.Tensor = self.W_Query(input_embeddings)
        keys: torch.Tensor = self.W_Key(input_embeddings)
        values: torch.Tensor = self.W_Value(input_embeddings)

        # [batch, num_tokens, num_heads, head_dim]
        queries: torch.Tensor = queries.view(batch, num_tokens, self.num_heads, self.head_dim)
        keys: torch.Tensor = keys.view(batch, num_tokens, self.num_heads, self.head_dim)
        values: torch.Tensor = values.view(batch, num_tokens, self.num_heads, self.head_dim)

        # [batch, num_heads, num_tokens, head_dim]
        queries = queries.transpose(dim0=1, dim1=2)
        keys = keys.transpose(dim0=1, dim1=2)
        values = values.transpose(dim0=1, dim1=2)

        '''
        queries: [batch, num_heads, num_tokens, head_dim]
        keys.tarnspose: [batch, num_heads, head_dim, num_tokens]
        attention_scores: [batch, num_heads, num_tokens, num_tokens]
        '''
        attention_scores: torch.Tensor = queries @ keys.transpose(dim0=2, dim1=3)
        mask = self.mask.bool()[:num_tokens, :num_tokens]
        attention_scores.masked_fill_(mask, -torch.inf)
        attention_weights: torch.Tensor = torch.softmax(attention_scores/(keys.shape[-1]**0.5), dim=-1)
        attention_weights = self.dropout(attention_weights)

        '''
        attention_weights: [batch, num_heads, num_tokens, num_tokens]
        values: [batch, num_heads, num_tokens, head_dim]
        attention_weights @ values: [batch, num_heads, num_tokens, head_dim]
        (attention_weights @ values).transpose: [batch, num_tokens, num_heads, head_dim]
        '''
        context_vector: torch.Tensor = (attention_weights @ values).transpose(dim0=1, dim1=2)
        context_vector = context_vector.contiguous().view(batch, num_tokens, emb_dim)

        return self.out_proj(context_vector)
    

