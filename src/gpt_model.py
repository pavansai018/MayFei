import torch
import torch.nn as nn
from transformer import Transformer
from layer_norm import LayerNorm

class GPTModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.token_embeddings: nn.Embedding = nn.Embedding(num_embeddings=cfg['vocab_size'], embedding_dim=cfg['emb_dim'])
        self.position_embeddings: nn.Embedding = nn.Embedding(num_embeddings=cfg['context_length'], embedding_dim=cfg['emb_dim'])
        self.dropout: nn.Dropout = nn.Dropout(p=cfg['drop_rate'])

        self.transformer_blocks: nn.Sequential = nn.Sequential(
            *[Transformer(cfg=cfg) for i in range(cfg['num_layers'])]
        )

        self.final_norm: LayerNorm = LayerNorm(cfg=cfg)
        self.out_head: nn.Linear = nn.Linear(in_features=cfg['emb_dim'], out_features=cfg['vocab_size'])

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_tokens.shape

        tok_emb: torch.Tensor = self.token_embeddings(input_tokens)
        pos_ids: torch.Tensor = torch.arange(start=0, end=seq_len, device=input_tokens.device)
        pos_emb: torch.Tensor = self.position_embeddings(pos_ids)
        x = tok_emb + pos_emb

        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        x = self.out_head(x)
        return x