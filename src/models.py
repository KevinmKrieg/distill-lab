from __future__ import annotations

import math
from typing import Dict

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class CausalTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int, config: Dict) -> None:
        super().__init__()
        d_model = config["d_model"]
        ff_dim = d_model * config.get("ff_mult", 4)
        self.hidden_size = d_model
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionalEncoding(d_model, max_seq_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config["n_heads"],
            dim_feedforward=ff_dim,
            dropout=config.get("dropout", 0.1),
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config["n_layers"])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        seq_len = input_ids.size(1)
        hidden = self.token_embedding(input_ids)
        hidden = self.position_encoding(hidden)
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        key_padding_mask = ~attention_mask
        hidden = self.encoder(hidden, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        return {"logits": logits, "hidden_states": hidden}


def build_model(vocab_size: int, max_seq_len: int, config: Dict) -> CausalTransformerLM:
    return CausalTransformerLM(vocab_size=vocab_size, max_seq_len=max_seq_len, config=config)
