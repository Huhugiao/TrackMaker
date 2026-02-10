from typing import Optional

import torch
import torch.nn as nn


class AttackerGRUPredictor(nn.Module):
    """Predict attacker relative state (distance/bearing in [0, 1]) from partial measurements."""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid(),
        )

    def forward(self, seq_input: torch.Tensor, hidden: Optional[torch.Tensor] = None):
        # seq_input shape: [B, T, input_dim]
        gru_out, next_hidden = self.gru(seq_input, hidden)
        pred = self.head(gru_out[:, -1, :])
        return pred, next_hidden
