"""Radar encoder."""

import torch.nn as nn

from configs.skill_config import NetParameters


class RadarEncoder(nn.Module):
    """Encode raw radar vectors into a compact embedding."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NetParameters.RADAR_DIM, 256),
            nn.Tanh(),
            nn.Linear(256, NetParameters.RADAR_EMBED_DIM),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


__all__ = ["RadarEncoder"]
