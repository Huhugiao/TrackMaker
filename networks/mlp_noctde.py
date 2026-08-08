"""MLP actor-critic network without CTDE critic inputs."""

import numpy as np
import torch
import torch.nn as nn

from configs.skill_config import NetParameters

from .radar_encoder import RadarEncoder


class DefenderNetMLPNoCTDE(nn.Module):
    """Defender actor-critic network with non-CTDE critic."""

    def __init__(self, action_dim=None):
        super().__init__()
        self.hidden_dim = NetParameters.HIDDEN_DIM
        self.num_layers = getattr(NetParameters, "NUM_HIDDEN_LAYERS", 3)
        self.action_dim = int(NetParameters.ACTION_DIM if action_dim is None else action_dim)
        self.radar_encoder = RadarEncoder()

        self.actor_backbone = self._build_mlp(
            NetParameters.ACTOR_VECTOR_LEN,
            self.hidden_dim,
            self.num_layers,
        )
        self.policy_mean = nn.Linear(self.hidden_dim, self.action_dim)
        self.log_std = nn.Parameter(torch.zeros(self.action_dim))
        self.critic_backbone = self._build_mlp(
            NetParameters.ACTOR_VECTOR_LEN,
            self.hidden_dim,
            self.num_layers,
        )
        self.value_head = nn.Linear(self.hidden_dim, 1)
        self.apply(self._init_weights)

    @staticmethod
    def _build_mlp(input_dim, hidden_dim, num_layers):
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        return nn.Sequential(*layers)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def _encode_actor_obs(self, obs):
        actor_obs = obs[:, :NetParameters.ACTOR_RAW_LEN]
        scalar = actor_obs[:, :NetParameters.ACTOR_SCALAR_LEN]
        radar = actor_obs[:, NetParameters.ACTOR_SCALAR_LEN:NetParameters.ACTOR_RAW_LEN]
        radar_emb = self.radar_encoder(radar)
        return torch.cat([scalar, radar_emb], dim=-1)

    def forward(self, actor_obs, critic_obs):
        actor_in = self._encode_actor_obs(actor_obs)
        critic_in = self._encode_actor_obs(critic_obs)
        actor_state = self.actor_backbone(actor_in)
        mean = self.policy_mean(actor_state)
        log_std = self.log_std.expand_as(mean)
        critic_state = self.critic_backbone(critic_in)
        value = self.value_head(critic_state)
        return mean, value, log_std

    def act(self, actor_obs, critic_obs):
        mean, value, log_std = self.forward(actor_obs, critic_obs)
        std = torch.exp(log_std)
        eps = torch.randn_like(mean)
        pre_tanh = mean + eps * std
        action = torch.tanh(pre_tanh)
        log_det_jac = torch.log(1.0 - torch.tanh(pre_tanh) ** 2 + 1e-6)
        log_prob = (torch.distributions.Normal(mean, std).log_prob(pre_tanh) - log_det_jac).sum(dim=-1)
        return action, log_prob, pre_tanh, value

    def critic_value(self, critic_obs):
        critic_in = self._encode_actor_obs(critic_obs)
        critic_state = self.critic_backbone(critic_in)
        return self.value_head(critic_state)


__all__ = ["DefenderNetMLPNoCTDE"]
