"""MLP actor-critic network."""

import numpy as np
import torch
import torch.nn as nn

from configs.skill_config import NetParameters

from .radar_encoder import RadarEncoder


class DefenderNetMLP(nn.Module):
    """Defender actor-critic network with CTDE critic."""

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
            NetParameters.CRITIC_VECTOR_LEN,
            self.hidden_dim,
            self.num_layers,
        )
        self.value_head = nn.Linear(self.hidden_dim, 1)
        self.apply(self._init_weights)

    def _build_mlp(self, input_dim, hidden_dim, num_layers):
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        return nn.Sequential(*layers)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def _encode_observation(self, obs, is_critic=False):
        if is_critic:
            defender_end = NetParameters.ACTOR_RAW_LEN
            defender_scalar = obs[:, :NetParameters.ACTOR_SCALAR_LEN]
            defender_radar = obs[:, NetParameters.ACTOR_SCALAR_LEN:defender_end]
            defender_radar_emb = self.radar_encoder(defender_radar)
            defender_part = torch.cat([defender_scalar, defender_radar_emb], dim=-1)

            attacker_start = defender_end
            attacker_scalar = obs[:, attacker_start:attacker_start + NetParameters.PRIVILEGED_SCALAR_LEN]
            attacker_radar = obs[:, attacker_start + NetParameters.PRIVILEGED_SCALAR_LEN:]
            attacker_radar_emb = self.radar_encoder(attacker_radar)
            attacker_part = torch.cat([attacker_scalar, attacker_radar_emb], dim=-1)
            return torch.cat([defender_part, attacker_part], dim=-1)

        scalar = obs[:, :NetParameters.ACTOR_SCALAR_LEN]
        radar = obs[:, NetParameters.ACTOR_SCALAR_LEN:]
        radar_emb = self.radar_encoder(radar)
        return torch.cat([scalar, radar_emb], dim=-1)

    def forward(self, actor_obs, critic_obs):
        actor_in = self._encode_observation(actor_obs, is_critic=False)
        critic_in = self._encode_observation(critic_obs, is_critic=True)
        actor_state = self.actor_backbone(actor_in)
        mean = self.policy_mean(actor_state)
        log_std = self.log_std.expand_as(mean)
        critic_state = self.critic_backbone(critic_in)
        value = self.value_head(critic_state)
        return mean, value, log_std

    def act(self, actor_obs, critic_obs):
        mean, value, log_std = self.forward(actor_obs, critic_obs)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        eps = torch.randn_like(mean)
        pre_tanh = mean + eps * std
        action = torch.tanh(pre_tanh)
        log_det_jac = torch.log(1.0 - torch.tanh(pre_tanh) ** 2 + 1e-6)
        log_prob = (dist.log_prob(pre_tanh) - log_det_jac).sum(dim=-1)
        return action, log_prob, pre_tanh, value

    def critic_value(self, critic_obs):
        critic_in = self._encode_observation(critic_obs, is_critic=True)
        critic_state = self.critic_backbone(critic_in)
        return self.value_head(critic_state)


__all__ = ["DefenderNetMLP"]
