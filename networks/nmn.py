"""Neural modular network."""

import numpy as np
import torch
import torch.nn as nn

from configs.skill_config import NetParameters

from .common import _NMNStageMixin
from .radar_encoder import RadarEncoder


class DefenderNetNMN(_NMNStageMixin, nn.Module):
    """Neural modular network with non-CTDE critic."""

    def __init__(self):
        super().__init__()
        self._init_nmn_stage_control()

        self.radar_encoder = RadarEncoder()
        tracking_dim = NetParameters.ACTOR_SCALAR_LEN
        obstacle_dim = NetParameters.RADAR_EMBED_DIM
        branch_dim = NetParameters.NMN_BRANCH_DIM
        merged_dim = NetParameters.NMN_MERGED_DIM

        self.tracking_branch = nn.Sequential(nn.Linear(tracking_dim, branch_dim), nn.Tanh())
        self.obstacle_branch = nn.Sequential(nn.Linear(obstacle_dim, branch_dim), nn.Tanh())
        self.merged_layer = nn.Sequential(nn.Linear(branch_dim * 2, merged_dim), nn.Tanh())

        self.policy_mean = nn.Linear(merged_dim, NetParameters.ACTION_DIM)
        self.log_std = nn.Parameter(torch.zeros(NetParameters.ACTION_DIM))

        critic_input = NetParameters.ACTOR_VECTOR_LEN
        critic_hidden = NetParameters.NMN_CRITIC_HIDDEN
        critic_layers = NetParameters.NMN_CRITIC_LAYERS
        self.critic_backbone = self._build_mlp(critic_input, critic_hidden, critic_layers)
        self.value_head = nn.Linear(critic_hidden, 1)
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
        scalar = obs[:, :NetParameters.ACTOR_SCALAR_LEN]
        radar = obs[:, NetParameters.ACTOR_SCALAR_LEN:]
        radar = self._prepare_nmn_radar(radar)
        radar_emb = self.radar_encoder(radar)
        return scalar, radar_emb

    def forward(self, actor_obs, critic_obs):
        scalar, radar_emb = self._encode_actor_obs(actor_obs)
        tracking_feat = self.tracking_branch(scalar)
        obstacle_feat = self.obstacle_branch(radar_emb)
        merged = self.merged_layer(torch.cat([tracking_feat, obstacle_feat], dim=-1))
        mean = self.policy_mean(merged)
        log_std = self.log_std.expand_as(mean)

        critic_in = torch.cat([scalar, radar_emb], dim=-1)
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
        actor_obs = critic_obs[:, :NetParameters.ACTOR_RAW_LEN]
        scalar, radar_emb = self._encode_actor_obs(actor_obs)
        critic_in = torch.cat([scalar, radar_emb], dim=-1)
        critic_state = self.critic_backbone(critic_in)
        return self.value_head(critic_state)


__all__ = ["DefenderNetNMN"]
