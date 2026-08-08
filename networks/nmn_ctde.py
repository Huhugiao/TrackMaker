"""Neural modular network with CTDE critic inputs."""

import numpy as np
import torch
import torch.nn as nn

from configs.skill_config import NetParameters

from .common import _NMNStageMixin
from .radar_encoder import RadarEncoder


class DefenderNetNMNCTDE(_NMNStageMixin, nn.Module):
    """Neural modular network with CTDE critic."""

    def __init__(self):
        super().__init__()
        self._init_nmn_stage_control()

        self.actor_radar_encoder = RadarEncoder()
        self.critic_defender_radar_encoder = RadarEncoder()
        self.critic_attacker_radar_encoder = RadarEncoder()

        self.hidden_dim = int(NetParameters.HIDDEN_DIM)
        self.num_layers = int(getattr(NetParameters, "NUM_HIDDEN_LAYERS", 3))

        tracking_dim = int(NetParameters.ACTOR_SCALAR_LEN)
        obstacle_dim = int(NetParameters.RADAR_EMBED_DIM)
        branch_dim = int(NetParameters.NMN_BRANCH_DIM)
        merged_dim = int(NetParameters.NMN_MERGED_DIM)

        self.tracking_branch = nn.Sequential(nn.Linear(tracking_dim, branch_dim), nn.Tanh())
        self.obstacle_branch = nn.Sequential(nn.Linear(obstacle_dim, branch_dim), nn.Tanh())
        self.merged_layer = nn.Sequential(nn.Linear(branch_dim * 2, merged_dim), nn.Tanh())

        self.policy_mean = nn.Linear(merged_dim, NetParameters.ACTION_DIM)
        self.log_std = nn.Parameter(torch.zeros(NetParameters.ACTION_DIM))
        self.critic_backbone = self._build_mlp(NetParameters.CRITIC_VECTOR_LEN, self.hidden_dim, self.num_layers)
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
        radar = self._prepare_nmn_radar(radar)
        radar_emb = self.actor_radar_encoder(radar)
        return scalar, radar_emb

    def _encode_critic_obs(self, obs):
        defender_obs = obs[:, :NetParameters.ACTOR_RAW_LEN]
        defender_scalar = defender_obs[:, :NetParameters.ACTOR_SCALAR_LEN]
        defender_radar = defender_obs[:, NetParameters.ACTOR_SCALAR_LEN:NetParameters.ACTOR_RAW_LEN]
        defender_radar = self._prepare_nmn_radar(defender_radar)
        defender_radar_emb = self.critic_defender_radar_encoder(defender_radar)
        defender_part = torch.cat([defender_scalar, defender_radar_emb], dim=-1)

        attacker_start = NetParameters.ACTOR_RAW_LEN
        if obs.shape[-1] >= NetParameters.CRITIC_RAW_LEN:
            attacker_scalar = obs[:, attacker_start:attacker_start + NetParameters.PRIVILEGED_SCALAR_LEN]
            attacker_radar = obs[:, attacker_start + NetParameters.PRIVILEGED_SCALAR_LEN:attacker_start + NetParameters.PRIVILEGED_RAW_LEN]
        else:
            batch_size = obs.shape[0]
            attacker_scalar = torch.zeros(batch_size, NetParameters.PRIVILEGED_SCALAR_LEN, device=obs.device, dtype=obs.dtype)
            attacker_radar = torch.zeros(batch_size, NetParameters.RADAR_DIM, device=obs.device, dtype=obs.dtype)
        attacker_radar = self._prepare_nmn_radar(attacker_radar)
        attacker_radar_emb = self.critic_attacker_radar_encoder(attacker_radar)
        attacker_part = torch.cat([attacker_scalar, attacker_radar_emb], dim=-1)
        return torch.cat([defender_part, attacker_part], dim=-1)

    def forward(self, actor_obs, critic_obs):
        scalar, radar_emb = self._encode_actor_obs(actor_obs)
        tracking_feat = self.tracking_branch(scalar)
        obstacle_feat = self.obstacle_branch(radar_emb)
        merged = self.merged_layer(torch.cat([tracking_feat, obstacle_feat], dim=-1))
        mean = self.policy_mean(merged)
        log_std = self.log_std.expand_as(mean)

        critic_in = self._encode_critic_obs(critic_obs)
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
        critic_in = self._encode_critic_obs(critic_obs)
        critic_state = self.critic_backbone(critic_in)
        return self.value_head(critic_state)


__all__ = ["DefenderNetNMNCTDE"]
