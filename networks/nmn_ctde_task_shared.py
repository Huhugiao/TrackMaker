"""Neural modular network with CTDE critic and shared task branches."""

import numpy as np
import torch
import torch.nn as nn

from configs.skill_config import NetParameters

from .common import _NMNStageMixin
from .radar_encoder import RadarEncoder


class DefenderNetNMNCTDETaskShared(_NMNStageMixin, nn.Module):
    """NMN CTDE network with shared radar/task encoders across actor and critic."""

    def __init__(self):
        super().__init__()
        self._init_nmn_stage_control()

        self.shared_radar_encoder = RadarEncoder()

        self.hidden_dim = int(NetParameters.HIDDEN_DIM)
        self.num_layers = int(getattr(NetParameters, "NUM_HIDDEN_LAYERS", 3))

        tracking_dim = int(NetParameters.ACTOR_SCALAR_LEN)
        obstacle_dim = int(NetParameters.RADAR_EMBED_DIM)
        branch_dim = int(NetParameters.NMN_BRANCH_DIM)
        merged_dim = int(NetParameters.NMN_MERGED_DIM)
        privileged_dim = int(NetParameters.PRIVILEGED_LEN)
        privileged_branch_dim = int(NetParameters.NMN_BRANCH_DIM)

        self.shared_tracking_branch = nn.Sequential(nn.Linear(tracking_dim, branch_dim), nn.Tanh())
        self.shared_obstacle_branch = nn.Sequential(nn.Linear(obstacle_dim, branch_dim), nn.Tanh())

        self.actor_merge_layer = nn.Sequential(nn.Linear(branch_dim * 2, merged_dim), nn.Tanh())
        self.policy_mean = nn.Linear(merged_dim, NetParameters.ACTION_DIM)
        self.log_std = nn.Parameter(torch.zeros(NetParameters.ACTION_DIM))

        self.critic_task_merge_layer = nn.Sequential(nn.Linear(branch_dim * 2, merged_dim), nn.Tanh())
        self.critic_privileged_branch = nn.Sequential(
            nn.Linear(privileged_dim, privileged_branch_dim),
            nn.Tanh(),
        )
        critic_input_dim = merged_dim + privileged_branch_dim
        self.critic_backbone = self._build_mlp(critic_input_dim, self.hidden_dim, self.num_layers)
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

    def _encode_radar(self, radar):
        radar = self._prepare_nmn_radar(radar)
        return self.shared_radar_encoder(radar)

    def _extract_task_features(self, scalar, radar_emb):
        tracking_feat = self.shared_tracking_branch(scalar)
        obstacle_feat = self.shared_obstacle_branch(radar_emb)
        return tracking_feat, obstacle_feat

    def _encode_actor_obs(self, obs):
        actor_obs = obs[:, :NetParameters.ACTOR_RAW_LEN]
        scalar = actor_obs[:, :NetParameters.ACTOR_SCALAR_LEN]
        radar = actor_obs[:, NetParameters.ACTOR_SCALAR_LEN:NetParameters.ACTOR_RAW_LEN]
        return scalar, self._encode_radar(radar)

    def _encode_privileged_obs(self, obs):
        attacker_start = NetParameters.ACTOR_RAW_LEN
        if obs.shape[-1] >= NetParameters.CRITIC_RAW_LEN:
            attacker_scalar = obs[:, attacker_start:attacker_start + NetParameters.PRIVILEGED_SCALAR_LEN]
            attacker_radar = obs[
                :,
                attacker_start + NetParameters.PRIVILEGED_SCALAR_LEN:attacker_start + NetParameters.PRIVILEGED_RAW_LEN,
            ]
        else:
            batch_size = obs.shape[0]
            attacker_scalar = torch.zeros(
                batch_size,
                NetParameters.PRIVILEGED_SCALAR_LEN,
                device=obs.device,
                dtype=obs.dtype,
            )
            attacker_radar = torch.zeros(
                batch_size,
                NetParameters.RADAR_DIM,
                device=obs.device,
                dtype=obs.dtype,
            )
        return attacker_scalar, self._encode_radar(attacker_radar)

    def forward(self, actor_obs, critic_obs):
        actor_scalar, actor_radar_emb = self._encode_actor_obs(actor_obs)
        actor_tracking_feat, actor_obstacle_feat = self._extract_task_features(actor_scalar, actor_radar_emb)
        actor_feat = self.actor_merge_layer(torch.cat([actor_tracking_feat, actor_obstacle_feat], dim=-1))
        mean = self.policy_mean(actor_feat)
        log_std = self.log_std.expand_as(mean)

        defender_scalar, defender_radar_emb = self._encode_actor_obs(critic_obs)
        critic_tracking_feat, critic_obstacle_feat = self._extract_task_features(defender_scalar, defender_radar_emb)
        critic_task_feat = self.critic_task_merge_layer(
            torch.cat([critic_tracking_feat, critic_obstacle_feat], dim=-1)
        )
        attacker_scalar, attacker_radar_emb = self._encode_privileged_obs(critic_obs)
        privileged_feat = self.critic_privileged_branch(torch.cat([attacker_scalar, attacker_radar_emb], dim=-1))
        critic_state = self.critic_backbone(torch.cat([critic_task_feat, privileged_feat], dim=-1))
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
        _, value, _ = self.forward(critic_obs[:, :NetParameters.ACTOR_RAW_LEN], critic_obs)
        return value


__all__ = ["DefenderNetNMNCTDETaskShared"]
