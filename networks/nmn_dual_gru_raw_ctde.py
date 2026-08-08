"""Dual-GRU NMN with raw radar actor and CTDE recurrent critic."""

import torch
import torch.nn as nn

from configs.skill_config import NetParameters

from .nmn_dual_gru_raw import DefenderNetNMNDualGRURaw


class DefenderNetNMNDualGRURawCTDE(DefenderNetNMNDualGRURaw):
    """Actor is partial-observation dual-GRU; critic uses defender + privileged attacker obs."""

    def __init__(self, action_dim=None):
        super().__init__(action_dim=action_dim)
        critic_scalar_dim = NetParameters.ACTOR_SCALAR_LEN + NetParameters.PRIVILEGED_SCALAR_LEN
        critic_radar_dim = NetParameters.RADAR_DIM * 2
        self.critic_tracking_gru = nn.GRU(critic_scalar_dim, self.hidden_dim, batch_first=True)
        self.critic_obstacle_gru = nn.GRU(critic_radar_dim, self.hidden_dim, batch_first=True)
        self._init_gru(self.critic_tracking_gru)
        self._init_gru(self.critic_obstacle_gru)

    @staticmethod
    def _init_gru(gru):
        for name, param in gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

    def _split_critic_obs(self, obs):
        defender = obs[..., :NetParameters.ACTOR_RAW_LEN]
        attacker_start = NetParameters.ACTOR_RAW_LEN
        attacker_end = attacker_start + NetParameters.PRIVILEGED_RAW_LEN
        attacker = obs[..., attacker_start:attacker_end]

        defender_scalar = defender[..., :NetParameters.ACTOR_SCALAR_LEN]
        defender_radar = defender[..., NetParameters.ACTOR_SCALAR_LEN:NetParameters.ACTOR_RAW_LEN]
        attacker_scalar = attacker[..., :NetParameters.PRIVILEGED_SCALAR_LEN]
        attacker_radar = attacker[..., NetParameters.PRIVILEGED_SCALAR_LEN:NetParameters.PRIVILEGED_RAW_LEN]

        defender_radar = self._prepare_nmn_radar(defender_radar)
        attacker_radar = self._prepare_nmn_radar(attacker_radar)
        scalar = torch.cat([defender_scalar, attacker_scalar], dim=-1)
        radar = torch.cat([defender_radar, attacker_radar], dim=-1)
        return scalar, radar

    def forward_sequence(self, actor_obs_seq, critic_obs_seq, dones=None, actor_hidden=None, critic_hidden=None):
        actor_scalar, actor_radar = self._split_actor_obs(actor_obs_seq)
        critic_scalar, critic_radar = self._split_critic_obs(critic_obs_seq)

        actor_tracking, actor_obstacle, _ = self._unroll_pair(
            self.actor_tracking_gru,
            self.actor_obstacle_gru,
            actor_scalar,
            actor_radar,
            dones=dones,
            hidden=actor_hidden,
        )
        critic_tracking, critic_obstacle, _ = self._unroll_pair(
            self.critic_tracking_gru,
            self.critic_obstacle_gru,
            critic_scalar,
            critic_radar,
            dones=dones,
            hidden=critic_hidden,
        )

        actor_state = self.actor_merge(torch.cat([actor_tracking, actor_obstacle], dim=-1))
        critic_state = self.critic_merge(torch.cat([critic_tracking, critic_obstacle], dim=-1))
        mean = self.policy_mean(actor_state)
        value = self.value_head(critic_state)
        log_std = self._bounded_log_std().view(1, 1, -1).expand_as(mean)
        return mean, value, log_std

    def forward_recurrent(self, actor_obs, critic_obs, actor_hidden=None, critic_hidden=None):
        if actor_obs.dim() == 1:
            actor_obs = actor_obs.unsqueeze(0)
        if critic_obs.dim() == 1:
            critic_obs = critic_obs.unsqueeze(0)

        actor_scalar, actor_radar = self._split_actor_obs(actor_obs)
        critic_scalar, critic_radar = self._split_critic_obs(critic_obs)

        actor_tracking_hidden, actor_obstacle_hidden = self._split_hidden(actor_hidden)
        critic_tracking_hidden, critic_obstacle_hidden = self._split_hidden(critic_hidden)
        if actor_obstacle_hidden is not None:
            actor_obstacle_hidden = actor_obstacle_hidden * self.obstacle_decay
        if critic_obstacle_hidden is not None:
            critic_obstacle_hidden = critic_obstacle_hidden * self.obstacle_decay

        actor_tracking, next_actor_tracking = self.actor_tracking_gru(actor_scalar.unsqueeze(1), actor_tracking_hidden)
        actor_obstacle, next_actor_obstacle = self.actor_obstacle_gru(actor_radar.unsqueeze(1), actor_obstacle_hidden)
        critic_tracking, next_critic_tracking = self.critic_tracking_gru(critic_scalar.unsqueeze(1), critic_tracking_hidden)
        critic_obstacle, next_critic_obstacle = self.critic_obstacle_gru(critic_radar.unsqueeze(1), critic_obstacle_hidden)

        actor_state = self.actor_merge(torch.cat([actor_tracking.squeeze(1), actor_obstacle.squeeze(1)], dim=-1))
        critic_state = self.critic_merge(torch.cat([critic_tracking.squeeze(1), critic_obstacle.squeeze(1)], dim=-1))
        mean = self.policy_mean(actor_state)
        value = self.value_head(critic_state)
        log_std = self._bounded_log_std().unsqueeze(0).expand_as(mean)
        next_actor_hidden = self._pack_hidden(next_actor_tracking, next_actor_obstacle)
        next_critic_hidden = self._pack_hidden(next_critic_tracking, next_critic_obstacle)
        return mean, value, log_std, next_actor_hidden, next_critic_hidden

    def critic_value_recurrent(self, critic_obs, critic_hidden=None):
        if critic_obs.dim() == 1:
            critic_obs = critic_obs.unsqueeze(0)
        critic_scalar, critic_radar = self._split_critic_obs(critic_obs)
        critic_tracking_hidden, critic_obstacle_hidden = self._split_hidden(critic_hidden)
        if critic_obstacle_hidden is not None:
            critic_obstacle_hidden = critic_obstacle_hidden * self.obstacle_decay
        critic_tracking, next_critic_tracking = self.critic_tracking_gru(critic_scalar.unsqueeze(1), critic_tracking_hidden)
        critic_obstacle, next_critic_obstacle = self.critic_obstacle_gru(critic_radar.unsqueeze(1), critic_obstacle_hidden)
        critic_state = self.critic_merge(torch.cat([critic_tracking.squeeze(1), critic_obstacle.squeeze(1)], dim=-1))
        value = self.value_head(critic_state)
        next_hidden = self._pack_hidden(next_critic_tracking, next_critic_obstacle)
        return value, next_hidden


__all__ = ["DefenderNetNMNDualGRURawCTDE"]
