"""Dual-GRU NMN using raw radar for obstacle memory."""

import numpy as np
import torch
import torch.nn as nn

from configs.skill_config import NetParameters

from .common import _NMNStageMixin


class DefenderNetNMNDualGRURaw(_NMNStageMixin, nn.Module):
    """Separate tracking/obstacle GRUs with raw radar obstacle input."""

    def __init__(self, action_dim=None):
        super().__init__()
        self._init_nmn_stage_control()
        self.action_dim = int(NetParameters.ACTION_DIM if action_dim is None else action_dim)
        self.tracking_hidden_dim = int(getattr(NetParameters, "NMN_DUAL_GRU_TRACKING_HIDDEN", NetParameters.NMN_BRANCH_DIM))
        self.obstacle_hidden_dim = int(getattr(NetParameters, "NMN_DUAL_GRU_OBSTACLE_HIDDEN", NetParameters.NMN_BRANCH_DIM))
        if self.tracking_hidden_dim != self.obstacle_hidden_dim:
            raise ValueError("nmn_dual_gru_raw currently requires equal tracking/obstacle hidden dims")
        self.hidden_dim = self.tracking_hidden_dim
        self.obstacle_decay = float(getattr(NetParameters, "NMN_DUAL_GRU_OBSTACLE_DECAY", 0.75))
        self.obstacle_decay = float(np.clip(self.obstacle_decay, 0.0, 1.0))
        self.is_recurrent = True

        self.actor_tracking_gru = nn.GRU(NetParameters.ACTOR_SCALAR_LEN, self.hidden_dim, batch_first=True)
        self.actor_obstacle_gru = nn.GRU(NetParameters.RADAR_DIM, self.hidden_dim, batch_first=True)
        self.actor_merge = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, NetParameters.NMN_MERGED_DIM),
            nn.Tanh(),
        )
        self.policy_mean = nn.Linear(NetParameters.NMN_MERGED_DIM, self.action_dim)
        initial_log_std = float(getattr(NetParameters, "NMN_DUAL_GRU_INITIAL_LOG_STD", 0.0))
        self.log_std = nn.Parameter(torch.full((self.action_dim,), initial_log_std))
        self.min_log_std = float(getattr(NetParameters, "NMN_DUAL_GRU_MIN_LOG_STD", -20.0))
        self.max_log_std = float(getattr(NetParameters, "NMN_DUAL_GRU_MAX_LOG_STD", 2.0))
        if self.min_log_std > self.max_log_std:
            self.min_log_std, self.max_log_std = self.max_log_std, self.min_log_std

        self.critic_tracking_gru = nn.GRU(NetParameters.ACTOR_SCALAR_LEN, self.hidden_dim, batch_first=True)
        self.critic_obstacle_gru = nn.GRU(NetParameters.RADAR_DIM, self.hidden_dim, batch_first=True)
        self.critic_merge = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, NetParameters.NMN_MERGED_DIM),
            nn.Tanh(),
        )
        self.value_head = nn.Linear(NetParameters.NMN_MERGED_DIM, 1)

        self.apply(self._init_weights)
        self._init_policy_head()

    def _bounded_log_std(self):
        return torch.clamp(self.log_std, min=self.min_log_std, max=self.max_log_std)

    def _apply_initial_mean_bias(self):
        initial_mean_bias = getattr(NetParameters, "NMN_DUAL_GRU_INITIAL_MEAN_BIAS", (0.0, 0.0))
        initial_mean_bias = tuple(float(v) for v in initial_mean_bias)
        if len(initial_mean_bias) != self.action_dim:
            initial_mean_bias = tuple(0.0 for _ in range(self.action_dim))
        with torch.no_grad():
            self.policy_mean.bias.copy_(torch.tensor(initial_mean_bias, dtype=self.policy_mean.bias.dtype))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def _init_policy_head(self):
        gain = float(getattr(NetParameters, "NMN_DUAL_GRU_POLICY_HEAD_GAIN", np.sqrt(2)))
        nn.init.orthogonal_(self.policy_mean.weight, gain=gain)
        if self.policy_mean.bias is not None:
            nn.init.constant_(self.policy_mean.bias, 0.0)
        self._apply_initial_mean_bias()

    def recurrent_hidden_spec(self, role: str):
        return (2, self.hidden_dim)

    def _split_actor_obs(self, obs):
        actor_obs = obs[..., :NetParameters.ACTOR_RAW_LEN]
        scalar = actor_obs[..., :NetParameters.ACTOR_SCALAR_LEN]
        radar = actor_obs[..., NetParameters.ACTOR_SCALAR_LEN:NetParameters.ACTOR_RAW_LEN]
        radar = self._prepare_nmn_radar(radar)
        return scalar, radar

    def _split_hidden(self, hidden):
        if hidden is None:
            return None, None
        if hidden.dim() != 3:
            raise ValueError(f"invalid recurrent hidden shape: {tuple(hidden.shape)}")
        if hidden.shape[0] < 2:
            zeros = torch.zeros_like(hidden[:1])
            hidden = torch.cat([hidden[:1], zeros], dim=0)
        return hidden[0:1].contiguous(), hidden[1:2].contiguous()

    @staticmethod
    def _pack_hidden(tracking_hidden, obstacle_hidden):
        if tracking_hidden is None and obstacle_hidden is None:
            return None
        if tracking_hidden is None:
            tracking_hidden = torch.zeros_like(obstacle_hidden)
        if obstacle_hidden is None:
            obstacle_hidden = torch.zeros_like(tracking_hidden)
        return torch.cat([tracking_hidden, obstacle_hidden], dim=0)

    def _unroll_pair(self, tracking_gru, obstacle_gru, scalar_seq, radar_seq, dones=None, hidden=None):
        bsz, tlen, _ = scalar_seq.shape
        tracking_hidden, obstacle_hidden = self._split_hidden(hidden)
        tracking_outputs = []
        obstacle_outputs = []
        dones = None if dones is None else dones.float()

        for t in range(tlen):
            if t > 0 and dones is not None:
                keep = (1.0 - dones[:, t - 1]).view(1, bsz, 1)
                if tracking_hidden is not None:
                    tracking_hidden = tracking_hidden * keep
                if obstacle_hidden is not None:
                    obstacle_hidden = obstacle_hidden * keep
            if obstacle_hidden is not None:
                obstacle_hidden = obstacle_hidden * self.obstacle_decay

            tracking_out, tracking_hidden = tracking_gru(scalar_seq[:, t:t + 1, :], tracking_hidden)
            obstacle_out, obstacle_hidden = obstacle_gru(radar_seq[:, t:t + 1, :], obstacle_hidden)
            tracking_outputs.append(tracking_out)
            obstacle_outputs.append(obstacle_out)

        return (
            torch.cat(tracking_outputs, dim=1),
            torch.cat(obstacle_outputs, dim=1),
            self._pack_hidden(tracking_hidden, obstacle_hidden),
        )

    def forward_sequence(self, actor_obs_seq, critic_obs_seq, dones=None, actor_hidden=None, critic_hidden=None):
        actor_scalar, actor_radar = self._split_actor_obs(actor_obs_seq)
        critic_scalar, critic_radar = self._split_actor_obs(critic_obs_seq)

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
        critic_scalar, critic_radar = self._split_actor_obs(critic_obs)

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

    def forward(self, actor_obs, critic_obs):
        if actor_obs.dim() == 3 and critic_obs.dim() == 3:
            return self.forward_sequence(actor_obs, critic_obs, dones=None)
        mean, value, log_std, _, _ = self.forward_recurrent(actor_obs, critic_obs)
        return mean, value, log_std

    def act_recurrent(self, actor_obs, critic_obs, actor_hidden=None, critic_hidden=None):
        mean, value, log_std, next_actor_hidden, next_critic_hidden = self.forward_recurrent(
            actor_obs,
            critic_obs,
            actor_hidden=actor_hidden,
            critic_hidden=critic_hidden,
        )
        std = torch.exp(log_std)
        pre_tanh = mean + torch.randn_like(mean) * std
        action = torch.tanh(pre_tanh)
        log_det_jac = torch.log(1.0 - torch.tanh(pre_tanh) ** 2 + 1e-6)
        log_prob = (torch.distributions.Normal(mean, std).log_prob(pre_tanh) - log_det_jac).sum(dim=-1)
        return action, log_prob, pre_tanh, value, next_actor_hidden, next_critic_hidden

    def act(self, actor_obs, critic_obs):
        action, log_prob, pre_tanh, value, _, _ = self.act_recurrent(actor_obs, critic_obs)
        return action, log_prob, pre_tanh, value

    def critic_value_recurrent(self, critic_obs, critic_hidden=None):
        if critic_obs.dim() == 1:
            critic_obs = critic_obs.unsqueeze(0)
        critic_scalar, critic_radar = self._split_actor_obs(critic_obs)
        critic_tracking_hidden, critic_obstacle_hidden = self._split_hidden(critic_hidden)
        if critic_obstacle_hidden is not None:
            critic_obstacle_hidden = critic_obstacle_hidden * self.obstacle_decay
        critic_tracking, next_critic_tracking = self.critic_tracking_gru(critic_scalar.unsqueeze(1), critic_tracking_hidden)
        critic_obstacle, next_critic_obstacle = self.critic_obstacle_gru(critic_radar.unsqueeze(1), critic_obstacle_hidden)
        critic_state = self.critic_merge(torch.cat([critic_tracking.squeeze(1), critic_obstacle.squeeze(1)], dim=-1))
        value = self.value_head(critic_state)
        next_hidden = self._pack_hidden(next_critic_tracking, next_critic_obstacle)
        return value, next_hidden

    def critic_value(self, critic_obs):
        value, _ = self.critic_value_recurrent(critic_obs, critic_hidden=None)
        return value


__all__ = ["DefenderNetNMNDualGRURaw"]
