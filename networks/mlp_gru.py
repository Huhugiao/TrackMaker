"""GRU-based MLP actor-critic network."""

import numpy as np
import torch
import torch.nn as nn

from configs.skill_config import NetParameters

from .common import _unroll_gru_with_dones
from .radar_encoder import RadarEncoder


class DefenderNetMLPGRU(nn.Module):
    """Defender actor-critic network with GRU actor and CTDE critic."""

    def __init__(self, action_dim=None):
        super().__init__()
        self.hidden_dim = int(NetParameters.HIDDEN_DIM)
        self.action_dim = int(NetParameters.ACTION_DIM if action_dim is None else action_dim)
        self.is_recurrent = True

        self.radar_encoder = RadarEncoder()
        self.actor_in_proj = nn.Linear(NetParameters.ACTOR_VECTOR_LEN, self.hidden_dim)
        self.actor_gru = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.policy_mean = nn.Linear(self.hidden_dim, self.action_dim)
        self.log_std = nn.Parameter(torch.zeros(self.action_dim))

        self.critic_in_proj = nn.Linear(NetParameters.CRITIC_VECTOR_LEN, self.hidden_dim)
        self.critic_gru = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.value_head = nn.Linear(self.hidden_dim, 1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def _encode_observation(self, obs, is_critic=False):
        if is_critic:
            defender_obs = obs[..., :NetParameters.ACTOR_RAW_LEN]
            defender_scalar = defender_obs[..., :NetParameters.ACTOR_SCALAR_LEN]
            defender_radar = defender_obs[..., NetParameters.ACTOR_SCALAR_LEN:NetParameters.ACTOR_RAW_LEN]
            defender_radar_emb = self.radar_encoder(defender_radar)
            defender_part = torch.cat([defender_scalar, defender_radar_emb], dim=-1)

            attacker_start = NetParameters.ACTOR_RAW_LEN
            attacker_scalar = obs[..., attacker_start:attacker_start + NetParameters.PRIVILEGED_SCALAR_LEN]
            attacker_radar = obs[..., attacker_start + NetParameters.PRIVILEGED_SCALAR_LEN:]
            attacker_radar_emb = self.radar_encoder(attacker_radar)
            attacker_part = torch.cat([attacker_scalar, attacker_radar_emb], dim=-1)
            return torch.cat([defender_part, attacker_part], dim=-1)

        actor_obs = obs[..., :NetParameters.ACTOR_RAW_LEN]
        scalar = actor_obs[..., :NetParameters.ACTOR_SCALAR_LEN]
        radar = actor_obs[..., NetParameters.ACTOR_SCALAR_LEN:NetParameters.ACTOR_RAW_LEN]
        radar_emb = self.radar_encoder(radar)
        return torch.cat([scalar, radar_emb], dim=-1)

    def forward_sequence(self, actor_obs_seq, critic_obs_seq, dones=None, actor_hidden=None, critic_hidden=None):
        actor_in = self._encode_observation(actor_obs_seq, is_critic=False)
        critic_in = self._encode_observation(critic_obs_seq, is_critic=True)
        actor_x = torch.tanh(self.actor_in_proj(actor_in))
        critic_x = torch.tanh(self.critic_in_proj(critic_in))

        if dones is None:
            actor_out, _ = self.actor_gru(actor_x, actor_hidden)
            critic_out, _ = self.critic_gru(critic_x, critic_hidden)
        else:
            actor_out = _unroll_gru_with_dones(self.actor_gru, actor_x, dones, initial_hidden=actor_hidden)
            critic_out = _unroll_gru_with_dones(self.critic_gru, critic_x, dones, initial_hidden=critic_hidden)

        mean = self.policy_mean(actor_out)
        value = self.value_head(critic_out)
        log_std = self.log_std.view(1, 1, -1).expand_as(mean)
        return mean, value, log_std

    def forward_recurrent(self, actor_obs, critic_obs, actor_hidden=None, critic_hidden=None):
        actor_in = self._encode_observation(actor_obs, is_critic=False)
        critic_in = self._encode_observation(critic_obs, is_critic=True)
        actor_x = torch.tanh(self.actor_in_proj(actor_in)).unsqueeze(1)
        critic_x = torch.tanh(self.critic_in_proj(critic_in)).unsqueeze(1)

        actor_out, next_actor_hidden = self.actor_gru(actor_x, actor_hidden)
        critic_out, next_critic_hidden = self.critic_gru(critic_x, critic_hidden)
        actor_state = actor_out.squeeze(1)
        critic_state = critic_out.squeeze(1)
        mean = self.policy_mean(actor_state)
        value = self.value_head(critic_state)
        log_std = self.log_std.unsqueeze(0).expand_as(mean)
        return mean, value, log_std, next_actor_hidden, next_critic_hidden

    def forward(self, actor_obs, critic_obs):
        if actor_obs.dim() == 3 and critic_obs.dim() == 3:
            return self.forward_sequence(actor_obs, critic_obs, dones=None)
        if actor_obs.dim() == 1:
            actor_obs = actor_obs.unsqueeze(0)
        if critic_obs.dim() == 1:
            critic_obs = critic_obs.unsqueeze(0)
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
        eps = torch.randn_like(mean)
        pre_tanh = mean + eps * std
        action = torch.tanh(pre_tanh)
        log_det_jac = torch.log(1.0 - torch.tanh(pre_tanh) ** 2 + 1e-6)
        log_prob = (torch.distributions.Normal(mean, std).log_prob(pre_tanh) - log_det_jac).sum(dim=-1)
        return action, log_prob, pre_tanh, value, next_actor_hidden, next_critic_hidden

    def act(self, actor_obs, critic_obs):
        action, log_prob, pre_tanh, value, _, _ = self.act_recurrent(actor_obs, critic_obs)
        return action, log_prob, pre_tanh, value

    def critic_value_recurrent(self, critic_obs, critic_hidden=None):
        critic_in = self._encode_observation(critic_obs, is_critic=True)
        critic_x = torch.tanh(self.critic_in_proj(critic_in)).unsqueeze(1)
        critic_out, next_hidden = self.critic_gru(critic_x, critic_hidden)
        value = self.value_head(critic_out.squeeze(1))
        return value, next_hidden

    def critic_value(self, critic_obs):
        if critic_obs.dim() == 1:
            critic_obs = critic_obs.unsqueeze(0)
        value, _ = self.critic_value_recurrent(critic_obs, critic_hidden=None)
        return value


__all__ = ["DefenderNetMLPGRU"]
