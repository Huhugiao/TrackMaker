"""Top-level HRL GRU policy."""

import numpy as np
import torch
import torch.nn as nn

from configs.skill_config import NetParameters

from .classifier_heads import HRLTopSkillClassifier, HRLTopSkillClassifierWithContext
from .common import _unroll_gru_with_dones
from .radar_encoder import RadarEncoder


class DefenderNetHRLTopGRU(nn.Module):
    """HRL top-level GRU actor-critic with a CTDE critic."""

    def __init__(self, action_dim=None):
        super().__init__()
        self.hidden_dim = int(NetParameters.HIDDEN_DIM)
        self.action_dim = int(NetParameters.ACTION_DIM if action_dim is None else action_dim)
        self.is_recurrent = True

        self.radar_encoder = RadarEncoder()

        self.actor_in_proj = nn.Linear(NetParameters.ACTOR_VECTOR_LEN, self.hidden_dim)
        self.actor_gru = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.policy_mean = nn.Linear(self.hidden_dim, self.action_dim)
        self.skill_classifier = HRLTopSkillClassifier(self.hidden_dim, 2)
        self.skill_classifier_with_context = HRLTopSkillClassifierWithContext(
            self.hidden_dim,
            context_dim=6,
            num_skills=2,
        )
        self.privileged_classifier = HRLTopSkillClassifier(self.hidden_dim, 2)
        self.register_buffer("use_skill_classifier_for_action", torch.zeros((), dtype=torch.bool))
        self.register_buffer("use_contextual_skill_head_for_action", torch.zeros((), dtype=torch.bool))
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

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        legacy_prefix = prefix + "privileged_probe."
        current_prefix = prefix + "privileged_classifier."
        for key in list(state_dict.keys()):
            if key.startswith(legacy_prefix):
                mapped_key = current_prefix + key[len(legacy_prefix):]
                state_dict[mapped_key] = state_dict.pop(key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _encode_observation(self, obs, is_critic=False):
        if obs.dim() == 2:
            obs_flat = obs
            prefix_shape = obs.shape[:-1]
        elif obs.dim() == 3:
            prefix_shape = obs.shape[:-1]
            obs_flat = obs.reshape(-1, obs.shape[-1])
        else:
            raise ValueError(f"Invalid obs shape: {tuple(obs.shape)}")

        if is_critic:
            defender_end = NetParameters.ACTOR_RAW_LEN
            defender_scalar = obs_flat[:, :NetParameters.ACTOR_SCALAR_LEN]
            defender_radar = obs_flat[:, NetParameters.ACTOR_SCALAR_LEN:defender_end]
            defender_radar_emb = self.radar_encoder(defender_radar)
            defender_part = torch.cat([defender_scalar, defender_radar_emb], dim=-1)

            attacker_start = defender_end
            attacker_scalar = obs_flat[:, attacker_start:attacker_start + NetParameters.PRIVILEGED_SCALAR_LEN]
            attacker_radar = obs_flat[:, attacker_start + NetParameters.PRIVILEGED_SCALAR_LEN:]
            attacker_radar_emb = self.radar_encoder(attacker_radar)
            attacker_part = torch.cat([attacker_scalar, attacker_radar_emb], dim=-1)
            encoded = torch.cat([defender_part, attacker_part], dim=-1)
        else:
            actor_flat = obs_flat[:, :NetParameters.ACTOR_RAW_LEN]
            scalar = actor_flat[:, :NetParameters.ACTOR_SCALAR_LEN]
            radar = actor_flat[:, NetParameters.ACTOR_SCALAR_LEN:NetParameters.ACTOR_RAW_LEN]
            radar_emb = self.radar_encoder(radar)
            encoded = torch.cat([scalar, radar_emb], dim=-1)

        return encoded.view(*prefix_shape, -1)

    def _run_recurrent_cores(
        self,
        actor_x,
        critic_x,
        dones=None,
        actor_hidden=None,
        critic_hidden=None,
    ):
        if dones is None:
            return self.actor_gru(actor_x, actor_hidden)[0], self.critic_gru(critic_x, critic_hidden)[0]

        actor_out = _unroll_gru_with_dones(
            self.actor_gru,
            actor_x,
            dones,
            initial_hidden=actor_hidden,
        )
        critic_out = _unroll_gru_with_dones(
            self.critic_gru,
            critic_x,
            dones,
            initial_hidden=critic_hidden,
        )
        return actor_out, critic_out

    def set_skill_classifier_for_action(self, enabled: bool):
        self.use_skill_classifier_for_action.fill_(bool(enabled))

    def set_contextual_skill_head_for_action(self, enabled: bool):
        self.use_contextual_skill_head_for_action.fill_(bool(enabled))

    def classify_skills(self, actor_feat):
        return self.skill_classifier(actor_feat)

    def classify_skills_with_context(self, actor_feat, behavior_context):
        return self.skill_classifier_with_context(actor_feat, behavior_context)

    def classify_skills_privileged(self, critic_feat):
        return self.privileged_classifier(critic_feat)

    def _maybe_inject_skill_logits(self, mean, actor_feat):
        skill_dim = int(getattr(NetParameters, "HRL_NUM_SKILLS", 2))
        if self.action_dim > (skill_dim + 1):
            return mean
        if self.action_dim < 3 or not bool(self.use_skill_classifier_for_action.item()):
            return mean
        mean = mean.clone()
        mean[..., :2] = self.classify_skills(actor_feat)
        return mean

    def _project_actor_critic_inputs(self, actor_obs, critic_obs):
        actor_in = self._encode_observation(actor_obs, is_critic=False)
        critic_in = self._encode_observation(critic_obs, is_critic=True)
        actor_x = torch.tanh(self.actor_in_proj(actor_in))
        critic_x = torch.tanh(self.critic_in_proj(critic_in))
        return actor_x, critic_x

    def forward_sequence(
        self,
        actor_obs_seq,
        critic_obs_seq,
        dones=None,
        actor_hidden=None,
        critic_hidden=None,
    ):
        actor_x, critic_x = self._project_actor_critic_inputs(actor_obs_seq, critic_obs_seq)
        actor_out, critic_out = self._run_recurrent_cores(
            actor_x,
            critic_x,
            dones=dones,
            actor_hidden=actor_hidden,
            critic_hidden=critic_hidden,
        )
        mean = self.policy_mean(actor_out)
        mean = self._maybe_inject_skill_logits(mean, actor_out)
        value = self.value_head(critic_out)
        log_std = self.log_std.view(1, 1, -1).expand_as(mean)
        return mean, value, log_std

    def forward_sequence_with_features(
        self,
        actor_obs_seq,
        critic_obs_seq,
        dones=None,
        actor_hidden=None,
        critic_hidden=None,
    ):
        actor_x, critic_x = self._project_actor_critic_inputs(actor_obs_seq, critic_obs_seq)
        actor_out, critic_out = self._run_recurrent_cores(
            actor_x,
            critic_x,
            dones=dones,
            actor_hidden=actor_hidden,
            critic_hidden=critic_hidden,
        )
        mean = self.policy_mean(actor_out)
        mean = self._maybe_inject_skill_logits(mean, actor_out)
        value = self.value_head(critic_out)
        log_std = self.log_std.view(1, 1, -1).expand_as(mean)
        return mean, value, log_std, actor_out, critic_out

    def forward_recurrent(self, actor_obs, critic_obs, actor_hidden=None, critic_hidden=None):
        actor_x, critic_x = self._project_actor_critic_inputs(actor_obs, critic_obs)
        actor_out, next_actor_hidden = self.actor_gru(actor_x.unsqueeze(1), actor_hidden)
        critic_out, next_critic_hidden = self.critic_gru(critic_x.unsqueeze(1), critic_hidden)

        actor_feat = actor_out.squeeze(1)
        critic_feat = critic_out.squeeze(1)
        mean = self.policy_mean(actor_feat)
        mean = self._maybe_inject_skill_logits(mean, actor_feat)
        value = self.value_head(critic_feat)
        log_std = self.log_std.unsqueeze(0).expand_as(mean)
        return mean, value, log_std, next_actor_hidden, next_critic_hidden

    def forward_recurrent_with_features(self, actor_obs, critic_obs, actor_hidden=None, critic_hidden=None):
        actor_x, critic_x = self._project_actor_critic_inputs(actor_obs, critic_obs)
        actor_out, next_actor_hidden = self.actor_gru(actor_x.unsqueeze(1), actor_hidden)
        critic_out, next_critic_hidden = self.critic_gru(critic_x.unsqueeze(1), critic_hidden)

        actor_feat = actor_out.squeeze(1)
        critic_feat = critic_out.squeeze(1)
        mean = self.policy_mean(actor_feat)
        mean = self._maybe_inject_skill_logits(mean, actor_feat)
        value = self.value_head(critic_feat)
        log_std = self.log_std.unsqueeze(0).expand_as(mean)
        return mean, value, log_std, actor_feat, critic_feat, next_actor_hidden, next_critic_hidden

    def forward_privileged_recurrent(self, critic_obs, critic_hidden=None):
        critic_in = self._encode_observation(critic_obs, is_critic=True)
        critic_x = torch.tanh(self.critic_in_proj(critic_in)).unsqueeze(1)
        critic_out, next_critic_hidden = self.critic_gru(critic_x, critic_hidden)
        critic_feat = critic_out.squeeze(1)
        skill_logits = self.classify_skills_privileged(critic_feat)
        value = self.value_head(critic_feat)
        return skill_logits, value, critic_feat, next_critic_hidden

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

    def forward(self, actor_obs, critic_obs):
        if actor_obs.dim() == 3 and critic_obs.dim() == 3:
            return self.forward_sequence(actor_obs, critic_obs, dones=None)

        if actor_obs.dim() == 1:
            actor_obs = actor_obs.unsqueeze(0)
        if critic_obs.dim() == 1:
            critic_obs = critic_obs.unsqueeze(0)
        mean, value, log_std, _, _ = self.forward_recurrent(actor_obs, critic_obs)
        return mean, value, log_std

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


__all__ = ["DefenderNetHRLTopGRU"]
