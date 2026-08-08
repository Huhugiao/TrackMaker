"""Top-level HRL dual-GRU policy using raw radar for obstacle memory."""

import numpy as np
import torch
import torch.nn as nn

from configs.skill_config import NetParameters

from .classifier_heads import HRLTopSkillClassifier, HRLTopSkillClassifierWithContext


class DefenderNetHRLTopDualGRURaw(nn.Module):
    """HRL top actor-critic with separate task and obstacle recurrent cores."""

    def __init__(self, action_dim=None):
        super().__init__()
        self.hidden_dim = int(NetParameters.HIDDEN_DIM)
        self.branch_hidden_dim = int(
            getattr(NetParameters, "HRL_TOP_DUAL_GRU_BRANCH_HIDDEN", NetParameters.NMN_DUAL_GRU_TRACKING_HIDDEN)
        )
        self.obstacle_decay = float(
            getattr(NetParameters, "HRL_TOP_DUAL_GRU_OBSTACLE_DECAY", NetParameters.NMN_DUAL_GRU_OBSTACLE_DECAY)
        )
        self.obstacle_decay = float(np.clip(self.obstacle_decay, 0.0, 1.0))
        self.action_dim = int(NetParameters.ACTION_DIM if action_dim is None else action_dim)
        self.is_recurrent = True

        self.actor_task_gru = nn.GRU(NetParameters.ACTOR_SCALAR_LEN, self.branch_hidden_dim, batch_first=True)
        self.actor_obstacle_gru = nn.GRU(NetParameters.RADAR_DIM, self.branch_hidden_dim, batch_first=True)
        self.actor_merge = nn.Sequential(
            nn.Linear(self.branch_hidden_dim * 2, self.hidden_dim),
            nn.Tanh(),
        )
        self.policy_mean = nn.Linear(self.hidden_dim, self.action_dim)
        self.skill_classifier = HRLTopSkillClassifier(self.hidden_dim, 2)
        self.skill_classifier_with_context = HRLTopSkillClassifierWithContext(
            self.hidden_dim,
            context_dim=6,
            num_skills=2,
        )
        self.register_buffer("use_skill_classifier_for_action", torch.zeros((), dtype=torch.bool))
        self.register_buffer("use_contextual_skill_head_for_action", torch.zeros((), dtype=torch.bool))
        self.log_std = nn.Parameter(torch.zeros(self.action_dim))

        critic_task_dim = int(NetParameters.ACTOR_SCALAR_LEN + NetParameters.PRIVILEGED_SCALAR_LEN)
        critic_obstacle_dim = int(NetParameters.RADAR_DIM * 2)
        self.critic_task_gru = nn.GRU(critic_task_dim, self.branch_hidden_dim, batch_first=True)
        self.critic_obstacle_gru = nn.GRU(critic_obstacle_dim, self.branch_hidden_dim, batch_first=True)
        self.critic_merge = nn.Sequential(
            nn.Linear(self.branch_hidden_dim * 2, self.hidden_dim),
            nn.Tanh(),
        )
        self.value_head = nn.Linear(self.hidden_dim, 1)
        self.privileged_classifier = HRLTopSkillClassifier(self.hidden_dim, 2)

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

    def recurrent_hidden_spec(self, _role: str):
        return (2, self.branch_hidden_dim)

    def _split_actor_obs(self, obs):
        actor_obs = obs[..., :NetParameters.ACTOR_RAW_LEN]
        scalar = actor_obs[..., :NetParameters.ACTOR_SCALAR_LEN]
        radar = actor_obs[..., NetParameters.ACTOR_SCALAR_LEN:NetParameters.ACTOR_RAW_LEN]
        return scalar, radar

    def _split_critic_obs(self, obs):
        defender_obs = obs[..., :NetParameters.ACTOR_RAW_LEN]
        defender_scalar = defender_obs[..., :NetParameters.ACTOR_SCALAR_LEN]
        defender_radar = defender_obs[..., NetParameters.ACTOR_SCALAR_LEN:NetParameters.ACTOR_RAW_LEN]

        attacker_start = NetParameters.ACTOR_RAW_LEN
        attacker_scalar = obs[..., attacker_start:attacker_start + NetParameters.PRIVILEGED_SCALAR_LEN]
        attacker_radar = obs[..., attacker_start + NetParameters.PRIVILEGED_SCALAR_LEN:]

        task = torch.cat([defender_scalar, attacker_scalar], dim=-1)
        obstacle = torch.cat([defender_radar, attacker_radar], dim=-1)
        return task, obstacle

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
    def _pack_hidden(task_hidden, obstacle_hidden):
        if task_hidden is None and obstacle_hidden is None:
            return None
        if task_hidden is None:
            task_hidden = torch.zeros_like(obstacle_hidden)
        if obstacle_hidden is None:
            obstacle_hidden = torch.zeros_like(task_hidden)
        return torch.cat([task_hidden, obstacle_hidden], dim=0)

    def _unroll_pair(self, task_gru, obstacle_gru, task_seq, obstacle_seq, dones=None, hidden=None):
        bsz, tlen, _ = task_seq.shape
        task_hidden, obstacle_hidden = self._split_hidden(hidden)
        task_outputs = []
        obstacle_outputs = []
        dones = None if dones is None else dones.float()

        for t in range(tlen):
            if t > 0 and dones is not None:
                keep = (1.0 - dones[:, t - 1]).view(1, bsz, 1)
                if task_hidden is not None:
                    task_hidden = task_hidden * keep
                if obstacle_hidden is not None:
                    obstacle_hidden = obstacle_hidden * keep
            if obstacle_hidden is not None:
                obstacle_hidden = obstacle_hidden * self.obstacle_decay

            task_out, task_hidden = task_gru(task_seq[:, t:t + 1, :], task_hidden)
            obstacle_out, obstacle_hidden = obstacle_gru(obstacle_seq[:, t:t + 1, :], obstacle_hidden)
            task_outputs.append(task_out)
            obstacle_outputs.append(obstacle_out)

        return (
            torch.cat(task_outputs, dim=1),
            torch.cat(obstacle_outputs, dim=1),
            self._pack_hidden(task_hidden, obstacle_hidden),
        )

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

    @staticmethod
    def _behavior_context_from_actor_obs(actor_obs_seq):
        dist = actor_obs_seq[..., 0]
        bearing = actor_obs_seq[..., 1]
        visible = actor_obs_seq[..., 3]
        unobserved = actor_obs_seq[..., 4]
        target_dist = actor_obs_seq[..., 5]
        target_bearing = actor_obs_seq[..., 6]

        delta_dist = torch.zeros_like(dist)
        delta_bearing = torch.zeros_like(bearing)
        if actor_obs_seq.shape[-2] > 1:
            delta_dist[..., 1:] = dist[..., 1:] - dist[..., :-1]
            raw_delta = bearing[..., 1:] - bearing[..., :-1]
            delta_bearing[..., 1:] = torch.remainder(raw_delta + 1.0, 2.0) - 1.0

        defender_attacker = torch.clamp(0.5 * (dist + 1.0), 0.0, 1.0)
        defender_target = torch.clamp(0.5 * (target_dist + 1.0), 0.0, 1.0)
        rel_bearing = (bearing - target_bearing) * np.pi
        attacker_target_sq = (
            defender_attacker.pow(2)
            + defender_target.pow(2)
            - 2.0 * defender_attacker * defender_target * torch.cos(rel_bearing)
        )
        attacker_target = torch.sqrt(attacker_target_sq.clamp_min(0.0))
        urgency_base = torch.sigmoid(4.0 * (defender_target - attacker_target))
        uncertainty = (1.0 - visible).clamp(0.0, 1.0) * torch.clamp(0.5 * (unobserved + 1.0), 0.0, 1.0)
        urgency = torch.clamp(urgency_base + 0.35 * uncertainty * (1.0 - urgency_base), 0.0, 1.0)

        return torch.stack(
            [
                delta_dist,
                delta_bearing,
                torch.abs(delta_dist),
                torch.abs(delta_bearing),
                visible,
                urgency,
            ],
            dim=-1,
        )

    def _maybe_inject_skill_logits_with_context(self, mean, actor_feat, actor_obs_seq):
        skill_dim = int(getattr(NetParameters, "HRL_NUM_SKILLS", 2))
        if self.action_dim > (skill_dim + 1):
            return mean
        if self.action_dim < 3:
            return mean
        if bool(self.use_contextual_skill_head_for_action.item()):
            context = self._behavior_context_from_actor_obs(actor_obs_seq)
            if mean.dim() == 2 and context.dim() == 3:
                context = context[:, -1, :]
            mean = mean.clone()
            mean[..., :2] = self.classify_skills_with_context(actor_feat, context)
            return mean
        return self._maybe_inject_skill_logits(mean, actor_feat)

    def _maybe_inject_skill_logits(self, mean, actor_feat):
        skill_dim = int(getattr(NetParameters, "HRL_NUM_SKILLS", 2))
        if self.action_dim > (skill_dim + 1):
            return mean
        if self.action_dim < 3 or not bool(self.use_skill_classifier_for_action.item()):
            return mean
        mean = mean.clone()
        mean[..., :2] = self.classify_skills(actor_feat)
        return mean

    def _actor_sequence_features(self, actor_obs_seq, dones=None, actor_hidden=None):
        actor_task, actor_obstacle = self._split_actor_obs(actor_obs_seq)
        task_out, obstacle_out, next_hidden = self._unroll_pair(
            self.actor_task_gru,
            self.actor_obstacle_gru,
            actor_task,
            actor_obstacle,
            dones=dones,
            hidden=actor_hidden,
        )
        return self.actor_merge(torch.cat([task_out, obstacle_out], dim=-1)), next_hidden

    def _critic_sequence_features(self, critic_obs_seq, dones=None, critic_hidden=None):
        critic_task, critic_obstacle = self._split_critic_obs(critic_obs_seq)
        task_out, obstacle_out, next_hidden = self._unroll_pair(
            self.critic_task_gru,
            self.critic_obstacle_gru,
            critic_task,
            critic_obstacle,
            dones=dones,
            hidden=critic_hidden,
        )
        return self.critic_merge(torch.cat([task_out, obstacle_out], dim=-1)), next_hidden

    def _actor_step_features(self, actor_obs, actor_hidden=None):
        if actor_obs.dim() == 1:
            actor_obs = actor_obs.unsqueeze(0)
        actor_task, actor_obstacle = self._split_actor_obs(actor_obs)
        task_hidden, obstacle_hidden = self._split_hidden(actor_hidden)
        if obstacle_hidden is not None:
            obstacle_hidden = obstacle_hidden * self.obstacle_decay
        task_out, next_task = self.actor_task_gru(actor_task.unsqueeze(1), task_hidden)
        obstacle_out, next_obstacle = self.actor_obstacle_gru(actor_obstacle.unsqueeze(1), obstacle_hidden)
        feat = self.actor_merge(torch.cat([task_out.squeeze(1), obstacle_out.squeeze(1)], dim=-1))
        return feat, self._pack_hidden(next_task, next_obstacle)

    def _critic_step_features(self, critic_obs, critic_hidden=None):
        if critic_obs.dim() == 1:
            critic_obs = critic_obs.unsqueeze(0)
        critic_task, critic_obstacle = self._split_critic_obs(critic_obs)
        task_hidden, obstacle_hidden = self._split_hidden(critic_hidden)
        if obstacle_hidden is not None:
            obstacle_hidden = obstacle_hidden * self.obstacle_decay
        task_out, next_task = self.critic_task_gru(critic_task.unsqueeze(1), task_hidden)
        obstacle_out, next_obstacle = self.critic_obstacle_gru(critic_obstacle.unsqueeze(1), obstacle_hidden)
        feat = self.critic_merge(torch.cat([task_out.squeeze(1), obstacle_out.squeeze(1)], dim=-1))
        return feat, self._pack_hidden(next_task, next_obstacle)

    def forward_sequence(
        self,
        actor_obs_seq,
        critic_obs_seq,
        dones=None,
        actor_hidden=None,
        critic_hidden=None,
    ):
        actor_feat, _ = self._actor_sequence_features(
            actor_obs_seq,
            dones=dones,
            actor_hidden=actor_hidden,
        )
        critic_feat, _ = self._critic_sequence_features(
            critic_obs_seq,
            dones=dones,
            critic_hidden=critic_hidden,
        )
        mean = self.policy_mean(actor_feat)
        mean = self._maybe_inject_skill_logits_with_context(mean, actor_feat, actor_obs_seq)
        value = self.value_head(critic_feat)
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
        actor_feat, _ = self._actor_sequence_features(
            actor_obs_seq,
            dones=dones,
            actor_hidden=actor_hidden,
        )
        critic_feat, _ = self._critic_sequence_features(
            critic_obs_seq,
            dones=dones,
            critic_hidden=critic_hidden,
        )
        mean = self.policy_mean(actor_feat)
        mean = self._maybe_inject_skill_logits_with_context(mean, actor_feat, actor_obs_seq)
        value = self.value_head(critic_feat)
        log_std = self.log_std.view(1, 1, -1).expand_as(mean)
        return mean, value, log_std, actor_feat, critic_feat

    def forward_recurrent(self, actor_obs, critic_obs, actor_hidden=None, critic_hidden=None):
        actor_feat, next_actor_hidden = self._actor_step_features(actor_obs, actor_hidden=actor_hidden)
        critic_feat, next_critic_hidden = self._critic_step_features(critic_obs, critic_hidden=critic_hidden)
        mean = self.policy_mean(actor_feat)
        mean = self._maybe_inject_skill_logits_with_context(mean, actor_feat, actor_obs.unsqueeze(1)).squeeze(1)
        value = self.value_head(critic_feat)
        log_std = self.log_std.unsqueeze(0).expand_as(mean)
        return mean, value, log_std, next_actor_hidden, next_critic_hidden

    def forward_recurrent_with_features(self, actor_obs, critic_obs, actor_hidden=None, critic_hidden=None):
        actor_feat, next_actor_hidden = self._actor_step_features(actor_obs, actor_hidden=actor_hidden)
        critic_feat, next_critic_hidden = self._critic_step_features(critic_obs, critic_hidden=critic_hidden)
        mean = self.policy_mean(actor_feat)
        mean = self._maybe_inject_skill_logits_with_context(mean, actor_feat, actor_obs.unsqueeze(1)).squeeze(1)
        value = self.value_head(critic_feat)
        log_std = self.log_std.unsqueeze(0).expand_as(mean)
        return mean, value, log_std, actor_feat, critic_feat, next_actor_hidden, next_critic_hidden

    def forward_privileged_recurrent(self, critic_obs, critic_hidden=None):
        critic_feat, next_hidden = self._critic_step_features(critic_obs, critic_hidden=critic_hidden)
        skill_logits = self.classify_skills_privileged(critic_feat)
        value = self.value_head(critic_feat)
        return skill_logits, value, critic_feat, next_hidden

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
        critic_feat, next_hidden = self._critic_step_features(critic_obs, critic_hidden=critic_hidden)
        value = self.value_head(critic_feat)
        return value, next_hidden

    def critic_value(self, critic_obs):
        if critic_obs.dim() == 1:
            critic_obs = critic_obs.unsqueeze(0)
        value, _ = self.critic_value_recurrent(critic_obs, critic_hidden=None)
        return value


__all__ = ["DefenderNetHRLTopDualGRURaw"]
