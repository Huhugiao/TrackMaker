"""
TAD PPO Model

包含:
- Model: PPO Actor-Critic模型封装类
  - 推理: step(), evaluate()
  - 训练: train() - 纯RL训练
  - 训练: train_mixed() - IL+RL混合训练（加权组合，不使用梯度投影）
  - 权重管理: get_weights(), set_weights()

IL+RL混合架构:
- 使用IL权重进行加权组合: total_loss = il_weight * il_loss + (1 - il_weight) * rl_loss
- IL权重使用余弦退火，从初始值逐渐衰减到最终值
"""

import importlib
import os
import numpy as np
import torch
from configs.skill_config import NetParameters, TrainingParameters
from networks import DefenderNetMLP, DefenderNetNMN, create_network
from utils.top_policy_calibration import apply_chase_logit_bias


class Model(object):
    """
    PPO Actor-Critic 模型
    """
    
    @staticmethod
    def _angle_limit() -> float:
        map_cfg = importlib.import_module('configs.map_config')
        limit = float(getattr(map_cfg, 'defender_max_angular_speed',
                      getattr(map_cfg, 'max_turn_deg', 45.0)))
        return max(1.0, limit)
    
    @staticmethod
    def to_normalized_action(pair):
        max_turn = Model._angle_limit()
        angle_norm = float(np.clip(pair[0] / max_turn, -1.0, 1.0))
        speed_norm = float(np.clip(pair[1], 0.0, 1.0) * 2.0 - 1.0)
        return np.array([angle_norm, speed_norm], dtype=np.float32)
    
    @staticmethod
    def to_pre_tanh(action_normalized):
        clipped = np.clip(action_normalized, -0.999999, 0.999999)
        return np.arctanh(clipped).astype(np.float32)
    
    @staticmethod
    def from_normalized(action_normalized):
        max_turn = Model._angle_limit()
        angle = float(np.clip(action_normalized[0], -1.0, 1.0) * max_turn)
        speed = float(np.clip((action_normalized[1] + 1.0) * 0.5, 0.0, 1.0))
        return angle, speed
    
    def __init__(self, device, global_model=False, network_type='nmn'):
        self.device = device
        self.network_type = network_type
        self.network = create_network(network_type).to(device)
        self.is_recurrent = bool(getattr(self.network, 'is_recurrent', False))
        self.is_discrete_policy = bool(getattr(self.network, 'is_discrete_policy', False))
        self.policy_anchor_network = None
        self.policy_anchor_coef = 0.0
        self._actor_hidden = None
        self._critic_hidden = None
        self._prev_actor_obs_for_context = None
        
        if global_model:
            self.net_optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=TrainingParameters.lr
            )
        else:
            self.net_optimizer = None
            
        self.network.train()
        self.current_lr = TrainingParameters.lr
        self._init_policy_anchor()

    def _init_policy_anchor(self):
        enable = bool(getattr(TrainingParameters, 'POLICY_ANCHOR_ENABLE', False))
        coef = float(getattr(TrainingParameters, 'POLICY_ANCHOR_COEF', 0.0))
        checkpoint = getattr(TrainingParameters, 'POLICY_ANCHOR_CHECKPOINT', None)
        if not enable or coef <= 0.0 or not checkpoint:
            return
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f'POLICY_ANCHOR_CHECKPOINT not found: {checkpoint}')

        anchor_type = str(getattr(TrainingParameters, 'POLICY_ANCHOR_NETWORK_TYPE', 'mlp')).strip().lower()
        anchor = create_network(anchor_type).to(self.device)
        loaded = torch.load(checkpoint, map_location='cpu', weights_only=False)
        state_dict = loaded.get('model', loaded) if isinstance(loaded, dict) else loaded
        self._load_state_dict_compat(anchor, state_dict)
        anchor.eval()
        for param in anchor.parameters():
            param.requires_grad_(False)
        self.policy_anchor_network = anchor
        self.policy_anchor_coef = coef
        print(
            f"[PolicyAnchor] enabled: network={anchor_type}, coef={coef:.4g}, "
            f"checkpoint={checkpoint}"
        )

    def reset_recurrent_state(self):
        self._actor_hidden = None
        self._critic_hidden = None
        self._prev_actor_obs_for_context = None

    # Backward-compatible hook name used by evaluation code.
    def reset_gru_sequence(self):
        self.reset_recurrent_state()

    def set_nmn_stage(self, stage: int):
        if hasattr(self.network, 'set_nmn_stage'):
            self.network.set_nmn_stage(stage)

    def get_nmn_stage(self):
        if hasattr(self.network, 'get_nmn_stage'):
            return int(self.network.get_nmn_stage())
        return None
    
    def get_weights(self):
        return {name: param.cpu() for name, param in self.network.state_dict().items()}

    @staticmethod
    def _load_state_dict_compat(module, state_dict):
        incompatible = module.load_state_dict(state_dict, strict=False)
        allowed_exact = {
            'skill_classifier.fc.weight',
            'skill_classifier.fc.bias',
            'use_skill_classifier_for_action',
            'use_contextual_skill_head_for_action',
            'use_changepoint_head_for_action',
            'use_hysteresis_for_action',
            'hrl_top_marker',
            'discrete_policy_marker',
        }
        allowed_prefixes = (
            'actor_gru_refine.',
            'critic_gru_refine.',
            'scalar_policy_residual.',
            'skill_classifier.pooled_proj.',
            'skill_classifier_with_context.base_classifier.pooled_proj.',
            'privileged_classifier.pooled_proj.',
            'change_classifier.',
            'change_classifier_with_context.',
        )

        def _is_allowed(key: str) -> bool:
            return key in allowed_exact or any(key.startswith(prefix) for prefix in allowed_prefixes)

        missing = {key for key in incompatible.missing_keys if not _is_allowed(key)}
        unexpected = {key for key in incompatible.unexpected_keys if not _is_allowed(key)}
        if missing or unexpected:
            raise RuntimeError(
                f'state_dict incompatible: missing={sorted(missing)}, unexpected={sorted(unexpected)}'
            )

    def set_weights(self, weights):
        self._load_state_dict_compat(self.network, weights)
        self.reset_recurrent_state()
    
    def _to_tensor(self, vector):
        if isinstance(vector, np.ndarray):
            input_vector = torch.from_numpy(vector).float().to(self.device)
        elif torch.is_tensor(vector):
            input_vector = vector.to(self.device).float()
        else:
            input_vector = torch.tensor(vector, dtype=torch.float32, device=self.device)
        if input_vector.dim() == 1:
            input_vector = input_vector.unsqueeze(0)
        return torch.nan_to_num(input_vector)
    
    @staticmethod
    def _log_prob_from_pre_tanh(pre_tanh, mean, log_std):
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        base_log_prob = dist.log_prob(pre_tanh)
        log_det_jac = torch.log(1.0 - torch.tanh(pre_tanh) ** 2 + 1e-6)
        return (base_log_prob - log_det_jac).sum(dim=-1)

    def _policy_entropy(self, _policy_output, log_std):
        return (0.5 * (1.0 + np.log(2 * np.pi)) + log_std).sum(dim=-1)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    def _policy_action_from_output(self, policy_output, value, log_std, greedy: bool = False):
        pre_tanh = policy_output if greedy else policy_output + torch.exp(log_std) * torch.randn_like(policy_output)
        action = torch.tanh(pre_tanh)
        log_prob = self._log_prob_from_pre_tanh(pre_tanh, policy_output, log_std)
        return action, log_prob, pre_tanh, value

    def _policy_deterministic_action(self, policy_output, _log_std):
        return torch.tanh(policy_output)

    def _compute_auxiliary_loss(self, actor_obs, critic_obs, mask=None):
        if not hasattr(self.network, 'compute_auxiliary_loss'):
            return None
        aux_loss = self.network.compute_auxiliary_loss(actor_obs, critic_obs, mask=mask)
        if aux_loss is None:
            return None
        if not torch.is_tensor(aux_loss):
            aux_loss = torch.as_tensor(aux_loss, dtype=torch.float32, device=self.device)
        return torch.nan_to_num(aux_loss)

    def _scale_auxiliary_loss(self, aux_loss: torch.Tensor) -> torch.Tensor:
        coef = float(getattr(self.network, 'auxiliary_loss_coef', 1.0))
        return aux_loss * coef

    def _to_aux_target_tensors(self, aux_targets):
        if aux_targets is None:
            return None
        return {
            str(key): torch.as_tensor(value, dtype=torch.float32, device=self.device)
            for key, value in dict(aux_targets).items()
        }

    @staticmethod
    def _reshape_aux_targets(aux_targets, num_envs: int, rollout_steps: int):
        if aux_targets is None:
            return None
        shaped = {}
        expected = int(num_envs * rollout_steps)
        for key, value in aux_targets.items():
            if value.shape[0] != expected:
                raise ValueError(
                    f"aux target {key} shape mismatch: dataset={value.shape[0]}, "
                    f"num_envs*rollout_steps={expected}"
                )
            shaped[key] = value.reshape(num_envs, rollout_steps)
        return shaped

    def _policy_anchor_loss(self, actor_obs, critic_obs, mean, mask):
        if self.policy_anchor_network is None or self.policy_anchor_coef <= 0.0:
            return None
        with torch.no_grad():
            if hasattr(self.policy_anchor_network, 'forward_recurrent'):
                anchor_mean, _anchor_value, _anchor_log_std, _ah, _ch = self.policy_anchor_network.forward_recurrent(
                    actor_obs,
                    critic_obs,
                    actor_hidden=None,
                    critic_hidden=None,
                )
            else:
                anchor_mean, _anchor_value, _anchor_log_std = self.policy_anchor_network(actor_obs, critic_obs)
            anchor_action = torch.tanh(anchor_mean)
        pred_action = torch.tanh(mean)
        mse = (pred_action - anchor_action).pow(2).sum(dim=-1)
        mask = mask.to(dtype=mse.dtype)
        return self.policy_anchor_coef * (mse * mask).sum() / mask.sum().clamp_min(1.0)

    def _multitask_auxiliary_loss(self, aux_outputs, aux_targets, mask, coefs=None):
        if aux_targets is None or not hasattr(self.network, 'multitask_auxiliary_loss'):
            return None, {}
        if coefs is None:
            coefs = {
                'chase_value': float(getattr(TrainingParameters, 'MULTITASK_CHASE_VALUE_COEF', 0.05)),
                'baseline_value': float(getattr(TrainingParameters, 'MULTITASK_BASELINE_VALUE_COEF', 0.05)),
                'collision': float(getattr(TrainingParameters, 'MULTITASK_COLLISION_COEF', 0.02)),
            }
        aux_loss, aux_parts = self.network.multitask_auxiliary_loss(
            aux_outputs,
            aux_targets,
            mask=mask,
            coefs=coefs,
        )
        return torch.nan_to_num(aux_loss), aux_parts

    def _discrete_action_dim(self) -> int:
        return int(getattr(self.network, 'discrete_action_dim', getattr(NetParameters, 'HRL_NUM_SKILLS', 2)))

    def _extract_discrete_logits(self, policy_output: torch.Tensor) -> torch.Tensor:
        return policy_output[..., :self._discrete_action_dim()]

    def _categorical_dist(self, policy_output: torch.Tensor):
        return torch.distributions.Categorical(logits=self._extract_discrete_logits(policy_output))

    def _top_observable_opportunity_risk(self, actor_tensor: torch.Tensor):
        obs = actor_tensor
        dist = obs[:, 0]
        bearing = obs[:, 1]
        visible = obs[:, 3].clamp(0.0, 1.0)
        unobserved = obs[:, 4]
        target_dist = obs[:, 5]
        target_bearing = obs[:, 6]
        urgency = self._observable_urgency(dist, bearing, visible, unobserved, target_dist, target_bearing)

        defender_attacker = torch.clamp(0.5 * (dist + 1.0), 0.0, 1.0)
        close_attacker = torch.clamp(
            (self._env_float("HRL_TOP_OPPORTUNITY_DIST", 0.48) - defender_attacker)
            / max(1e-6, self._env_float("HRL_TOP_OPPORTUNITY_DIST", 0.48)),
            0.0,
            1.0,
        )
        low_urgency = torch.clamp(
            (self._env_float("HRL_TOP_OPPORTUNITY_MAX_URGENCY", 0.55) - urgency)
            / max(1e-6, self._env_float("HRL_TOP_OPPORTUNITY_MAX_URGENCY", 0.55)),
            0.0,
            1.0,
        )

        obstacle_risk = torch.zeros_like(urgency)
        obstacle_safe = torch.ones_like(urgency)
        if obs.shape[-1] > 7:
            radar = obs[:, 7:]
            radar_min_01 = torch.clamp(0.5 * (torch.min(radar, dim=-1).values + 1.0), 0.0, 1.0)
            danger_dist = self._env_float("HRL_TOP_RISK_RADAR_DIST", 0.08)
            safe_dist = self._env_float("HRL_TOP_SAFE_RADAR_DIST", 0.18)
            obstacle_risk = torch.clamp((danger_dist - radar_min_01) / max(1e-6, danger_dist), 0.0, 1.0)
            obstacle_safe = torch.clamp((radar_min_01 - danger_dist) / max(1e-6, safe_dist - danger_dist), 0.0, 1.0)

        opportunity = visible * close_attacker * low_urgency * obstacle_safe
        urgency_risk = torch.clamp(
            (urgency - self._env_float("HRL_TOP_RISK_MIN_URGENCY", 0.58))
            / max(1e-6, 1.0 - self._env_float("HRL_TOP_RISK_MIN_URGENCY", 0.58)),
            0.0,
            1.0,
        )
        risk = torch.maximum(urgency_risk, obstacle_risk)
        return opportunity, risk

    def _discrete_policy_regularization(self, dist, mask: torch.Tensor, actor_obs: torch.Tensor = None):
        """Optional top-policy anti-collapse regularizers for discrete HRL PPO."""
        mask = mask.to(dtype=torch.float32)
        mask_sum = mask.sum().clamp_min(1.0)
        probs = dist.probs
        entropy = dist.entropy()
        entropy_mean = (entropy * mask).sum() / mask_sum
        mean_probs = (probs * mask.unsqueeze(-1)).sum(dim=0) / mask_sum
        max_prob_mean = (probs.max(dim=-1).values * mask).sum() / mask_sum

        reg_loss = torch.zeros((), dtype=probs.dtype, device=probs.device)

        entropy_floor = float(getattr(TrainingParameters, 'HRL_TOP_MIN_ENTROPY', 0.0))
        entropy_floor_coef = float(getattr(TrainingParameters, 'HRL_TOP_ENTROPY_FLOOR_COEF', 0.0))
        if entropy_floor_coef > 0.0 and entropy_floor > 0.0:
            reg_loss = reg_loss + entropy_floor_coef * torch.relu(
                torch.as_tensor(entropy_floor, dtype=probs.dtype, device=probs.device) - entropy_mean
            ).pow(2)

        max_prob_threshold = float(getattr(TrainingParameters, 'HRL_TOP_MAX_MEAN_MAX_PROB', 1.0))
        max_prob_coef = float(getattr(TrainingParameters, 'HRL_TOP_MAX_MEAN_MAX_PROB_COEF', 0.0))
        if max_prob_coef > 0.0 and max_prob_threshold < 1.0:
            reg_loss = reg_loss + max_prob_coef * torch.relu(
                max_prob_mean - torch.as_tensor(max_prob_threshold, dtype=probs.dtype, device=probs.device)
            ).pow(2)

        bounds_coef = float(getattr(TrainingParameters, 'HRL_TOP_MEAN_PROB_BOUNDS_COEF', 0.0))
        min_chase_prob = float(getattr(TrainingParameters, 'HRL_TOP_MIN_CHASE_PROB', 0.0))
        max_chase_prob = float(getattr(TrainingParameters, 'HRL_TOP_MAX_CHASE_PROB', 1.0))
        chase_prob_std = torch.zeros((), dtype=probs.dtype, device=probs.device)
        if bounds_coef > 0.0 and probs.shape[-1] == 2:
            chase_prob = mean_probs[1]
            lower = torch.relu(torch.as_tensor(min_chase_prob, dtype=probs.dtype, device=probs.device) - chase_prob)
            upper = torch.relu(chase_prob - torch.as_tensor(max_chase_prob, dtype=probs.dtype, device=probs.device))
            reg_loss = reg_loss + bounds_coef * (lower.pow(2) + upper.pow(2))
            centered = probs[:, 1] - chase_prob
            chase_prob_std = torch.sqrt(((centered.pow(2) * mask).sum() / mask_sum).clamp_min(1e-8))

        std_floor = float(getattr(TrainingParameters, 'HRL_TOP_CHASE_PROB_STD_FLOOR', 0.0))
        std_floor_coef = float(getattr(TrainingParameters, 'HRL_TOP_CHASE_PROB_STD_FLOOR_COEF', 0.0))
        if std_floor_coef > 0.0 and std_floor > 0.0 and probs.shape[-1] == 2:
            reg_loss = reg_loss + std_floor_coef * torch.relu(
                torch.as_tensor(std_floor, dtype=probs.dtype, device=probs.device) - chase_prob_std
            ).pow(2)

        deploy_chase_rate = torch.zeros((), dtype=probs.dtype, device=probs.device)
        hard_deploy_chase_rate = torch.zeros((), dtype=probs.dtype, device=probs.device)
        deploy_rate_coef = float(getattr(TrainingParameters, 'HRL_TOP_DEPLOY_CHASE_RATE_COEF', 0.0))
        if deploy_rate_coef > 0.0 and probs.shape[-1] == 2:
            deploy_threshold = float(getattr(TrainingParameters, 'HRL_TOP_DEPLOY_CHASE_THRESHOLD', 0.60))
            deploy_temp = max(1e-3, float(getattr(TrainingParameters, 'HRL_TOP_DEPLOY_CHASE_TEMP', 0.03)))
            soft_selected = torch.sigmoid((probs[:, 1] - deploy_threshold) / deploy_temp)
            deploy_chase_rate = (soft_selected * mask).sum() / mask_sum
            hard_selected = (probs[:, 1] >= deploy_threshold).to(dtype=probs.dtype)
            hard_deploy_chase_rate = (hard_selected * mask).sum() / mask_sum
            min_deploy_rate = float(getattr(TrainingParameters, 'HRL_TOP_MIN_DEPLOY_CHASE_RATE', 0.0))
            max_deploy_rate = float(getattr(TrainingParameters, 'HRL_TOP_MAX_DEPLOY_CHASE_RATE', 1.0))
            lower = torch.relu(
                torch.as_tensor(min_deploy_rate, dtype=probs.dtype, device=probs.device) - deploy_chase_rate
            )
            upper = torch.relu(
                deploy_chase_rate - torch.as_tensor(max_deploy_rate, dtype=probs.dtype, device=probs.device)
            )
            reg_loss = reg_loss + deploy_rate_coef * (lower.pow(2) + upper.pow(2))

        opportunity_chase_rate = torch.zeros((), dtype=probs.dtype, device=probs.device)
        risk_chase_rate = torch.zeros((), dtype=probs.dtype, device=probs.device)
        opportunity_chase_prob = torch.zeros((), dtype=probs.dtype, device=probs.device)
        risk_chase_prob = torch.zeros((), dtype=probs.dtype, device=probs.device)
        opportunity_mass = torch.zeros((), dtype=probs.dtype, device=probs.device)
        risk_mass = torch.zeros((), dtype=probs.dtype, device=probs.device)
        cond_coef = float(getattr(TrainingParameters, 'HRL_TOP_CONDITIONAL_DEPLOY_COEF', 0.0))
        cond_prob_coef = float(getattr(TrainingParameters, 'HRL_TOP_CONDITIONAL_PROB_COEF', 0.0))
        if (cond_coef > 0.0 or cond_prob_coef > 0.0) and probs.shape[-1] == 2 and actor_obs is not None:
            cond_threshold = float(getattr(TrainingParameters, 'HRL_TOP_CONDITIONAL_DEPLOY_THRESHOLD', 0.60))
            cond_temp = max(1e-3, float(getattr(TrainingParameters, 'HRL_TOP_CONDITIONAL_DEPLOY_TEMP', 0.01)))
            soft_selected = torch.sigmoid((probs[:, 1] - cond_threshold) / cond_temp)
            opportunity, risk = self._top_observable_opportunity_risk(actor_obs)
            opportunity_w = opportunity.to(dtype=probs.dtype) * mask
            risk_w = risk.to(dtype=probs.dtype) * mask
            opportunity_mass = opportunity_w.sum()
            risk_mass = risk_w.sum()
            min_mass = float(getattr(TrainingParameters, 'HRL_TOP_CONDITIONAL_MIN_MASS', 16.0))
            if float(opportunity_mass.detach().item()) >= min_mass:
                opportunity_chase_rate = (soft_selected * opportunity_w).sum() / opportunity_mass.clamp_min(1e-6)
                opportunity_chase_prob = (probs[:, 1] * opportunity_w).sum() / opportunity_mass.clamp_min(1e-6)
                min_opp_rate = float(getattr(TrainingParameters, 'HRL_TOP_OPPORTUNITY_MIN_CHASE_RATE', 0.20))
                max_opp_rate = float(getattr(TrainingParameters, 'HRL_TOP_OPPORTUNITY_MAX_CHASE_RATE', 0.65))
                if cond_coef > 0.0:
                    reg_loss = reg_loss + cond_coef * (
                        torch.relu(torch.as_tensor(min_opp_rate, dtype=probs.dtype, device=probs.device) - opportunity_chase_rate).pow(2)
                        + torch.relu(opportunity_chase_rate - torch.as_tensor(max_opp_rate, dtype=probs.dtype, device=probs.device)).pow(2)
                    )
                opp_prob_target = float(getattr(TrainingParameters, 'HRL_TOP_OPPORTUNITY_CHASE_PROB_TARGET', 0.62))
                if cond_prob_coef > 0.0:
                    reg_loss = reg_loss + cond_prob_coef * torch.relu(
                        torch.as_tensor(opp_prob_target, dtype=probs.dtype, device=probs.device) - opportunity_chase_prob
                    ).pow(2)
            if float(risk_mass.detach().item()) >= min_mass:
                risk_chase_rate = (soft_selected * risk_w).sum() / risk_mass.clamp_min(1e-6)
                risk_chase_prob = (probs[:, 1] * risk_w).sum() / risk_mass.clamp_min(1e-6)
                max_risk_rate = float(getattr(TrainingParameters, 'HRL_TOP_RISK_MAX_CHASE_RATE', 0.08))
                if cond_coef > 0.0:
                    reg_loss = reg_loss + cond_coef * torch.relu(
                        risk_chase_rate - torch.as_tensor(max_risk_rate, dtype=probs.dtype, device=probs.device)
                    ).pow(2)
                risk_prob_target = float(getattr(TrainingParameters, 'HRL_TOP_RISK_CHASE_PROB_TARGET', 0.38))
                if cond_prob_coef > 0.0:
                    reg_loss = reg_loss + cond_prob_coef * torch.relu(
                        risk_chase_prob - torch.as_tensor(risk_prob_target, dtype=probs.dtype, device=probs.device)
                    ).pow(2)

        stats = {
            'reg_loss': float(reg_loss.detach().item()),
            'entropy': float(entropy_mean.detach().item()),
            'max_prob': float(max_prob_mean.detach().item()),
            'chase_prob': float(mean_probs[1].detach().item()) if probs.shape[-1] == 2 else 0.0,
            'chase_prob_std': float(chase_prob_std.detach().item()) if probs.shape[-1] == 2 else 0.0,
            'deploy_chase_rate': float(deploy_chase_rate.detach().item()) if probs.shape[-1] == 2 else 0.0,
            'hard_deploy_chase_rate': float(hard_deploy_chase_rate.detach().item()) if probs.shape[-1] == 2 else 0.0,
            'opportunity_chase_rate': float(opportunity_chase_rate.detach().item()) if probs.shape[-1] == 2 else 0.0,
            'risk_chase_rate': float(risk_chase_rate.detach().item()) if probs.shape[-1] == 2 else 0.0,
            'opportunity_chase_prob': float(opportunity_chase_prob.detach().item()) if probs.shape[-1] == 2 else 0.0,
            'risk_chase_prob': float(risk_chase_prob.detach().item()) if probs.shape[-1] == 2 else 0.0,
            'opportunity_mass': float(opportunity_mass.detach().item()) if probs.shape[-1] == 2 else 0.0,
            'risk_mass': float(risk_mass.detach().item()) if probs.shape[-1] == 2 else 0.0,
        }
        return reg_loss, stats

    def _format_discrete_action(self, action_idx: torch.Tensor) -> torch.Tensor:
        if action_idx.dim() == 0:
            action_idx = action_idx.unsqueeze(0)
        return action_idx.to(dtype=torch.float32).unsqueeze(-1)

    @staticmethod
    def _env_bool(name: str, default: bool = False) -> bool:
        raw = os.environ.get(name)
        if raw is None or str(raw).strip() == "":
            return bool(default)
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        raw = os.environ.get(name)
        if raw is None or str(raw).strip() == "":
            return float(default)
        try:
            return float(raw)
        except ValueError:
            return float(default)

    def _select_discrete_greedy_action_idx(
        self,
        logits: torch.Tensor,
        actor_tensor: torch.Tensor = None,
    ) -> torch.Tensor:
        logits = apply_chase_logit_bias(
            logits,
            skill_names=("baseline", "chase"),
            chase_logit_bias=self._env_float("HRL_TOP_CHASE_LOGIT_BIAS", 0.0),
        )
        if logits.shape[-1] == 2:
            threshold_raw = os.environ.get('HRL_TOP_GREEDY_CHASE_THRESHOLD', '').strip()
            if threshold_raw:
                try:
                    threshold = float(threshold_raw)
                except ValueError:
                    threshold = 0.5
                threshold = float(np.clip(threshold, 0.0, 1.0))
                probs = torch.softmax(logits, dim=-1)
                return (probs[..., 1] >= threshold).to(dtype=torch.long)
        return torch.argmax(logits, dim=-1)

    def _use_contextual_skill_head_for_action(self) -> bool:
        return bool(
            hasattr(self.network, 'classify_skills_with_context')
            and hasattr(self.network, 'forward_recurrent_with_features')
            and bool(getattr(self.network, 'use_contextual_skill_head_for_action', torch.zeros((), dtype=torch.bool)).item())
        )

    def _build_step_behavior_context(self, actor_tensor: torch.Tensor) -> torch.Tensor:
        dist = actor_tensor[:, 0]
        bearing = actor_tensor[:, 1]
        visible = actor_tensor[:, 3]
        unobserved = actor_tensor[:, 4]
        target_dist = actor_tensor[:, 5]
        target_bearing = actor_tensor[:, 6]

        prev = self._prev_actor_obs_for_context
        if prev is None:
            delta_dist = torch.zeros_like(dist)
            delta_bearing = torch.zeros_like(bearing)
        else:
            prev = prev.to(device=actor_tensor.device, dtype=actor_tensor.dtype)
            delta_dist = dist - prev[:, 0]
            raw_delta = bearing - prev[:, 1]
            delta_bearing = torch.remainder(raw_delta + 1.0, 2.0) - 1.0

        return torch.stack(
            [
                delta_dist,
                delta_bearing,
                torch.abs(delta_dist),
                torch.abs(delta_bearing),
                visible,
                self._observable_urgency(dist, bearing, visible, unobserved, target_dist, target_bearing),
            ],
            dim=-1,
        )

    @staticmethod
    def _observable_urgency(dist, bearing, visible, unobserved, target_dist, target_bearing):
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
        return torch.clamp(urgency_base + 0.35 * uncertainty * (1.0 - urgency_base), 0.0, 1.0)

    def _forward_discrete_policy(self, actor_tensor, critic_tensor):
        if self.is_recurrent and self._use_contextual_skill_head_for_action():
            policy_output, value, _log_std, actor_feat, _critic_feat, next_actor_hidden, next_critic_hidden = self.network.forward_recurrent_with_features(
                actor_tensor,
                critic_tensor,
                actor_hidden=self._actor_hidden,
                critic_hidden=self._critic_hidden,
            )
            behavior_context = self._build_step_behavior_context(actor_tensor)
            policy_output = policy_output.clone()
            policy_output[..., :self._discrete_action_dim()] = self.network.classify_skills_with_context(actor_feat, behavior_context)
        elif self.is_recurrent and hasattr(self.network, 'forward_recurrent'):
            policy_output, value, _log_std, next_actor_hidden, next_critic_hidden = self.network.forward_recurrent(
                actor_tensor,
                critic_tensor,
                actor_hidden=self._actor_hidden,
                critic_hidden=self._critic_hidden,
            )
        else:
            policy_output, value, _log_std = self.network(actor_tensor, critic_tensor)
            next_actor_hidden = None
            next_critic_hidden = None

        self._actor_hidden = next_actor_hidden.detach() if next_actor_hidden is not None else None
        self._critic_hidden = next_critic_hidden.detach() if next_critic_hidden is not None else None
        self._prev_actor_obs_for_context = actor_tensor.detach().clone()
        return policy_output, value

    def _prepare_discrete_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.dim() == 0:
            actions = actions.unsqueeze(0)
        if actions.dim() > 1:
            actions = actions[..., 0]
        return actions.to(dtype=torch.long).clamp_(0, self._discrete_action_dim() - 1)

    @staticmethod
    def _reshape_recurrent_hidden_buffer(hidden_states, num_envs: int, rollout_steps: int):
        if hidden_states is None:
            return None
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(1)
        if hidden_states.dim() != 3:
            raise ValueError(f'Invalid recurrent hidden shape: {tuple(hidden_states.shape)}')
        expected = int(num_envs * rollout_steps)
        if hidden_states.shape[0] != expected:
            raise ValueError(
                f"recurrent hidden shape mismatch: dataset={hidden_states.shape[0]}, "
                f"num_envs*rollout_steps={expected}"
            )
        return hidden_states.reshape(num_envs, rollout_steps, hidden_states.shape[1], hidden_states.shape[2])

    @staticmethod
    def _select_recurrent_chunk_hidden(hidden_seq, env_idx, t0: int):
        if hidden_seq is None:
            return None
        hidden = hidden_seq[env_idx, t0]
        return hidden.permute(1, 0, 2).contiguous()

    @staticmethod
    def _select_recurrent_chunk_hidden_by_index(hidden_seq, env_idx, t0_idx):
        if hidden_seq is None:
            return None
        hidden = hidden_seq[env_idx, t0_idx]
        return hidden.permute(1, 0, 2).contiguous()

    @staticmethod
    def _iter_recurrent_chunk_minibatches(num_envs: int, rollout_steps: int, tbptt_steps: int,
                                          minibatch_transitions: int, device):
        starts = torch.arange(0, int(rollout_steps), int(tbptt_steps), device=device)
        envs = torch.arange(int(num_envs), device=device)
        chunk_envs = envs.repeat_interleave(starts.numel())
        chunk_starts = starts.repeat(int(num_envs))
        perm = torch.randperm(chunk_envs.numel(), device=device)
        chunks_per_minibatch = max(1, int(minibatch_transitions // max(1, int(tbptt_steps))))
        for start in range(0, perm.numel(), chunks_per_minibatch):
            idx = perm[start:start + chunks_per_minibatch]
            if idx.numel() > 0:
                yield chunk_envs[idx], chunk_starts[idx]

    @staticmethod
    def _gather_recurrent_chunks(seq, env_idx, t0_idx, tbptt_steps: int, fill_value: float = 0.0):
        chunk_len = int(tbptt_steps)
        if int(env_idx.numel()) == 0:
            return seq.new_empty((0, chunk_len, *seq.shape[2:]))
        env_idx = env_idx.to(device=seq.device, dtype=torch.long)
        t0_idx = t0_idx.to(device=seq.device, dtype=torch.long)
        offsets = torch.arange(chunk_len, device=seq.device, dtype=torch.long)
        time_idx = t0_idx.unsqueeze(1) + offsets.unsqueeze(0)
        valid = time_idx < int(seq.shape[1])
        safe_time_idx = time_idx.clamp_max(max(0, int(seq.shape[1]) - 1))
        gathered = seq[env_idx.unsqueeze(1), safe_time_idx]
        if bool(valid.all()):
            return gathered
        out = seq.new_full((int(env_idx.numel()), chunk_len, *seq.shape[2:]), fill_value)
        valid_view = valid.view(valid.shape[0], valid.shape[1], *([1] * (gathered.dim() - 2)))
        return torch.where(valid_view, gathered, out)

    def _train_recurrent_ppo(
        self,
        actor_obs,
        critic_obs,
        returns,
        values,
        actions,
        old_log_probs,
        mask,
        dones,
        actor_hiddens,
        critic_hiddens,
        num_envs: int,
        rollout_steps: int,
        tbptt_steps: int,
        aux_targets=None,
    ):
        dataset_size = actor_obs.shape[0]
        if dataset_size != int(num_envs * rollout_steps):
            raise ValueError(
                f"recurrent train shape mismatch: dataset={dataset_size}, "
                f"num_envs*rollout_steps={int(num_envs * rollout_steps)}"
            )

        raw_advantages = returns - values.squeeze(-1)
        valid_mask = mask > 0
        if valid_mask.sum() > 1:
            adv_std = float(raw_advantages[valid_mask].std().item())
            adv_mean = float(raw_advantages[valid_mask].mean().item())
            advantages = ((raw_advantages - adv_mean) / (adv_std + 1e-8))
        else:
            adv_std = 0.0
            adv_mean = 0.0
            advantages = raw_advantages * 0.0
        advantages = advantages * mask

        actor_obs_seq = actor_obs.reshape(num_envs, rollout_steps, -1)
        critic_obs_seq = critic_obs.reshape(num_envs, rollout_steps, -1)
        actions_seq = actions.reshape(num_envs, rollout_steps, -1)
        old_log_probs_seq = old_log_probs.reshape(num_envs, rollout_steps)
        returns_seq = returns.reshape(num_envs, rollout_steps)
        values_seq = values.reshape(num_envs, rollout_steps, -1)
        advantages_seq = advantages.reshape(num_envs, rollout_steps)
        mask_seq = mask.reshape(num_envs, rollout_steps)
        dones_seq = dones.reshape(num_envs, rollout_steps)
        actor_hidden_seq = self._reshape_recurrent_hidden_buffer(actor_hiddens, num_envs, rollout_steps)
        critic_hidden_seq = self._reshape_recurrent_hidden_buffer(critic_hiddens, num_envs, rollout_steps)
        aux_targets_seq = self._reshape_aux_targets(aux_targets, num_envs, rollout_steps)

        minibatch_transitions = min(TrainingParameters.MINIBATCH_SIZE, dataset_size)
        n_epochs = int(TrainingParameters.N_EPOCHS)

        sum_policy_loss = 0.0
        sum_entropy_loss = 0.0
        sum_value_loss = 0.0
        sum_aux_loss = 0.0
        sum_anchor_loss = 0.0
        sum_adv_action_bc_loss = 0.0
        sum_approx_kl = 0.0
        sum_clipfrac = 0.0
        sum_grad_norm = 0.0
        sum_discrete_reg = 0.0
        sum_mean_chase_prob = 0.0
        sum_mean_max_prob = 0.0
        sum_chase_prob_std = 0.0
        sum_deploy_chase_rate = 0.0
        sum_hard_deploy_chase_rate = 0.0
        n_updates = 0

        for _ in range(n_epochs):
            for env_idx, t0_idx in self._iter_recurrent_chunk_minibatches(
                num_envs,
                rollout_steps,
                tbptt_steps,
                minibatch_transitions,
                self.device,
            ):
                mb_actor_seq = self._gather_recurrent_chunks(actor_obs_seq, env_idx, t0_idx, tbptt_steps)
                mb_critic_seq = self._gather_recurrent_chunks(critic_obs_seq, env_idx, t0_idx, tbptt_steps)
                mb_actions_seq = self._gather_recurrent_chunks(actions_seq, env_idx, t0_idx, tbptt_steps)
                mb_old_log_probs_seq = self._gather_recurrent_chunks(old_log_probs_seq.unsqueeze(-1), env_idx, t0_idx, tbptt_steps).squeeze(-1)
                mb_returns_seq = self._gather_recurrent_chunks(returns_seq.unsqueeze(-1), env_idx, t0_idx, tbptt_steps).squeeze(-1)
                mb_values_seq = self._gather_recurrent_chunks(values_seq, env_idx, t0_idx, tbptt_steps)
                mb_advantages_seq = self._gather_recurrent_chunks(advantages_seq.unsqueeze(-1), env_idx, t0_idx, tbptt_steps).squeeze(-1)
                mb_mask_seq = self._gather_recurrent_chunks(mask_seq.unsqueeze(-1), env_idx, t0_idx, tbptt_steps).squeeze(-1)
                mb_dones_seq = self._gather_recurrent_chunks(dones_seq.unsqueeze(-1), env_idx, t0_idx, tbptt_steps).squeeze(-1)
                mb_aux_targets = None
                if aux_targets_seq is not None:
                    mb_aux_targets = {
                        key: self._gather_recurrent_chunks(value.unsqueeze(-1), env_idx, t0_idx, tbptt_steps).squeeze(-1)
                        for key, value in aux_targets_seq.items()
                    }
                mb_actor_hidden = self._select_recurrent_chunk_hidden_by_index(actor_hidden_seq, env_idx, t0_idx)
                mb_critic_hidden = self._select_recurrent_chunk_hidden_by_index(critic_hidden_seq, env_idx, t0_idx)

                self.net_optimizer.zero_grad(set_to_none=True)
                if mb_aux_targets is not None and hasattr(self.network, 'forward_sequence_with_aux'):
                    mean_seq, value_seq, log_std_seq, aux_outputs = self.network.forward_sequence_with_aux(
                        mb_actor_seq,
                        mb_critic_seq,
                        dones=mb_dones_seq,
                        actor_hidden=mb_actor_hidden,
                        critic_hidden=mb_critic_hidden,
                    )
                else:
                    mean_seq, value_seq, log_std_seq = self.network.forward_sequence(
                        mb_actor_seq,
                        mb_critic_seq,
                        dones=mb_dones_seq,
                        actor_hidden=mb_actor_hidden,
                        critic_hidden=mb_critic_hidden,
                    )
                    aux_outputs = None

                mean_flat = mean_seq.reshape(-1, mean_seq.shape[-1])
                value_flat = value_seq.reshape(-1)
                log_std_flat = log_std_seq.reshape(-1, log_std_seq.shape[-1])
                actions_flat = mb_actions_seq.reshape(-1, mb_actions_seq.shape[-1])
                old_log_probs_flat = mb_old_log_probs_seq.reshape(-1)
                returns_flat = mb_returns_seq.reshape(-1)
                old_values_flat = mb_values_seq.reshape(-1)
                advantages_flat = mb_advantages_seq.reshape(-1)
                mask_flat = mb_mask_seq.reshape(-1)
                mask_sum = mask_flat.sum().clamp_min(1.0)

                new_log_probs = self._log_prob_from_pre_tanh(actions_flat, mean_flat, log_std_flat)
                ratio = torch.exp(new_log_probs - old_log_probs_flat)
                ratio = torch.clamp(ratio, 0.0, TrainingParameters.RATIO_CLAMP_MAX)
                surr1 = ratio * advantages_flat
                surr2 = torch.clamp(
                    ratio,
                    1.0 - TrainingParameters.CLIP_RANGE,
                    1.0 + TrainingParameters.CLIP_RANGE,
                ) * advantages_flat
                policy_loss_t = -torch.min(surr1, surr2).mul(mask_flat).sum() / mask_sum

                ent = (0.5 * (1.0 + np.log(2 * np.pi)) + log_std_flat).sum(dim=-1)
                entropy_loss_t = -(ent * mask_flat).sum() / mask_sum

                value_clipped = old_values_flat + torch.clamp(
                    value_flat - old_values_flat,
                    -TrainingParameters.VALUE_CLIP_RANGE,
                    TrainingParameters.VALUE_CLIP_RANGE,
                )
                v_loss1 = (value_flat - returns_flat) ** 2
                v_loss2 = (value_clipped - returns_flat) ** 2
                value_loss_t = (torch.max(v_loss1, v_loss2) * mask_flat).sum() / mask_sum

                total_loss_t = (
                    policy_loss_t
                    + TrainingParameters.EX_VALUE_COEF * value_loss_t
                    + TrainingParameters.ENTROPY_COEF * entropy_loss_t
                )
                adv_action_bc_loss_t = None
                adv_action_bc_coef = float(getattr(TrainingParameters, 'ADV_ACTION_BC_COEF', 0.0))
                if adv_action_bc_coef > 0.0:
                    with torch.no_grad():
                        positive_adv = torch.clamp(advantages_flat, min=0.0)
                        max_weight = float(getattr(TrainingParameters, 'ADV_ACTION_BC_MAX_WEIGHT', 3.0))
                        if max_weight > 0.0:
                            positive_adv = torch.clamp(positive_adv, max=max_weight)
                        target_action = torch.tanh(actions_flat)
                    pred_action = torch.tanh(mean_flat)
                    action_bc = (pred_action - target_action).pow(2).sum(dim=-1)
                    weighted_mask = positive_adv * mask_flat
                    adv_action_bc_loss_t = (
                        adv_action_bc_coef
                        * (action_bc * weighted_mask).sum()
                        / weighted_mask.sum().clamp_min(1.0)
                    )
                    total_loss_t = total_loss_t + adv_action_bc_loss_t
                aux_loss_t, _aux_parts = self._multitask_auxiliary_loss(
                    aux_outputs,
                    mb_aux_targets,
                    mb_mask_seq,
                )
                if aux_loss_t is not None:
                    total_loss_t = total_loss_t + aux_loss_t
                anchor_loss_t = self._policy_anchor_loss(
                    mb_actor_seq.reshape(-1, mb_actor_seq.shape[-1]),
                    mb_critic_seq.reshape(-1, mb_critic_seq.shape[-1]),
                    mean_flat,
                    mask_flat,
                )
                if anchor_loss_t is not None:
                    total_loss_t = total_loss_t + anchor_loss_t
                total_loss_t.backward()

                with torch.no_grad():
                    mb_kl = (old_log_probs_flat - new_log_probs).mean().item()
                    mb_cf = (torch.abs(ratio - 1.0) > TrainingParameters.CLIP_RANGE).float().mean().item()

                gn = float(torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    TrainingParameters.MAX_GRAD_NORM
                ).item())
                for _, param in self.network.named_parameters():
                    if param.grad is not None:
                        param.grad = torch.nan_to_num(param.grad)
                self.net_optimizer.step()

                sum_policy_loss += float(policy_loss_t.item())
                sum_entropy_loss += float(entropy_loss_t.item())
                sum_value_loss += float(value_loss_t.item())
                if aux_loss_t is not None:
                    sum_aux_loss += float(aux_loss_t.detach().item())
                if anchor_loss_t is not None:
                    sum_anchor_loss += float(anchor_loss_t.detach().item())
                if adv_action_bc_loss_t is not None:
                    sum_adv_action_bc_loss += float(adv_action_bc_loss_t.detach().item())
                sum_approx_kl += float(mb_kl)
                sum_clipfrac += float(mb_cf)
                sum_grad_norm += float(gn)
                n_updates += 1

            if n_updates > 0 and abs(sum_approx_kl / n_updates) > 0.03:
                break

        if n_updates > 0:
            policy_loss = sum_policy_loss / n_updates
            entropy_loss = sum_entropy_loss / n_updates
            value_loss = sum_value_loss / n_updates
            aux_loss_value = sum_aux_loss / n_updates
            anchor_loss_value = sum_anchor_loss / n_updates
            adv_action_bc_loss_value = sum_adv_action_bc_loss / n_updates
            approx_kl = sum_approx_kl / n_updates
            clipfrac = sum_clipfrac / n_updates
            grad_norm = sum_grad_norm / n_updates
            total_loss = (
                policy_loss
                + TrainingParameters.EX_VALUE_COEF * value_loss
                + TrainingParameters.ENTROPY_COEF * entropy_loss
                + aux_loss_value
                + anchor_loss_value
                + adv_action_bc_loss_value
            )
        else:
            policy_loss = 0.0
            entropy_loss = 0.0
            value_loss = 0.0
            aux_loss_value = 0.0
            anchor_loss_value = 0.0
            adv_action_bc_loss_value = 0.0
            approx_kl = 0.0
            clipfrac = 0.0
            grad_norm = 0.0
            total_loss = 0.0

        losses = [total_loss, policy_loss, entropy_loss, value_loss,
                  adv_std, approx_kl, 0.0, clipfrac, grad_norm, adv_mean]
        return {
            'losses': losses,
            'il_loss': None,
            'il_filter_ratio': None,
            'aux_loss': aux_loss_value,
            'anchor_loss': anchor_loss_value,
            'adv_action_bc_loss': adv_action_bc_loss_value,
        }

    def _train_discrete_ppo(self, actor_obs, critic_obs, returns, values, actions, old_log_probs, mask, advantages,
                            adv_mean: float, adv_std: float):
        dataset_size = actor_obs.shape[0]
        minibatch_size = min(TrainingParameters.MINIBATCH_SIZE, dataset_size)
        n_epochs = int(TrainingParameters.N_EPOCHS)

        sum_policy_loss = 0.0
        sum_entropy_loss = 0.0
        sum_value_loss = 0.0
        sum_approx_kl = 0.0
        sum_clipfrac = 0.0
        sum_grad_norm = 0.0
        sum_discrete_reg = 0.0
        sum_mean_chase_prob = 0.0
        sum_mean_max_prob = 0.0
        sum_chase_prob_std = 0.0
        sum_deploy_chase_rate = 0.0
        sum_hard_deploy_chase_rate = 0.0
        sum_opportunity_chase_rate = 0.0
        sum_risk_chase_rate = 0.0
        sum_opportunity_chase_prob = 0.0
        sum_risk_chase_prob = 0.0
        sum_opportunity_mass = 0.0
        sum_risk_mass = 0.0
        n_updates = 0

        action_indices = self._prepare_discrete_actions(actions)

        for _ in range(n_epochs):
            indices = torch.randperm(dataset_size, device=self.device)
            for start in range(0, dataset_size, minibatch_size):
                end = min(start + minibatch_size, dataset_size)
                mb_idx = indices[start:end]

                mb_actor_obs = actor_obs[mb_idx]
                mb_critic_obs = critic_obs[mb_idx]
                mb_actions = action_indices[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_values = values[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_mask = mask[mb_idx]

                self.net_optimizer.zero_grad(set_to_none=True)

                policy_output, value_flat, _ = self.network(mb_actor_obs, mb_critic_obs)
                new_values = value_flat.squeeze(-1)
                dist = self._categorical_dist(policy_output)
                new_log_probs = dist.log_prob(mb_actions)

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                ratio = torch.clamp(ratio, 0.0, TrainingParameters.RATIO_CLAMP_MAX)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - TrainingParameters.CLIP_RANGE,
                    1.0 + TrainingParameters.CLIP_RANGE,
                ) * mb_advantages
                mb_mask_sum = mb_mask.sum().clamp_min(1.0)
                policy_loss_t = -(torch.min(surr1, surr2) * mb_mask).sum() / mb_mask_sum

                entropy_loss_t = -(dist.entropy() * mb_mask).sum() / mb_mask_sum

                value_clipped = mb_values.squeeze(-1) + torch.clamp(
                    new_values - mb_values.squeeze(-1),
                    -TrainingParameters.VALUE_CLIP_RANGE,
                    TrainingParameters.VALUE_CLIP_RANGE,
                )
                v_loss1 = (new_values - mb_returns) ** 2
                v_loss2 = (value_clipped - mb_returns) ** 2
                value_loss_t = (torch.max(v_loss1, v_loss2) * mb_mask).sum() / mb_mask_sum

                total_loss_t = (
                    policy_loss_t
                    + TrainingParameters.EX_VALUE_COEF * value_loss_t
                    + TrainingParameters.ENTROPY_COEF * entropy_loss_t
                )
                discrete_reg_t, discrete_reg_stats = self._discrete_policy_regularization(
                    dist,
                    mb_mask,
                    actor_obs=mb_actor_obs,
                )
                total_loss_t = total_loss_t + discrete_reg_t
                total_loss_t.backward()

                with torch.no_grad():
                    mb_kl = (mb_old_log_probs - new_log_probs).mean().item()
                    mb_cf = (torch.abs(ratio - 1.0) > TrainingParameters.CLIP_RANGE).float().mean().item()

                gn = float(torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    TrainingParameters.MAX_GRAD_NORM,
                ).item())
                for _, param in self.network.named_parameters():
                    if param.grad is not None:
                        param.grad = torch.nan_to_num(param.grad)
                self.net_optimizer.step()

                sum_policy_loss += float(policy_loss_t.item())
                sum_entropy_loss += float(entropy_loss_t.item())
                sum_value_loss += float(value_loss_t.item())
                sum_approx_kl += float(mb_kl)
                sum_clipfrac += float(mb_cf)
                sum_grad_norm += float(gn)
                sum_discrete_reg += float(discrete_reg_stats['reg_loss'])
                sum_mean_chase_prob += float(discrete_reg_stats['chase_prob'])
                sum_mean_max_prob += float(discrete_reg_stats['max_prob'])
                sum_chase_prob_std += float(discrete_reg_stats['chase_prob_std'])
                sum_deploy_chase_rate += float(discrete_reg_stats.get('deploy_chase_rate', 0.0))
                sum_hard_deploy_chase_rate += float(discrete_reg_stats.get('hard_deploy_chase_rate', 0.0))
                sum_opportunity_chase_rate += float(discrete_reg_stats.get('opportunity_chase_rate', 0.0))
                sum_risk_chase_rate += float(discrete_reg_stats.get('risk_chase_rate', 0.0))
                sum_opportunity_chase_prob += float(discrete_reg_stats.get('opportunity_chase_prob', 0.0))
                sum_risk_chase_prob += float(discrete_reg_stats.get('risk_chase_prob', 0.0))
                sum_opportunity_mass += float(discrete_reg_stats.get('opportunity_mass', 0.0))
                sum_risk_mass += float(discrete_reg_stats.get('risk_mass', 0.0))
                n_updates += 1

            if n_updates > 0 and abs(sum_approx_kl / n_updates) > 0.03:
                break

        if n_updates > 0:
            policy_loss = sum_policy_loss / n_updates
            entropy_loss = sum_entropy_loss / n_updates
            value_loss = sum_value_loss / n_updates
            approx_kl = sum_approx_kl / n_updates
            clipfrac = sum_clipfrac / n_updates
            grad_norm = sum_grad_norm / n_updates
            discrete_reg = sum_discrete_reg / n_updates
            mean_chase_prob = sum_mean_chase_prob / n_updates
            mean_max_prob = sum_mean_max_prob / n_updates
            chase_prob_std = sum_chase_prob_std / n_updates
            deploy_chase_rate = sum_deploy_chase_rate / n_updates
            hard_deploy_chase_rate = sum_hard_deploy_chase_rate / n_updates
            opportunity_chase_rate = sum_opportunity_chase_rate / n_updates
            risk_chase_rate = sum_risk_chase_rate / n_updates
            opportunity_chase_prob = sum_opportunity_chase_prob / n_updates
            risk_chase_prob = sum_risk_chase_prob / n_updates
            opportunity_mass = sum_opportunity_mass / n_updates
            risk_mass = sum_risk_mass / n_updates
            total_loss = (
                policy_loss
                + TrainingParameters.EX_VALUE_COEF * value_loss
                + TrainingParameters.ENTROPY_COEF * entropy_loss
                + discrete_reg
            )
        else:
            policy_loss = 0.0
            entropy_loss = 0.0
            value_loss = 0.0
            approx_kl = 0.0
            clipfrac = 0.0
            grad_norm = 0.0
            discrete_reg = 0.0
            mean_chase_prob = 0.0
            mean_max_prob = 0.0
            chase_prob_std = 0.0
            deploy_chase_rate = 0.0
            hard_deploy_chase_rate = 0.0
            opportunity_chase_rate = 0.0
            risk_chase_rate = 0.0
            opportunity_chase_prob = 0.0
            risk_chase_prob = 0.0
            opportunity_mass = 0.0
            risk_mass = 0.0
            total_loss = 0.0

        losses = [total_loss, policy_loss, entropy_loss, value_loss,
                  adv_std, approx_kl, discrete_reg, clipfrac, grad_norm, adv_mean]
        return {
            'losses': losses,
            'il_loss': None,
            'il_filter_ratio': None,
            'aux_loss': None,
            'discrete_reg_loss': discrete_reg,
            'mean_chase_prob': mean_chase_prob,
            'mean_max_prob': mean_max_prob,
            'chase_prob_std': chase_prob_std,
            'deploy_chase_rate': deploy_chase_rate,
            'hard_deploy_chase_rate': hard_deploy_chase_rate,
            'opportunity_chase_rate': opportunity_chase_rate,
            'risk_chase_rate': risk_chase_rate,
            'opportunity_chase_prob': opportunity_chase_prob,
            'risk_chase_prob': risk_chase_prob,
            'opportunity_mass': opportunity_mass,
            'risk_mass': risk_mass,
        }

    def _train_recurrent_discrete_ppo(
        self,
        actor_obs,
        critic_obs,
        returns,
        values,
        actions,
        old_log_probs,
        mask,
        dones,
        actor_hiddens,
        critic_hiddens,
        num_envs: int,
        rollout_steps: int,
        tbptt_steps: int,
        aux_targets=None,
    ):
        dataset_size = actor_obs.shape[0]
        if dataset_size != int(num_envs * rollout_steps):
            raise ValueError(
                f"recurrent train shape mismatch: dataset={dataset_size}, "
                f"num_envs*rollout_steps={int(num_envs * rollout_steps)}"
            )

        raw_advantages = returns - values.squeeze(-1)
        valid_mask = mask > 0
        if valid_mask.sum() > 1:
            adv_std = float(raw_advantages[valid_mask].std().item())
            adv_mean = float(raw_advantages[valid_mask].mean().item())
            advantages = ((raw_advantages - adv_mean) / (adv_std + 1e-8))
        else:
            adv_std = 0.0
            adv_mean = 0.0
            advantages = raw_advantages * 0.0
        advantages = advantages * mask

        actor_obs_seq = actor_obs.reshape(num_envs, rollout_steps, -1)
        critic_obs_seq = critic_obs.reshape(num_envs, rollout_steps, -1)
        actions_seq = actions.reshape(num_envs, rollout_steps, -1)
        old_log_probs_seq = old_log_probs.reshape(num_envs, rollout_steps)
        returns_seq = returns.reshape(num_envs, rollout_steps)
        values_seq = values.reshape(num_envs, rollout_steps, -1)
        advantages_seq = advantages.reshape(num_envs, rollout_steps)
        mask_seq = mask.reshape(num_envs, rollout_steps)
        dones_seq = dones.reshape(num_envs, rollout_steps)
        actor_hidden_seq = self._reshape_recurrent_hidden_buffer(actor_hiddens, num_envs, rollout_steps)
        critic_hidden_seq = self._reshape_recurrent_hidden_buffer(critic_hiddens, num_envs, rollout_steps)
        aux_targets_seq = self._reshape_aux_targets(aux_targets, num_envs, rollout_steps)

        minibatch_transitions = min(TrainingParameters.MINIBATCH_SIZE, dataset_size)
        envs_per_minibatch = max(1, int(minibatch_transitions // max(1, tbptt_steps)))
        n_epochs = int(TrainingParameters.N_EPOCHS)

        sum_policy_loss = 0.0
        sum_entropy_loss = 0.0
        sum_value_loss = 0.0
        sum_approx_kl = 0.0
        sum_clipfrac = 0.0
        sum_grad_norm = 0.0
        sum_discrete_reg = 0.0
        sum_mean_chase_prob = 0.0
        sum_mean_max_prob = 0.0
        sum_chase_prob_std = 0.0
        sum_deploy_chase_rate = 0.0
        sum_hard_deploy_chase_rate = 0.0
        sum_opportunity_chase_rate = 0.0
        sum_risk_chase_rate = 0.0
        sum_opportunity_chase_prob = 0.0
        sum_risk_chase_prob = 0.0
        sum_opportunity_mass = 0.0
        sum_risk_mass = 0.0
        n_updates = 0

        for _ in range(n_epochs):
            env_perm = torch.randperm(num_envs, device=self.device)
            for t0 in range(0, rollout_steps, tbptt_steps):
                t1 = min(t0 + tbptt_steps, rollout_steps)
                for e0 in range(0, num_envs, envs_per_minibatch):
                    env_idx = env_perm[e0:e0 + envs_per_minibatch]
                    if env_idx.numel() == 0:
                        continue

                    mb_actor_seq = actor_obs_seq[env_idx, t0:t1, :]
                    mb_critic_seq = critic_obs_seq[env_idx, t0:t1, :]
                    mb_actions_seq = actions_seq[env_idx, t0:t1, :]
                    mb_old_log_probs_seq = old_log_probs_seq[env_idx, t0:t1]
                    mb_returns_seq = returns_seq[env_idx, t0:t1]
                    mb_values_seq = values_seq[env_idx, t0:t1, :]
                    mb_advantages_seq = advantages_seq[env_idx, t0:t1]
                    mb_mask_seq = mask_seq[env_idx, t0:t1]
                    mb_dones_seq = dones_seq[env_idx, t0:t1]
                    mb_actor_hidden = self._select_recurrent_chunk_hidden(actor_hidden_seq, env_idx, t0)
                    mb_critic_hidden = self._select_recurrent_chunk_hidden(critic_hidden_seq, env_idx, t0)

                    self.net_optimizer.zero_grad(set_to_none=True)
                    policy_output_seq, value_seq, _ = self.network.forward_sequence(
                        mb_actor_seq,
                        mb_critic_seq,
                        dones=mb_dones_seq,
                        actor_hidden=mb_actor_hidden,
                        critic_hidden=mb_critic_hidden,
                    )

                    logits_flat = self._extract_discrete_logits(
                        policy_output_seq.reshape(-1, policy_output_seq.shape[-1])
                    )
                    value_flat = value_seq.reshape(-1)
                    actions_flat = self._prepare_discrete_actions(
                        mb_actions_seq.reshape(-1, mb_actions_seq.shape[-1])
                    )
                    old_log_probs_flat = mb_old_log_probs_seq.reshape(-1)
                    returns_flat = mb_returns_seq.reshape(-1)
                    old_values_flat = mb_values_seq.reshape(-1)
                    advantages_flat = mb_advantages_seq.reshape(-1)
                    mask_flat = mb_mask_seq.reshape(-1)
                    mask_sum = mask_flat.sum().clamp_min(1.0)

                    dist = torch.distributions.Categorical(logits=logits_flat)
                    new_log_probs = dist.log_prob(actions_flat)
                    ratio = torch.exp(new_log_probs - old_log_probs_flat)
                    ratio = torch.clamp(ratio, 0.0, TrainingParameters.RATIO_CLAMP_MAX)
                    surr1 = ratio * advantages_flat
                    surr2 = torch.clamp(
                        ratio,
                        1.0 - TrainingParameters.CLIP_RANGE,
                        1.0 + TrainingParameters.CLIP_RANGE,
                    ) * advantages_flat
                    policy_loss_t = -(torch.min(surr1, surr2) * mask_flat).sum() / mask_sum

                    entropy_loss_t = -(dist.entropy() * mask_flat).sum() / mask_sum

                    value_clipped = old_values_flat + torch.clamp(
                        value_flat - old_values_flat,
                        -TrainingParameters.VALUE_CLIP_RANGE,
                        TrainingParameters.VALUE_CLIP_RANGE,
                    )
                    v_loss1 = (value_flat - returns_flat) ** 2
                    v_loss2 = (value_clipped - returns_flat) ** 2
                    value_loss_t = (torch.max(v_loss1, v_loss2) * mask_flat).sum() / mask_sum

                    total_loss_t = (
                        policy_loss_t
                        + TrainingParameters.EX_VALUE_COEF * value_loss_t
                        + TrainingParameters.ENTROPY_COEF * entropy_loss_t
                    )
                    discrete_reg_t, discrete_reg_stats = self._discrete_policy_regularization(
                        dist,
                        mask_flat,
                        actor_obs=mb_actor_seq.reshape(-1, mb_actor_seq.shape[-1]),
                    )
                    total_loss_t = total_loss_t + discrete_reg_t
                    total_loss_t.backward()

                    with torch.no_grad():
                        mb_kl = (old_log_probs_flat - new_log_probs).mean().item()
                        mb_cf = (torch.abs(ratio - 1.0) > TrainingParameters.CLIP_RANGE).float().mean().item()

                    gn = float(torch.nn.utils.clip_grad_norm_(
                        self.network.parameters(),
                        TrainingParameters.MAX_GRAD_NORM,
                    ).item())
                    for _, param in self.network.named_parameters():
                        if param.grad is not None:
                            param.grad = torch.nan_to_num(param.grad)
                    self.net_optimizer.step()

                    sum_policy_loss += float(policy_loss_t.item())
                    sum_entropy_loss += float(entropy_loss_t.item())
                    sum_value_loss += float(value_loss_t.item())
                    sum_approx_kl += float(mb_kl)
                    sum_clipfrac += float(mb_cf)
                    sum_grad_norm += float(gn)
                    sum_discrete_reg += float(discrete_reg_stats['reg_loss'])
                    sum_mean_chase_prob += float(discrete_reg_stats['chase_prob'])
                    sum_mean_max_prob += float(discrete_reg_stats['max_prob'])
                    sum_chase_prob_std += float(discrete_reg_stats['chase_prob_std'])
                    sum_deploy_chase_rate += float(discrete_reg_stats.get('deploy_chase_rate', 0.0))
                    sum_hard_deploy_chase_rate += float(discrete_reg_stats.get('hard_deploy_chase_rate', 0.0))
                    sum_opportunity_chase_rate += float(discrete_reg_stats.get('opportunity_chase_rate', 0.0))
                    sum_risk_chase_rate += float(discrete_reg_stats.get('risk_chase_rate', 0.0))
                    sum_opportunity_chase_prob += float(discrete_reg_stats.get('opportunity_chase_prob', 0.0))
                    sum_risk_chase_prob += float(discrete_reg_stats.get('risk_chase_prob', 0.0))
                    sum_opportunity_mass += float(discrete_reg_stats.get('opportunity_mass', 0.0))
                    sum_risk_mass += float(discrete_reg_stats.get('risk_mass', 0.0))
                    n_updates += 1

            if n_updates > 0 and abs(sum_approx_kl / n_updates) > 0.03:
                break

        if n_updates > 0:
            policy_loss = sum_policy_loss / n_updates
            entropy_loss = sum_entropy_loss / n_updates
            value_loss = sum_value_loss / n_updates
            approx_kl = sum_approx_kl / n_updates
            clipfrac = sum_clipfrac / n_updates
            grad_norm = sum_grad_norm / n_updates
            discrete_reg = sum_discrete_reg / n_updates
            mean_chase_prob = sum_mean_chase_prob / n_updates
            mean_max_prob = sum_mean_max_prob / n_updates
            chase_prob_std = sum_chase_prob_std / n_updates
            deploy_chase_rate = sum_deploy_chase_rate / n_updates
            hard_deploy_chase_rate = sum_hard_deploy_chase_rate / n_updates
            opportunity_chase_rate = sum_opportunity_chase_rate / n_updates
            risk_chase_rate = sum_risk_chase_rate / n_updates
            opportunity_chase_prob = sum_opportunity_chase_prob / n_updates
            risk_chase_prob = sum_risk_chase_prob / n_updates
            opportunity_mass = sum_opportunity_mass / n_updates
            risk_mass = sum_risk_mass / n_updates
            total_loss = (
                policy_loss
                + TrainingParameters.EX_VALUE_COEF * value_loss
                + TrainingParameters.ENTROPY_COEF * entropy_loss
                + discrete_reg
            )
        else:
            policy_loss = 0.0
            entropy_loss = 0.0
            value_loss = 0.0
            approx_kl = 0.0
            clipfrac = 0.0
            grad_norm = 0.0
            discrete_reg = 0.0
            mean_chase_prob = 0.0
            mean_max_prob = 0.0
            chase_prob_std = 0.0
            deploy_chase_rate = 0.0
            hard_deploy_chase_rate = 0.0
            opportunity_chase_rate = 0.0
            risk_chase_rate = 0.0
            opportunity_chase_prob = 0.0
            risk_chase_prob = 0.0
            opportunity_mass = 0.0
            risk_mass = 0.0
            total_loss = 0.0

        losses = [total_loss, policy_loss, entropy_loss, value_loss,
                  adv_std, approx_kl, discrete_reg, clipfrac, grad_norm, adv_mean]
        return {
            'losses': losses,
            'il_loss': None,
            'il_filter_ratio': None,
            'aux_loss': None,
            'discrete_reg_loss': discrete_reg,
            'mean_chase_prob': mean_chase_prob,
            'mean_max_prob': mean_max_prob,
            'chase_prob_std': chase_prob_std,
            'deploy_chase_rate': deploy_chase_rate,
            'hard_deploy_chase_rate': hard_deploy_chase_rate,
            'opportunity_chase_rate': opportunity_chase_rate,
            'risk_chase_rate': risk_chase_rate,
            'opportunity_chase_prob': opportunity_chase_prob,
            'risk_chase_prob': risk_chase_prob,
            'opportunity_mass': opportunity_mass,
            'risk_mass': risk_mass,
        }

    def _train_recurrent_mixed(
        self,
        actor_obs,
        critic_obs,
        returns,
        values,
        actions,
        old_log_probs,
        expert_actions,
        il_weight: float,
        mask,
        dones,
        actor_hiddens,
        critic_hiddens,
        num_envs: int,
        rollout_steps: int,
        tbptt_steps: int,
        aux_targets=None,
    ):
        dataset_size = actor_obs.shape[0]
        if dataset_size != int(num_envs * rollout_steps):
            raise ValueError(
                f"recurrent mixed train shape mismatch: dataset={dataset_size}, "
                f"num_envs*rollout_steps={int(num_envs * rollout_steps)}"
            )
        if self.is_discrete_policy:
            raise NotImplementedError('Recurrent mixed training does not support discrete policies.')

        raw_advantages = returns - values.squeeze(-1)
        valid_mask = mask > 0
        if valid_mask.sum() > 1:
            adv_std = float(raw_advantages[valid_mask].std().item())
            adv_mean = float(raw_advantages[valid_mask].mean().item())
            advantages = ((raw_advantages - adv_mean) / (adv_std + 1e-8))
        else:
            adv_std = 0.0
            adv_mean = 0.0
            advantages = raw_advantages * 0.0
        advantages = advantages * mask

        actor_obs_seq = actor_obs.reshape(num_envs, rollout_steps, -1)
        critic_obs_seq = critic_obs.reshape(num_envs, rollout_steps, -1)
        actions_seq = actions.reshape(num_envs, rollout_steps, -1)
        old_log_probs_seq = old_log_probs.reshape(num_envs, rollout_steps)
        returns_seq = returns.reshape(num_envs, rollout_steps)
        values_seq = values.reshape(num_envs, rollout_steps, -1)
        advantages_seq = advantages.reshape(num_envs, rollout_steps)
        expert_actions_seq = expert_actions.reshape(num_envs, rollout_steps, -1)
        mask_seq = mask.reshape(num_envs, rollout_steps)
        dones_seq = dones.reshape(num_envs, rollout_steps)
        actor_hidden_seq = self._reshape_recurrent_hidden_buffer(actor_hiddens, num_envs, rollout_steps)
        critic_hidden_seq = self._reshape_recurrent_hidden_buffer(critic_hiddens, num_envs, rollout_steps)
        aux_targets_seq = self._reshape_aux_targets(aux_targets, num_envs, rollout_steps)

        minibatch_transitions = min(TrainingParameters.MINIBATCH_SIZE, dataset_size)
        envs_per_minibatch = max(1, int(minibatch_transitions // max(1, tbptt_steps)))
        n_epochs = int(TrainingParameters.N_EPOCHS)
        rl_weight = 1.0 - float(il_weight)

        sum_policy_loss = 0.0
        sum_entropy_loss = 0.0
        sum_value_loss = 0.0
        sum_il_loss = 0.0
        sum_total_loss = 0.0
        sum_rl_loss = 0.0
        sum_aux_loss = 0.0
        sum_approx_kl = 0.0
        sum_clipfrac = 0.0
        sum_grad_norm = 0.0
        n_updates = 0

        for _ in range(n_epochs):
            env_perm = torch.randperm(num_envs, device=self.device)
            for t0 in range(0, rollout_steps, tbptt_steps):
                t1 = min(t0 + tbptt_steps, rollout_steps)
                for e0 in range(0, num_envs, envs_per_minibatch):
                    env_idx = env_perm[e0:e0 + envs_per_minibatch]
                    if env_idx.numel() == 0:
                        continue

                    mb_actor_seq = actor_obs_seq[env_idx, t0:t1, :]
                    mb_critic_seq = critic_obs_seq[env_idx, t0:t1, :]
                    mb_actions_seq = actions_seq[env_idx, t0:t1, :]
                    mb_old_log_probs_seq = old_log_probs_seq[env_idx, t0:t1]
                    mb_returns_seq = returns_seq[env_idx, t0:t1]
                    mb_values_seq = values_seq[env_idx, t0:t1, :]
                    mb_advantages_seq = advantages_seq[env_idx, t0:t1]
                    mb_expert_seq = expert_actions_seq[env_idx, t0:t1, :]
                    mb_mask_seq = mask_seq[env_idx, t0:t1]
                    mb_dones_seq = dones_seq[env_idx, t0:t1]
                    mb_aux_targets = None
                    if aux_targets_seq is not None:
                        mb_aux_targets = {
                            key: value[env_idx, t0:t1]
                            for key, value in aux_targets_seq.items()
                        }
                    mb_actor_hidden = self._select_recurrent_chunk_hidden(actor_hidden_seq, env_idx, t0)
                    mb_critic_hidden = self._select_recurrent_chunk_hidden(critic_hidden_seq, env_idx, t0)

                    self.net_optimizer.zero_grad(set_to_none=True)
                    if mb_aux_targets is not None and hasattr(self.network, 'forward_sequence_with_aux'):
                        mean_seq, value_seq, log_std_seq, aux_outputs = self.network.forward_sequence_with_aux(
                            mb_actor_seq,
                            mb_critic_seq,
                            dones=mb_dones_seq,
                            actor_hidden=mb_actor_hidden,
                            critic_hidden=mb_critic_hidden,
                        )
                    else:
                        mean_seq, value_seq, log_std_seq = self.network.forward_sequence(
                            mb_actor_seq,
                            mb_critic_seq,
                            dones=mb_dones_seq,
                            actor_hidden=mb_actor_hidden,
                            critic_hidden=mb_critic_hidden,
                        )
                        aux_outputs = None

                    mean_flat = mean_seq.reshape(-1, mean_seq.shape[-1])
                    value_flat = value_seq.reshape(-1)
                    log_std_flat = log_std_seq.reshape(-1, log_std_seq.shape[-1])
                    actions_flat = mb_actions_seq.reshape(-1, mb_actions_seq.shape[-1])
                    old_log_probs_flat = mb_old_log_probs_seq.reshape(-1)
                    returns_flat = mb_returns_seq.reshape(-1)
                    old_values_flat = mb_values_seq.reshape(-1)
                    advantages_flat = mb_advantages_seq.reshape(-1)
                    expert_flat = mb_expert_seq.reshape(-1, mb_expert_seq.shape[-1])
                    mask_flat = mb_mask_seq.reshape(-1)
                    mask_sum = mask_flat.sum().clamp_min(1.0)

                    pred_actions = torch.tanh(mean_flat)
                    il_mse = ((pred_actions - expert_flat) ** 2).sum(dim=-1)
                    il_loss_t = (il_mse * mask_flat).sum() / mask_sum

                    new_log_probs = self._log_prob_from_pre_tanh(actions_flat, mean_flat, log_std_flat)
                    ratio = torch.exp(new_log_probs - old_log_probs_flat)
                    ratio = torch.clamp(ratio, 0.0, TrainingParameters.RATIO_CLAMP_MAX)
                    surr1 = ratio * advantages_flat
                    surr2 = torch.clamp(
                        ratio,
                        1.0 - TrainingParameters.CLIP_RANGE,
                        1.0 + TrainingParameters.CLIP_RANGE,
                    ) * advantages_flat
                    policy_loss_t = -(torch.min(surr1, surr2) * mask_flat).sum() / mask_sum

                    ent = (0.5 * (1.0 + np.log(2 * np.pi)) + log_std_flat).sum(dim=-1)
                    entropy_loss_t = -(ent * mask_flat).sum() / mask_sum

                    value_clipped = old_values_flat + torch.clamp(
                        value_flat - old_values_flat,
                        -TrainingParameters.VALUE_CLIP_RANGE,
                        TrainingParameters.VALUE_CLIP_RANGE,
                    )
                    v_loss1 = (value_flat - returns_flat) ** 2
                    v_loss2 = (value_clipped - returns_flat) ** 2
                    value_loss_t = (torch.max(v_loss1, v_loss2) * mask_flat).sum() / mask_sum

                    rl_loss_t = (
                        policy_loss_t
                        + TrainingParameters.EX_VALUE_COEF * value_loss_t
                        + TrainingParameters.ENTROPY_COEF * entropy_loss_t
                    )
                    total_loss_t = float(il_weight) * il_loss_t + rl_weight * rl_loss_t
                    aux_loss_t, _aux_parts = self._multitask_auxiliary_loss(
                        aux_outputs,
                        mb_aux_targets,
                        mb_mask_seq,
                    )
                    if aux_loss_t is not None:
                        total_loss_t = total_loss_t + aux_loss_t
                    total_loss_t.backward()

                    with torch.no_grad():
                        mb_kl = (old_log_probs_flat - new_log_probs).mean().item()
                        mb_cf = (torch.abs(ratio - 1.0) > TrainingParameters.CLIP_RANGE).float().mean().item()

                    gn = float(torch.nn.utils.clip_grad_norm_(
                        self.network.parameters(),
                        TrainingParameters.MAX_GRAD_NORM,
                    ).item())
                    for _, param in self.network.named_parameters():
                        if param.grad is not None:
                            param.grad = torch.nan_to_num(param.grad)
                    self.net_optimizer.step()

                    sum_policy_loss += float(policy_loss_t.item())
                    sum_entropy_loss += float(entropy_loss_t.item())
                    sum_value_loss += float(value_loss_t.item())
                    sum_il_loss += float(il_loss_t.item())
                    if aux_loss_t is not None:
                        sum_aux_loss += float(aux_loss_t.detach().item())
                    sum_total_loss += float(total_loss_t.item())
                    sum_rl_loss += float(rl_loss_t.item())
                    sum_approx_kl += float(mb_kl)
                    sum_clipfrac += float(mb_cf)
                    sum_grad_norm += float(gn)
                    n_updates += 1

            if n_updates > 0 and abs(sum_approx_kl / n_updates) > 0.03:
                break

        if n_updates > 0:
            policy_loss = sum_policy_loss / n_updates
            entropy_loss = sum_entropy_loss / n_updates
            value_loss = sum_value_loss / n_updates
            il_loss_value = sum_il_loss / n_updates
            total_loss = sum_total_loss / n_updates
            approx_kl_avg = sum_approx_kl / n_updates
            clipfrac_avg = sum_clipfrac / n_updates
            grad_norm_avg = sum_grad_norm / n_updates
            rl_loss_avg = sum_rl_loss / n_updates
            aux_loss_value = sum_aux_loss / n_updates
        else:
            policy_loss = 0.0
            entropy_loss = 0.0
            value_loss = 0.0
            il_loss_value = 0.0
            total_loss = 0.0
            approx_kl_avg = 0.0
            clipfrac_avg = 0.0
            grad_norm_avg = 0.0
            rl_loss_avg = 0.0
            aux_loss_value = 0.0

        losses = [total_loss, policy_loss, entropy_loss, value_loss,
                  adv_std, approx_kl_avg, 0.0, clipfrac_avg, grad_norm_avg, adv_mean]
        return {
            'losses': losses,
            'il_loss': il_loss_value,
            'il_weight': float(il_weight),
            'rl_loss': rl_loss_avg,
            'aux_loss': aux_loss_value,
        }

    @torch.no_grad()
    def step(self, actor_obs, critic_obs):
        actor_tensor = self._to_tensor(actor_obs)
        critic_tensor = self._to_tensor(critic_obs)
        if self.is_discrete_policy:
            policy_output, value = self._forward_discrete_policy(actor_tensor, critic_tensor)
            dist = self._categorical_dist(policy_output)
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)
            action = self._format_discrete_action(action_idx)
            pre_tanh = action.clone()
        elif self.is_recurrent and hasattr(self.network, 'act_recurrent'):
            action, log_prob, pre_tanh, value, next_actor_hidden, next_critic_hidden = self.network.act_recurrent(
                actor_tensor, critic_tensor,
                actor_hidden=self._actor_hidden,
                critic_hidden=self._critic_hidden,
            )
            self._actor_hidden = next_actor_hidden.detach() if next_actor_hidden is not None else None
            self._critic_hidden = next_critic_hidden.detach() if next_critic_hidden is not None else None
        else:
            policy_output, value, log_std_or_aux = self.network(actor_tensor, critic_tensor)
            action, log_prob, pre_tanh, value = self._policy_action_from_output(
                policy_output,
                value,
                log_std_or_aux,
                greedy=False,
            )

        return action[0].cpu().numpy(), pre_tanh[0].cpu().numpy(),                float(value.item()), float(log_prob.item())

    @torch.no_grad()
    def evaluate(self, actor_obs, critic_obs, greedy=True):
        actor_tensor = self._to_tensor(actor_obs)
        critic_tensor = self._to_tensor(critic_obs)
        if self.is_discrete_policy:
            policy_output, value = self._forward_discrete_policy(actor_tensor, critic_tensor)
            logits = self._extract_discrete_logits(policy_output)
            action_idx = (
                self._select_discrete_greedy_action_idx(logits, actor_tensor=actor_tensor)
                if greedy else torch.distributions.Categorical(logits=logits).sample()
            )
            dist = torch.distributions.Categorical(logits=logits)
            log_prob = dist.log_prob(action_idx)
            action = self._format_discrete_action(action_idx)
            pre_tanh = action.clone()
        elif self.is_recurrent and hasattr(self.network, 'forward_recurrent'):
            mean, value, log_std, next_actor_hidden, next_critic_hidden = self.network.forward_recurrent(
                actor_tensor, critic_tensor,
                actor_hidden=self._actor_hidden,
                critic_hidden=self._critic_hidden,
            )
            self._actor_hidden = next_actor_hidden.detach() if next_actor_hidden is not None else None
            self._critic_hidden = next_critic_hidden.detach() if next_critic_hidden is not None else None
            pre_tanh = mean if greedy else mean + torch.exp(log_std) * torch.randn_like(mean)
            action = torch.tanh(pre_tanh)
            log_prob = self._log_prob_from_pre_tanh(pre_tanh, mean, log_std)
        else:
            policy_output, value, log_std_or_aux = self.network(actor_tensor, critic_tensor)
            action, log_prob, pre_tanh, value = self._policy_action_from_output(
                policy_output,
                value,
                log_std_or_aux,
                greedy=greedy,
            )

        return action[0].cpu().numpy(), pre_tanh[0].cpu().numpy(),                float(value.item()), float(log_prob.item())

    def train(self, actor_obs=None, critic_obs=None, returns=None, values=None,
              actions=None, old_log_probs=None, mask=None,
              actor_hiddens=None, critic_hiddens=None,
              aux_targets=None,
              writer=None, global_step=None, perf_dict=None,
              dones=None, num_envs=None, rollout_steps=None, tbptt_steps=None):
        """
        纯RL训练（PPO）- Mini-batch多轮更新
        
        标准PPO流程:
        1. 在full batch上计算并标准化advantage
        2. 对数据做N_EPOCHS轮随机shuffle
        3. 每轮内按MINIBATCH_SIZE切分，每个mini-batch做一次梯度更新
        """
        total_loss = 0.0
        policy_loss = 0.0
        entropy_loss = 0.0
        value_loss = 0.0
        approx_kl = 0.0
        clipfrac = 0.0
        grad_norm = 0.0
        adv_mean = 0.0
        adv_std = 0.0
        
        if actor_obs is not None:
            actor_obs = torch.as_tensor(actor_obs, dtype=torch.float32, device=self.device)
            critic_obs = torch.as_tensor(critic_obs, dtype=torch.float32, device=self.device)
            returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
            values = torch.as_tensor(values, dtype=torch.float32, device=self.device)
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
            old_log_probs = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
            if actor_hiddens is not None:
                actor_hiddens = torch.as_tensor(actor_hiddens, dtype=torch.float32, device=self.device)
            if critic_hiddens is not None:
                critic_hiddens = torch.as_tensor(critic_hiddens, dtype=torch.float32, device=self.device)
            aux_targets = self._to_aux_target_tensors(aux_targets)
            
            if actor_obs.dim() == 1:
                actor_obs = actor_obs.unsqueeze(0)
            if critic_obs.dim() == 1:
                critic_obs = critic_obs.unsqueeze(0)
            if returns.dim() == 0:
                returns = returns.unsqueeze(0)
            if values.dim() == 0:
                values = values.unsqueeze(0)
            if old_log_probs.dim() == 0:
                old_log_probs = old_log_probs.unsqueeze(0)
            if actions.dim() == 1:
                actions = actions.unsqueeze(0)
            if mask is None:
                mask = torch.ones_like(returns, dtype=torch.float32, device=self.device)
            else:
                mask = torch.as_tensor(mask, dtype=torch.float32, device=self.device)
                if mask.dim() == 0:
                    mask = mask.unsqueeze(0)
            
            # ====== 在full batch上计算并标准化advantage ======
            raw_advantages = returns - values.squeeze(-1)
            valid_mask = mask > 0
            if valid_mask.sum() > 1:
                adv_std = float(raw_advantages[valid_mask].std().item())
                adv_mean = float(raw_advantages[valid_mask].mean().item())
                advantages = ((raw_advantages - adv_mean) / (adv_std + 1e-8))
            else:
                advantages = raw_advantages * 0.0
            advantages = advantages * mask

            if self.is_recurrent:
                if dones is None or num_envs is None or rollout_steps is None:
                    raise ValueError(
                        'Recurrent network training requires dones, num_envs, and rollout_steps.'
                    )
                dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
                if dones.dim() == 0:
                    dones = dones.unsqueeze(0)
                tbptt = int(tbptt_steps) if tbptt_steps is not None else int(getattr(NetParameters, 'CONTEXT_WINDOW', 32))
                if self.is_discrete_policy:
                    return self._train_recurrent_discrete_ppo(
                        actor_obs=actor_obs,
                        critic_obs=critic_obs,
                        returns=returns,
                        values=values,
                        actions=actions,
                        old_log_probs=old_log_probs,
                        mask=mask,
                        dones=dones,
                        actor_hiddens=actor_hiddens,
                        critic_hiddens=critic_hiddens,
                        num_envs=int(num_envs),
                        rollout_steps=int(rollout_steps),
                        tbptt_steps=max(1, tbptt),
                    )
                return self._train_recurrent_ppo(
                    actor_obs=actor_obs,
                    critic_obs=critic_obs,
                    returns=returns,
                    values=values,
                    actions=actions,
                    old_log_probs=old_log_probs,
                    mask=mask,
                    dones=dones,
                    actor_hiddens=actor_hiddens,
                    critic_hiddens=critic_hiddens,
                    aux_targets=aux_targets,
                    num_envs=int(num_envs),
                    rollout_steps=int(rollout_steps),
                    tbptt_steps=max(1, tbptt),
                )
            
            if self.is_discrete_policy:
                return self._train_discrete_ppo(
                    actor_obs=actor_obs,
                    critic_obs=critic_obs,
                    returns=returns,
                    values=values,
                    actions=actions,
                    old_log_probs=old_log_probs,
                    mask=mask,
                    advantages=advantages,
                    adv_mean=adv_mean,
                    adv_std=adv_std,
                )

            # ====== Mini-batch多轮更新 ======
            dataset_size = actor_obs.shape[0]
            minibatch_size = min(TrainingParameters.MINIBATCH_SIZE, dataset_size)
            n_epochs = int(TrainingParameters.N_EPOCHS)
            
            # 累计统计量
            sum_policy_loss = 0.0
            sum_entropy_loss = 0.0
            sum_value_loss = 0.0
            sum_aux_loss = 0.0
            sum_approx_kl = 0.0
            sum_clipfrac = 0.0
            sum_grad_norm = 0.0
            n_updates = 0
            
            for epoch in range(n_epochs):
                indices = torch.randperm(dataset_size, device=self.device)
                
                for start in range(0, dataset_size, minibatch_size):
                    end = min(start + minibatch_size, dataset_size)
                    mb_idx = indices[start:end]
                    
                    mb_actor_obs = actor_obs[mb_idx]
                    mb_critic_obs = critic_obs[mb_idx]
                    mb_actions = actions[mb_idx]
                    mb_old_log_probs = old_log_probs[mb_idx]
                    mb_returns = returns[mb_idx]
                    mb_values = values[mb_idx]
                    mb_advantages = advantages[mb_idx]
                    mb_mask = mask[mb_idx]
                    
                    self.net_optimizer.zero_grad(set_to_none=True)
                    
                    policy_output, value_flat, log_std_or_aux = self.network(mb_actor_obs, mb_critic_obs)
                    new_values = value_flat.squeeze(-1)
                    new_log_probs = self._log_prob_from_pre_tanh(mb_actions, policy_output, log_std_or_aux)
                    
                    # Policy loss (clipped surrogate)
                    ratio = torch.exp(new_log_probs - mb_old_log_probs)
                    ratio = torch.clamp(ratio, 0.0, TrainingParameters.RATIO_CLAMP_MAX)
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1.0 - TrainingParameters.CLIP_RANGE,
                                       1.0 + TrainingParameters.CLIP_RANGE) * mb_advantages
                    mb_mask_sum = mb_mask.sum().clamp_min(1.0)
                    policy_loss_t = -torch.min(surr1, surr2).sum() / mb_mask_sum
                    
                    # Entropy loss
                    ent = self._policy_entropy(policy_output, log_std_or_aux)
                    entropy_loss_t = -(ent * mb_mask).sum() / mb_mask_sum
                    
                    # Value loss (clipped)
                    value_clipped = mb_values.squeeze(-1) + torch.clamp(
                        new_values - mb_values.squeeze(-1),
                        -TrainingParameters.VALUE_CLIP_RANGE,
                        TrainingParameters.VALUE_CLIP_RANGE
                    )
                    v_loss1 = (new_values - mb_returns) ** 2
                    v_loss2 = (value_clipped - mb_returns) ** 2
                    value_loss_t = (torch.max(v_loss1, v_loss2) * mb_mask).sum() / mb_mask_sum
                    
                    total_loss_t = (policy_loss_t +
                                   TrainingParameters.EX_VALUE_COEF * value_loss_t +
                                   TrainingParameters.ENTROPY_COEF * entropy_loss_t)
                    aux_loss_t = self._compute_auxiliary_loss(mb_actor_obs, mb_critic_obs, mask=mb_mask)
                    if aux_loss_t is not None:
                        total_loss_t = total_loss_t + self._scale_auxiliary_loss(aux_loss_t)
                    total_loss_t.backward()
                    
                    with torch.no_grad():
                        mb_kl = (mb_old_log_probs - new_log_probs).mean().item()
                        mb_cf = (torch.abs(ratio - 1.0) > TrainingParameters.CLIP_RANGE).float().mean().item()
                    
                    gn = float(torch.nn.utils.clip_grad_norm_(
                        self.network.parameters(),
                        TrainingParameters.MAX_GRAD_NORM
                    ).item())
                    
                    for name, param in self.network.named_parameters():
                        if param.grad is not None:
                            param.grad = torch.nan_to_num(param.grad)
                    
                    self.net_optimizer.step()
                    
                    sum_policy_loss += policy_loss_t.item()
                    sum_entropy_loss += entropy_loss_t.item()
                    sum_value_loss += value_loss_t.item()
                    if aux_loss_t is not None:
                        sum_aux_loss += float(self._scale_auxiliary_loss(aux_loss_t.detach()).item())
                    sum_approx_kl += mb_kl
                    sum_clipfrac += mb_cf
                    sum_grad_norm += gn
                    n_updates += 1
                
                # Early stopping: 如果KL散度过大，停止后续epoch
                avg_kl_so_far = sum_approx_kl / max(n_updates, 1)
                if abs(avg_kl_so_far) > 0.03:
                    break
            
            # 计算平均统计量
            if n_updates > 0:
                policy_loss = sum_policy_loss / n_updates
                entropy_loss = sum_entropy_loss / n_updates
                value_loss = sum_value_loss / n_updates
                approx_kl = sum_approx_kl / n_updates
                clipfrac = sum_clipfrac / n_updates
                grad_norm = sum_grad_norm / n_updates
                aux_loss_value = sum_aux_loss / n_updates
                total_loss = (policy_loss +
                             TrainingParameters.EX_VALUE_COEF * value_loss +
                             TrainingParameters.ENTROPY_COEF * entropy_loss +
                             aux_loss_value)
            else:
                policy_loss = 0.0
                entropy_loss = 0.0
                value_loss = 0.0
                approx_kl = 0.0
                clipfrac = 0.0
                grad_norm = 0.0
                aux_loss_value = 0.0
                total_loss = 0.0
        
        losses = [total_loss, policy_loss, entropy_loss, value_loss,
                  adv_std, approx_kl, 0.0, clipfrac, grad_norm, adv_mean]
        return {
            'losses': losses,
            'il_loss': None,
            'il_filter_ratio': None,
            'aux_loss': aux_loss_value,
        }
    
    def train_mixed(self, actor_obs, critic_obs, actions, old_log_probs,
                    returns, values, expert_actions, il_weight,
                    mask=None, actor_hiddens=None, critic_hiddens=None,
                    aux_targets=None,
                    writer=None, global_step=None,
                    dones=None, num_envs=None, rollout_steps=None, tbptt_steps=None):
        """
        IL+RL混合训练（加权组合）- Mini-batch多轮更新
        
        total_loss = il_weight * il_loss + (1 - il_weight) * rl_loss
        """
        # 转换为tensor
        actor_obs = torch.as_tensor(actor_obs, dtype=torch.float32, device=self.device)
        critic_obs = torch.as_tensor(critic_obs, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        values = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        expert_actions = torch.as_tensor(expert_actions, dtype=torch.float32, device=self.device)
        if actor_hiddens is not None:
            actor_hiddens = torch.as_tensor(actor_hiddens, dtype=torch.float32, device=self.device)
        if critic_hiddens is not None:
            critic_hiddens = torch.as_tensor(critic_hiddens, dtype=torch.float32, device=self.device)
        aux_targets = self._to_aux_target_tensors(aux_targets)
        
        # 确保维度正确
        if actor_obs.dim() == 1:
            actor_obs = actor_obs.unsqueeze(0)
        if critic_obs.dim() == 1:
            critic_obs = critic_obs.unsqueeze(0)
        if returns.dim() == 0:
            returns = returns.unsqueeze(0)
        if values.dim() == 0:
            values = values.unsqueeze(0)
        if old_log_probs.dim() == 0:
            old_log_probs = old_log_probs.unsqueeze(0)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        if expert_actions.dim() == 1:
            expert_actions = expert_actions.unsqueeze(0)
        
        if mask is None:
            mask = torch.ones_like(returns, dtype=torch.float32, device=self.device)
        else:
            mask = torch.as_tensor(mask, dtype=torch.float32, device=self.device)
            if mask.dim() == 0:
                mask = mask.unsqueeze(0)

        if self.is_recurrent:
            if dones is None or num_envs is None or rollout_steps is None:
                raise ValueError(
                    'Recurrent mixed training requires dones, num_envs, and rollout_steps.'
                )
            dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
            if dones.dim() == 0:
                dones = dones.unsqueeze(0)
            tbptt = int(tbptt_steps) if tbptt_steps is not None else int(getattr(NetParameters, 'CONTEXT_WINDOW', 32))
            return self._train_recurrent_mixed(
                actor_obs=actor_obs,
                critic_obs=critic_obs,
                returns=returns,
                values=values,
                actions=actions,
                old_log_probs=old_log_probs,
                expert_actions=expert_actions,
                il_weight=il_weight,
                mask=mask,
                dones=dones,
                actor_hiddens=actor_hiddens,
                critic_hiddens=critic_hiddens,
                aux_targets=aux_targets,
                num_envs=int(num_envs),
                rollout_steps=int(rollout_steps),
                tbptt_steps=max(1, tbptt),
            )

        # ====== 在full batch上计算并标准化advantage ======
        raw_advantages = returns - values.squeeze(-1)
        valid_mask = mask > 0
        if valid_mask.sum() > 1:
            adv_std = float(raw_advantages[valid_mask].std().item())
            adv_mean = float(raw_advantages[valid_mask].mean().item())
            advantages = ((raw_advantages - adv_mean) / (adv_std + 1e-8))
        else:
            adv_std = 0.0
            adv_mean = 0.0
            advantages = raw_advantages * 0.0
        advantages = advantages * mask
        
        # ====== Mini-batch多轮更新 ======
        dataset_size = actor_obs.shape[0]
        minibatch_size = min(TrainingParameters.MINIBATCH_SIZE, dataset_size)
        n_epochs = int(TrainingParameters.N_EPOCHS)
        rl_weight = 1.0 - il_weight
        
        sum_policy_loss = 0.0
        sum_entropy_loss = 0.0
        sum_value_loss = 0.0
        sum_il_loss = 0.0
        sum_total_loss = 0.0
        sum_rl_loss = 0.0
        sum_aux_loss = 0.0
        sum_approx_kl = 0.0
        sum_clipfrac = 0.0
        sum_grad_norm = 0.0
        n_updates = 0
        
        for epoch in range(n_epochs):
            indices = torch.randperm(dataset_size, device=self.device)
            
            for start in range(0, dataset_size, minibatch_size):
                end = min(start + minibatch_size, dataset_size)
                mb_idx = indices[start:end]
                
                mb_actor_obs = actor_obs[mb_idx]
                mb_critic_obs = critic_obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_values = values[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_mask = mask[mb_idx]
                mb_expert_actions = expert_actions[mb_idx]
                
                self.net_optimizer.zero_grad(set_to_none=True)
                
                policy_output, value_flat, log_std_or_aux = self.network(mb_actor_obs, mb_critic_obs)
                new_values = value_flat.squeeze(-1)
                new_log_probs = self._log_prob_from_pre_tanh(mb_actions, policy_output, log_std_or_aux)
                mb_mask_sum = mb_mask.sum().clamp_min(1.0)
                
                # ========== IL Loss ==========
                pred_actions = self._policy_deterministic_action(policy_output, log_std_or_aux)
                il_mse = ((pred_actions - mb_expert_actions) ** 2).sum(dim=-1)
                il_loss_t = (il_mse * mb_mask).sum() / mb_mask_sum
                
                # ========== RL Loss (PPO) ==========
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                ratio = torch.clamp(ratio, 0.0, TrainingParameters.RATIO_CLAMP_MAX)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - TrainingParameters.CLIP_RANGE,
                                   1.0 + TrainingParameters.CLIP_RANGE) * mb_advantages
                policy_loss_t = -torch.min(surr1, surr2).sum() / mb_mask_sum
                
                ent = self._policy_entropy(policy_output, log_std_or_aux)
                entropy_loss_t = -(ent * mb_mask).sum() / mb_mask_sum
                
                value_clipped = mb_values.squeeze(-1) + torch.clamp(
                    new_values - mb_values.squeeze(-1),
                    -TrainingParameters.VALUE_CLIP_RANGE,
                    TrainingParameters.VALUE_CLIP_RANGE
                )
                v_loss1 = (new_values - mb_returns) ** 2
                v_loss2 = (value_clipped - mb_returns) ** 2
                value_loss_t = (torch.max(v_loss1, v_loss2) * mb_mask).sum() / mb_mask_sum
                
                rl_loss_t = (policy_loss_t +
                            TrainingParameters.EX_VALUE_COEF * value_loss_t +
                            TrainingParameters.ENTROPY_COEF * entropy_loss_t)
                
                # ========== 加权组合 ==========
                total_loss_t = il_weight * il_loss_t + rl_weight * rl_loss_t
                aux_loss_t = self._compute_auxiliary_loss(mb_actor_obs, mb_critic_obs, mask=mb_mask)
                if aux_loss_t is not None:
                    total_loss_t = total_loss_t + self._scale_auxiliary_loss(aux_loss_t)
                total_loss_t.backward()
                
                with torch.no_grad():
                    mb_kl = (mb_old_log_probs - new_log_probs).mean().item()
                    mb_cf = (torch.abs(ratio - 1.0) > TrainingParameters.CLIP_RANGE).float().mean().item()
                
                gn = float(torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    TrainingParameters.MAX_GRAD_NORM
                ).item())
                
                for name, param in self.network.named_parameters():
                    if param.grad is not None:
                        param.grad = torch.nan_to_num(param.grad)
                
                self.net_optimizer.step()
                
                sum_policy_loss += policy_loss_t.item()
                sum_entropy_loss += entropy_loss_t.item()
                sum_value_loss += value_loss_t.item()
                sum_il_loss += il_loss_t.item()
                sum_rl_loss += rl_loss_t.item()
                if aux_loss_t is not None:
                    sum_aux_loss += float(self._scale_auxiliary_loss(aux_loss_t.detach()).item())
                sum_total_loss += total_loss_t.item()
                sum_approx_kl += mb_kl
                sum_clipfrac += mb_cf
                sum_grad_norm += gn
                n_updates += 1
            
            # Early stopping
            avg_kl_so_far = sum_approx_kl / max(n_updates, 1)
            if abs(avg_kl_so_far) > 0.03:
                break
        
        # 计算平均统计量
        if n_updates > 0:
            policy_loss = sum_policy_loss / n_updates
            entropy_loss = sum_entropy_loss / n_updates
            value_loss = sum_value_loss / n_updates
            il_loss_value = sum_il_loss / n_updates
            total_loss = sum_total_loss / n_updates
            approx_kl_avg = sum_approx_kl / n_updates
            clipfrac_avg = sum_clipfrac / n_updates
            grad_norm_avg = sum_grad_norm / n_updates
            rl_loss_avg = sum_rl_loss / n_updates
            aux_loss_value = sum_aux_loss / n_updates
        else:
            policy_loss = 0.0
            entropy_loss = 0.0
            value_loss = 0.0
            il_loss_value = 0.0
            total_loss = 0.0
            approx_kl_avg = 0.0
            clipfrac_avg = 0.0
            grad_norm_avg = 0.0
            rl_loss_avg = 0.0
            aux_loss_value = 0.0
        
        losses = [total_loss, policy_loss, entropy_loss, value_loss,
                  adv_std, approx_kl_avg, 0.0, clipfrac_avg, grad_norm_avg, adv_mean]
        
        return {
            'losses': losses,
            'il_loss': il_loss_value,
            'il_weight': il_weight,
            'rl_loss': rl_loss_avg,
            'aux_loss': aux_loss_value,
        }
    
    def update_learning_rate(self, new_lr):
        self.current_lr = float(new_lr)
        for group in self.net_optimizer.param_groups:
            group['lr'] = self.current_lr
            
    def save(self, path, step=None, reward=None, extra_metadata=None):
        """
        保存模型检查点
        
        Args:
            path: 保存路径
            step: 当前训练步数 (用于RETRAIN恢复)
            reward: 当前最佳奖励 (用于RETRAIN恢复)
            extra_metadata: 额外的checkpoint元数据
        """
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        checkpoint = {
            'model': self.get_weights(),
            'step': step if step is not None else 0,
            'reward': reward if reward is not None else -float('inf'),
            'network_type': self.network_type,
            'hrl_num_skills': int(getattr(NetParameters, 'HRL_NUM_SKILLS', 2)),
            'hrl_duration_bins': tuple(int(v) for v in getattr(NetParameters, 'HRL_DURATION_BINS', (1,))),
            'hrl_top_action_dim': int(getattr(NetParameters, 'HRL_TOP_ACTION_DIM', 3)),
            'hrl_top_discrete_action_dim': int(
                getattr(NetParameters, 'HRL_TOP_DISCRETE_ACTION_DIM', getattr(NetParameters, 'HRL_NUM_SKILLS', 2))
            ),
        }
        nmn_stage = self.get_nmn_stage()
        if nmn_stage is not None:
            checkpoint['nmn_cl_stage'] = int(nmn_stage)
        if isinstance(extra_metadata, dict):
            checkpoint.update(extra_metadata)
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            self._load_state_dict_compat(self.network, checkpoint['model'])
        else:
            self._load_state_dict_compat(self.network, checkpoint)
        self.reset_recurrent_state()
