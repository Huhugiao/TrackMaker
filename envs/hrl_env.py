import math
import copy
import os
import random
import sys
from typing import Optional, Dict
import gymnasium as gym
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import map_config
from envs.tad_env import TADEnv
from configs.skill_config import NetParameters
from networks import create_network
from skill.util import build_critic_observation
from policies.attacker_global import (
    SUPPORTED_STRATEGIES,
    TRAINING_STRATEGIES,
    AttackerGlobalPolicy,
)


class HRLEnv(gym.Env):
    """HRL env with macro top-level actions.

    Current top-level policy can operate in two modes:
      - legacy discrete-over-skills
      - joint discrete macro actions over (skill, duration_bin)

    For backward compatibility, this env still accepts old logit vectors:
      - scalar / shape(1,): discrete skill index
      - shape(N,) or shape(N+1,): legacy skill logits (+ optional hold control)
    """

    def __init__(
        self,
        protect_model_path,
        chase_model_path=None,
        primary_skill_name: str = 'protect',
        attacker_strategy='random',
        attacker_strategy_pool=None,
        attacker_strategy_params: Optional[Dict] = None,
        attacker_policy_kwargs: Optional[Dict] = None,
        device='cpu',
        hold_min: int = 1,
        hold_max: int = 1,
        macro_gamma: float = 0.95,
        disable_hold_control: bool = True,
        macro_duration_bins=None,
        macro_duration_cost: float = 0.0,
        enable_early_interrupt: bool = False,
        early_interrupt_min_steps: int = 1,
        early_interrupt_visibility_change: bool = True,
        early_interrupt_primary_urgency: float = 0.60,
        early_interrupt_chase_urgency: float = 0.40,
        defender_hard_action_mask: bool = False,
        defender_hard_action_mask_params: Optional[Dict] = None,
    ):
        super().__init__()
        self.env = TADEnv(reward_mode='hrl')
        if defender_hard_action_mask and hasattr(self.env, 'configure_hard_action_mask'):
            self.env.configure_hard_action_mask(
                True,
                role='defender',
                **dict(defender_hard_action_mask_params or {}),
            )

        self.device = torch.device(device)
        self.attacker_strategy_mode = attacker_strategy
        self.attacker_strategy_pool = self._resolve_attacker_strategy_pool(attacker_strategy_pool)
        self.attacker_strategy_params = dict(attacker_strategy_params or {})
        self.attacker_policy_kwargs = dict(attacker_policy_kwargs or {})
        self._static_attacker_action = np.zeros(2, dtype=np.float32)

        self.hold_min = int(max(1, hold_min))
        self.hold_max = int(max(self.hold_min, hold_max))
        self.macro_gamma = float(macro_gamma)
        self.disable_hold_control = bool(disable_hold_control)
        self.macro_duration_bins = self._resolve_macro_duration_bins(macro_duration_bins)
        self.num_duration_bins = int(len(self.macro_duration_bins))
        self.use_joint_macro_action = bool(self.num_duration_bins > 1)
        self.macro_duration_cost = float(max(0.0, macro_duration_cost))
        self.enable_early_interrupt = bool(enable_early_interrupt)
        self.early_interrupt_min_steps = int(max(1, early_interrupt_min_steps))
        self.early_interrupt_visibility_change = bool(early_interrupt_visibility_change)
        self.early_interrupt_primary_urgency = float(np.clip(early_interrupt_primary_urgency, 0.0, 1.0))
        self.early_interrupt_chase_urgency = float(np.clip(early_interrupt_chase_urgency, 0.0, 1.0))
        self.top_obs_dim = int(NetParameters.ACTOR_RAW_LEN)
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.top_obs_dim,),
            dtype=np.float32,
        )

        primary_skill_name = str(primary_skill_name).strip().lower()
        if primary_skill_name != 'protect':
            raise ValueError(f'primary_skill_name must be "protect", got {primary_skill_name!r}')
        self.primary_skill_name = primary_skill_name
        self.primary_net = self._load_skill_model(protect_model_path, skill_name=self.primary_skill_name)
        # Backward-compatible attribute name.
        self.protect_net = self.primary_net

        if chase_model_path is None:
            raise ValueError('chase model path is required.')
        if not os.path.exists(chase_model_path):
            raise FileNotFoundError(f'chase model not found: {chase_model_path}')
        self.chase_net = self._load_skill_model(chase_model_path, skill_name='chase')
        self.skill_names = [self.primary_skill_name, 'chase']
        self.skill_nets = [self.primary_net, self.chase_net]
        self.num_skills = int(len(self.skill_names))
        self.num_macro_actions = int(self.num_skills * self.num_duration_bins)
        self._reset_skill_states()

        # Keep global NetParameters in sync for top-level network creation.
        NetParameters.HRL_NUM_SKILLS = self.num_skills
        NetParameters.HRL_DURATION_BINS = tuple(int(v) for v in self.macro_duration_bins)
        NetParameters.HRL_NUM_DURATION_BINS = self.num_duration_bins
        NetParameters.HRL_TOP_DISCRETE_ACTION_DIM = (
            self.num_macro_actions if self.use_joint_macro_action else self.num_skills
        )
        NetParameters.HRL_TOP_ACTION_DIM = int(
            NetParameters.HRL_TOP_DISCRETE_ACTION_DIM if self.use_joint_macro_action else (self.num_skills + 1)
        )
        if self.use_joint_macro_action or self.disable_hold_control:
            self.action_space = gym.spaces.Discrete(int(NetParameters.HRL_TOP_DISCRETE_ACTION_DIM))
        else:
            self.action_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(NetParameters.HRL_TOP_ACTION_DIM,),
                dtype=np.float32,
            )

        self.attacker_policy = self._make_attacker_policy()

        self.last_urgency = 0.0
        self.has_seen_attacker_once = False

        # Urgency feature hyper-parameters (observation-derived; no privileged state).
        self.urgency_sigmoid_k = 4.0
        self.urgency_uncertainty_beta = 0.35

        self._attacker_speed = float(getattr(map_config, 'attacker_speed', 2.0))
        self._defender_speed = float(getattr(map_config, 'defender_speed', 2.6))
        agent_radius = float(getattr(map_config, 'agent_radius', 8.0))
        target_radius = float(getattr(map_config, 'target_radius', 16.0))
        # Same geometric thresholds used by env reward/termination logic.
        self._attacker_target_capture_radius = agent_radius + target_radius
        self._defender_target_reach_radius = agent_radius + target_radius

        self.step_count = 0
        self.cached_obs = None
        self.cached_skill_obs = None

    def _adaptive_task_metrics(self):
        sim = self.env
        defender = getattr(sim, 'defender', None)
        attacker = getattr(sim, 'attacker', None)
        target = getattr(sim, 'target', None)
        if not defender or not attacker or not target:
            return None
        d_da = float(math.hypot(float(defender['x']) - float(attacker['x']), float(defender['y']) - float(attacker['y'])))
        d_at = float(math.hypot(float(attacker['x']) - float(target['x']), float(attacker['y']) - float(target['y'])))
        width = float(getattr(sim, 'width', 800.0))
        height = float(getattr(sim, 'height', 600.0))
        diag = max(1.0, float(math.hypot(width, height)))
        defender_speed = max(1e-6, float(getattr(sim, 'defender_speed', self._defender_speed)))
        attacker_speed = max(1e-6, float(getattr(sim, 'attacker_speed', self._attacker_speed)))
        t_def_att = d_da / defender_speed
        t_att_tgt = d_at / attacker_speed
        return {
            'd_da': d_da,
            'd_at': d_at,
            'diag': diag,
            'margin': float(t_att_tgt - t_def_att),
        }

    def _online_adaptive_reward(self, selected_skill: str, start_metrics, end_metrics, info: Dict) -> float:
        """Optional training-only dense reward for online HRL PPO.

        This is not a label or a mask: the top policy still receives only PPO
        gradients from the selected action's rollout. The shaping provides a
        smoother risk/efficiency signal in the multi-regime training launcher.
        """
        if not self._env_bool('HRL_ONLINE_ADAPTIVE_REWARD_ENABLE', False):
            return 0.0
        if start_metrics is None or end_metrics is None:
            return 0.0

        risk_margin = self._env_float('HRL_ONLINE_RISK_MARGIN', 24.0)
        safe_margin = self._env_float('HRL_ONLINE_SAFE_MARGIN', 56.0)
        scale = max(1e-6, self._env_float('HRL_ONLINE_MARGIN_SCALE', 10.0))
        risk_gate = self._sigmoid((risk_margin - float(start_metrics['margin'])) / scale)
        safe_gate = self._sigmoid((float(start_metrics['margin']) - safe_margin) / scale)

        diag = max(1.0, float(start_metrics['diag']))
        chase_progress = (float(start_metrics['d_da']) - float(end_metrics['d_da'])) / diag
        target_delay = (float(end_metrics['d_at']) - float(start_metrics['d_at'])) / diag
        margin_gain = (float(end_metrics['margin']) - float(start_metrics['margin'])) / 80.0

        reward = 0.0
        reward += self._env_float('HRL_ONLINE_RISK_MARGIN_WEIGHT', 5.0) * risk_gate * margin_gain
        reward += self._env_float('HRL_ONLINE_RISK_DELAY_WEIGHT', 2.0) * risk_gate * target_delay
        reward += self._env_float('HRL_ONLINE_SAFE_CHASE_WEIGHT', 2.5) * safe_gate * chase_progress
        reward += self._env_float('HRL_ONLINE_BASE_CHASE_WEIGHT', 0.35) * chase_progress

        skill = str(selected_skill).strip().lower()
        if skill == 'chase' and risk_gate > 0.5 and margin_gain < 0.0:
            reward -= self._env_float('HRL_ONLINE_RISK_CHASE_PENALTY', 0.08) * risk_gate
        if skill == 'protect' and safe_gate > 0.6 and chase_progress < 0.0:
            reward -= self._env_float('HRL_ONLINE_SAFE_STALL_PENALTY', 0.03) * safe_gate

        if bool(info.get('defender_collision', False)):
            reward -= self._env_float('HRL_ONLINE_COLLISION_EXTRA_PENALTY', 1.0)

        return float(reward)

    @staticmethod
    def _clone_hidden_dict(hidden_dict):
        cloned = {}
        for key, value in dict(hidden_dict or {}).items():
            if torch.is_tensor(value):
                cloned[key] = value.detach().clone()
            else:
                cloned[key] = copy.deepcopy(value)
        return cloned

    @staticmethod
    def _copy_obs(obs):
        if obs is None:
            return None
        if isinstance(obs, tuple):
            return tuple(HRLEnv._copy_obs(x) for x in obs)
        if isinstance(obs, list):
            return [HRLEnv._copy_obs(x) for x in obs]
        if isinstance(obs, np.ndarray):
            return obs.copy()
        return copy.deepcopy(obs)

    def snapshot_state(self):
        """Capture HRL wrapper state plus the underlying TAD simulator state."""
        return {
            'env': self.env.snapshot_state(),
            'attacker_policy': copy.deepcopy(self.attacker_policy),
            'static_attacker_action': np.asarray(self._static_attacker_action, dtype=np.float32).copy(),
            'last_urgency': float(self.last_urgency),
            'has_seen_attacker_once': bool(self.has_seen_attacker_once),
            'skill_actor_hidden': self._clone_hidden_dict(self._skill_actor_hidden),
            'skill_critic_hidden': self._clone_hidden_dict(self._skill_critic_hidden),
            'step_count': int(self.step_count),
            'cached_obs': self._copy_obs(self.cached_obs),
            'cached_skill_obs': self._copy_obs(self.cached_skill_obs),
        }

    def restore_state(self, state):
        """Restore a state captured by snapshot_state()."""
        self.env.restore_state(state['env'])
        self.attacker_policy = copy.deepcopy(state['attacker_policy'])
        self._static_attacker_action = np.asarray(
            state.get('static_attacker_action', self._static_attacker_action),
            dtype=np.float32,
        ).copy()
        self.last_urgency = float(state.get('last_urgency', 0.0))
        self.has_seen_attacker_once = bool(state.get('has_seen_attacker_once', False))
        self._skill_actor_hidden = self._clone_hidden_dict(state.get('skill_actor_hidden', {}))
        self._skill_critic_hidden = self._clone_hidden_dict(state.get('skill_critic_hidden', {}))
        self.step_count = int(state.get('step_count', 0))
        self.cached_obs = self._copy_obs(state.get('cached_obs'))
        self.cached_skill_obs = self._copy_obs(state.get('cached_skill_obs'))

    def _reset_skill_states(self):
        self._skill_actor_hidden = {name: None for name in getattr(self, 'skill_names', [])}
        self._skill_critic_hidden = {name: None for name in getattr(self, 'skill_names', [])}

    def _current_attacker_style(self) -> str:
        if self.attacker_policy is None:
            return str(self.attacker_strategy_mode)
        style = getattr(self.attacker_policy, 'strategy', self.attacker_strategy_mode)
        return str(style)

    def _make_attacker_policy(self):
        if self.attacker_strategy_mode == 'static':
            return None
        if self.attacker_strategy_mode == 'random':
            pool = self.attacker_strategy_pool or TRAINING_STRATEGIES
            strategy = random.choice(pool)
        else:
            strategy = self.attacker_strategy_mode
        return AttackerGlobalPolicy(
            env_width=self.env.width,
            env_height=self.env.height,
            attacker_speed=float(getattr(map_config, 'attacker_speed', self.env.attacker_speed)),
            attacker_max_turn=float(getattr(map_config, 'attacker_max_angular_speed', 12.0)),
            strategy=strategy,
            strategy_params=self.attacker_strategy_params,
            **self.attacker_policy_kwargs,
        )

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        raw = os.environ.get(name, None)
        if raw is None or str(raw).strip() == '':
            return bool(default)
        return str(raw).strip().lower() in ('1', 'true', 'yes', 'on', 'y')

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        raw = os.environ.get(name, None)
        if raw is None or str(raw).strip() == '':
            return float(default)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return float(default)

    def _regime_skill_prior_reward(self, selected_skill: str) -> float:
        """Weak training-only prior to prevent regime-adaptive HRL collapse.

        It is disabled by default. The prior is deliberately small relative to
        sparse task rewards; it only nudges skill usage so the top policy has a
        reason to explore non-conservative options in favorable regimes.
        """
        if not self._env_bool('HRL_REGIME_SKILL_PRIOR_ENABLE', False):
            return 0.0
        weight = self._env_float('HRL_REGIME_SKILL_PRIOR_WEIGHT', 1.0)
        margin_regime = str(getattr(self.env, 'current_margin_regime', getattr(self.env, 'current_regime', 'default'))).strip().lower()
        speed_regime = str(getattr(self.env, 'current_speed_regime', 'neutral')).strip().lower()
        skill = str(selected_skill).strip().lower()
        table = {
            'advantage': {
                'chase': 0.020,
                'protect': 0.010,
                'protect': -0.030,
            },
            'neutral': {
                'chase': 0.010,
                'protect': 0.006,
                'protect': -0.012,
            },
            'disadvantage': {
                'protect': 0.014,
                'protect': 0.010,
                'chase': -0.004,
            },
        }
        prior = float(table.get(margin_regime, {}).get(skill, 0.0))
        if speed_regime == 'advantage':
            if skill == 'chase':
                prior += self._env_float('HRL_REGIME_SPEED_ADV_CHASE_PRIOR', 0.006)
            elif skill == 'protect':
                prior -= self._env_float('HRL_REGIME_SPEED_ADV_PROTECT_PRIOR_PENALTY', 0.006)
        elif speed_regime == 'disadvantage':
            if skill == 'chase':
                prior -= self._env_float('HRL_REGIME_SPEED_DISADV_CHASE_PRIOR_PENALTY', 0.006)
            elif skill == 'protect':
                prior += self._env_float('HRL_REGIME_SPEED_DISADV_PROTECT_PRIOR', 0.006)
        # Extra guard against "unknown opponent => always protect": before the
        # attacker has been observed, Protect is not rewarded unless the sampled
        # geometry is actually disadvantageous.
        if skill == 'protect' and margin_regime != 'disadvantage' and not bool(self.has_seen_attacker_once):
            prior -= self._env_float('HRL_REGIME_UNKNOWN_PROTECT_PENALTY', 0.010)
        return float(weight * prior)

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-float(np.clip(x, -20.0, 20.0))))

    @staticmethod
    def _resolve_attacker_strategy_pool(pool_cfg):
        if pool_cfg is None:
            return None
        if isinstance(pool_cfg, str):
            raw = [s.strip() for s in pool_cfg.split(',')]
        elif isinstance(pool_cfg, (list, tuple, set, np.ndarray)):
            raw = [str(s).strip() for s in list(pool_cfg)]
        else:
            raise ValueError(f'Invalid attacker_strategy_pool type: {type(pool_cfg)}')

        normalized = []
        allow_duplicates = HRLEnv._env_bool('HRL_ATTACKER_POOL_ALLOW_DUPLICATES', False)
        for strategy in raw:
            if not strategy:
                continue
            key = str(strategy).lower()
            if key not in SUPPORTED_STRATEGIES:
                raise ValueError(
                    f'Unsupported attacker strategy in pool: {strategy}. '
                    f'Valid={SUPPORTED_STRATEGIES}'
                )
            if allow_duplicates or key not in normalized:
                normalized.append(key)
        return tuple(normalized) if normalized else None

    @staticmethod
    def _resolve_macro_duration_bins(duration_bins_cfg):
        if duration_bins_cfg is None:
            return (1,)
        if isinstance(duration_bins_cfg, str):
            raw = [s.strip() for s in duration_bins_cfg.split(',')]
        elif isinstance(duration_bins_cfg, (list, tuple, set, np.ndarray)):
            raw = list(duration_bins_cfg)
        else:
            raise ValueError(f'Invalid macro_duration_bins type: {type(duration_bins_cfg)}')

        normalized = []
        for value in raw:
            if value is None or str(value).strip() == '':
                continue
            steps = int(value)
            if steps < 1:
                raise ValueError(f'macro duration must be >= 1, got {value}')
            if steps not in normalized:
                normalized.append(steps)
        normalized.sort()
        return tuple(normalized) if normalized else (1,)

    def _denorm_distance(self, normalized_distance: float) -> float:
        map_diagonal = math.hypot(self.env.width, self.env.height)
        nd = float(np.clip(normalized_distance, -1.0, 1.0))
        return (nd + 1.0) * 0.5 * map_diagonal

    @staticmethod
    def _denorm_bearing_deg(normalized_bearing: float) -> float:
        nb = float(np.clip(normalized_bearing, -1.0, 1.0))
        return nb * 180.0

    def _compute_urgency(self, defender_obs: np.ndarray) -> float:
        """
        Compute protect urgency from defender-observable signals only.

        Uses:
        - attacker estimate from visible obs / GRU-injected obs
        - target relative observation
        - visibility + unobserved time as uncertainty term
        """
        src = np.asarray(defender_obs, dtype=np.float32).reshape(-1)
        if src.shape[0] < 7:
            return 0.0

        attacker_dist = self._denorm_distance(src[0])
        attacker_bearing_deg = self._denorm_bearing_deg(src[1])
        is_visible = bool(float(src[3]) > 0.5)
        unobserved_norm = float(np.clip(src[4], -1.0, 1.0))
        target_dist = self._denorm_distance(src[5])
        target_bearing_deg = self._denorm_bearing_deg(src[6])

        # Conservative opening behavior when attacker has never been observed.
        if (not self.has_seen_attacker_once) and (not is_visible):
            return 1.0

        rel_delta = ((attacker_bearing_deg - target_bearing_deg + 180.0) % 360.0) - 180.0
        delta_rad = math.radians(rel_delta)
        attacker_target_dist_sq = (
            attacker_dist * attacker_dist
            + target_dist * target_dist
            - 2.0 * attacker_dist * target_dist * math.cos(delta_rad)
        )
        attacker_target_dist = math.sqrt(max(0.0, attacker_target_dist_sq))

        t_attacker = max(0.0, attacker_target_dist - self._attacker_target_capture_radius) / max(self._attacker_speed, 1e-6)
        t_defender = max(0.0, target_dist - self._defender_target_reach_radius) / max(self._defender_speed, 1e-6)

        # Higher urgency when defender is slower than attacker to secure target.
        urgency_base = self._sigmoid(self.urgency_sigmoid_k * (t_defender - t_attacker))

        # Visibility-aware uncertainty boost (high when unseen for long).
        unseen_ratio = 0.5 * (unobserved_norm + 1.0)  # [-1,1] -> [0,1]
        uncertainty = 0.0 if is_visible else float(np.clip(unseen_ratio, 0.0, 1.0))
        urgency = urgency_base + self.urgency_uncertainty_beta * uncertainty * (1.0 - urgency_base)
        return float(np.clip(urgency, 0.0, 1.0))

    @staticmethod
    def _compat_numpy_checkpoint_load(path, device):
        import numpy as _np
        import sys as _sys

        if not hasattr(_np, '_core'):
            _sys.modules['numpy._core'] = _np.core
            _sys.modules['numpy._core.multiarray'] = _np.core.multiarray
        return torch.load(path, map_location=device, weights_only=False)

    @staticmethod
    def _display_bottom_network_type(network_type: str) -> str:
        return 'mlp' if str(network_type) == 'mlp_noctde' else str(network_type)

    @staticmethod
    def _detect_network_type(state_dict):
        keys = set(state_dict.keys())
        if 'shared_tracking_branch.0.weight' in keys:
            return 'nmn_ctde_task_shared'
        if 'shared_radar_encoder.net.0.weight' in keys:
            return 'nmn_ctde_shared'
        has_tracking = any('tracking_branch' in k for k in keys)
        has_actor_backbone = any('actor_backbone' in k for k in keys)
        has_actor_gru = any('actor_gru' in k for k in keys)
        if has_tracking and has_actor_gru:
            return 'nmn_gru'
        if has_actor_gru:
            action_dim = None
            if 'log_std' in state_dict and hasattr(state_dict['log_std'], 'shape'):
                action_dim = int(state_dict['log_std'].shape[0])
            elif 'policy_mean.weight' in state_dict and hasattr(state_dict['policy_mean.weight'], 'shape'):
                action_dim = int(state_dict['policy_mean.weight'].shape[0])
            if action_dim is not None and action_dim >= 3:
                return 'hrl_top_gru'
            return 'mlp_gru'
        if has_tracking:
            critic_in_dim = None
            if 'critic_backbone.0.weight' in state_dict and hasattr(state_dict['critic_backbone.0.weight'], 'shape'):
                critic_in_dim = int(state_dict['critic_backbone.0.weight'].shape[1])
            if critic_in_dim == NetParameters.CRITIC_VECTOR_LEN:
                return 'nmn_ctde'
            return 'nmn'
        if has_actor_backbone:
            critic_in_dim = None
            if 'critic_backbone.0.weight' in state_dict and hasattr(state_dict['critic_backbone.0.weight'], 'shape'):
                critic_in_dim = int(state_dict['critic_backbone.0.weight'].shape[1])
            if critic_in_dim == NetParameters.ACTOR_VECTOR_LEN:
                return 'mlp_noctde'
            return 'mlp_ctde'
        return 'nmn'

    def _load_skill_model(self, model_path, skill_name='skill'):
        if model_path is None:
            raise ValueError(f'{skill_name} model path is required.')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'{skill_name} model not found: {model_path}')

        checkpoint = self._compat_numpy_checkpoint_load(model_path, self.device)
        state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
        network_type = None
        if isinstance(checkpoint, dict) and checkpoint.get('network_type'):
            network_type = str(checkpoint['network_type']).strip().lower()
        if not network_type:
            network_type = self._detect_network_type(state_dict)
        if network_type in {'hrl_top', 'hrl_top_noctde', 'hrl_top_gru'}:
            raise ValueError(
                f'{skill_name} skill checkpoint points to a top-level HRL model '
                f'({network_type}) instead of a bottom skill model: {model_path}'
            )

        net = create_network(network_type).to(self.device)
        net.load_state_dict(state_dict)
        net.eval()
        print(
            f"[HRLEnv] Loaded {skill_name} model "
            f"({self._display_bottom_network_type(network_type)}) from {model_path}"
        )
        return net

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits)
        exps = np.exp(np.clip(shifted, -20.0, 20.0))
        return exps / (np.sum(exps) + 1e-8)

    def _decode_legacy_top_action(self, action: np.ndarray):
        if action.size == 1:
            skill_idx = int(np.clip(np.rint(float(action[0])), 0, self.num_skills - 1))
            skill_probs = np.zeros((self.num_skills,), dtype=np.float32)
            skill_probs[skill_idx] = 1.0
            hold_steps = 1 if self.disable_hold_control else self.hold_min
            return skill_idx, hold_steps, skill_probs

        valid_sizes = {self.num_skills, self.num_skills + 1}
        if self.num_skills == 3:
            valid_sizes.update({2, 3})
        if action.size not in valid_sizes:
            raise ValueError(f'Top-level action has invalid dim {action.shape}, expected one of {[1] + sorted(valid_sizes)}')

        if action.size in (2, 3) and self.num_skills == 3:
            skill_logits = action[:2]
            hold_index = 2 if action.size == 3 else None
        else:
            skill_logits = action[:self.num_skills]
            hold_index = self.num_skills if action.size > self.num_skills else None

        skill_probs = self._softmax(skill_logits)
        skill_idx = int(np.argmax(skill_probs))

        if self.disable_hold_control:
            return skill_idx, 1, skill_probs

        if hold_index is None:
            hold_steps = self.hold_min
        else:
            hold_ratio = 0.5 * (float(np.clip(action[hold_index], -1.0, 1.0)) + 1.0)
            hold_steps = int(round(self.hold_min + hold_ratio * (self.hold_max - self.hold_min)))
            hold_steps = int(np.clip(hold_steps, self.hold_min, self.hold_max))

        return skill_idx, hold_steps, skill_probs

    def _decode_macro_action_index(self, macro_idx: int, macro_probs: Optional[np.ndarray] = None):
        macro_idx = int(np.clip(int(macro_idx), 0, self.num_macro_actions - 1))
        duration_idx = int(macro_idx % self.num_duration_bins)
        skill_idx = int(macro_idx // self.num_duration_bins)
        hold_steps = int(self.macro_duration_bins[duration_idx])
        skill_probs = np.zeros((self.num_skills,), dtype=np.float32)
        skill_probs[skill_idx] = 1.0
        if macro_probs is None:
            macro_probs = np.zeros((self.num_macro_actions,), dtype=np.float32)
            macro_probs[macro_idx] = 1.0
        return skill_idx, hold_steps, skill_probs, duration_idx, macro_idx, macro_probs.astype(np.float32)

    def _nearest_duration_index(self, hold_steps: int) -> int:
        bins = np.asarray(self.macro_duration_bins, dtype=np.int32)
        return int(np.argmin(np.abs(bins - int(hold_steps))))

    def _decode_top_action(self, top_action: np.ndarray):
        action = np.asarray(top_action, dtype=np.float32).reshape(-1)
        if action.size == 0:
            raise ValueError('Top-level action is empty.')

        if self.use_joint_macro_action:
            if action.size == 1:
                return self._decode_macro_action_index(int(np.rint(float(action[0]))))
            if action.size == self.num_macro_actions:
                macro_probs = self._softmax(action)
                macro_idx = int(np.argmax(macro_probs))
                return self._decode_macro_action_index(macro_idx, macro_probs=macro_probs)

            skill_idx, legacy_hold_steps, skill_probs = self._decode_legacy_top_action(action)
            duration_idx = self._nearest_duration_index(legacy_hold_steps)
            macro_idx = int(skill_idx * self.num_duration_bins + duration_idx)
            hold_steps = int(self.macro_duration_bins[duration_idx])
            macro_probs = np.zeros((self.num_macro_actions,), dtype=np.float32)
            macro_probs[macro_idx] = 1.0
            return skill_idx, hold_steps, skill_probs, duration_idx, macro_idx, macro_probs

        skill_idx, hold_steps, skill_probs = self._decode_legacy_top_action(action)
        return skill_idx, hold_steps, skill_probs, None, skill_idx, skill_probs.astype(np.float32)

    def _skill_action_from_net(self, skill_name, net, defender_obs, attacker_obs):
        action, *_ = self._skill_action_info_from_net(
            skill_name,
            net,
            defender_obs,
            attacker_obs,
        )
        return action

    @staticmethod
    def _normal_log_prob_from_pre_tanh(pre_tanh, mean, log_std):
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        base_log_prob = dist.log_prob(pre_tanh)
        log_det_jac = torch.log(1.0 - torch.tanh(pre_tanh) ** 2 + 1e-6)
        return (base_log_prob - log_det_jac).sum(dim=-1)

    @staticmethod
    def _serialize_recurrent_hidden_for_net(net, hidden, role: str):
        if not bool(getattr(net, 'is_recurrent', False)):
            return None
        if hasattr(net, 'recurrent_hidden_spec'):
            num_layers, hidden_size = net.recurrent_hidden_spec(role)
            if hidden is None:
                return np.zeros((int(num_layers), int(hidden_size)), dtype=np.float32)
        else:
            gru = getattr(net, f'{role}_gru', None)
            if gru is None:
                return None
            if hidden is None:
                return np.zeros((int(gru.num_layers), int(gru.hidden_size)), dtype=np.float32)

        hidden_cpu = hidden.detach().to(device='cpu', dtype=torch.float32)
        if hidden_cpu.dim() == 3:
            hidden_cpu = hidden_cpu[:, 0, :]
        if hidden_cpu.dim() != 2:
            raise ValueError(f'Invalid {role} recurrent hidden shape: {tuple(hidden_cpu.shape)}')
        return hidden_cpu.numpy().copy()

    def _skill_action_info_from_net(self, skill_name, net, defender_obs, attacker_obs):
        with torch.no_grad():
            obs_tensor = torch.as_tensor(defender_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            critic_obs = build_critic_observation(defender_obs, attacker_obs)
            critic_tensor = torch.as_tensor(critic_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            is_recurrent = bool(getattr(net, 'is_recurrent', False))
            trainable = (not is_recurrent) or hasattr(net, 'forward_sequence')
            actor_hidden_in = None
            critic_hidden_in = None
            if is_recurrent and hasattr(net, 'forward_recurrent'):
                actor_hidden = self._skill_actor_hidden.get(skill_name)
                critic_hidden = self._skill_critic_hidden.get(skill_name)
                actor_hidden_in = self._serialize_recurrent_hidden_for_net(net, actor_hidden, 'actor')
                critic_hidden_in = self._serialize_recurrent_hidden_for_net(net, critic_hidden, 'critic')
                mean, value, log_std, next_actor_hidden, next_critic_hidden = net.forward_recurrent(
                    obs_tensor,
                    critic_tensor,
                    actor_hidden=actor_hidden,
                    critic_hidden=critic_hidden,
                )
                self._skill_actor_hidden[skill_name] = (
                    next_actor_hidden.detach() if next_actor_hidden is not None else None
                )
                self._skill_critic_hidden[skill_name] = (
                    next_critic_hidden.detach() if next_critic_hidden is not None else None
                )
            else:
                mean, value, log_std = net(obs_tensor, critic_tensor)
            pre_tanh = mean
            action = torch.tanh(pre_tanh)
            log_prob = self._normal_log_prob_from_pre_tanh(pre_tanh, mean, log_std)
            return (
                action.cpu().numpy()[0],
                pre_tanh.cpu().numpy()[0],
                float(value.squeeze(-1).cpu().numpy().item()),
                float(log_prob.cpu().numpy().item()),
                bool(trainable),
                actor_hidden_in,
                critic_hidden_in,
            )

    def _skill_bootstrap_value(self, skill_name, net, defender_obs, attacker_obs):
        with torch.no_grad():
            obs_tensor = torch.as_tensor(defender_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            critic_obs = build_critic_observation(defender_obs, attacker_obs)
            critic_tensor = torch.as_tensor(critic_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            if bool(getattr(net, 'is_recurrent', False)):
                critic_hidden = self._skill_critic_hidden.get(skill_name)
                if hasattr(net, 'critic_value_recurrent'):
                    value, _ = net.critic_value_recurrent(critic_tensor, critic_hidden=critic_hidden)
                    return float(value.squeeze(-1).cpu().numpy().item())
                if hasattr(net, 'forward_recurrent'):
                    actor_hidden = self._skill_actor_hidden.get(skill_name)
                    _mean, value, _log_std, _next_actor_hidden, _next_critic_hidden = net.forward_recurrent(
                        obs_tensor,
                        critic_tensor,
                        actor_hidden=actor_hidden,
                        critic_hidden=critic_hidden,
                    )
                    return float(value.squeeze(-1).cpu().numpy().item())
                return 0.0
            if hasattr(net, 'critic_value'):
                value = net.critic_value(critic_tensor)
            else:
                _mean, value, _log_std = net(obs_tensor, critic_tensor)
            return float(value.squeeze(-1).cpu().numpy().item())

    def set_skill_weights(self, skill_weights: Dict[str, Dict[str, torch.Tensor]]):
        for name, weights in dict(skill_weights or {}).items():
            key = str(name).strip().lower()
            if key not in self.skill_names:
                continue
            idx = self.skill_names.index(key)
            state_dict = {}
            for k, v in dict(weights).items():
                if isinstance(v, torch.Tensor):
                    state_dict[k] = v.clone().detach().to(self.device)
                else:
                    state_dict[k] = torch.as_tensor(v, device=self.device)
            self.skill_nets[idx].load_state_dict(state_dict, strict=False)
            self.skill_nets[idx].eval()
        self._reset_skill_states()

    def _process_observation(self, raw_obs):
        defender_obs, attacker_obs = raw_obs
        skill_processed = (
            np.asarray(defender_obs, dtype=np.float32),
            np.asarray(attacker_obs, dtype=np.float32),
        )
        if skill_processed[0].shape[0] > 3 and float(skill_processed[0][3]) > 0.5:
            self.has_seen_attacker_once = True
        self.last_urgency = self._compute_urgency(skill_processed[0])
        self.cached_skill_obs = skill_processed
        processed = (skill_processed[0], skill_processed[1])
        self.cached_obs = processed
        return processed

    @staticmethod
    def _obs_visible(defender_obs: np.ndarray) -> bool:
        return bool(np.asarray(defender_obs).shape[0] > 3 and float(defender_obs[3]) > 0.5)

    @staticmethod
    def _duration_penalty_units(duration_idx: Optional[int]) -> float:
        return float(max(0, int(duration_idx))) if duration_idx is not None else 0.0

    def _is_primary_skill(self, skill_name: str) -> bool:
        return str(skill_name).strip().lower() != 'chase'

    def _should_interrupt_macro(
        self,
        selected_skill: str,
        hold_steps: int,
        primitive_steps: int,
        macro_start_visible: bool,
    ):
        if (not self.enable_early_interrupt) or primitive_steps >= int(hold_steps):
            return False, None
        if primitive_steps < self.early_interrupt_min_steps:
            return False, None

        defender_obs, _ = self.cached_skill_obs
        current_visible = self._obs_visible(defender_obs)
        if self.early_interrupt_visibility_change and current_visible != bool(macro_start_visible):
            return True, 'visibility_change'

        current_urgency = float(self.last_urgency)
        if self._is_primary_skill(selected_skill):
            if current_urgency <= self.early_interrupt_chase_urgency:
                return True, 'urgency_low'
        else:
            if current_urgency >= self.early_interrupt_primary_urgency:
                return True, 'urgency_high'

        return False, None

    def reset(self, seed=None, options=None):
        raw_obs, info = self.env.reset(seed=seed, options=options)

        # Recreate after TADEnv.reset because regime randomization may change
        # map_config.attacker_speed for this episode.
        self.attacker_policy = self._make_attacker_policy()

        self.last_urgency = 0.0
        self.has_seen_attacker_once = False
        self._reset_skill_states()

        self.step_count = 0
        obs = self._process_observation(raw_obs)
        return obs, info

    def macro_step(self, top_action, attacker_action=None):
        """Execute one top-level decision for multiple primitive env steps."""
        if self.cached_obs is None or self.cached_skill_obs is None:
            self._process_observation(self.env.current_obs)

        skill_idx, hold_steps, skill_probs, duration_idx, macro_idx, macro_probs = self._decode_top_action(top_action)
        macro_start_metrics = self._adaptive_task_metrics()

        macro_reward = 0.0
        raw_reward_sum = 0.0
        discount_acc = 1.0
        primitive_steps = 0

        terminated = False
        truncated = False
        info = {}
        selected_skill = self.skill_names[min(skill_idx, self.num_skills - 1)]
        macro_start_visible = self._obs_visible(self.cached_skill_obs[0])
        interrupt_reason = None
        joint_low_samples = []

        for _ in range(hold_steps):
            defender_obs, attacker_obs = self.cached_skill_obs

            skill_actions = {}
            skill_action_info = {}
            for name, net in zip(self.skill_names, self.skill_nets):
                action, pre_tanh, value, log_prob, trainable, actor_hidden, critic_hidden = self._skill_action_info_from_net(
                    name, net, defender_obs, attacker_obs
                )
                skill_actions[name] = action
                sample = {
                    'obs': np.asarray(defender_obs, dtype=np.float32).copy(),
                    'critic_obs': np.asarray(build_critic_observation(defender_obs, attacker_obs), dtype=np.float32),
                    'action': np.asarray(pre_tanh, dtype=np.float32).copy(),
                    'value': float(value),
                    'log_prob': float(log_prob),
                    'trainable': bool(trainable),
                }
                if actor_hidden is not None and critic_hidden is not None:
                    sample['actor_hidden'] = np.asarray(actor_hidden, dtype=np.float32).copy()
                    sample['critic_hidden'] = np.asarray(critic_hidden, dtype=np.float32).copy()
                skill_action_info[name] = sample

            final_action = skill_actions[selected_skill]
            attacker_style = self._current_attacker_style()

            if attacker_action is not None:
                a_attacker = np.asarray(attacker_action, dtype=np.float32).reshape(-1)
            elif self.attacker_policy is None:
                a_attacker = self._static_attacker_action
            elif hasattr(self.attacker_policy, 'get_action_with_info'):
                a_attacker, _ = self.attacker_policy.get_action_with_info(attacker_obs)
            else:
                a_attacker = self.attacker_policy.get_action(attacker_obs)

            next_raw_obs, reward, terminated, truncated, info = self.env.step(
                action=final_action,
                attacker_action=a_attacker,
            )
            step_done = bool(terminated or truncated)
            skill_rewards = dict(info.get('skill_rewards', {}) or {})

            primitive_steps += 1
            self.step_count += 1
            raw_reward_sum += float(reward)
            macro_reward += discount_acc * float(reward)
            discount_acc *= self.macro_gamma

            self._process_observation(next_raw_obs)
            selected_sample = skill_action_info.get(selected_skill)
            if selected_sample is not None and selected_sample.get('trainable', False):
                selected_net = self.skill_nets[self.skill_names.index(selected_skill)]
                next_defender_obs, next_attacker_obs = self.cached_skill_obs
                next_value = 0.0 if step_done else self._skill_bootstrap_value(
                    selected_skill,
                    selected_net,
                    next_defender_obs,
                    next_attacker_obs,
                )
                skill_reward = float(skill_rewards.get(selected_skill, reward))
                selected_sample = dict(selected_sample)
                selected_sample.update({
                    'skill': selected_skill,
                    'reward': skill_reward,
                    'return': skill_reward + float(self.macro_gamma) * float(next_value) * (1.0 - float(step_done)),
                    'done': bool(step_done),
                })
                joint_low_samples.append(selected_sample)

            for name, act in skill_actions.items():
                info[f'a_{name}'] = act
            info['selected_skill'] = selected_skill
            info['attacker_style'] = attacker_style

            if terminated or truncated:
                break

            should_interrupt, interrupt_reason = self._should_interrupt_macro(
                selected_skill=selected_skill,
                hold_steps=hold_steps,
                primitive_steps=primitive_steps,
                macro_start_visible=macro_start_visible,
            )
            if should_interrupt:
                break

        duration_penalty = self.macro_duration_cost * self._duration_penalty_units(duration_idx)
        macro_reward -= duration_penalty
        regime_skill_prior = self._regime_skill_prior_reward(selected_skill)
        macro_reward += regime_skill_prior
        adaptive_reward = self._online_adaptive_reward(
            selected_skill,
            macro_start_metrics,
            self._adaptive_task_metrics(),
            info,
        )
        macro_reward += adaptive_reward
        info['top_skill_idx'] = int(skill_idx)
        info['top_skill_probs'] = skill_probs.astype(np.float32)
        info['top_macro_idx'] = int(macro_idx)
        info['top_macro_probs'] = macro_probs.astype(np.float32)
        info['top_duration_idx'] = int(duration_idx) if duration_idx is not None else 0
        info['top_duration_steps'] = int(hold_steps)
        info['top_hold_steps'] = int(hold_steps)
        info['macro_steps'] = int(primitive_steps)
        info['macro_reward'] = float(macro_reward)
        info['raw_reward_sum'] = float(raw_reward_sum)
        info['macro_duration_penalty'] = float(duration_penalty)
        info['regime_skill_prior_reward'] = float(regime_skill_prior)
        info['online_adaptive_reward'] = float(adaptive_reward)
        info['macro_interrupted'] = bool(interrupt_reason is not None)
        info['macro_interrupt_reason'] = interrupt_reason
        info['macro_discount'] = float(self.macro_gamma ** primitive_steps)
        info['skill_names'] = tuple(self.skill_names)
        info['duration_bins'] = tuple(int(v) for v in self.macro_duration_bins)
        info['urgency'] = float(self.last_urgency)
        info['regime'] = str(getattr(self.env, 'current_regime', 'default'))
        info['regime_info'] = dict(getattr(self.env, 'current_regime_info', {}) or {})
        info['joint_low_samples'] = joint_low_samples
        if 'attacker_style' not in info:
            attacker_style = self._current_attacker_style()
            info['attacker_style'] = attacker_style

        return self.cached_obs, float(macro_reward), bool(terminated), bool(truncated), info

    def step(self, action, attacker_action=None):
        return self.macro_step(action, attacker_action=attacker_action)
