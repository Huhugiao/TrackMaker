"""
TAD PPO Runner - Ray分布式采样Worker
"""

import os
import time
import ray
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional

from attacker.learned_policy import (
    DEFAULT_LEARNED_ATTACKER_ALIAS,
    LearnedAttackerPolicy,
    normalize_learned_attacker_alias,
    normalize_learned_attacker_specs,
)
from attacker.frozen_pool import PROGRAMMATIC_ATTACKER_ALIASES
from configs.skill_config import SetupParameters, TrainingParameters, NetParameters, RecordingParameters
from networks import DefenderNetMLP, create_network
from skill.util import build_critic_observation, update_perf, get_adjusted_n_envs

from configs import map_config
from configs.map_config import EnvParameters
from envs.tad_env import TADEnv
from policies import AttackerGlobalPolicy, AttackerStaticPolicy
from policies.attacker_global import SUPPORTED_STRATEGIES, TRAINING_STRATEGIES
from policies.defender_global import DefenderGlobalPolicy


ATTACKER_POLICY_REGISTRY = {
    'attacker_global': AttackerGlobalPolicy,
    'attacker_static': AttackerStaticPolicy,
    'attacker_learned': LearnedAttackerPolicy,
}


SUPPORTED_TRAINING_MODES = ('rl', 'mixed')


def resolve_training_mode(mode: str) -> str:
    """Normalize supported D1 training modes and reject the removed IL path."""
    normalized = str(mode).strip().lower()
    if normalized == 'il':
        raise ValueError(
            "TRAINING_MODE='il' and its deprecated pure-imitation rollout were "
            "removed; use 'mixed' for expert-guided PPO or 'rl' for PPO only."
        )
    if normalized not in SUPPORTED_TRAINING_MODES:
        raise ValueError(
            f'Unsupported TRAINING_MODE={mode!r}; valid modes are '
            f'{SUPPORTED_TRAINING_MODES}.'
        )
    return normalized


class RewardNormalizer:
    """
    Running Return Normalization (OpenAI baselines style)
    
    维护一个折扣回报 G_t 的 running variance，用 1/sqrt(var + eps) 缩放 reward。
    只做 scale，不做 shift（不减均值），避免改变最优策略。
    
    原理：
    - 在训练前期，reward 的方差可能很大（如 protect1 的 -100 累积惩罚）
    - 通过除以 sqrt(var) 将 reward 缩放到稳定范围
    - running mean/var 使用 Welford 在线算法更新
    
    warmup_steps: 前 N 步不标准化，用于收集足够的统计数据
    clip_range: 标准化后的 reward 裁剪范围，防止极端值
    """
    def __init__(self, gamma=0.99, epsilon=1e-8, warmup_steps=100, clip_range=10.0):
        self.gamma = gamma
        self.epsilon = epsilon
        self.warmup_steps = warmup_steps
        self.clip_range = clip_range
        # Running statistics (Welford online algorithm)
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        # 折扣回报的 running estimate
        self.ret = 0.0  # 当前 episode 的折扣回报 G_t
    
    def update(self, reward, done):
        """
        更新 running statistics 并返回标准化后的 reward
        
        Args:
            reward: 原始 reward
            done: episode 是否结束
        Returns:
            normalized reward (reward / sqrt(var + eps)), clipped to [-clip_range, clip_range]
        """
        # 更新折扣回报 (当 done 时重置)
        self.ret = self.ret * self.gamma * (1.0 - float(done)) + reward
        
        # Welford online update for mean and variance
        self.count += 1
        delta = self.ret - self.mean
        self.mean += delta / self.count
        delta2 = self.ret - self.mean
        # Numerically stable variance update
        if self.count > 1:
            self.var += (delta * delta2 - self.var) / self.count
        else:
            self.var = 0.0
        # 确保 var >= 0
        self.var = max(self.var, 0.0)
        
        # Warmup: 统计量不稳定时直接返回原始 reward
        if self.count < self.warmup_steps:
            return reward
        
        # Normalize: 只 scale，不 shift
        std = max(self.var ** 0.5, self.epsilon)
        normalized = reward / std
        
        # Clip to prevent extreme values
        normalized = max(-self.clip_range, min(self.clip_range, normalized))
        return normalized
    
    def reset_ret(self):
        """Reset discounted return tracker (called on episode reset)"""
        self.ret = 0.0


@ray.remote(num_cpus=1, num_gpus=0)
class Runner:
    """Runner defaults to CPU inference; speed tests may override to GPU."""
    def __init__(self, meta_agent_id: int, env_configs: Dict = None, network_type: str = 'nmn'):
        self.meta_agent_id = meta_agent_id
        self.env_configs = env_configs or {}
        self.seed = int(self.env_configs.get('seed', getattr(SetupParameters, 'SEED', 1234)))
        self.train_use_random_seed = bool(
            self.env_configs.get(
                'train_use_random_seed',
                getattr(SetupParameters, 'TRAIN_USE_RANDOM_SEED', True),
            )
        )
        self.eval_use_random_seed = bool(
            self.env_configs.get(
                'eval_use_random_seed',
                getattr(SetupParameters, 'EVAL_USE_RANDOM_SEED', True),
            )
        )
        self.eval_fixed_seed = int(
            self.env_configs.get(
                'eval_fixed_seed',
                getattr(SetupParameters, 'EVAL_FIXED_SEED', 42),
            )
        )
        self._apply_network_env_overrides()
        self.training_mode = resolve_training_mode(
            self.env_configs.get(
                'training_mode',
                getattr(TrainingParameters, 'TRAINING_MODE', 'rl'),
            )
        )
        self.skill_mode = str(
            self.env_configs.get('skill_mode', getattr(SetupParameters, 'SKILL_MODE', ''))
        ).strip().lower()
        self.fixed_attacker_strategy = self.env_configs.get('attacker_strategy')
        self.learned_attacker_alias = normalize_learned_attacker_alias(
            self.env_configs.get(
                'learned_attacker_alias',
                getattr(
                    SetupParameters,
                    'TRAIN_LEARNED_ATTACKER_ALIAS',
                    DEFAULT_LEARNED_ATTACKER_ALIAS,
                ),
            )
        )
        if self.learned_attacker_alias in SUPPORTED_STRATEGIES or self.learned_attacker_alias in {
            'static',
            'random',
        }:
            raise ValueError(
                f'learned_attacker_alias conflicts with a built-in strategy: '
                f'{self.learned_attacker_alias!r}'
            )
        specs_cfg = self.env_configs.get(
            'learned_attacker_specs',
            getattr(SetupParameters, 'TRAIN_LEARNED_ATTACKER_SPECS', None),
        )
        self.learned_attacker_specs = normalize_learned_attacker_specs(specs_cfg)
        self.learned_attacker_aliases = tuple(
            dict.fromkeys(
                (self.learned_attacker_alias, *self.learned_attacker_specs.keys())
            )
        )
        self.programmatic_attacker_aliases = tuple(PROGRAMMATIC_ATTACKER_ALIASES)
        for alias in self.learned_attacker_specs:
            if (
                alias in SUPPORTED_STRATEGIES
                or alias in self.programmatic_attacker_aliases
                or alias in {'static', 'random'}
            ):
                raise ValueError(
                    f'learned attacker alias conflicts with a built-in strategy: {alias!r}'
                )
        pool_cfg = self.env_configs.get(
            'attacker_strategy_pool',
            getattr(SetupParameters, 'TRAIN_ATTACKER_STRATEGIES', None),
        )
        self.attacker_strategy_pool = self._resolve_attacker_strategy_pool(
            pool_cfg,
            learned_alias=self.learned_attacker_alias,
            learned_aliases=self.learned_attacker_aliases,
        )
        requested_learned_aliases = {
            str(strategy).strip().lower()
            for strategy in (self.attacker_strategy_pool or ())
            if str(strategy).strip().lower() in self.learned_attacker_aliases
        }
        if (
            self.fixed_attacker_strategy is not None
            and str(self.fixed_attacker_strategy).strip().lower()
            in self.learned_attacker_aliases
        ):
            requested_learned_aliases.add(
                str(self.fixed_attacker_strategy).strip().lower()
            )
        missing_specs = sorted(
            requested_learned_aliases - set(self.learned_attacker_specs)
        )
        if missing_specs:
            raise ValueError(
                f'attacker strategies {missing_specs} require learned_attacker_specs'
            )
        weights_cfg = self.env_configs.get(
            'attacker_strategy_pool_weights',
            getattr(SetupParameters, 'TRAIN_ATTACKER_STRATEGY_WEIGHTS', None),
        )
        self.attacker_strategy_pool_weights = self._resolve_attacker_strategy_weights(
            weights_cfg,
            self.attacker_strategy_pool,
        )
        self.runner_use_gpu = bool(self.env_configs.get('runner_use_gpu', False))
        self.device = torch.device(
            'cuda' if self.runner_use_gpu and torch.cuda.is_available() else 'cpu'
        )
        self.obstacle_density = self._resolve_obstacle_density(
            self.env_configs.get(
                'obstacle_density',
                getattr(SetupParameters, 'OBSTACLE_DENSITY', getattr(map_config, 'DEFAULT_OBSTACLE_DENSITY', None)),
            )
        )
        self.nmn_cl_stage = self._resolve_nmn_stage(self.env_configs.get('nmn_cl_stage', 2))

        self.local_network = create_network(network_type).to(self.device)
        self._apply_nmn_stage_to_local_network()
        self.local_network.eval()
        self._is_recurrent_policy = bool(getattr(self.local_network, 'is_recurrent', False))
        self._has_multitask_aux = bool(hasattr(self.local_network, 'multitask_auxiliary_loss'))
        self.actor_hidden = None
        self.critic_hidden = None
        
        self._init_env()
        
        self.opponent_policies = self._create_opponent_policies()
        self.current_opponent_key = None
        
        self._reset()
        
        # 奖励标准化器
        self.reward_normalizer = RewardNormalizer(
            gamma=TrainingParameters.GAMMA
        ) if TrainingParameters.REWARD_NORMALIZATION else None

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        raw = os.environ.get(name)
        if raw is None or str(raw).strip() == '':
            return float(default)
        return float(raw)

    @staticmethod
    def _env_float_pair(name: str, default):
        raw = os.environ.get(name)
        if raw is None or str(raw).strip() == '':
            return default
        parts = [p.strip() for p in str(raw).replace(';', ',').split(',') if p.strip()]
        if len(parts) != 2:
            raise ValueError(f"{name} must contain two comma-separated floats, got {raw!r}")
        return (float(parts[0]), float(parts[1]))

    @classmethod
    def _apply_network_env_overrides(cls):
        """Keep Ray worker networks consistent with train_skill.py overrides."""
        if os.environ.get("NMN_DUAL_GRU_INITIAL_LOG_STD") is not None:
            NetParameters.NMN_DUAL_GRU_INITIAL_LOG_STD = cls._env_float(
                "NMN_DUAL_GRU_INITIAL_LOG_STD",
                getattr(NetParameters, "NMN_DUAL_GRU_INITIAL_LOG_STD", 0.0),
            )
        if os.environ.get("NMN_DUAL_GRU_MIN_LOG_STD") is not None:
            NetParameters.NMN_DUAL_GRU_MIN_LOG_STD = cls._env_float(
                "NMN_DUAL_GRU_MIN_LOG_STD",
                getattr(NetParameters, "NMN_DUAL_GRU_MIN_LOG_STD", -20.0),
            )
        if os.environ.get("NMN_DUAL_GRU_MAX_LOG_STD") is not None:
            NetParameters.NMN_DUAL_GRU_MAX_LOG_STD = cls._env_float(
                "NMN_DUAL_GRU_MAX_LOG_STD",
                getattr(NetParameters, "NMN_DUAL_GRU_MAX_LOG_STD", 2.0),
            )
        if os.environ.get("NMN_DUAL_GRU_POLICY_HEAD_GAIN") is not None:
            NetParameters.NMN_DUAL_GRU_POLICY_HEAD_GAIN = cls._env_float(
                "NMN_DUAL_GRU_POLICY_HEAD_GAIN",
                getattr(NetParameters, "NMN_DUAL_GRU_POLICY_HEAD_GAIN", np.sqrt(2)),
            )
        NetParameters.NMN_DUAL_GRU_INITIAL_MEAN_BIAS = cls._env_float_pair(
            "NMN_DUAL_GRU_INITIAL_MEAN_BIAS",
            getattr(NetParameters, "NMN_DUAL_GRU_INITIAL_MEAN_BIAS", (0.0, 0.0)),
        )

    @staticmethod
    def _resolve_attacker_strategy_pool(
        pool_cfg,
        learned_alias: str = DEFAULT_LEARNED_ATTACKER_ALIAS,
        learned_aliases=None,
    ):
        """Parse optional attacker training pool from config."""
        if pool_cfg is None:
            return None
        if isinstance(pool_cfg, str):
            raw = [s.strip() for s in pool_cfg.split(',')]
        elif isinstance(pool_cfg, (list, tuple, set, np.ndarray)):
            raw = [str(s).strip() for s in list(pool_cfg)]
        else:
            raise ValueError(f'Invalid attacker_strategy_pool type: {type(pool_cfg)}')

        valid_learned_aliases = {normalize_learned_attacker_alias(learned_alias)}
        if learned_aliases is not None:
            valid_learned_aliases.update(
                normalize_learned_attacker_alias(alias)
                for alias in learned_aliases
            )
        normalized = []
        for strategy in raw:
            if not strategy:
                continue
            key = str(strategy).lower()
            if (
                key not in SUPPORTED_STRATEGIES
                and key not in PROGRAMMATIC_ATTACKER_ALIASES
                and key not in valid_learned_aliases
            ):
                raise ValueError(
                    f'Unsupported attacker strategy in pool: {strategy}. '
                    f'Valid={tuple(SUPPORTED_STRATEGIES) + tuple(PROGRAMMATIC_ATTACKER_ALIASES) + tuple(sorted(valid_learned_aliases))}'
                )
            if key not in normalized:
                normalized.append(key)
        return tuple(normalized) if normalized else None

    @staticmethod
    def _resolve_obstacle_density(density_cfg):
        if density_cfg is None:
            return None
        density = str(density_cfg).strip().lower()
        valid_levels = tuple(getattr(map_config.ObstacleDensity, 'ALL_LEVELS', ()))
        if valid_levels and density not in valid_levels:
            raise ValueError(f'Invalid obstacle_density: {density_cfg}. Valid={valid_levels}')
        return density

    @staticmethod
    def _resolve_attacker_strategy_weights(weights_cfg, pool):
        if weights_cfg is None or pool is None:
            return None
        if isinstance(weights_cfg, str):
            raw = [s.strip() for s in weights_cfg.split(',') if s.strip()]
        elif isinstance(weights_cfg, (list, tuple, np.ndarray)):
            raw = list(weights_cfg)
        else:
            raise ValueError(f'Invalid attacker_strategy_pool_weights type: {type(weights_cfg)}')
        if len(raw) != len(pool):
            raise ValueError(
                f'attacker_strategy_pool_weights length mismatch: '
                f'weights={len(raw)}, pool={len(pool)}'
            )
        weights = np.asarray([float(v) for v in raw], dtype=np.float64)
        if np.any(weights < 0.0):
            raise ValueError(f'attacker_strategy_pool_weights must be non-negative: {weights_cfg!r}')
        total = float(weights.sum())
        if total <= 0.0:
            raise ValueError(f'attacker_strategy_pool_weights sum must be positive: {weights_cfg!r}')
        return weights / total

    def set_attacker_strategy_pool_weights(self, weights_cfg):
        """Update attacker sampling weights without rebuilding the worker."""
        self.attacker_strategy_pool_weights = self._resolve_attacker_strategy_weights(
            weights_cfg,
            self.attacker_strategy_pool,
        )
        return None if self.attacker_strategy_pool_weights is None else self.attacker_strategy_pool_weights.tolist()

    @staticmethod
    def _resolve_nmn_stage(stage_cfg) -> int:
        stage = int(stage_cfg)
        if stage not in (1, 2):
            raise ValueError(f'nmn_cl_stage must be 1 or 2, got {stage_cfg!r}')
        return stage

    def _apply_nmn_stage_to_local_network(self):
        if hasattr(self.local_network, 'set_nmn_stage'):
            self.local_network.set_nmn_stage(self.nmn_cl_stage)

    def _apply_obstacle_density(self):
        if self.obstacle_density is not None and hasattr(map_config, 'set_obstacle_density'):
            map_config.set_obstacle_density(self.obstacle_density)
    
    def _init_env(self):
        self._apply_obstacle_density()

        if 'episode_len' in self.env_configs and self.env_configs.get('episode_len') is not None:
            EnvParameters.EPISODE_LEN = int(self.env_configs.get('episode_len'))

        reward_mode_cfg = self.env_configs.get('reward_mode', None)
        if reward_mode_cfg is None:
            if self.skill_mode == 'baseline':
                reward_mode = 'baseline'
            else:
                reward_mode = self.skill_mode
        else:
            reward_mode = str(reward_mode_cfg).strip().lower()
            if reward_mode != 'baseline' and self.skill_mode == 'baseline':
                reward_mode = 'baseline'

        self.env = TADEnv(
            reward_mode=reward_mode,
            emit_skill_rewards=bool(hasattr(self.local_network, 'multitask_auxiliary_loss')),
        )

        self.expert_policy = None
        if self.training_mode == 'mixed':
            expert_skill_mode = str(
                self.env_configs.get('expert_skill_mode', self.skill_mode)
            ).strip().lower()
            if expert_skill_mode not in ('protect', 'protect1', 'protect2', 'chase'):
                expert_skill_mode = 'chase'
            self.expert_policy = DefenderGlobalPolicy(
                env_width=self.env.width,
                env_height=self.env.height,
                defender_speed=self.env.defender_speed,
                defender_max_turn=getattr(map_config, 'defender_max_angular_speed', 6.0),
                skill_mode=expert_skill_mode,
            )
    
    def _create_opponent_policies(self) -> Dict[str, Any]:
        """创建对手策略池"""
        # 规则策略始终可用；每个 learned checkpoint 每个 worker 只加载一次。
        policies = {
            'attacker_global': ATTACKER_POLICY_REGISTRY['attacker_global'],
            'attacker_static': ATTACKER_POLICY_REGISTRY['attacker_static'],
        }
        for alias, spec in self.learned_attacker_specs.items():
            policies[self._learned_policy_key(alias)] = ATTACKER_POLICY_REGISTRY[
                'attacker_learned'
            ](
                checkpoint_path=spec['checkpoint'],
                device=self.device,
                alias=alias,
                reward_style=spec.get('reward_style'),
            )
        return policies

    @staticmethod
    def _learned_policy_key(alias: str) -> str:
        return f'attacker_learned:{normalize_learned_attacker_alias(alias)}'

    def _is_learned_attacker(self, strategy: str) -> bool:
        return str(strategy).strip().lower() in self.learned_attacker_aliases

    def _is_programmatic_attacker(self, strategy: str) -> bool:
        aliases = getattr(
            self,
            'programmatic_attacker_aliases',
            tuple(PROGRAMMATIC_ATTACKER_ALIASES),
        )
        return str(strategy).strip().lower() in aliases

    @staticmethod
    def _programmatic_policy_key(alias: str) -> str:
        return f'attacker_programmatic:{str(alias).strip().lower()}'

    @staticmethod
    def _create_programmatic_attacker(alias: str, seed: int):
        # Import after TADEnv/policies are initialized; the legacy env package
        # has an import-order dependency through policies.attacker_global.
        from attacker_heuristics.registry import create_policy

        return create_policy(
            PROGRAMMATIC_ATTACKER_ALIASES[str(alias).strip().lower()],
            seed=int(seed),
        )
    
    def _sample_opponent_policy(self) -> Tuple[str, Optional[str]]:
        """
        采样对手策略
        
        Returns:
            (policy_key, strategy): policy_key 是 ATTACKER_POLICY_REGISTRY 中的键，
                                   strategy 是 AttackerGlobalPolicy 的具体策略（如 'default', 'zigzag' 等）
        """
        skill_mode = self.skill_mode
        
        forced_strategy = self.fixed_attacker_strategy
        if forced_strategy is not None:
            forced = str(forced_strategy).lower()
            if self._is_learned_attacker(forced):
                return self._learned_policy_key(forced), forced
            if self._is_programmatic_attacker(forced):
                return self._programmatic_policy_key(forced), forced
            if forced == 'static':
                return 'attacker_static', None
            if forced == 'random':
                pool = self.attacker_strategy_pool or TRAINING_STRATEGIES
                probs = self.attacker_strategy_pool_weights if self.attacker_strategy_pool else None
                strategy = np.random.choice(pool, p=probs)
                if self._is_learned_attacker(strategy):
                    return self._learned_policy_key(strategy), str(strategy)
                if self._is_programmatic_attacker(strategy):
                    return self._programmatic_policy_key(strategy), str(strategy)
                return 'attacker_global', strategy
            if forced in SUPPORTED_STRATEGIES:
                return 'attacker_global', forced

        if self.attacker_strategy_pool:
            strategy = np.random.choice(self.attacker_strategy_pool, p=self.attacker_strategy_pool_weights)
            if self._is_learned_attacker(strategy):
                return self._learned_policy_key(strategy), str(strategy)
            if self._is_programmatic_attacker(strategy):
                return self._programmatic_policy_key(strategy), str(strategy)
            return 'attacker_global', strategy

        if skill_mode == 'protect1':
            return 'attacker_static', None  # 阶段1: 静止对手
        else:
            # protect2, chase, 其他模式: 从默认训练策略集中随机选择
            strategy = np.random.choice(TRAINING_STRATEGIES)
            return 'attacker_global', strategy
    
    def _reset(self, for_eval: bool = False, episode_idx: int = 0):
        """
        重置环境和对手策略
        
        Args:
            for_eval: 是否为评估模式（使用评估种子设置）
            episode_idx: 当前episode索引（用于固定种子时区分不同episode）
        """
        self._apply_obstacle_density()
        self._apply_nmn_stage_to_local_network()

        policy_key, strategy = self._sample_opponent_policy()
        self.current_opponent_key = policy_key
        policy_cls = self.opponent_policies.get(policy_key)
        learned_policy = str(policy_key).startswith('attacker_learned:')
        programmatic_policy = str(policy_key).startswith('attacker_programmatic:')
        if learned_policy and policy_cls is None:
            raise ValueError(
                f'attacker strategy {strategy!r} requires a matching '
                'learned_attacker_specs checkpoint'
            )
        
        reset_seed = self._resolve_reset_seed(for_eval=for_eval, episode_idx=episode_idx)

        # Instantiate policy with strategy if applicable.
        if programmatic_policy:
            policy_seed = (
                int(reset_seed)
                if reset_seed is not None
                else int(np.random.randint(0, 2**31 - 1))
            )
            self.attacker_policy = self._create_programmatic_attacker(
                str(strategy),
                policy_seed,
            )
        elif learned_policy:
            self.attacker_policy = policy_cls
        elif policy_key == 'attacker_global' and strategy is not None:
            self.attacker_policy = policy_cls(
                env_width=self.env.width,
                env_height=self.env.height,
                attacker_speed=self.env.attacker_speed,
                attacker_max_turn=getattr(map_config, 'attacker_max_angular_speed', 12.0),
                strategy=strategy
            )
        else:
            self.attacker_policy = policy_cls(
                env_width=self.env.width,
                env_height=self.env.height,
                attacker_speed=self.env.attacker_speed,
                attacker_max_turn=getattr(map_config, 'attacker_max_angular_speed', 12.0)
            )
        if programmatic_policy:
            self.attacker_policy.reset(seed=reset_seed)
        else:
            self.attacker_policy.reset()

        obs, _ = self.env.reset(seed=reset_seed)
        self.defender_obs, self.attacker_obs = obs
        
        self.done = False
        self.episode_reward = 0.0
        self.episode_len = 0
        self.actor_hidden = None
        self.critic_hidden = None
        self.anchor_actor_hidden = None
        self.anchor_critic_hidden = None
        # 重置标准化器的折扣回报追踪
        if hasattr(self, 'reward_normalizer') and self.reward_normalizer is not None:
            self.reward_normalizer.reset_ret()

    def _resolve_reset_seed(self, for_eval: bool = False, episode_idx: int = 0):
        if for_eval:
            if self.eval_use_random_seed:
                return None
            return int(self.eval_fixed_seed) + int(episode_idx)
        if self.train_use_random_seed:
            return None
        return int(self.seed) + int(self.meta_agent_id) * 100

    def set_weights(self, weights):
        state_dict = {}
        for k, v in weights.items():
            if isinstance(v, torch.Tensor):
                state_dict[k] = v.clone().detach().to(self.device)
            else:
                state_dict[k] = torch.as_tensor(v, device=self.device)
        self.local_network.load_state_dict(state_dict)
        self.local_network.eval()
        self.actor_hidden = None
        self.critic_hidden = None

    def set_dual_gru_log_std_bounds(self, min_log_std=None, max_log_std=None):
        if min_log_std is not None:
            NetParameters.NMN_DUAL_GRU_MIN_LOG_STD = float(min_log_std)
            if hasattr(self.local_network, "min_log_std"):
                self.local_network.min_log_std = float(min_log_std)
        if max_log_std is not None:
            NetParameters.NMN_DUAL_GRU_MAX_LOG_STD = float(max_log_std)
            if hasattr(self.local_network, "max_log_std"):
                self.local_network.max_log_std = float(max_log_std)

    def _serialize_recurrent_hidden(self, hidden, role: str):
        if hasattr(self.local_network, 'recurrent_hidden_spec'):
            num_layers, hidden_size = self.local_network.recurrent_hidden_spec(role)
            if hidden is None:
                return np.zeros((int(num_layers), int(hidden_size)), dtype=np.float32)
            hidden_cpu = hidden.detach().to(device='cpu', dtype=torch.float32)
            if hidden_cpu.dim() == 3:
                hidden_cpu = hidden_cpu[:, 0, :]
            return hidden_cpu.numpy().copy()
        gru = getattr(self.local_network, f'{role}_gru', None)
        if gru is None:
            return None
        if hidden is None:
            return np.zeros((gru.num_layers, gru.hidden_size), dtype=np.float32)
        hidden_cpu = hidden.detach().to(device='cpu', dtype=torch.float32)
        if hidden_cpu.dim() == 3:
            hidden_cpu = hidden_cpu[:, 0, :]
        return hidden_cpu.numpy().copy()

    @staticmethod
    def _discounted_component_returns(rewards, dones, gamma: float):
        returns = np.zeros_like(rewards, dtype=np.float32)
        running = 0.0
        for t in reversed(range(len(rewards))):
            running = float(rewards[t]) + float(gamma) * running * (1.0 - float(dones[t]))
            returns[t] = running
        return returns

    def _policy_act(self, obs_t, critic_obs_t):
        with torch.no_grad():
            if self._is_recurrent_policy and hasattr(self.local_network, 'act_recurrent'):
                actions, log_probs, pre_tanh, values, next_actor_hidden, next_critic_hidden = (
                    self.local_network.act_recurrent(
                        obs_t,
                        critic_obs_t,
                        actor_hidden=self.actor_hidden,
                        critic_hidden=self.critic_hidden,
                    )
                )
                self.actor_hidden = next_actor_hidden.detach() if next_actor_hidden is not None else None
                self.critic_hidden = next_critic_hidden.detach() if next_critic_hidden is not None else None
            else:
                actions, log_probs, pre_tanh, values = self.local_network.act(obs_t, critic_obs_t)
        return actions, log_probs, pre_tanh, values

    def _policy_eval_action(self, obs_t, critic_obs_t, greedy: bool = True):
        with torch.no_grad():
            if self._is_recurrent_policy and hasattr(self.local_network, 'forward_recurrent'):
                mean, _value, log_std, next_actor_hidden, next_critic_hidden = self.local_network.forward_recurrent(
                    obs_t,
                    critic_obs_t,
                    actor_hidden=self.actor_hidden,
                    critic_hidden=self.critic_hidden,
                )
                self.actor_hidden = next_actor_hidden.detach() if next_actor_hidden is not None else None
                self.critic_hidden = next_critic_hidden.detach() if next_critic_hidden is not None else None
                if greedy:
                    return torch.tanh(mean)
                pre_tanh = mean + torch.exp(log_std) * torch.randn_like(mean)
                return torch.tanh(pre_tanh)
            if greedy:
                policy_output, _, _log_std = self.local_network(obs_t, critic_obs_t)
                return torch.tanh(policy_output)
            actions, _, _, _ = self.local_network.act(obs_t, critic_obs_t)
            return actions

    def _critic_bootstrap_value(self, critic_obs_t):
        with torch.no_grad():
            if self._is_recurrent_policy and hasattr(self.local_network, 'critic_value_recurrent'):
                value, _ = self.local_network.critic_value_recurrent(
                    critic_obs_t,
                    critic_hidden=self.critic_hidden,
                )
            else:
                value = self.local_network.critic_value(critic_obs_t)
        return value

    def run(self, num_steps: int, profile: bool = False) -> Dict[str, np.ndarray]:
        mb_obs = []
        mb_critic_obs = []
        mb_actions = []
        mb_log_probs = []
        mb_values = []
        mb_rewards = []
        mb_dones = []
        mb_expert_actions = []
        mb_actor_hiddens = [] if self._is_recurrent_policy else None
        mb_critic_hiddens = [] if self._is_recurrent_policy else None
        mb_chase_rewards = [] if self._has_multitask_aux else None
        mb_baseline_rewards = [] if self._has_multitask_aux else None
        mb_collision_labels = [] if self._has_multitask_aux else None

        perf = {'per_r': [], 'per_episode_len': [], 'win': []}

        timings = None
        profiled_keys = (
            'critic_obs', 'tensorize', 'policy_inference', 'expert_policy',
            'opponent_policy', 'env_step', 'reward_norm', 'buffer_ops',
            'episode_reset', 'bootstrap_value', 'gae_compute', 'pack_numpy'
        )
        if profile:
            timings = {k: 0.0 for k in profiled_keys}
            rollout_start = time.perf_counter()
            finished_episodes = 0

        for _ in range(num_steps):
            if profile:
                t0 = time.perf_counter()
            critic_obs = build_critic_observation(self.defender_obs, self.attacker_obs)
            if profile:
                timings['critic_obs'] += time.perf_counter() - t0

            if profile:
                t0 = time.perf_counter()
            obs_t = torch.tensor(self.defender_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            critic_obs_t = torch.tensor(critic_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            if profile:
                timings['tensorize'] += time.perf_counter() - t0

            if profile:
                t0 = time.perf_counter()
            if self._is_recurrent_policy:
                actor_hidden_in = self._serialize_recurrent_hidden(self.actor_hidden, 'actor')
                critic_hidden_in = self._serialize_recurrent_hidden(self.critic_hidden, 'critic')
            actions, log_probs, pre_tanh, values = self._policy_act(obs_t, critic_obs_t)
            if profile:
                timings['policy_inference'] += time.perf_counter() - t0

            if profile:
                t0 = time.perf_counter()
            tanh_action = actions.cpu().numpy().flatten()
            pre_tanh_action = pre_tanh.cpu().numpy().flatten()
            log_prob = log_probs.cpu().numpy().item()
            value = values.cpu().numpy().item()
            mb_obs.append(self.defender_obs.copy())
            mb_critic_obs.append(critic_obs.copy())
            mb_actions.append(pre_tanh_action)
            mb_log_probs.append(log_prob)
            mb_values.append(value)
            if self._is_recurrent_policy:
                mb_actor_hiddens.append(actor_hidden_in)
                mb_critic_hiddens.append(critic_hidden_in)
            if profile:
                timings['buffer_ops'] += time.perf_counter() - t0

            # Mixed training adds expert actions to the PPO rollout.
            if profile:
                t0 = time.perf_counter()
            if self.training_mode == 'mixed':
                priv_state = self.env.get_privileged_state()
                expert_action = self.expert_policy.get_action(self.defender_obs, priv_state)
            else:
                expert_action = np.zeros(NetParameters.ACTION_DIM, dtype=np.float32)
            mb_expert_actions.append(expert_action)
            if profile:
                timings['expert_policy'] += time.perf_counter() - t0

            # Get attacker action
            if profile:
                t0 = time.perf_counter()
            attacker_action = self.attacker_policy.get_action(self.attacker_obs)
            if profile:
                timings['opponent_policy'] += time.perf_counter() - t0

            # Step environment
            if profile:
                t0 = time.perf_counter()
            obs, reward, terminated, truncated, info = self.env.step(tanh_action, attacker_action)
            if profile:
                timings['env_step'] += time.perf_counter() - t0
            done = terminated or truncated
            if self._has_multitask_aux:
                skill_rewards = dict(info.get('skill_rewards', {}) or {})
                mb_chase_rewards.append(float(skill_rewards.get('chase', 0.0)))
                mb_baseline_rewards.append(float(skill_rewards.get('baseline', 0.0)))
                mb_collision_labels.append(1.0 if bool(info.get('defender_collision', False)) else 0.0)

            self.defender_obs, self.attacker_obs = obs
            self.done = done
            self.episode_reward += reward
            self.episode_len += 1

            # 奖励标准化: reward / sqrt(running_var(G_t))
            if profile:
                t0 = time.perf_counter()
            if self.reward_normalizer is not None:
                norm_reward = self.reward_normalizer.update(reward, done)
            else:
                norm_reward = reward
            if profile:
                timings['reward_norm'] += time.perf_counter() - t0
            mb_rewards.append(norm_reward)
            mb_dones.append(done)

            if done:
                one_ep = {
                    'episode_reward': self.episode_reward,
                    'num_step': self.episode_len,
                    'win': info.get('win', False)
                }
                update_perf(one_ep, perf)
                perf['win'].append(one_ep['win'])
                if profile:
                    t0 = time.perf_counter()
                self._reset()
                if profile:
                    timings['episode_reset'] += time.perf_counter() - t0
                    finished_episodes += 1

        if profile:
            t0 = time.perf_counter()
        last_critic_obs = build_critic_observation(self.defender_obs, self.attacker_obs)
        obs_t = torch.tensor(self.defender_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        critic_obs_t = torch.tensor(last_critic_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        last_value = self._critic_bootstrap_value(critic_obs_t).cpu().numpy().item()
        if profile:
            timings['bootstrap_value'] += time.perf_counter() - t0

        if profile:
            t0 = time.perf_counter()
        mb_obs = np.array(mb_obs, dtype=np.float32)
        mb_critic_obs = np.array(mb_critic_obs, dtype=np.float32)
        mb_actions = np.array(mb_actions, dtype=np.float32)
        mb_log_probs = np.array(mb_log_probs, dtype=np.float32)
        mb_values = np.array(mb_values, dtype=np.float32)
        mb_rewards = np.array(mb_rewards, dtype=np.float32)
        mb_dones = np.array(mb_dones, dtype=np.float32)
        mb_expert_actions = np.array(mb_expert_actions, dtype=np.float32)
        if self._has_multitask_aux:
            mb_chase_rewards = np.array(mb_chase_rewards, dtype=np.float32)
            mb_baseline_rewards = np.array(mb_baseline_rewards, dtype=np.float32)
            mb_collision_labels = np.array(mb_collision_labels, dtype=np.float32)
        if self._is_recurrent_policy:
            mb_actor_hiddens = np.array(mb_actor_hiddens, dtype=np.float32)
            mb_critic_hiddens = np.array(mb_critic_hiddens, dtype=np.float32)
        if profile:
            timings['pack_numpy'] += time.perf_counter() - t0

        mb_advs = np.zeros_like(mb_rewards)
        mb_returns = np.zeros_like(mb_rewards)
        lastgaelam = 0.0

        if profile:
            t0 = time.perf_counter()
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_value = last_value
            else:
                next_value = mb_values[t + 1]

            done_t = mb_dones[t]
            delta = mb_rewards[t] + TrainingParameters.GAMMA * next_value * (1.0 - done_t) - mb_values[t]
            lastgaelam = delta + TrainingParameters.GAMMA * TrainingParameters.LAM * (1.0 - done_t) * lastgaelam
            mb_advs[t] = lastgaelam
        if profile:
            timings['gae_compute'] += time.perf_counter() - t0

        mb_returns = mb_advs + mb_values

        ret = {
            'obs': mb_obs,
            'critic_obs': mb_critic_obs,
            'actions': mb_actions,
            'log_probs': mb_log_probs,
            'values': mb_values,
            'returns': mb_returns,
            'advs': mb_advs,
            'dones': mb_dones,
            'expert_actions': mb_expert_actions,
            'perf': perf,
            'action_speed_norm_mean': float(np.mean(np.tanh(mb_actions[:, 1]))) if mb_actions.size else 0.0,
            'action_speed_norm_p10': float(np.percentile(np.tanh(mb_actions[:, 1]), 10)) if mb_actions.size else 0.0,
            'action_speed_norm_p90': float(np.percentile(np.tanh(mb_actions[:, 1]), 90)) if mb_actions.size else 0.0,
            'action_abs_turn_mean': float(np.mean(np.abs(np.tanh(mb_actions[:, 0])))) if mb_actions.size else 0.0,
        }
        if self._is_recurrent_policy:
            ret['actor_hiddens'] = mb_actor_hiddens
            ret['critic_hiddens'] = mb_critic_hiddens
        if self._has_multitask_aux:
            aux_return_clip = float(getattr(TrainingParameters, 'MULTITASK_AUX_RETURN_CLIP', 20.0))
            chase_returns = self._discounted_component_returns(
                mb_chase_rewards,
                mb_dones,
                TrainingParameters.GAMMA,
            )
            baseline_returns = self._discounted_component_returns(
                mb_baseline_rewards,
                mb_dones,
                TrainingParameters.GAMMA,
            )
            if aux_return_clip > 0.0:
                chase_returns = np.clip(chase_returns, -aux_return_clip, aux_return_clip)
                baseline_returns = np.clip(baseline_returns, -aux_return_clip, aux_return_clip)
            ret['aux_targets'] = {
                'chase_returns': chase_returns.astype(np.float32),
                'baseline_returns': baseline_returns.astype(np.float32),
                'collision_labels': mb_collision_labels,
            }
        if profile:
            profiled_total = sum(timings[k] for k in profiled_keys)
            timings['profiled_total'] = profiled_total
            timings['rollout_total'] = time.perf_counter() - rollout_start
            timings['untracked'] = max(0.0, timings['rollout_total'] - profiled_total)
            timings['num_steps'] = float(num_steps)
            timings['finished_episodes'] = float(finished_episodes)
            ret['timings'] = timings
        return ret

    def evaluate(
        self,
        num_episodes: int = 5,
        greedy: bool = True,
        record_gif: bool = False,
        attacker_strategy: Optional[str] = None,
    ) -> Dict:
        perf = {'per_r': [], 'per_episode_len': [], 'win': []}
        outcome_counts = {}
        frames = [] if record_gif else None
        trajectory_data = None  # trajectory data for the first episode (for static plot)
        previous_fixed_attacker_strategy = self.fixed_attacker_strategy
        if attacker_strategy is not None and str(attacker_strategy).strip():
            self.fixed_attacker_strategy = str(attacker_strategy).strip().lower()
        
        try:
            for ep_idx in range(num_episodes):
                self._reset(for_eval=True, episode_idx=ep_idx)
                ep_reward = 0.0
                ep_len = 0
                ep_frames = []

                # Record initial positions for first episode's trajectory plot
                record_traj = (ep_idx == 0)
                if record_traj:
                    priv = self.env.get_privileged_state()
                    target_pos = (priv['target']['center_x'], priv['target']['center_y'])
                    ep_def_traj = [(priv['defender']['center_x'], priv['defender']['center_y'])]
                    ep_atk_traj = [(priv['attacker']['center_x'], priv['attacker']['center_y'])]
                    ep_def_theta = [priv['defender'].get('theta', 0.0)]
                    ep_atk_theta = [priv['attacker'].get('theta', 0.0)]

                info = {}
                while not self.done:
                    if record_gif and ep_idx == 0:
                        # Use the compact scene renderer for diagnostic GIFs;
                        # static trajectory figures keep the academic style.
                        frame = self.env.render(mode='rgb_array', style='pygame')
                        if frame is not None:
                            ep_frames.append(frame)

                    critic_obs = build_critic_observation(self.defender_obs, self.attacker_obs)
                    obs_t = torch.tensor(self.defender_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    critic_obs_t = torch.tensor(critic_obs, dtype=torch.float32, device=self.device).unsqueeze(0)

                    action = self._policy_eval_action(obs_t, critic_obs_t, greedy=greedy).cpu().numpy().flatten()

                    attacker_action = self.attacker_policy.get_action(self.attacker_obs)
                    obs, reward, terminated, truncated, info = self.env.step(action, attacker_action)
                    done = terminated or truncated

                    self.defender_obs, self.attacker_obs = obs
                    self.done = done
                    ep_reward += reward
                    ep_len += 1

                    # Record trajectory
                    if record_traj:
                        priv = self.env.get_privileged_state()
                        ep_def_traj.append((priv['defender']['center_x'], priv['defender']['center_y']))
                        ep_atk_traj.append((priv['attacker']['center_x'], priv['attacker']['center_y']))
                        ep_def_theta.append(priv['defender'].get('theta', 0.0))
                        ep_atk_theta.append(priv['attacker'].get('theta', 0.0))

                    if ep_len >= EnvParameters.EPISODE_LEN:
                        break

                one_ep = {
                    'episode_reward': ep_reward,
                    'num_step': ep_len,
                    'win': info.get('win', False)
                }
                update_perf(one_ep, perf)
                perf['win'].append(one_ep['win'])
                terminal_reason = str(info.get('reason', 'unknown')).strip().lower() or 'unknown'
                outcome_counts[terminal_reason] = outcome_counts.get(terminal_reason, 0) + 1

                if record_gif and ep_idx == 0 and ep_frames:
                    frames = ep_frames

                # Save trajectory data from first episode
                if record_traj:
                    from configs import map_config as _mc
                    trajectory_data = {
                        'defender_traj': ep_def_traj,
                        'attacker_traj': ep_atk_traj,
                        'defender_theta': ep_def_theta,
                        'attacker_theta': ep_atk_theta,
                        'target_pos': target_pos,
                        'obstacles': list(getattr(_mc, 'obstacles', [])),
                        'width': getattr(_mc, 'width', 640),
                        'height': getattr(_mc, 'height', 640),
                        'win': one_ep['win'],
                        'skill_mode': self.skill_mode,
                        'episode_len': ep_len,
                        'episode_reward': ep_reward,
                        'capture_radius': getattr(_mc, 'capture_radius', 20),
                        'capture_sector_angle_deg': getattr(_mc, 'capture_sector_angle_deg', 30),
                    }
        finally:
            self.fixed_attacker_strategy = previous_fixed_attacker_strategy
            for policy_key, learned_policy in self.opponent_policies.items():
                if str(policy_key).startswith('attacker_learned:'):
                    learned_policy.reset()
            self._reset()
        
        total_outcomes = max(1, int(sum(outcome_counts.values())))
        return {
            'perf': perf,
            'outcome_counts': outcome_counts,
            'outcome_rates': {
                reason: float(count) / float(total_outcomes)
                for reason, count in outcome_counts.items()
            },
            'frames': frames,
            'trajectory_data': trajectory_data,
        }
