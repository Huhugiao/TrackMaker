"""
D vs A Evaluation Script with Suite & Interactive Modes
Revived and Enhanced.
"""

import os
import sys
import argparse
import glob
import json
import time
import re
import numpy as np
import torch
from datetime import datetime
from typing import Dict, Optional, Tuple, List, Union

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import map_config
from configs.map_config import EnvParameters, set_obstacle_density, set_map_layout
from configs.paths import CHECKPOINTS_DIR, EVAL_OUTPUT_DIR
from envs import env_lib
from attacker.learned_policy import LearnedAttackerPolicy
from skill.model import Model
from skill.util import (
    build_critic_observation,
    get_device,
    make_gif,
    make_trajectory_plot,
    print_device_info,
)
from configs.skill_config import SetupParameters, NetParameters


def _display_network_type(network_type: str) -> str:
    return 'mlp' if str(network_type) == 'mlp_noctde' else str(network_type)


def _read_env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, None)
    if raw is None:
        return bool(default)
    text = str(raw).strip().lower()
    if not text:
        return bool(default)
    if text in ('1', 'true', 'yes', 'on'):
        return True
    if text in ('0', 'false', 'no', 'off'):
        return False
    return bool(default)


def _read_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, None)
    if raw is None:
        return int(default)
    text = str(raw).strip()
    if not text:
        return int(default)
    try:
        return int(text)
    except (TypeError, ValueError):
        return int(default)


def _read_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, None)
    if raw is None:
        return float(default)
    text = str(raw).strip()
    if not text:
        return float(default)
    try:
        return float(text)
    except (TypeError, ValueError):
        return float(default)


def _read_env_float_list(name: str, default):
    raw = os.environ.get(name, None)
    if raw is None:
        return tuple(float(v) for v in default)
    text = str(raw).strip()
    if not text:
        return tuple(float(v) for v in default)
    values = []
    for token in text.split(','):
        item = token.strip()
        if not item:
            continue
        try:
            values.append(float(item))
        except (TypeError, ValueError):
            return tuple(float(v) for v in default)
    return tuple(values) if values else tuple(float(v) for v in default)


def _current_task_margin(env: object) -> Optional[float]:
    base_env = env.env if hasattr(env, 'env') else env
    defender = getattr(base_env, 'defender', None)
    attacker = getattr(base_env, 'attacker', None)
    target = getattr(base_env, 'target', None)
    if not defender or not attacker or not target:
        return None
    defender_speed = max(1e-6, float(getattr(base_env, 'defender_speed', getattr(map_config, 'defender_speed', 2.6))))
    attacker_speed = max(1e-6, float(getattr(base_env, 'attacker_speed', getattr(map_config, 'attacker_speed', 2.0))))
    try:
        metrics = compute_path_risk_metrics(
            state={'defender': defender, 'attacker': attacker, 'target': target},
            defender_speed=defender_speed,
            attacker_speed=attacker_speed,
            width=float(getattr(base_env, 'width', 640.0)),
            height=float(getattr(base_env, 'height', 640.0)),
            grid_size=Model._env_float("HRL_TOP_ASTAR_GRID_SIZE", 8.0),
            obstacle_padding=Model._env_float("HRL_TOP_ASTAR_OBSTACLE_PADDING", 12.0),
            obstacles=getattr(map_config, 'obstacles', []),
            metric="astar",
        )
        return float(metrics["margin"])
    except Exception:
        pass
    d_da = float(np.hypot(float(defender['x']) - float(attacker['x']), float(defender['y']) - float(attacker['y'])))
    d_at = float(np.hypot(float(attacker['x']) - float(target['x']), float(attacker['y']) - float(target['y'])))
    return float((d_at / attacker_speed) - (d_da / defender_speed))

# Import Rule Policies
from policies import (
    AttackerGlobalPolicy,
    DefenderGlobalPolicy,
    DefenderHRLApolloniusLabelPolicy,
    DefenderHRLRulePolicy,
    create_reach_avoid_defender_policy,
)

# Import Environments
# Note: We use specific environments for different strategies to ensure correct observation/action spaces
from envs.hrl_env import HRLEnv
from envs.protect_env import ProtectEnv
from configs.hrl_config import HRLEnvTrainParameters
from envs.tad_env import TADEnv, TrackingEnv  # Fallback/Standard
from utils.path_risk import compute_path_risk_metrics

# --- Default Paths Configuration ---
DEFAULT_MODEL_PATHS = {
    # Retained Chapter 2 A* top policy.
    'hrl': str(CHECKPOINTS_DIR / "hrl_ch2_m1_astar_cached_top_20260606_170036" / "best_model.pth"),
    # Active Protect skill.
    'protect': str(CHECKPOINTS_DIR / "defender_protect_mlp_ctde_frozen6_20260721_105148" / "best_balanced_model.pth"),
    'chase': str(CHECKPOINTS_DIR / "defender_chase_nmn_dual_gru_raw_dense_05-05-19-12" / "final_model.pth"),
}

# --- HRL Skill Eval Defaults (edit here) ---
# 技能个数固定为 2：protect+chase。
HRL_EVAL_NUM_SKILLS = 2
# HRL技能路径统一由 DEFAULT_MODEL_PATHS 指定（单一来源）。
HRL_EVAL_PROTECT_SKILL_PATH = str(
    CHECKPOINTS_DIR / "defender_protect_mlp_ctde_repro_20260526" / "final_model.pth"
)
HRL_EVAL_CHASE_SKILL_PATH = DEFAULT_MODEL_PATHS.get('chase')

ALL_ATTACKER_STRATEGIES = [
    'default',
    'evasive',
]
LEARNED_ATTACKER_STRATEGIES = ['attacker_rl']
DEFAULT_MAP_LAYOUT = getattr(map_config, 'MapLayout').DEFAULT

NAV_RULE_DEFENDER_STRATEGIES = ['astar_to_attacker', 'astar_to_target']
REACH_AVOID_RULE_DEFENDER_STRATEGIES = [
    'cbf_qp',
    'cbf_qp_local',
    'cbf_qp_local_obs',
]
ALL_RULE_DEFENDER_STRATEGIES = NAV_RULE_DEFENDER_STRATEGIES + REACH_AVOID_RULE_DEFENDER_STRATEGIES
HRL_DEFENDER_STRATEGIES = ['hrl']
HRL_RULE_DEFENDER_STRATEGIES = ['hrl_rule_geo_trend', 'hrl_rule_apollonius_label']
RL_DEFENDER_STRATEGIES = [
    'rl', 'hrl',
    'protect', 'chase'
]
TEST_DEFENDER_STRATEGIES = ['hrl', 'hrl_rule_geo_trend', 'hrl_rule_apollonius_label', 'protect', 'chase']

def _is_hrl_like_strategy(strategy: str) -> bool:
    return strategy in HRL_DEFENDER_STRATEGIES or strategy in HRL_RULE_DEFENDER_STRATEGIES


def _predicted_top_skill_from_action(action: np.ndarray) -> str:
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if arr.size >= 2:
        return "chase" if float(arr[1]) >= float(arr[0]) else "protect"
    if arr.size == 1:
        return "chase" if int(np.rint(float(arr[0]))) == 1 else "protect"
    return "unknown"


def _hrl_step_trace_row(
    *,
    episode: int,
    step: int,
    seed: Optional[int],
    attacker: str,
    speed_regime: str,
    margin_regime: str,
    action: np.ndarray,
    selected_skill: Optional[str],
    risk_margin: Optional[float],
) -> dict:
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    action_format = "two_skill_scores" if arr.size >= 2 else "discrete_skill_idx" if arr.size == 1 else "unknown"
    protect_logit = float(arr[0]) if arr.size >= 2 else None
    chase_logit = float(arr[1]) if arr.size >= 2 else None
    return {
        "episode": int(episode),
        "step": int(step),
        "seed": None if seed is None else int(seed),
        "attacker": str(attacker or "unknown"),
        "speed_regime": str(speed_regime or "unknown"),
        "margin_regime": str(margin_regime or "unknown"),
        "cell": f"{str(speed_regime or 'unknown')}/{str(margin_regime or 'unknown')}",
        "action_dim": int(arr.size),
        "action_format": action_format,
        "top_action_0": float(arr[0]) if arr.size >= 1 else None,
        "top_action_1": float(arr[1]) if arr.size >= 2 else None,
        "protect_logit": protect_logit,
        "chase_logit": chase_logit,
        "predicted_top_skill": _predicted_top_skill_from_action(action),
        "selected_skill": str(selected_skill or "unknown"),
        "risk_margin": None if risk_margin is None else float(risk_margin),
    }


def _apply_network_type_constraints(defender_strategy: str, network_type: Optional[str]) -> Optional[str]:
    return network_type

# --- HRL Evaluation Hold Parameters (edit here when needed) ---
# 说明：评估阶段与HRL训练配置对齐。
HRL_EVAL_HOLD_MIN = int(HRLEnvTrainParameters.HOLD_MIN)
HRL_EVAL_HOLD_MAX = int(HRLEnvTrainParameters.HOLD_MAX)
HRL_EVAL_DISABLE_HOLD_CONTROL = bool(HRLEnvTrainParameters.DISABLE_HOLD_CONTROL)

import ray
from skill.util import get_adjusted_n_envs, get_ray_temp_dir


def _disable_env_hard_mask(env):
    base_env = env.env if hasattr(env, 'env') else env
    if hasattr(base_env, 'set_hard_action_mask'):
        base_env.set_hard_action_mask(False)
    else:
        setattr(base_env, 'hard_action_mask', False)


def _get_hrl_eval_defender_mask_params() -> Optional[Dict]:
    if not _read_env_bool('VS_HRL_SAFE_MASK_ENABLE', True):
        return None
    return {
        'obstacle_margin': _read_env_float('VS_HRL_SAFE_MASK_MARGIN', 0.0),
        'radius_scale': _read_env_float('VS_HRL_SAFE_MASK_RADIUS_SCALE', 0.993),
        'lookahead_steps': _read_env_int('VS_HRL_SAFE_MASK_LOOKAHEAD', 1),
        'speed_cost_weight': _read_env_float('VS_HRL_SAFE_MASK_SPEED_COST_WEIGHT', 0.35),
        'max_deviation_cost': _read_env_float('VS_HRL_SAFE_MASK_MAX_DEVIATION_COST', -1.0),
        'allow_zero_fallback': _read_env_bool('VS_HRL_SAFE_MASK_ALLOW_ZERO_FALLBACK', True),
        'speed_scales': _read_env_float_list(
            'VS_HRL_SAFE_MASK_SPEED_SCALES',
            (1.0, 0.8, 0.6, 0.4, 0.2, 0.0),
        ),
        'angle_fracs': _read_env_float_list(
            'VS_HRL_SAFE_MASK_ANGLE_FRACS',
            (0.0, -0.2, 0.2, -0.4, 0.4, -0.6, 0.6, -1.0, 1.0),
        ),
        'clearance_extras': _read_env_float_list(
            'VS_HRL_SAFE_MASK_CLEARANCE_EXTRAS',
            (0.0, 4.0, 8.0),
        ),
    }


def _maybe_enable_eval_defender_mask(env, defender_strategy: str):
    strategy = str(defender_strategy).strip().lower()
    enable_for_hrl = strategy == 'hrl'
    enable_for_chase = strategy == 'chase' and _read_env_bool('VS_CHASE_SAFE_MASK_ENABLE', False)
    enable_for_protect = strategy == 'protect' and _read_env_bool('VS_PROTECT_SAFE_MASK_ENABLE', False)
    if not (enable_for_hrl or enable_for_chase or enable_for_protect):
        return
    params = _get_hrl_eval_defender_mask_params()
    if not params:
        return
    base_env = env.env if hasattr(env, 'env') else env
    if hasattr(base_env, 'configure_hard_action_mask'):
        base_env.configure_hard_action_mask(True, role='defender', **params)
        label = 'HRL' if enable_for_hrl else ('Chase' if enable_for_chase else 'Protect')
        print(f"[Eval {label} SafeMask] {params}")


def _allow_parallel_learned_attacker() -> bool:
    return _read_env_bool('VS_PARALLEL_LEARNED_ATTACKER', True)


def _learned_attacker_worker_cap(default_workers: int) -> int:
    cap = _read_env_int('VS_PARALLEL_LEARNED_ATTACKER_WORKERS', default_workers)
    return int(max(1, cap))


def _configure_torch_eval_runtime():
    num_threads = _read_env_int('VS_EVAL_TORCH_NUM_THREADS', 1)
    if num_threads <= 0:
        return
    try:
        torch.set_num_threads(int(num_threads))
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(int(num_threads))
    except Exception:
        pass


def _configure_eval_environment():
    """固定为默认地图/环境配置。"""
    if hasattr(map_config, 'set_extra_obstacles'):
        map_config.set_extra_obstacles([])

    set_map_layout(DEFAULT_MAP_LAYOUT)
    density_override = str(os.environ.get('VS_OBSTACLE_DENSITY', '')).strip().lower()
    density_level = density_override or str(map_config.DEFAULT_OBSTACLE_DENSITY)
    set_obstacle_density(density_level)
    disable_obstacle_randomization = str(
        os.environ.get('VS_DISABLE_OBSTACLE_RANDOMIZATION', '')
    ).strip().lower() in ('1', 'true', 'yes', 'on')
    if disable_obstacle_randomization and hasattr(map_config, 'set_obstacle_randomization'):
        map_config.set_obstacle_randomization(enabled=False, jitter_px=0, seed=None)
    map_config.regenerate_obstacles(density_level=map_config.current_obstacle_density)

    env_lib.build_occupancy(
        width=map_config.width,
        height=map_config.height,
        cell=map_config.pixel_size,
        obstacles=getattr(map_config, 'obstacles', []),
    )


def _safe_filename_token(text: str) -> str:
    token = re.sub(r'[^0-9A-Za-z_.-]+', '_', str(text)).strip('_.')
    return token or 'unknown'


def _resolve_attacker_strategy_spec(
    attacker_strategy: str,
    attacker_strategy_params: Optional[Dict] = None,
) -> Tuple[str, Optional[Dict], str]:
    raw_strategy = str(attacker_strategy).strip()
    normalized_strategy = raw_strategy.lower()
    base_params = dict(attacker_strategy_params or {})
    return normalized_strategy, (base_params or None), (raw_strategy or normalized_strategy)


def _resolve_episode_length(info: Dict, env: object) -> int:
    """Resolve episode length robustly across wrapped and raw envs."""
    if not isinstance(info, dict):
        info = {}

    for key in ('episode_length', 'episode_len', 'step', 'step_count'):
        v = info.get(key, None)
        if v is not None:
            try:
                iv = int(v)
                if iv > 0:
                    return iv
            except (TypeError, ValueError):
                pass

    # Fallback: check common env attributes.
    candidates = [env]
    if hasattr(env, 'env'):
        candidates.append(getattr(env, 'env'))
    if hasattr(env, 'unwrapped'):
        candidates.append(getattr(env, 'unwrapped'))

    for obj in candidates:
        if obj is None:
            continue
        for attr in ('step_count', 'step'):
            if hasattr(obj, attr):
                try:
                    iv = int(getattr(obj, attr))
                    if iv > 0:
                        return iv
                except (TypeError, ValueError):
                    pass
    return 0


def _resolve_selected_skill_from_info(info: Dict) -> Optional[str]:
    if not isinstance(info, dict):
        return None
    selected_skill = info.get('selected_skill')
    if isinstance(selected_skill, str):
        selected_skill = selected_skill.strip().lower()
        if selected_skill in ('protect', 'chase'):
            return selected_skill

    top_idx = info.get('top_skill_idx')
    try:
        top_idx = int(top_idx)
    except (TypeError, ValueError):
        top_idx = None

    if top_idx is None or top_idx < 0:
        return None

    skill_names = info.get('skill_names')
    if isinstance(skill_names, (list, tuple)) and top_idx < len(skill_names):
        name = str(skill_names[top_idx]).strip().lower()
        if name in ('protect', 'chase'):
            return name

    # Two-skill action order is Protect, Chase.
    if top_idx == 0:
        return 'protect'
    if top_idx == 1:
        return 'chase'
    return None


def _find_latest_checkpoint(model_prefixes: List[str]) -> Optional[str]:
    candidates = []
    for prefix in model_prefixes:
        patterns = [
            str(CHECKPOINTS_DIR / f'{prefix}_*' / 'best_model.pth'),
            os.path.join('models', f'{prefix}_*', 'best_model.pth'),
        ]
        for pattern in patterns:
            candidates.extend(glob.glob(pattern))
    candidates = [p for p in candidates if os.path.isfile(p)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _find_latest_online_hrl_checkpoint() -> Optional[str]:
    """Find retained Chapter 2 HRL checkpoints."""
    preferred = str(
        CHECKPOINTS_DIR
        / 'hrl_ch2_m1_astar_cached_top_20260606_170036'
        / 'best_model.pth'
    )
    if os.path.isfile(preferred):
        return preferred
    patterns = [
        str(CHECKPOINTS_DIR / 'hrl_ch2_*_top_*' / 'best_model.pth'),
        os.path.join('models', 'hrl_ch2_*_top_*', 'best_model.pth'),
    ]
    candidates: List[str] = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))
    candidates = [p for p in candidates if os.path.isfile(p)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _default_model_path(strategy: str) -> Optional[str]:
    path = DEFAULT_MODEL_PATHS.get(strategy)
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{strategy}模型路径不存在: {path}")
        return path

    if strategy == 'hrl':
        return _find_latest_online_hrl_checkpoint()

    prefix_map = {
        'protect': ['defender_protect_mlp_ctde_frozen6', 'defender_protect_dense'],
        'chase': ['defender_chase_nmn_dual_gru_raw_dense'],
    }
    prefixes = prefix_map.get(strategy)
    if not prefixes:
        return None
    return _find_latest_checkpoint(prefixes)


def _resolve_hrl_skill_paths(
    strategy: str = 'hrl',
    protect_path: Optional[str] = None,
    chase_path: Optional[str] = None,
    num_skills: Optional[int] = None,
) -> Tuple[Optional[str], Optional[str]]:
    if num_skills is not None:
        num_skills = int(num_skills)
        if num_skills != 2:
            raise ValueError(f'HRL evaluation is fixed to 2 skills, got {num_skills}')

    protect_path = None if protect_path is None else str(protect_path).strip() or None
    chase_path = None if chase_path is None else str(chase_path).strip() or None
    cfg_protect = None if HRL_EVAL_PROTECT_SKILL_PATH is None else str(HRL_EVAL_PROTECT_SKILL_PATH).strip() or None
    cfg_chase = None if HRL_EVAL_CHASE_SKILL_PATH is None else str(HRL_EVAL_CHASE_SKILL_PATH).strip() or None
    protect_path = protect_path or cfg_protect
    chase_path = chase_path or cfg_chase or _default_model_path('chase')

    if not protect_path:
        raise FileNotFoundError(f'{strategy}评估缺少 primary skill checkpoint')
    if not chase_path:
        raise FileNotFoundError(f'{strategy}评估缺少chase skill checkpoint')

    return protect_path, chase_path



def _init_ray():
    """初始化Ray集群（与训练使用相同配置）。"""
    if ray.is_initialized():
        return
    ray_tmp = get_ray_temp_dir()
    ray_num_cpus = os.cpu_count() or 4
    ray_num_gpus = 0  # 评估不使用GPU
    print(f"[Ray] Init with {ray_num_cpus} CPUs for evaluation")
    kwargs = dict(
        num_cpus=ray_num_cpus,
        num_gpus=ray_num_gpus,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False,
    )
    if ray_tmp:
        kwargs['_temp_dir'] = ray_tmp
    ray.init(**kwargs)


@ray.remote
class EvalWorker:
    """Ray远程评估Worker - 每个worker独立运行多个episode。"""

    def __init__(self, worker_id: int):
        self.worker_id = worker_id

    def run_episodes(
        self,
        defender_strategy: str,
        attacker_strategy: str,
        attacker_strategy_params: Optional[Dict],
        num_episodes: int,
        defender_checkpoint: str,
        attacker_checkpoint: Optional[str],
        network_type: str,
        seed_offset: int,
        hrl_num_skills: Optional[int] = None,
        hrl_protect_skill_path: Optional[str] = None,
        hrl_chase_skill_path: Optional[str] = None,
        use_privileged_classifier: bool = False,
        eval_use_random_seed: bool = True,
        eval_fixed_seed: int = 42,
    ) -> dict:
        """在worker中独立运行 num_episodes 个episode并返回统计数据。"""
        _configure_torch_eval_runtime()

        if network_type is None and defender_checkpoint and os.path.exists(defender_checkpoint):
            _, network_type, _ = _load_checkpoint(defender_checkpoint)
        network_type = _apply_network_type_constraints(defender_strategy, network_type)

        old_eval_use_random_seed = SetupParameters.EVAL_USE_RANDOM_SEED
        old_eval_fixed_seed = SetupParameters.EVAL_FIXED_SEED
        SetupParameters.EVAL_USE_RANDOM_SEED = bool(eval_use_random_seed)
        SetupParameters.EVAL_FIXED_SEED = int(eval_fixed_seed)
        try:
            return _run_serial_evaluation(
                defender_strategy=defender_strategy,
                attacker_strategy=attacker_strategy,
                attacker_strategy_params=attacker_strategy_params,
                num_episodes=num_episodes,
                defender_checkpoint=defender_checkpoint,
                attacker_checkpoint=attacker_checkpoint,
                device='cpu',
                network_type=network_type,
                save_gif=False,
                gif_episodes=0,
                seed_offset=seed_offset,
                collect_trajectory_episodes=0,
                hrl_num_skills=hrl_num_skills,
                hrl_protect_skill_path=hrl_protect_skill_path,
                hrl_chase_skill_path=hrl_chase_skill_path,
                use_privileged_classifier=use_privileged_classifier,
            )
        finally:
            SetupParameters.EVAL_USE_RANDOM_SEED = old_eval_use_random_seed
            SetupParameters.EVAL_FIXED_SEED = old_eval_fixed_seed


def _merge_stats(all_stats: list) -> dict:
    """合并多个worker返回的stats字典。"""
    merged = {}
    for key in all_stats[0]:
        merged[key] = []
        for s in all_stats:
            merged[key].extend(s[key])
    return merged


def _numeric_values(values: Optional[List[object]]) -> List[float]:
    out: List[float] = []
    for value in (values or []):
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            out.append(numeric)
    return out


def _mean_std(values: Optional[List[object]]) -> Tuple[Optional[float], Optional[float]]:
    numeric_values = _numeric_values(values)
    if not numeric_values:
        return None, None
    arr = np.asarray(numeric_values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))


def _mean_ci95(values: Optional[List[object]]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    numeric_values = _numeric_values(values)
    if not numeric_values:
        return None, None, None
    arr = np.asarray(numeric_values, dtype=np.float64)
    mean_value = float(np.mean(arr))
    n = int(arr.size)
    if n <= 1:
        return mean_value, mean_value, mean_value
    std_value = float(np.std(arr))
    half_width = 1.959963984540054 * std_value / np.sqrt(float(n))
    return mean_value, float(mean_value - half_width), float(mean_value + half_width)


def _mean_only(values: Optional[List[object]]) -> Optional[float]:
    mean_value, _ = _mean_std(values)
    return mean_value


def _proportion_ci95(values: Optional[List[object]]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    numeric_values = _numeric_values(values)
    if not numeric_values:
        return None, None, None

    arr = np.asarray(numeric_values, dtype=np.float64)
    mean_value = float(np.mean(arr))
    n = int(arr.size)
    if n <= 0:
        return None, None, None

    # 对 0/1 比例指标使用 Wilson 95% CI，避免出现负数或超过 1 的边界问题。
    if np.all((arr >= 0.0) & (arr <= 1.0)):
        z = 1.959963984540054
        z2 = z * z
        denom = 1.0 + z2 / n
        center = (mean_value + z2 / (2.0 * n)) / denom
        margin = (z * np.sqrt((mean_value * (1.0 - mean_value) / n) + (z2 / (4.0 * n * n)))) / denom
        return mean_value, float(max(0.0, center - margin)), float(min(1.0, center + margin))

    return _mean_ci95(numeric_values)


def _append_episode_outcome_stats(
    stats: Dict[str, List[int]],
    defender_strategy: str,
    defender_captured: bool,
    attacker_captured: bool,
    timeout: bool,
) -> None:
    """
    统一记录每个 episode 的胜负/平局统计。

    语义约定：
    - defender_wins: 防御者任务成功率
    - attacker_wins: 攻击者真实胜率（仅 attacker 真正达成终局时为 1）
    - draws: 双方都未达成真实胜利条件的回合

    注意：chase 任务里「未抓到 attacker」只是 defender 任务失败，
    不能把它误记成 attacker 胜利；否则 static attacker 也会出现虚假的 A 胜率。
    """
    if defender_strategy == 'chase':
        if defender_captured:
            stats['defender_wins'].append(1)
            stats['attacker_wins'].append(0)
            stats['draws'].append(0)
        elif attacker_captured:
            stats['defender_wins'].append(0)
            stats['attacker_wins'].append(1)
            stats['draws'].append(0)
        else:
            stats['defender_wins'].append(0)
            stats['attacker_wins'].append(0)
            stats['draws'].append(1)
        return

    if defender_captured or timeout:
        stats['defender_wins'].append(1)
        stats['attacker_wins'].append(0)
        stats['draws'].append(0)
    elif attacker_captured:
        stats['defender_wins'].append(0)
        stats['attacker_wins'].append(1)
        stats['draws'].append(0)
    else:
        stats['defender_wins'].append(0)
        stats['attacker_wins'].append(0)
        stats['draws'].append(1)


def _polyline_length(points: Optional[List[Tuple[float, float]]]) -> float:
    if not points or len(points) < 2:
        return 0.0
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return 0.0
    deltas = arr[1:] - arr[:-1]
    return float(np.linalg.norm(deltas, axis=1).sum())


def _extract_defender_path_length(env) -> Optional[float]:
    base_env = env.env if hasattr(env, 'env') else env
    start_pos = getattr(base_env, 'defender_start_pos', None)
    traj = getattr(base_env, 'defender_trajectory', None)

    path_points: List[Tuple[float, float]] = []
    if isinstance(start_pos, (list, tuple)) and len(start_pos) >= 2:
        path_points.append((float(start_pos[0]), float(start_pos[1])))

    if isinstance(traj, list):
        for point in traj:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                path_points.append((float(point[0]), float(point[1])))

    if not path_points:
        return None
    return _polyline_length(path_points)


def _fmt_metric_pm(
    mean_value: Optional[float],
    std_value: Optional[float],
    *,
    scale: float = 1.0,
    decimals: int = 1,
    suffix: str = '',
) -> str:
    if mean_value is None:
        return '-'
    std_val = 0.0 if std_value is None else float(std_value)
    mean_fmt = f"{float(mean_value) * scale:.{decimals}f}"
    std_fmt = f"{std_val * scale:.{decimals}f}"
    return f"{mean_fmt}±{std_fmt}{suffix}"


def _fmt_metric_value(
    value: Optional[float],
    *,
    scale: float = 1.0,
    decimals: int = 1,
    suffix: str = '',
) -> str:
    if value is None:
        return '-'
    return f"{float(value) * scale:.{decimals}f}{suffix}"


def _fmt_metric_ci(
    mean_value: Optional[float],
    ci_low: Optional[float],
    ci_high: Optional[float],
    *,
    scale: float = 1.0,
    decimals: int = 1,
    suffix: str = '',
) -> str:
    if mean_value is None:
        return '-'
    mean_fmt = _fmt_metric_value(mean_value, scale=scale, decimals=decimals, suffix=suffix)
    if ci_low is None or ci_high is None:
        return mean_fmt
    low_fmt = _fmt_metric_value(ci_low, scale=scale, decimals=decimals, suffix=suffix)
    high_fmt = _fmt_metric_value(ci_high, scale=scale, decimals=decimals, suffix=suffix)
    return f"{mean_fmt} [{low_fmt}, {high_fmt}]"


def _fmt_metric_ci_pm(
    mean_value: Optional[float],
    ci_low: Optional[float],
    ci_high: Optional[float],
    *,
    scale: float = 1.0,
    decimals: int = 1,
    suffix: str = '',
) -> str:
    if mean_value is None:
        return '-'
    mean_fmt = _fmt_metric_value(mean_value, scale=scale, decimals=decimals, suffix=suffix)
    if ci_low is None or ci_high is None:
        return mean_fmt
    half_width = max(abs(float(mean_value) - float(ci_low)), abs(float(ci_high) - float(mean_value)))
    pm_fmt = _fmt_metric_value(half_width, scale=scale, decimals=decimals, suffix=suffix)
    return f"{mean_fmt}±{pm_fmt}"


def _fmt_metric_ci_pm_from_row(
    raw_row: Dict[str, object],
    metric_key: str,
    *,
    scale: float = 1.0,
    decimals: int = 1,
    suffix: str = '',
) -> str:
    return _fmt_metric_ci_pm(
        raw_row.get(metric_key),
        raw_row.get(f'{metric_key}_ci_low'),
        raw_row.get(f'{metric_key}_ci_high'),
        scale=scale,
        decimals=decimals,
        suffix=suffix,
    )


def _fmt_metric_pm_from_row(
    raw_row: Dict[str, object],
    metric_key: str,
    *,
    scale: float = 1.0,
    decimals: int = 1,
    suffix: str = '',
) -> str:
    return _fmt_metric_pm(
        raw_row.get(metric_key),
        raw_row.get(f'{metric_key}_std'),
        scale=scale,
        decimals=decimals,
        suffix=suffix,
    )


def _build_average_row(
    rows: List[Dict[str, object]],
    *,
    defender_label: str = 'AVERAGE',
    attacker_label: str = 'ALL',
) -> Optional[Dict[str, object]]:
    if not rows:
        return None

    avg_row: Dict[str, object] = {}
    ordered_keys: List[str] = []
    seen_keys = set()
    for row in rows:
        for key in row.keys():
            if key not in seen_keys:
                seen_keys.add(key)
                ordered_keys.append(key)

    for key in ordered_keys:
        if key == 'defender':
            avg_row[key] = defender_label
        elif key == 'attacker':
            avg_row[key] = attacker_label
        else:
            avg_row[key] = _mean_only([row.get(key) for row in rows])
    return avg_row


SAVED_METRIC_EXCLUDE_KEYS = {
    'attacker_win_rate',
    'attacker_win_rate_ci_low',
    'attacker_win_rate_ci_high',
    'draw_rate',
    'draw_rate_ci_low',
    'draw_rate_ci_high',
    'attacker_capture_rate',
    'attacker_capture_rate_ci_low',
    'attacker_capture_rate_ci_high',
    'defender_path_length_mean',
    'defender_path_length_std',
}

HRL_SELECTION_RATE_PREFIXES = (
    'hrl_protect_selection_rate',
    'hrl_chase_selection_rate',
)

FORMATTED_SUMMARY_FIELD_ORDER = [
    'defender',
    'attacker',
    'episodes',
    'success_rate',
    'defender_capture_rate',
    'defender_collision_rate',
    'mean_episode_length',
    'hrl_protect_selection_rate',
    'hrl_chase_selection_rate',
]

RAW_MATCHUP_FIELD_ORDER = [
    'defender',
    'attacker',
    'episodes',
    'success_rate',
    'defender_capture_rate',
    'defender_collision_rate',
    'mean_episode_length',
    'hrl_protect_selection_rate',
    'hrl_chase_selection_rate',
]


def _metric_ci_triplet_keys(metric_key: str) -> List[str]:
    return [metric_key, f'{metric_key}_ci_low', f'{metric_key}_ci_high']


def _active_hrl_selection_prefixes(
    defender_strategy: Optional[str],
    *,
    hrl_num_skills: Optional[int] = None,
    row: Optional[Dict[str, object]] = None,
) -> List[str]:
    if not _is_hrl_like_strategy(str(defender_strategy).strip().lower() if defender_strategy is not None else ''):
        return []

    try:
        num_skills = int(hrl_num_skills) if hrl_num_skills is not None else None
    except (TypeError, ValueError):
        num_skills = None

    if num_skills == 2:
        return list(HRL_SELECTION_RATE_PREFIXES)

    active = []
    for prefix in HRL_SELECTION_RATE_PREFIXES:
        keys = _metric_ci_triplet_keys(prefix)
        if row is not None and any(row.get(key) is not None for key in keys):
            active.append(prefix)
    return active


def _prune_saved_metrics_dict(
    row: Dict[str, object],
    *,
    defender_strategy: Optional[str] = None,
    hrl_num_skills: Optional[int] = None,
) -> Dict[str, object]:
    pruned = dict(row)
    for key in SAVED_METRIC_EXCLUDE_KEYS:
        pruned.pop(key, None)

    active_prefixes = set(
        _active_hrl_selection_prefixes(
            defender_strategy if defender_strategy is not None else pruned.get('defender'),
            hrl_num_skills=hrl_num_skills,
            row=pruned,
        )
    )
    for prefix in HRL_SELECTION_RATE_PREFIXES:
        if prefix in active_prefixes:
            continue
        for key in _metric_ci_triplet_keys(prefix):
            pruned.pop(key, None)
    return pruned


def _format_saved_metrics_dict(row: Dict[str, object]) -> Dict[str, object]:
    formatted: Dict[str, object] = {}
    for key, value in row.items():
        if key.endswith('_ci_low') or key.endswith('_ci_high'):
            continue
        if key.endswith('_std'):
            continue
        ci_low_key = f'{key}_ci_low'
        ci_high_key = f'{key}_ci_high'
        std_key = f'{key}_std'
        if std_key in row:
            if key.endswith('_rate'):
                formatted[key] = _fmt_metric_pm_from_row(
                    row, key, scale=100.0, decimals=1, suffix='%'
                )
            else:
                formatted[key] = _fmt_metric_pm_from_row(row, key, decimals=1)
        elif ci_low_key in row or ci_high_key in row:
            if key.endswith('_rate'):
                formatted[key] = _fmt_metric_ci_pm_from_row(
                    row, key, scale=100.0, decimals=1, suffix='%'
                )
            elif key == 'mean_episode_length':
                formatted[key] = _fmt_metric_ci_pm_from_row(row, key, decimals=1)
            else:
                formatted[key] = _fmt_metric_ci_pm_from_row(row, key, decimals=3)
        else:
            formatted[key] = value
    return formatted


def _ordered_export_fieldnames(rows: List[Dict[str, object]], preferred_order: List[str]) -> List[str]:
    present = set()
    for row in rows:
        present.update(row.keys())

    ordered = [key for key in preferred_order if key in present]
    seen = set(ordered)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                ordered.append(key)
                seen.add(key)
    return ordered


def _append_unique_path(path_map: Dict[str, List[str]], key: str, path: Optional[str]) -> None:
    if not path:
        return
    bucket = path_map.setdefault(str(key), [])
    if path not in bucket:
        bucket.append(path)


def _format_suite_summary_row(raw_row: Dict[str, object]) -> Dict[str, object]:
    row = {
        'defender': raw_row['defender'],
        'attacker': raw_row['attacker'],
        'episodes': raw_row['episodes'],
        'success_rate': _fmt_metric_ci_pm_from_row(
            raw_row, 'success_rate', scale=100.0, decimals=1, suffix='%'
        ),
        'defender_capture_rate': _fmt_metric_ci_pm_from_row(
            raw_row, 'defender_capture_rate', scale=100.0, decimals=1, suffix='%'
        ),
        'defender_collision_rate': _fmt_metric_ci_pm_from_row(
            raw_row, 'defender_collision_rate', scale=100.0, decimals=1, suffix='%'
        ),
        'mean_episode_length': _fmt_metric_pm_from_row(
            raw_row, 'mean_episode_length', decimals=1
        ),
    }
    for prefix in HRL_SELECTION_RATE_PREFIXES:
        if prefix in raw_row:
            row[prefix] = _fmt_metric_pm_from_row(
                raw_row, prefix, scale=100.0, decimals=1, suffix='%'
            )
    return row


def _format_suite_average_row(raw_row: Dict[str, object]) -> Dict[str, object]:
    row = {
        'defender': raw_row['defender'],
        'attacker': raw_row['attacker'],
        'episodes': _fmt_metric_value(raw_row['episodes'], decimals=1),
        'success_rate': _fmt_metric_value(raw_row['success_rate'], scale=100.0, decimals=1, suffix='%'),
        'defender_capture_rate': _fmt_metric_value(raw_row['defender_capture_rate'], scale=100.0, decimals=1, suffix='%'),
        'defender_collision_rate': _fmt_metric_value(raw_row['defender_collision_rate'], scale=100.0, decimals=1, suffix='%'),
        'mean_episode_length': _fmt_metric_value(raw_row['mean_episode_length'], decimals=1),
    }
    for prefix in HRL_SELECTION_RATE_PREFIXES:
        if prefix in raw_row:
            row[prefix] = _fmt_metric_value(
                raw_row.get(prefix), scale=100.0, decimals=1, suffix='%'
            )
    return row


def _safe_detect_checkpoint_network_type(checkpoint_path: Optional[str]) -> Optional[str]:
    if not checkpoint_path:
        return None
    state_dict, network_type, _ = _load_checkpoint(checkpoint_path)
    if state_dict is None:
        return None
    return str(network_type)


def _load_checkpoint(checkpoint_path: str):
    """加载checkpoint并返回 (state_dict, network_type, arch_info)。"""
    try:
        # numpy版本兼容: 旧numpy(< 2.0)没有_core子包
        import numpy as _np
        if not hasattr(_np, '_core'):
            import sys as _sys
            _sys.modules['numpy._core'] = _np.core
            _sys.modules['numpy._core.multiarray'] = _np.core.multiarray

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        explicit_network_type = None
        if isinstance(checkpoint, dict) and checkpoint.get('network_type'):
            explicit_network_type = str(checkpoint['network_type']).strip().lower()
        if explicit_network_type:
            net_type = explicit_network_type
        else:
            keys = set(state_dict.keys())
            has_tracking = any('tracking_branch' in k for k in keys)
            has_actor_backbone = any('actor_backbone' in k for k in keys)
            has_actor_gru = any('actor_gru' in k for k in keys)
            has_hrl_top_marker = ('hrl_top_marker' in keys) or ('discrete_policy_marker' in keys)

            if 'belief_predictor.0.weight' in keys:
                net_type = 'nmn_ctde_task_shared_distill'
            elif 'shared_tracking_branch.0.weight' in keys:
                net_type = 'nmn_ctde_task_shared'
            elif 'shared_radar_encoder.net.0.weight' in keys:
                net_type = 'nmn_ctde_shared'
            elif 'actor_radar_encoder.net.0.weight' in keys and 'critic_radar_encoder.net.0.weight' in keys:
                net_type = 'nmn_no_shared_radar'
            elif 'actor_tracking_gru.weight_ih_l0' in keys and 'actor_obstacle_gru.weight_ih_l0' in keys:
                net_type = 'nmn_dual_gru_raw'
            elif has_tracking and has_actor_gru:
                net_type = 'nmn_gru'
            elif has_actor_gru:
                action_dim = None
                if 'log_std' in state_dict and hasattr(state_dict['log_std'], 'shape'):
                    action_dim = int(state_dict['log_std'].shape[0])
                elif 'policy_mean.weight' in state_dict and hasattr(state_dict['policy_mean.weight'], 'shape'):
                    action_dim = int(state_dict['policy_mean.weight'].shape[0])
                if has_hrl_top_marker or (action_dim is not None and action_dim >= 3):
                    net_type = 'hrl_top_gru'
                else:
                    net_type = 'mlp_gru'
            elif has_tracking:
                critic_input_dim = None
                if 'critic_backbone.0.weight' in state_dict and hasattr(state_dict['critic_backbone.0.weight'], 'shape'):
                    critic_input_dim = int(state_dict['critic_backbone.0.weight'].shape[1])
                net_type = 'nmn_ctde' if critic_input_dim == NetParameters.CRITIC_VECTOR_LEN else 'nmn'
            elif has_actor_backbone:
                # 区分普通MLP(2D action)与HRL顶层MLP(>=3D action)。
                action_dim = None
                critic_input_dim = None
                if 'log_std' in state_dict and hasattr(state_dict['log_std'], 'shape'):
                    action_dim = int(state_dict['log_std'].shape[0])
                elif 'policy_mean.weight' in state_dict and hasattr(state_dict['policy_mean.weight'], 'shape'):
                    action_dim = int(state_dict['policy_mean.weight'].shape[0])
                if 'critic_backbone.0.weight' in state_dict and hasattr(state_dict['critic_backbone.0.weight'], 'shape'):
                    critic_input_dim = int(state_dict['critic_backbone.0.weight'].shape[1])

                if has_hrl_top_marker or (action_dim is not None and action_dim >= 3):
                    net_type = 'hrl_top_noctde' if critic_input_dim == NetParameters.ACTOR_VECTOR_LEN else 'hrl_top'
                else:
                    net_type = 'mlp_noctde' if critic_input_dim == NetParameters.ACTOR_VECTOR_LEN else 'mlp_ctde'
            else:
                net_type = 'nmn'

        arch_info = {
            'hidden_dim': None,
            'action_dim': None,
            'hrl_num_skills': None,
            'hrl_duration_bins': None,
            'hrl_top_discrete_action_dim': None,
        }
        if 'actor_backbone.0.weight' in state_dict and hasattr(state_dict['actor_backbone.0.weight'], 'shape'):
            arch_info['hidden_dim'] = int(state_dict['actor_backbone.0.weight'].shape[0])
        elif 'actor_in_proj.weight' in state_dict and hasattr(state_dict['actor_in_proj.weight'], 'shape'):
            arch_info['hidden_dim'] = int(state_dict['actor_in_proj.weight'].shape[0])
        if 'log_std' in state_dict and hasattr(state_dict['log_std'], 'shape'):
            arch_info['action_dim'] = int(state_dict['log_std'].shape[0])
        elif 'policy_mean.weight' in state_dict and hasattr(state_dict['policy_mean.weight'], 'shape'):
            arch_info['action_dim'] = int(state_dict['policy_mean.weight'].shape[0])
        if isinstance(checkpoint, dict):
            if checkpoint.get('hrl_num_skills') is not None:
                arch_info['hrl_num_skills'] = int(checkpoint['hrl_num_skills'])
            if checkpoint.get('hrl_duration_bins') is not None:
                arch_info['hrl_duration_bins'] = tuple(int(v) for v in checkpoint['hrl_duration_bins'])
            if checkpoint.get('hrl_top_discrete_action_dim') is not None:
                arch_info['hrl_top_discrete_action_dim'] = int(checkpoint['hrl_top_discrete_action_dim'])

        return state_dict, net_type, arch_info
    except Exception as e:
        print(f"[警告] 无法加载checkpoint: {e}")
        return None, 'nmn', {
            'hidden_dim': None,
            'action_dim': None,
            'hrl_num_skills': None,
            'hrl_duration_bins': None,
            'hrl_top_discrete_action_dim': None,
        }


class Defenderevaluator:
    """Defender Strategy Evaluator Wrapper"""

    def __init__(
        self,
        strategy: str,
        checkpoint_path: Optional[str] = None,
        device: str = 'cpu',
        network_type: Optional[str] = None,
        use_privileged_classifier: bool = False,
    ):
        self.strategy = strategy
        self.use_privileged_classifier = bool(use_privileged_classifier)
        self._privileged_actor_hidden = None
        self._privileged_critic_hidden = None
        # 使用安全的GPU检测
        self.device = get_device(prefer_gpu=(device == 'cuda'))

        # 自动解析checkpoint
        if strategy in RL_DEFENDER_STRATEGIES and checkpoint_path is None:
            auto_path = _default_model_path(strategy)
            if auto_path is not None:
                checkpoint_path = auto_path
                print(f"[Defender] 自动加载模型: {checkpoint_path}")
            else:
                print(f"[Defender] 警告: 未找到 {strategy} 的默认模型")

        self.model = None

        if strategy in RL_DEFENDER_STRATEGIES:
            if checkpoint_path and os.path.exists(checkpoint_path):
                # 单次加载: 同时检测网络类型和获取权重
                state_dict, detected_type, arch_info = _load_checkpoint(checkpoint_path)
                if network_type is None:
                    network_type = detected_type
                network_type = _apply_network_type_constraints(strategy, network_type)
                self.network_type = network_type

                # 若checkpoint与当前全局网络超参不一致，临时按checkpoint规格构建网络。
                old_hidden = NetParameters.HIDDEN_DIM
                old_action_dim = NetParameters.ACTION_DIM
                old_hrl_top_action_dim = int(getattr(NetParameters, 'HRL_TOP_ACTION_DIM', 3))
                old_hrl_num_skills = int(getattr(NetParameters, 'HRL_NUM_SKILLS', 2))
                old_hrl_duration_bins = tuple(getattr(NetParameters, 'HRL_DURATION_BINS', (1,)))
                old_hrl_num_duration_bins = int(getattr(NetParameters, 'HRL_NUM_DURATION_BINS', len(old_hrl_duration_bins)))
                old_hrl_top_discrete_action_dim = int(
                    getattr(NetParameters, 'HRL_TOP_DISCRETE_ACTION_DIM', old_hrl_num_skills)
                )
                if arch_info.get('hidden_dim') is not None:
                    NetParameters.HIDDEN_DIM = int(arch_info['hidden_dim'])
                if network_type in ['hrl_top', 'hrl_top_noctde', 'hrl_top_gru', 'hrl_top_dual_gru_raw']:
                    if arch_info.get('action_dim') is not None:
                        top_action_dim = max(3, int(arch_info['action_dim']))
                        NetParameters.HRL_TOP_ACTION_DIM = top_action_dim
                    hrl_num_skills = arch_info.get('hrl_num_skills')
                    hrl_duration_bins = arch_info.get('hrl_duration_bins')
                    hrl_top_discrete_action_dim = arch_info.get('hrl_top_discrete_action_dim')
                    if hrl_num_skills is not None:
                        NetParameters.HRL_NUM_SKILLS = max(2, int(hrl_num_skills))
                    elif arch_info.get('action_dim') is not None:
                        NetParameters.HRL_NUM_SKILLS = max(2, int(top_action_dim - 1))
                    if hrl_duration_bins is not None:
                        NetParameters.HRL_DURATION_BINS = tuple(int(v) for v in hrl_duration_bins)
                        NetParameters.HRL_NUM_DURATION_BINS = len(NetParameters.HRL_DURATION_BINS)
                    else:
                        NetParameters.HRL_DURATION_BINS = (1,)
                        NetParameters.HRL_NUM_DURATION_BINS = 1
                    if hrl_top_discrete_action_dim is not None:
                        NetParameters.HRL_TOP_DISCRETE_ACTION_DIM = int(hrl_top_discrete_action_dim)
                    else:
                        NetParameters.HRL_TOP_DISCRETE_ACTION_DIM = int(
                            NetParameters.HRL_NUM_SKILLS * NetParameters.HRL_NUM_DURATION_BINS
                        )
                    NetParameters.ACTION_DIM = 2
                elif arch_info.get('action_dim') is not None:
                    NetParameters.ACTION_DIM = int(arch_info['action_dim'])

                self.model = Model(self.device, global_model=False, network_type=network_type)
                try:
                    if state_dict is not None:
                        self.model.set_weights(state_dict)
                    self.model.network.eval()
                    print(f"[Defender] 已加载RL模型 (网络={_display_network_type(network_type)}): {checkpoint_path}")
                    if self.use_privileged_classifier:
                        privileged_ok = (
                            _is_hrl_like_strategy(strategy)
                            and hasattr(self.model.network, 'classify_skills_privileged')
                            and (
                                hasattr(self.model.network, 'forward_privileged_recurrent')
                                or hasattr(self.model.network, 'forward_recurrent_with_features')
                            )
                        )
                        if not privileged_ok:
                            raise ValueError('use_privileged_classifier 仅支持带 classify_skills_privileged 的 HRL recurrent top 网络。')
                        print('[Defender] 当前使用 privileged classifier head 决策，仅走 privileged/critic 路径，不使用 actor action head')
                except Exception as e:
                    print(f"加载模型错误: {e}")
                    raise e
                finally:
                    NetParameters.HIDDEN_DIM = old_hidden
                    NetParameters.ACTION_DIM = old_action_dim
                    NetParameters.HRL_TOP_ACTION_DIM = old_hrl_top_action_dim
                    NetParameters.HRL_NUM_SKILLS = old_hrl_num_skills
                    NetParameters.HRL_DURATION_BINS = old_hrl_duration_bins
                    NetParameters.HRL_NUM_DURATION_BINS = old_hrl_num_duration_bins
                    NetParameters.HRL_TOP_DISCRETE_ACTION_DIM = old_hrl_top_discrete_action_dim
            else:
                raise ValueError(f"策略 {strategy} 需要有效的checkpoint")

        elif strategy == 'hrl_rule_geo_trend':
            self.model = DefenderHRLRulePolicy()
            print(f"[Defender] 使用规则上层HRL策略: {strategy}")

        elif strategy == 'hrl_rule_apollonius_label':
            self.model = DefenderHRLApolloniusLabelPolicy()
            print(f"[Defender] 使用阿波罗尼斯规则上层HRL策略: {strategy}")

        elif strategy == 'astar_to_attacker':
            # A*导航到攻击者
            self.model = DefenderGlobalPolicy(skill_mode='chase')
            print(f"[Defender] 使用A*导航策略(追击攻击者)")
            
        elif strategy == 'astar_to_target':
            # A*导航到目标
            self.model = DefenderGlobalPolicy(skill_mode='protect')
            print(f"[Defender] 使用A*导航策略(守护目标)")

        elif strategy in REACH_AVOID_RULE_DEFENDER_STRATEGIES:
            self.model = create_reach_avoid_defender_policy(strategy)
            print(f"[Defender] 使用Reach-Avoid规则基线: {strategy}")
            
        else:
            raise ValueError(f"未知的defender策略: {strategy}")

    def reset(self, env: Optional[object] = None):
        """Reset evaluator state"""
        self._privileged_actor_hidden = None
        self._privileged_critic_hidden = None
        if hasattr(self.model, 'reset'):
            self.model.reset()
        if hasattr(self.model, 'reset_gru_sequence'):
            self.model.reset_gru_sequence()

    def _get_privileged_classifier_action(self, obs: np.ndarray, critic_obs: np.ndarray) -> np.ndarray:
        critic_tensor = torch.as_tensor(critic_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if hasattr(self.model.network, 'forward_privileged_recurrent'):
                skill_logits, _value, _critic_feat, next_critic_hidden = self.model.network.forward_privileged_recurrent(
                    critic_tensor,
                    critic_hidden=self._privileged_critic_hidden,
                )
                next_actor_hidden = None
            else:
                actor_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                (
                    _mean,
                    _value,
                    _log_std,
                    _actor_feat,
                    critic_feat,
                    next_actor_hidden,
                    next_critic_hidden,
                ) = self.model.network.forward_recurrent_with_features(
                    actor_tensor,
                    critic_tensor,
                    actor_hidden=self._privileged_actor_hidden,
                    critic_hidden=self._privileged_critic_hidden,
                )
                skill_logits = self.model.network.classify_skills_privileged(critic_feat)
            skill_idx = int(torch.argmax(skill_logits, dim=-1).item())
        self._privileged_actor_hidden = next_actor_hidden.detach() if next_actor_hidden is not None else None
        self._privileged_critic_hidden = next_critic_hidden.detach() if next_critic_hidden is not None else None
        return np.asarray([skill_idx], dtype=np.float32)

    @staticmethod
    def _apply_defender_hard_obstacle_mask(action: np.ndarray, env: object) -> np.ndarray:
        """Apply defender-only hard obstacle mask for rule-based baselines."""
        if action is None:
            return action
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size != 2:
            return action

        base_env = env.env if hasattr(env, 'env') else env
        required = ['_control_to_physical', '_simulate_motion', '_encode_action_like_input', '_get_action_limits']
        if not all(hasattr(base_env, name) for name in required) or not hasattr(base_env, 'defender'):
            return action

        normalized_input = bool(np.all(np.abs(arr) <= 1.0 + 1e-6))
        physical = base_env._control_to_physical(arr, role='defender')
        if physical is None:
            return action

        orig_angle, orig_speed = float(physical[0]), float(physical[1])
        max_turn, max_speed, _ = base_env._get_action_limits('defender')
        ref_agent = base_env.defender
        agent_radius = float(getattr(map_config, 'agent_radius', getattr(base_env, 'pixel_size', 4.0) * 0.5))

        speed_scales = (1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1, 0.0)
        angle_scales = (0.0, -0.2, 0.2, -0.4, 0.4, -0.6, 0.6, -0.8, 0.8, -1.0, 1.0)
        best = None

        for s in speed_scales:
            cand_speed = float(np.clip(orig_speed * s, 0.0, max_speed))
            for a in angle_scales:
                cand_angle = float(np.clip(orig_angle + a * max_turn, -max_turn, max_turn))
                nx, ny = base_env._simulate_motion(ref_agent, cand_angle, cand_speed, role='defender')
                px = float(getattr(base_env, 'pixel_size', 4.0))
                cx = nx + px * 0.5
                cy = ny + px * 0.5
                if env_lib.is_point_blocked(cx, cy, padding=agent_radius):
                    continue

                angle_cost = abs(cand_angle - orig_angle) / (max_turn + 1e-6)
                speed_cost = abs(cand_speed - orig_speed) / (max_speed + 1e-6)
                cost = angle_cost + 0.35 * speed_cost
                if best is None or cost < best[0]:
                    best = (cost, cand_angle, cand_speed)

        if best is None:
            return base_env._encode_action_like_input(0.0, 0.0, 'defender', normalized_input)
        return base_env._encode_action_like_input(best[1], best[2], 'defender', normalized_input)

    def get_action(self, obs: np.ndarray, env: object, attacker_obs: np.ndarray = None) -> np.ndarray:
        """Get action from policy"""
        # RL策略（包括各种技能模型）
        if self.strategy in RL_DEFENDER_STRATEGIES:
            # PPO Model Evaluation
            # 兼容不同网络结构（MLP/NMN）
            if hasattr(self.model, 'update_gru_sequence'):
                try:
                    if hasattr(env, 'env') and hasattr(env.env, 'get_normalized_attacker_info'):
                        rel_x_norm, rel_y_norm, is_visible = env.env.get_normalized_attacker_info()
                        self.model.update_gru_sequence(rel_x_norm, rel_y_norm, is_visible)
                    elif hasattr(env, 'get_normalized_attacker_info'):
                        rel_x_norm, rel_y_norm, is_visible = env.get_normalized_attacker_info()
                        self.model.update_gru_sequence(rel_x_norm, rel_y_norm, is_visible)
                except (AttributeError, TypeError):
                    pass

            # 构建critic观测 - 兼容新的build_critic_observation签名
            try:
                if attacker_obs is not None:
                    # 使用完整的defender和attacker观测构建critic_obs
                    critic_obs = build_critic_observation(obs, attacker_obs)
                else:
                    # 回退到旧方式
                    critic_obs = build_critic_observation(obs)
            except Exception:
                # 如果构建失败，直接使用obs
                critic_obs = obs
            
            if self.use_privileged_classifier:
                return self._get_privileged_classifier_action(obs, critic_obs)
            with torch.no_grad():
                action, _, _, _ = self.model.evaluate(obs, critic_obs, greedy=True)
            return action

        elif self.strategy == 'hrl_rule_geo_trend':
            base_env = env.env if hasattr(env, 'env') else env
            if not hasattr(base_env, 'get_privileged_state'):
                raise ValueError(f"{self.strategy} 需要环境提供 privileged state")
            privileged_state = base_env.get_privileged_state()
            skill_names = getattr(env, 'skill_names', None)
            return self.model.get_action(obs, privileged_state, skill_names=skill_names)

        elif self.strategy == 'hrl_rule_apollonius_label':
            return self.model.get_action(obs, env, attacker_obs=attacker_obs)

        elif self.strategy in ALL_RULE_DEFENDER_STRATEGIES:
            # 规则策略允许读取 privileged state，但不允许策略层 hard-mask。
            if hasattr(env, 'get_privileged_state'):
                p_state = env.get_privileged_state()
            elif hasattr(env, 'env') and hasattr(env.env, 'get_privileged_state'):
                p_state = env.env.get_privileged_state()
            else:
                p_state = None
            
            if p_state:
                action = self.model.get_action(obs, p_state)
                return action
            else:
                return np.zeros(2)

        return np.zeros(2)


class Attackerevaluator:
    """Attacker Strategy Evaluator"""

    # 支持的策略列表
    VALID_STRATEGIES = [
        'default', 'evasive', 'attacker_apf', 'attacker_global',
        'attacker_rl', 'static', 'random',
    ]

    def __init__(
        self,
        strategy: str,
        env_width: float = None,
        env_height: float = None,
        attacker_speed: float = None,
        attacker_max_turn: float = None,
        strategy_params: Optional[Dict] = None,
        checkpoint_path: Optional[str] = None,
        device: str = 'cpu',
    ):
        self.strategy = strategy
        self.strategy_params = strategy_params or {}
        self.device = get_device(prefer_gpu=(device == 'cuda'))
        
        # 使用传入参数或默认值
        env_width = env_width if env_width is not None else map_config.width
        env_height = env_height if env_height is not None else map_config.height
        attacker_speed = attacker_speed if attacker_speed is not None else map_config.attacker_speed
        attacker_max_turn = attacker_max_turn if attacker_max_turn is not None else getattr(map_config, 'attacker_max_angular_speed', 12.0)
        
        if strategy == 'attacker_global':
            self.model = AttackerGlobalPolicy(
                env_width=env_width,
                env_height=env_height,
                attacker_speed=attacker_speed,
                attacker_max_turn=attacker_max_turn,
            )
        elif strategy in ['default', 'evasive']:
            # 核心策略 + 周期切换策略
            self.model = AttackerGlobalPolicy(
                env_width=env_width,
                env_height=env_height,
                attacker_speed=attacker_speed,
                attacker_max_turn=attacker_max_turn,
                strategy=strategy,
                strategy_params=self.strategy_params,
            )
        elif strategy == 'attacker_rl':
            if checkpoint_path is None or not os.path.exists(checkpoint_path):
                raise ValueError(f"attacker_rl requires checkpoint_path, got {checkpoint_path!r}")
            self.model = LearnedAttackerPolicy(
                checkpoint_path,
                device=self.device,
                alias='attacker_rl',
                reward_style=self.strategy_params.get('reward_style'),
            )
        elif strategy == 'static':
            self.model = None 
        elif strategy == 'random':
            self.model = None
        else:
            raise ValueError(f"Unknown attacker strategy: {strategy}. Valid strategies: {self.VALID_STRATEGIES}")

    def reset(self):
        if hasattr(self.model, 'reset'):
            self.model.reset()
        if hasattr(self.model, 'reset_recurrent_state'):
            self.model.reset_recurrent_state()
        if hasattr(self.model, 'reset_gru_sequence'):
            self.model.reset_gru_sequence()

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if self.strategy == 'static':
            return np.array([-1.0, -1.0], dtype=np.float32)
        if self.strategy == 'random':
            return np.zeros(2, dtype=np.float32)
        if self.strategy == 'attacker_rl':
            return np.asarray(self.model.get_action(obs), dtype=np.float32)
        return self.model.get_action(obs)


# --- Core Evaluation Function ---
def run_evaluation(
    defender_strategy: str,
    attacker_strategy: str,
    attacker_strategy_params: Optional[Dict] = None,
    attacker_display_name: Optional[str] = None,
    num_episodes: int = 100,
    defender_checkpoint: Optional[str] = None,
    attacker_checkpoint: Optional[str] = None,
    device: str = 'cpu',
    save_gif: bool = False,
    gif_path: Optional[str] = None,
    gif_episodes: int = 1,
    save_stats: bool = False,
    stats_path: Optional[str] = None,
    seed_offset: int = 0,
    network_type: Optional[str] = None,
    force_serial: bool = False,
    save_traj_png: bool = False,
    traj_png_count: int = 1,
    traj_png_path: Optional[str] = None,
    hrl_num_skills: Optional[int] = HRL_EVAL_NUM_SKILLS,
    hrl_protect_skill_path: Optional[str] = HRL_EVAL_PROTECT_SKILL_PATH,
    hrl_chase_skill_path: Optional[str] = HRL_EVAL_CHASE_SKILL_PATH,
    use_privileged_classifier: bool = False,
) -> Tuple[Dict, str]:
    _configure_torch_eval_runtime()

    resolved_attacker_strategy, resolved_attacker_strategy_params, resolved_attacker_display_name = (
        _resolve_attacker_strategy_spec(attacker_strategy, attacker_strategy_params)
    )
    attacker_label = attacker_display_name or resolved_attacker_display_name

    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] EVAL START: "
        f"D={defender_strategy} vs A={attacker_label} | map={DEFAULT_MAP_LAYOUT}"
    )
    if resolved_attacker_strategy_params:
        print(f"[Eval Attacker Params] {resolved_attacker_strategy_params}")
    if use_privileged_classifier:
        print('[Eval] 使用 privileged classifier head 决策，跳过 actor action head')
    if defender_strategy == 'hrl':
        print(
            f"[HRL Eval Hold] hold_min={HRL_EVAL_HOLD_MIN}, hold_max={HRL_EVAL_HOLD_MAX}, "
            f"disable_hold_control={HRL_EVAL_DISABLE_HOLD_CONTROL}"
        )

    # 自动检测网络类型（单次加载）
    if network_type is None and defender_checkpoint:
        _, network_type, _ = _load_checkpoint(defender_checkpoint)
    network_type = _apply_network_type_constraints(defender_strategy, network_type)

    traj_png_count = max(0, int(traj_png_count))
    need_traj_png = bool(save_traj_png and traj_png_count > 0)
    gif_episodes = min(max(0, int(gif_episodes)), int(num_episodes))
    need_gif_frames = bool(save_gif and gif_episodes > 0)
    episode_frames = []

    # ---- Ray 并行评估 ----
    n_workers = get_adjusted_n_envs(4)  # 基数4，高内存时自动扩展
    learned_attacker_parallel = (
        resolved_attacker_strategy in LEARNED_ATTACKER_STRATEGIES
        and _allow_parallel_learned_attacker()
    )
    use_parallel = (
        (not force_serial)
        and num_episodes > 1
        and (
            resolved_attacker_strategy not in LEARNED_ATTACKER_STRATEGIES
            or learned_attacker_parallel
        )
    )

    if use_parallel:
        _init_ray()
        # 将episodes均匀分配给workers
        n_workers = min(n_workers, num_episodes)
        if resolved_attacker_strategy in LEARNED_ATTACKER_STRATEGIES:
            if attacker_checkpoint is None or not os.path.exists(attacker_checkpoint):
                raise ValueError(f"并行学习型攻击者 requires checkpoint_path, got {attacker_checkpoint!r}")
            n_workers = min(n_workers, _learned_attacker_worker_cap(n_workers))
            print(
                f"[Ray] 学习型攻击者并行启用: checkpoint={attacker_checkpoint}, "
                f"workers={n_workers}"
            )
        episodes_per_worker = [num_episodes // n_workers] * n_workers
        for i in range(num_episodes % n_workers):
            episodes_per_worker[i] += 1

        workers = [EvalWorker.remote(i) for i in range(n_workers)]
        print(f"[Ray] 启动 {n_workers} 个评估worker, 共 {num_episodes} episodes")

        futures = []
        ep_offset = seed_offset
        for w, n_ep in zip(workers, episodes_per_worker):
            if n_ep > 0:
                futures.append(w.run_episodes.remote(
                    defender_strategy, resolved_attacker_strategy, resolved_attacker_strategy_params, n_ep,
                    defender_checkpoint, attacker_checkpoint, network_type, ep_offset,
                    hrl_num_skills, hrl_protect_skill_path, hrl_chase_skill_path,
                    use_privileged_classifier,
                    bool(SetupParameters.EVAL_USE_RANDOM_SEED),
                    int(SetupParameters.EVAL_FIXED_SEED),
                ))
            ep_offset += n_ep

        all_stats = ray.get(futures)
        stats = _merge_stats(all_stats)
        trajectory_data_list = []

        # 清理workers
        del workers

        # 并行模式下如需GIF或轨迹PNG，补跑少量串行episode抓取可视化数据
        serial_aux_episodes = max(
            gif_episodes if need_gif_frames else 0,
            traj_png_count if need_traj_png else 0,
        )
        if serial_aux_episodes > 0:
            serial_aux = _run_serial_evaluation(
                defender_strategy, resolved_attacker_strategy, attacker_strategy_params=resolved_attacker_strategy_params,
                num_episodes=serial_aux_episodes,
                defender_checkpoint=defender_checkpoint, attacker_checkpoint=attacker_checkpoint,
                device=device, network_type=network_type,
                save_gif=need_gif_frames, gif_episodes=gif_episodes, seed_offset=seed_offset,
                collect_trajectory_episodes=traj_png_count if need_traj_png else 0,
                hrl_num_skills=hrl_num_skills,
                hrl_protect_skill_path=hrl_protect_skill_path,
                hrl_chase_skill_path=hrl_chase_skill_path,
                use_privileged_classifier=use_privileged_classifier,
            )
            if need_gif_frames:
                episode_frames = serial_aux.pop('_frames', [])
            if need_traj_png:
                trajectory_data_list = serial_aux.pop('_trajectory_data_list', [])
                legacy_traj = serial_aux.pop('_trajectory_data', None)
                if legacy_traj is not None and not trajectory_data_list:
                    trajectory_data_list = [legacy_traj]

    else:
        # 串行模式（GIF或单episode）
        stats = _run_serial_evaluation(
            defender_strategy, resolved_attacker_strategy, resolved_attacker_strategy_params, num_episodes,
            defender_checkpoint, attacker_checkpoint, device, network_type,
            save_gif, gif_episodes, seed_offset,
            collect_trajectory_episodes=traj_png_count if need_traj_png else 0,
            hrl_num_skills=hrl_num_skills,
            hrl_protect_skill_path=hrl_protect_skill_path,
            hrl_chase_skill_path=hrl_chase_skill_path,
            use_privileged_classifier=use_privileged_classifier,
        )
        trajectory_data_list = stats.pop('_trajectory_data_list', [])
        episode_frames = stats.pop('_frames', [])
        # 兼容旧字段
        legacy_traj = stats.pop('_trajectory_data', None)
        if legacy_traj is not None and not trajectory_data_list:
            trajectory_data_list = [legacy_traj]

    # Final Compilation
    hrl_protect_selection_rate, hrl_protect_selection_rate_std = _mean_std(
        stats.get('episode_hrl_skill_protect_selection_rate', [])
    )
    hrl_chase_selection_rate, hrl_chase_selection_rate_std = _mean_std(
        stats.get('episode_hrl_skill_chase_selection_rate', [])
    )
    success_rate, success_rate_ci_low, success_rate_ci_high = _proportion_ci95(stats['defender_wins'])
    defender_win_rate = success_rate
    defender_win_rate_ci_low = success_rate_ci_low
    defender_win_rate_ci_high = success_rate_ci_high
    attacker_win_rate, attacker_win_rate_ci_low, attacker_win_rate_ci_high = _proportion_ci95(stats['attacker_wins'])
    draw_rate, draw_rate_ci_low, draw_rate_ci_high = _proportion_ci95(stats['draws'])
    defender_capture_rate, defender_capture_rate_ci_low, defender_capture_rate_ci_high = _proportion_ci95(
        stats['defender_captures']
    )
    attacker_capture_rate, attacker_capture_rate_ci_low, attacker_capture_rate_ci_high = _proportion_ci95(
        stats['attacker_captures']
    )
    defender_collision_rate, defender_collision_rate_ci_low, defender_collision_rate_ci_high = _proportion_ci95(
        stats['defender_collisions']
    )
    mean_episode_length, mean_episode_length_std = _mean_std(stats['episode_lengths'])
    defender_path_length_mean, defender_path_length_std = _mean_std(
        stats.get('episode_defender_path_length', [])
    )

    final_results = {
        'defender_strategy': defender_strategy,
        'attacker_strategy': resolved_attacker_strategy,
        'attacker_display_name': attacker_label,
        'attacker_strategy_params': resolved_attacker_strategy_params,
        'episodes': num_episodes,
        'success_rate': success_rate,
        'success_rate_ci_low': success_rate_ci_low,
        'success_rate_ci_high': success_rate_ci_high,
        'defender_win_rate': defender_win_rate,
        'defender_win_rate_ci_low': defender_win_rate_ci_low,
        'defender_win_rate_ci_high': defender_win_rate_ci_high,
        'attacker_win_rate': attacker_win_rate,
        'attacker_win_rate_ci_low': attacker_win_rate_ci_low,
        'attacker_win_rate_ci_high': attacker_win_rate_ci_high,
        'draw_rate': draw_rate,
        'draw_rate_ci_low': draw_rate_ci_low,
        'draw_rate_ci_high': draw_rate_ci_high,
        'defender_capture_rate': defender_capture_rate,
        'defender_capture_rate_ci_low': defender_capture_rate_ci_low,
        'defender_capture_rate_ci_high': defender_capture_rate_ci_high,
        'attacker_capture_rate': attacker_capture_rate,
        'attacker_capture_rate_ci_low': attacker_capture_rate_ci_low,
        'attacker_capture_rate_ci_high': attacker_capture_rate_ci_high,
        'defender_collision_rate': defender_collision_rate,
        'defender_collision_rate_ci_low': defender_collision_rate_ci_low,
        'defender_collision_rate_ci_high': defender_collision_rate_ci_high,
        'mean_episode_length': 0.0 if mean_episode_length is None else mean_episode_length,
        'mean_episode_length_std': 0.0 if mean_episode_length_std is None else mean_episode_length_std,
        'defender_path_length_mean': defender_path_length_mean,
        'defender_path_length_std': defender_path_length_std,
        'hrl_protect_selection_rate': hrl_protect_selection_rate,
        'hrl_protect_selection_rate_std': hrl_protect_selection_rate_std,
        'hrl_chase_selection_rate': hrl_chase_selection_rate,
        'hrl_chase_selection_rate_std': hrl_chase_selection_rate_std,
    }

    # GIF handling
    gif_out = None
    if save_gif and episode_frames:
        if not gif_path:
            attacker_file = _safe_filename_token(attacker_label)
            gif_path = str(EVAL_OUTPUT_DIR / f"{defender_strategy}_vs_{attacker_file}_{DEFAULT_MAP_LAYOUT}.gif")
        os.makedirs(os.path.dirname(gif_path) if os.path.dirname(gif_path) else '.', exist_ok=True)
        saved_count = 0
        for idx, reason, frames in episode_frames:
            p = gif_path.replace('.gif', f'_ep{idx}_{reason}.gif')
            make_gif(frames, p, fps=20, quality='medium')
            saved_count += 1
        print(f"  [GIF] Saved {saved_count} gifs")
        gif_out = gif_path

    if need_traj_png and trajectory_data_list:
        attacker_file = _safe_filename_token(attacker_label)
        default_traj_path = str(EVAL_OUTPUT_DIR / f"{defender_strategy}_vs_{attacker_file}_{DEFAULT_MAP_LAYOUT}.png")
        base_path = traj_png_path or default_traj_path
        os.makedirs(os.path.dirname(base_path) if os.path.dirname(base_path) else '.', exist_ok=True)

        if len(trajectory_data_list) == 1:
            make_trajectory_plot(trajectory_data_list[0], base_path, dpi=150)
            print(f"  [TRAJ] Saved trajectory plot to {base_path}")
        else:
            root, ext = os.path.splitext(base_path)
            ext = ext or '.png'
            saved_paths = []
            for idx, traj in enumerate(trajectory_data_list):
                reason = _safe_filename_token(traj.get('reason', 'unknown'))
                p = f"{root}_ep{idx}_{reason}{ext}"
                make_trajectory_plot(traj, p, dpi=150)
                saved_paths.append(p)
            print(f"  [TRAJ] Saved {len(saved_paths)} trajectory plots")

    if save_stats and stats_path:
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        stats_to_save = {
            k: v for k, v in stats.items()
            if k not in {'_frames', 'episode_defender_path_length'}
        }
        saved_final_results = _prune_saved_metrics_dict(
            final_results,
            defender_strategy=defender_strategy,
            hrl_num_skills=hrl_num_skills,
        )
        saved_final_results = _format_saved_metrics_dict(saved_final_results)
        with open(stats_path, 'w') as f:
            json.dump({**saved_final_results, 'raw': stats_to_save}, f, indent=2)
        print(f"  [Stats] Saved to {stats_path}")

    mode_str = f"parallel({n_workers}w)" if use_parallel else "serial"
    if use_parallel and need_gif_frames:
        mode_str += "+gif-serial"
    if use_parallel and need_traj_png:
        mode_str += "+traj-serial"
    print(
        f"  RESULT [{mode_str}]: "
        f"D-win={_fmt_metric_ci_pm_from_row(final_results, 'defender_win_rate', scale=100.0, decimals=1, suffix='%')} "
        f"A-win={_fmt_metric_ci_pm_from_row(final_results, 'attacker_win_rate', scale=100.0, decimals=1, suffix='%')} "
        f"draw={_fmt_metric_ci_pm_from_row(final_results, 'draw_rate', scale=100.0, decimals=1, suffix='%')} "
        f"mean_len={_fmt_metric_pm_from_row(final_results, 'mean_episode_length', decimals=1)} "
        f"path_len={_fmt_metric_pm(final_results['defender_path_length_mean'], final_results['defender_path_length_std'], decimals=1)}"
    )
    if _is_hrl_like_strategy(defender_strategy) and hrl_chase_selection_rate is not None:
        print(
            f"  HRL Skill选择占比: "
            f"protect={_fmt_metric_pm_from_row(final_results, 'hrl_protect_selection_rate', scale=100.0, decimals=1, suffix='%')} "
            f"chase={_fmt_metric_pm_from_row(final_results, 'hrl_chase_selection_rate', scale=100.0, decimals=1, suffix='%')}"
        )
    return final_results, gif_out


def _run_serial_evaluation(
    defender_strategy, attacker_strategy, attacker_strategy_params, num_episodes,
    defender_checkpoint, attacker_checkpoint, device, network_type,
    save_gif, gif_episodes, seed_offset, collect_trajectory_episodes=0,
    hrl_num_skills: Optional[int] = None,
    hrl_protect_skill_path: Optional[str] = None,
    hrl_chase_skill_path: Optional[str] = None,
    use_privileged_classifier: bool = False,
) -> dict:
    """串行运行episodes（用于GIF保存或单episode模式）。"""
    if network_type is None and defender_checkpoint and os.path.exists(defender_checkpoint):
        _, network_type, _ = _load_checkpoint(defender_checkpoint)
    network_type = _apply_network_type_constraints(defender_strategy, network_type)

    _configure_eval_environment()

    # Environment Selection
    is_gym_wrapper = False
    runtime_strategy = defender_strategy

    env_attacker_strategy = attacker_strategy
    if attacker_strategy in LEARNED_ATTACKER_STRATEGIES:
        env_attacker_strategy = 'static'

    if _is_hrl_like_strategy(defender_strategy):
        protect_path, chase_path = _resolve_hrl_skill_paths(
            strategy=defender_strategy,
            protect_path=hrl_protect_skill_path,
            chase_path=hrl_chase_skill_path,
            num_skills=hrl_num_skills,
        )
        env = HRLEnv(
            protect_model_path=protect_path,
            chase_model_path=chase_path,
            primary_skill_name='protect',
            attacker_strategy=env_attacker_strategy,
            attacker_strategy_params=attacker_strategy_params,
            device=device,
            hold_min=HRL_EVAL_HOLD_MIN,
            hold_max=HRL_EVAL_HOLD_MAX,
            disable_hold_control=HRL_EVAL_DISABLE_HOLD_CONTROL,
        )
        is_gym_wrapper = True
    elif runtime_strategy in RL_DEFENDER_STRATEGIES:
        env = ProtectEnv(
            attacker_strategy=env_attacker_strategy,
            attacker_strategy_params=attacker_strategy_params,
        )
        is_gym_wrapper = True
    elif defender_strategy in ALL_RULE_DEFENDER_STRATEGIES:
        env = TrackingEnv()
    else:
        env = TrackingEnv()

    # 全局硬掩码默认关闭；仅 HRL 在环境层按需开启 defender hard-mask。
    _disable_env_hard_mask(env)
    _maybe_enable_eval_defender_mask(env, defender_strategy)

    defender_eval = Defenderevaluator(
        defender_strategy,
        defender_checkpoint,
        device,
        network_type=network_type,
        use_privileged_classifier=use_privileged_classifier,
    )
    attacker_eval = Attackerevaluator(
        attacker_strategy,
        strategy_params=attacker_strategy_params,
        checkpoint_path=attacker_checkpoint,
        device=device,
    )

    stats = {
        'defender_wins': [], 'attacker_wins': [], 'draws': [],
        'reasons': [], 'episode_lengths': [],
        'defender_captures': [], 'attacker_captures': [],
        'defender_collisions': [],
        'episode_hrl_skill_protect_selection_rate': [],
        'episode_hrl_skill_chase_selection_rate': [],
        'episode_hrl_step_trace': [],
        'episode_defender_path_length': [],
    }
    episode_frames = []
    collect_trajectory_episodes = max(0, int(collect_trajectory_episodes))
    trajectory_data_list = []
    record_hrl_step_trace = (
        _is_hrl_like_strategy(defender_strategy)
        and _read_env_bool('VS_HRL_STEP_TRACE_ENABLE', False)
    )
    hrl_step_trace_max_steps = max(0, _read_env_int('VS_HRL_STEP_TRACE_MAX_STEPS', 0))

    for episode in range(num_episodes):
        if SetupParameters.EVAL_USE_RANDOM_SEED:
            current_seed = None
        else:
            current_seed = SetupParameters.EVAL_FIXED_SEED + episode + seed_offset

        obs, info = env.reset(seed=current_seed)
        if isinstance(obs, tuple) and len(obs) == 2:
            def_obs, att_obs = obs
        else:
            def_obs = obs; att_obs = obs

        defender_eval.reset(env)
        attacker_eval.reset()
        done = False
        frames = []
        ep_hrl_protect_selected = 0
        ep_hrl_chase_selected = 0
        record_traj = bool(episode < collect_trajectory_episodes)
        ep_def_traj = []
        ep_atk_traj = []
        ep_def_skill_trace = []
        ep_hrl_step_trace = []
        target_pos = None
        if record_traj:
            base_env = env.env if hasattr(env, 'env') else env
            if hasattr(base_env, 'get_privileged_state'):
                priv = base_env.get_privileged_state()
                ep_def_traj = [(priv['defender']['center_x'], priv['defender']['center_y'])]
                ep_atk_traj = [(priv['attacker']['center_x'], priv['attacker']['center_y'])]
                target_pos = (priv['target']['center_x'], priv['target']['center_y'])
            else:
                record_traj = False

        while not done:
            step_idx = ep_hrl_protect_selected + ep_hrl_chase_selected
            pre_step_risk_margin = _current_task_margin(env) if record_hrl_step_trace else None
            def_action = defender_eval.get_action(def_obs, env, att_obs)
            if is_gym_wrapper and attacker_strategy in LEARNED_ATTACKER_STRATEGIES:
                att_action = attacker_eval.get_action(att_obs)
                output = env.step(def_action, attacker_action=att_action)
            elif is_gym_wrapper:
                output = env.step(def_action)
            else:
                att_action = attacker_eval.get_action(att_obs)
                output = env.step(action=def_action, attacker_action=att_action)

            if len(output) == 5:
                next_obs, reward, term, trunc, info = output
            elif len(output) == 4:
                next_obs, reward, done_bool, info = output
                term = done_bool; trunc = False

            current_skill = None
            if _is_hrl_like_strategy(defender_strategy):
                current_skill = _resolve_selected_skill_from_info(info)

            if _is_hrl_like_strategy(defender_strategy):
                if current_skill == 'protect':
                    ep_hrl_protect_selected += 1
                elif current_skill == 'chase':
                    ep_hrl_chase_selected += 1
                if record_hrl_step_trace and (
                    hrl_step_trace_max_steps <= 0 or len(ep_hrl_step_trace) < hrl_step_trace_max_steps
                ):
                    ep_hrl_step_trace.append(
                        _hrl_step_trace_row(
                            episode=episode,
                            step=step_idx,
                            seed=current_seed,
                            attacker=attacker_strategy,
                            speed_regime=os.environ.get('TAD_SPEED_REGIME', 'unknown'),
                            margin_regime=os.environ.get('TAD_MARGIN_REGIME', 'unknown'),
                            action=def_action,
                            selected_skill=current_skill,
                            risk_margin=pre_step_risk_margin,
                        )
                    )

            if isinstance(next_obs, tuple) and len(next_obs) == 2:
                def_obs, att_obs = next_obs
            else:
                def_obs = next_obs

            done = term or trunc

            if save_gif and episode < gif_episodes and len(frames) < 2000:
                try:
                    if hasattr(env, 'env') and hasattr(env.env, 'render'):
                        f = env.env.render(mode='rgb_array', style='pygame')
                    elif hasattr(env, 'render'):
                        f = env.render(mode='rgb_array', style='pygame')
                    else:
                        f = None
                    if f is not None:
                        frames.append(f)
                except (NotImplementedError, TypeError):
                    pass

            if record_traj:
                base_env = env.env if hasattr(env, 'env') else env
                if hasattr(base_env, 'get_privileged_state'):
                    priv = base_env.get_privileged_state()
                    ep_def_traj.append((priv['defender']['center_x'], priv['defender']['center_y']))
                    ep_atk_traj.append((priv['attacker']['center_x'], priv['attacker']['center_y']))
                    if _is_hrl_like_strategy(defender_strategy):
                        ep_def_skill_trace.append(current_skill if current_skill in ('protect', 'chase') else 'unknown')

        # Record Stats
        reason = info.get('reason', 'unknown')
        stats['reasons'].append(reason)
        ep_len = _resolve_episode_length(info, env)
        stats['episode_lengths'].append(int(ep_len))
        ep_hrl_total_selected = ep_hrl_protect_selected + ep_hrl_chase_selected
        if _is_hrl_like_strategy(defender_strategy):
            if ep_hrl_total_selected > 0:
                stats['episode_hrl_skill_protect_selection_rate'].append(
                    float(ep_hrl_protect_selected) / float(ep_hrl_total_selected)
                )
                stats['episode_hrl_skill_chase_selection_rate'].append(
                    float(ep_hrl_chase_selected) / float(ep_hrl_total_selected)
                )
            else:
                stats['episode_hrl_skill_protect_selection_rate'].append(0.0)
                stats['episode_hrl_skill_chase_selection_rate'].append(0.0)
            if record_hrl_step_trace:
                stats['episode_hrl_step_trace'].append(ep_hrl_step_trace)
        stats['episode_defender_path_length'].append(_extract_defender_path_length(env))

        d_cap = 'defender_caught_attacker' in reason
        a_cap = 'attacker_caught_target' in reason or 'attacker_win' in reason
        d_col = 'defender_collision' in reason or 'defender_out' in reason
        timeout = 'timeout' in reason or 'time_limit' in reason or 'max_steps' in reason or 'truncated' in reason

        stats['defender_captures'].append(1 if d_cap else 0)
        stats['attacker_captures'].append(1 if a_cap else 0)
        stats['defender_collisions'].append(1 if d_col else 0)
        _append_episode_outcome_stats(
            stats=stats,
            defender_strategy=defender_strategy,
            defender_captured=d_cap,
            attacker_captured=a_cap,
            timeout=timeout,
        )

        if save_gif and episode < gif_episodes and len(frames) > 0:
            episode_frames.append((episode, reason, frames))

        if record_traj:
            trajectory_data_list.append({
                'defender_traj': ep_def_traj,
                'attacker_traj': ep_atk_traj,
                'target_pos': target_pos,
                'obstacles': list(getattr(map_config, 'obstacles', [])),
                'width': int(getattr(map_config, 'width', 640)),
                'height': int(getattr(map_config, 'height', 640)),
                'win': bool(stats['defender_wins'][-1]) if stats['defender_wins'] else False,
                'skill_mode': defender_strategy,
                'episode_len': int(ep_len),
                'reason': reason,
                'episode_idx': int(episode),
                'defender_skill_trace': ep_def_skill_trace,
            })

        if (episode + 1) % 10 == 0:
            print(f"  Ep {episode+1}/{num_episodes} | D-Win: {np.mean(stats['defender_wins'][-10:])*100:.0f}%")

    if episode_frames:
        stats['_frames'] = episode_frames
    if trajectory_data_list:
        stats['_trajectory_data_list'] = trajectory_data_list
    return stats

# --- Suite Mode ---
def _expand_all_attacker_configs(config_list: List[Dict], global_episodes: int) -> List[Dict]:
    """将attacker=all扩展为默认对手集合，每种沿用该配置episodes。"""
    expanded = []
    for config in config_list:
        attacker = config.get('attacker')
        # 兼容历史配置: random 视为 all
        if attacker in ['all', 'random']:
            per_attacker_episodes = int(config.get('episodes', global_episodes))
            for attacker_name in ALL_ATTACKER_STRATEGIES:
                item = dict(config)
                item['attacker'] = attacker_name
                item['episodes'] = per_attacker_episodes
                expanded.append(item)
        else:
            expanded.append(dict(config))
    return expanded


def run_suite(
    config_list: List[Dict],
    global_episodes=500,
    gif_episodes=0,
    save_traj_png: bool = False,
    traj_png_count: int = 1,
    hrl_num_skills: Optional[int] = HRL_EVAL_NUM_SKILLS,
    hrl_protect_skill_path: Optional[str] = HRL_EVAL_PROTECT_SKILL_PATH,
    hrl_chase_skill_path: Optional[str] = HRL_EVAL_CHASE_SKILL_PATH,
    use_privileged_classifier: bool = False,
):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    suite_dir = str(EVAL_OUTPUT_DIR / f"suite_{timestamp}")
    os.makedirs(suite_dir, exist_ok=True)

    expanded_configs = _expand_all_attacker_configs(config_list, global_episodes)
    if len(expanded_configs) != len(config_list):
        print(
            f"[Suite] 检测到 attacker=all，已展开为{len(ALL_ATTACKER_STRATEGIES)}种策略（不含比例切换），"
            f"每种使用该配置的episodes"
        )

    deduped_configs = []
    seen = set()
    for cfg in expanded_configs:
        resolved_attacker, resolved_params, display_attacker = _resolve_attacker_strategy_spec(
            cfg['attacker'],
            cfg.get('attacker_strategy_params'),
        )
        cfg = dict(cfg)
        cfg['resolved_attacker'] = resolved_attacker
        cfg['attacker_strategy_params'] = resolved_params
        cfg['attacker_display_name'] = display_attacker
        key = (
            cfg['defender'],
            cfg['attacker_display_name'],
            int(cfg.get('episodes', global_episodes)),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped_configs.append(cfg)
    if len(deduped_configs) != len(expanded_configs):
        print("[Suite] 检测到重复对阵，已自动去重")
    expanded_configs = deduped_configs

    summary_results = []
    matchup_results = []
    used_model_paths: Dict[str, List[str]] = {}

    print(f"\n开始批量评估: 共{len(expanded_configs)}个配置")

    for i, config in enumerate(expanded_configs):
        d_strat = config['defender']
        a_strat = config['attacker']
        resolved_a_strat = config.get('resolved_attacker', a_strat)
        resolved_a_params = config.get('attacker_strategy_params')
        display_a_strat = config.get('attacker_display_name', a_strat)
        display_a_token = _safe_filename_token(display_a_strat)
        n_episodes = int(config.get('episodes', global_episodes))
        checkpoint_path = config.get('checkpoint')
        attacker_checkpoint_path = config.get('attacker_checkpoint')
        if checkpoint_path is None:
            checkpoint_path = _default_model_path(d_strat)
        if d_strat in RL_DEFENDER_STRATEGIES and not checkpoint_path:
            print(
                f"\n[{i+1}/{len(expanded_configs)}] 配置: 防御者={d_strat} "
                f"vs 攻击者={display_a_strat}"
            )
            print(f"  [Skip] 缺少 {d_strat} checkpoint，请通过配置或 --checkpoint 指定")
            continue
        print(
            f"\n[{i+1}/{len(expanded_configs)}] 配置: 防御者={d_strat} "
            f"vs 攻击者={display_a_strat} | episodes={n_episodes}"
        )
        print(f"  checkpoint={checkpoint_path}")
        if attacker_checkpoint_path:
            print(f"  attacker_checkpoint={attacker_checkpoint_path}")
        _append_unique_path(used_model_paths, d_strat, checkpoint_path)

        metrics, _ = run_evaluation(
            defender_strategy=d_strat,
            attacker_strategy=resolved_a_strat,
            attacker_strategy_params=resolved_a_params,
            attacker_display_name=display_a_strat,
            num_episodes=n_episodes,
            defender_checkpoint=checkpoint_path,
            attacker_checkpoint=attacker_checkpoint_path,
            save_traj_png=save_traj_png,
            traj_png_count=traj_png_count,
            save_stats=True,
            stats_path=os.path.join(suite_dir, f'res_{d_strat}_vs_{display_a_token}.json'),
            save_gif=gif_episodes > 0,
            gif_episodes=gif_episodes,
            gif_path=os.path.join(suite_dir, f'{d_strat}_vs_{display_a_token}.gif'),
            traj_png_path=os.path.join(suite_dir, f'{d_strat}_vs_{display_a_token}.png') if save_traj_png else None,
            hrl_num_skills=hrl_num_skills,
            hrl_protect_skill_path=hrl_protect_skill_path,
            hrl_chase_skill_path=hrl_chase_skill_path,
            use_privileged_classifier=use_privileged_classifier,
        )

        row = {
            'defender': d_strat,
            'attacker': display_a_strat,
            'episodes': metrics['episodes'],
            'success_rate': metrics['success_rate'],
            'success_rate_ci_low': metrics['success_rate_ci_low'],
            'success_rate_ci_high': metrics['success_rate_ci_high'],
            'attacker_win_rate': metrics['attacker_win_rate'],
            'attacker_win_rate_ci_low': metrics['attacker_win_rate_ci_low'],
            'attacker_win_rate_ci_high': metrics['attacker_win_rate_ci_high'],
            'draw_rate': metrics['draw_rate'],
            'draw_rate_ci_low': metrics['draw_rate_ci_low'],
            'draw_rate_ci_high': metrics['draw_rate_ci_high'],
            'defender_capture_rate': metrics['defender_capture_rate'],
            'defender_capture_rate_ci_low': metrics['defender_capture_rate_ci_low'],
            'defender_capture_rate_ci_high': metrics['defender_capture_rate_ci_high'],
            'attacker_capture_rate': metrics['attacker_capture_rate'],
            'attacker_capture_rate_ci_low': metrics['attacker_capture_rate_ci_low'],
            'attacker_capture_rate_ci_high': metrics['attacker_capture_rate_ci_high'],
            'defender_collision_rate': metrics['defender_collision_rate'],
            'defender_collision_rate_ci_low': metrics['defender_collision_rate_ci_low'],
            'defender_collision_rate_ci_high': metrics['defender_collision_rate_ci_high'],
            'mean_episode_length': metrics['mean_episode_length'],
            'mean_episode_length_std': metrics['mean_episode_length_std'],
            'defender_path_length_mean': metrics['defender_path_length_mean'],
            'defender_path_length_std': metrics['defender_path_length_std'],
            'hrl_protect_selection_rate': metrics['hrl_protect_selection_rate'],
            'hrl_protect_selection_rate_std': metrics['hrl_protect_selection_rate_std'],
            'hrl_chase_selection_rate': metrics['hrl_chase_selection_rate'],
            'hrl_chase_selection_rate_std': metrics['hrl_chase_selection_rate_std'],
        }
        summary_results.append(row)
        matchup_results.append(row.copy())

    if not summary_results:
        print("\n未产生有效评估配置（可能全部被跳过）。")
        if ray.is_initialized():
            ray.shutdown()
        return

    # Save Summary CSV
    import csv
    csv_path = os.path.join(suite_dir, 'suite_summary.csv')
    saved_summary_results = [
        _prune_saved_metrics_dict(
            row,
            defender_strategy=row.get('defender'),
            hrl_num_skills=hrl_num_skills,
        )
        for row in summary_results
    ]
    saved_matchup_results = [
        _prune_saved_metrics_dict(
            row,
            defender_strategy=row.get('defender'),
            hrl_num_skills=hrl_num_skills,
        )
        for row in matchup_results
    ]

    formatted_summary_results = [_format_suite_summary_row(row) for row in saved_summary_results]
    average_summary_row = _build_average_row(saved_summary_results, defender_label='AVERAGE', attacker_label='ALL')
    formatted_average_summary_row = (
        _format_suite_average_row(average_summary_row) if average_summary_row is not None else None
    )
    formatted_matchup_results = [_format_suite_summary_row(row) for row in saved_matchup_results]
    model_path_info: Dict[str, object] = {
        'defender_models': {
            strategy: [
                {
                    'path': path,
                    'network_type': _safe_detect_checkpoint_network_type(path),
                }
                for path in paths
            ]
            for strategy, paths in used_model_paths.items()
        },
    }
    if any(cfg['defender'] in HRL_DEFENDER_STRATEGIES for cfg in expanded_configs):
        resolved_primary_path, resolved_chase_path = _resolve_hrl_skill_paths(
            strategy='hrl',
            protect_path=hrl_protect_skill_path,
            chase_path=hrl_chase_skill_path,
            num_skills=hrl_num_skills,
        )
        model_path_info['hrl_skills'] = {
            'num_skills': int(hrl_num_skills) if hrl_num_skills is not None else None,
            'primary_skill_name': 'protect',
            'primary_skill': {
                'path': resolved_primary_path,
                'network_type': _safe_detect_checkpoint_network_type(resolved_primary_path),
            },
            'chase_skill': {
                'path': resolved_chase_path,
                'network_type': _safe_detect_checkpoint_network_type(resolved_chase_path),
            },
        }
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        raw_writer = csv.writer(f)
        raw_writer.writerow(['model_info', json.dumps(model_path_info, ensure_ascii=False)])
        summary_fieldnames = _ordered_export_fieldnames(formatted_summary_results, FORMATTED_SUMMARY_FIELD_ORDER)
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(formatted_summary_results)
        if formatted_average_summary_row is not None:
            writer.writerow(formatted_average_summary_row)

    # Save Matchup CSV (同一个CSV内按attacker分表)
    matchup_csv_path = os.path.join(suite_dir, 'suite_matchups.csv')
    fieldnames = _ordered_export_fieldnames(formatted_matchup_results, RAW_MATCHUP_FIELD_ORDER)

    attacker_order = []
    for row in saved_matchup_results:
        attacker = row['attacker']
        if attacker not in attacker_order:
            attacker_order.append(attacker)

    with open(matchup_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        for idx, attacker in enumerate(attacker_order):
            writer.writerow([f'attacker={attacker}'])
            writer.writerow(fieldnames)

            rows = [r for r in saved_matchup_results if r['attacker'] == attacker]
            rows = sorted(rows, key=lambda x: x['defender'])
            for row in rows:
                formatted_row = _format_suite_summary_row(row)
                writer.writerow([formatted_row.get(k) for k in fieldnames])
            avg_row = _build_average_row(rows, defender_label='AVERAGE', attacker_label=attacker)
            if avg_row is not None:
                formatted_avg_row = _format_suite_average_row(avg_row)
                writer.writerow([formatted_avg_row.get(k) for k in fieldnames])

            if idx != len(attacker_order) - 1:
                writer.writerow([])

        overall_avg_row = _build_average_row(saved_matchup_results, defender_label='AVERAGE', attacker_label='ALL')
        if overall_avg_row is not None:
            writer.writerow([])
            writer.writerow(['overall_average'])
            writer.writerow(fieldnames)
            formatted_overall_avg_row = _format_suite_average_row(overall_avg_row)
            writer.writerow([formatted_overall_avg_row.get(k) for k in fieldnames])

    # 评估完成后关闭Ray
    if ray.is_initialized():
        ray.shutdown()

    print(f"\n评估完成! 汇总保存至 {csv_path}")
    print(f"对阵明细保存至 {matchup_csv_path} (按attacker分表)")
    print("\n========== 结果汇总 ==========")
    metric_width = 26
    print(
        f"{'Defender':<20} {'Attacker':<24} {'胜率':>{metric_width}} {'A胜率':>{metric_width}} "
        f"{'平局':>{metric_width}} {'D抓获':>{metric_width}} {'A抓获':>{metric_width}} "
        f"{'D碰撞':>{metric_width}} {'平均步数':>{metric_width}} {'平均路径':>{metric_width}} "
        f"{'Protect%':>{metric_width}} {'Chase%':>{metric_width}}"
    )
    print("-" * (20 + 24 + metric_width * 10 + 12))
    for res in summary_results:
        protect_str = _fmt_metric_pm_from_row(
            res,
            'hrl_protect_selection_rate',
            scale=100.0,
            decimals=1,
            suffix='%',
        )
        chase_str = _fmt_metric_pm_from_row(
            res,
            'hrl_chase_selection_rate',
            scale=100.0,
            decimals=1,
            suffix='%',
        )
        print(
            f"{res['defender']:<20} {res['attacker']:<24} "
            f"{_fmt_metric_ci_pm_from_row(res, 'success_rate', scale=100.0, decimals=1, suffix='%'):>{metric_width}} "
            f"{_fmt_metric_ci_pm_from_row(res, 'attacker_win_rate', scale=100.0, decimals=1, suffix='%'):>{metric_width}} "
            f"{_fmt_metric_ci_pm_from_row(res, 'draw_rate', scale=100.0, decimals=1, suffix='%'):>{metric_width}} "
            f"{_fmt_metric_ci_pm_from_row(res, 'defender_capture_rate', scale=100.0, decimals=1, suffix='%'):>{metric_width}} "
            f"{_fmt_metric_ci_pm_from_row(res, 'attacker_capture_rate', scale=100.0, decimals=1, suffix='%'):>{metric_width}} "
            f"{_fmt_metric_ci_pm_from_row(res, 'defender_collision_rate', scale=100.0, decimals=1, suffix='%'):>{metric_width}} "
            f"{_fmt_metric_pm_from_row(res, 'mean_episode_length', decimals=1):>{metric_width}} "
            f"{_fmt_metric_pm(res['defender_path_length_mean'], res['defender_path_length_std'], decimals=1):>{metric_width}} "
            f"{protect_str:>{metric_width}} {chase_str:>{metric_width}}"
        )

    overall_avg_row = _build_average_row(summary_results, defender_label='AVERAGE', attacker_label='ALL')
    if overall_avg_row is not None:
        print("-" * (20 + 24 + metric_width * 10 + 12))
        print(
            f"{overall_avg_row['defender']:<20} {overall_avg_row['attacker']:<24} "
            f"{_fmt_metric_value(overall_avg_row.get('success_rate'), scale=100.0, decimals=1, suffix='%'):>{metric_width}} "
            f"{_fmt_metric_value(overall_avg_row.get('attacker_win_rate'), scale=100.0, decimals=1, suffix='%'):>{metric_width}} "
            f"{_fmt_metric_value(overall_avg_row.get('draw_rate'), scale=100.0, decimals=1, suffix='%'):>{metric_width}} "
            f"{_fmt_metric_value(overall_avg_row.get('defender_capture_rate'), scale=100.0, decimals=1, suffix='%'):>{metric_width}} "
            f"{_fmt_metric_value(overall_avg_row.get('attacker_capture_rate'), scale=100.0, decimals=1, suffix='%'):>{metric_width}} "
            f"{_fmt_metric_value(overall_avg_row.get('defender_collision_rate'), scale=100.0, decimals=1, suffix='%'):>{metric_width}} "
            f"{_fmt_metric_value(overall_avg_row.get('mean_episode_length'), decimals=1):>{metric_width}} "
            f"{_fmt_metric_value(overall_avg_row.get('defender_path_length_mean'), decimals=1):>{metric_width}} "
            f"{_fmt_metric_value(overall_avg_row.get('hrl_protect_selection_rate'), scale=100.0, decimals=1, suffix='%'):>{metric_width}} "
            f"{_fmt_metric_value(overall_avg_row.get('hrl_chase_selection_rate'), scale=100.0, decimals=1, suffix='%'):>{metric_width}}"
        )

# --- 交互式模式 ---
def interactive_suite_mode():
    print("\n=== 防御者 vs 攻击者 评估系统 ===")

    # 仅保留学习型防御者策略（规则策略源码保留，但不提供评估入口）。
    defenders = TEST_DEFENDER_STRATEGIES
    # 评估时可选attacker策略 + all + static
    attackers = [
        'all',          # 展开为除比例切换外的全部策略，每种回合数=用户输入
        'default',      # 默认策略：A*寻路 + 适度避让
        'evasive',      # 规避策略：最大化距离并避开Defender视野
        'static'        # 静止不动
    ]

    defender_names = {
        'hrl': 'HRL分层策略(高层调度)',
        'hrl_rule_geo_trend': '规则上层HRL(几何+趋势)',
        'hrl_rule_apollonius_label': '规则上层HRL(阿波罗尼斯方向匹配)',
        'protect': 'Protect技能',
        'chase': 'Chase技能(NMN)',
    }
    attacker_names = {
        'all': (
            f'默认全量策略({len(ALL_ATTACKER_STRATEGIES)}种; '
            f'每种回合数=你输入的episodes)'
        ),
        'default': '默认策略(A*+适度避让)',
        'evasive': '规避策略(避视野)',
        'static': '静止不动'
    }
    
    def multi_select(options, names, prompt, exclude_when_select_all: Optional[List[str]] = None):
        """多选函数，返回选中的选项列表"""
        print(f"\n{prompt} (输入序号，多选用逗号分隔，如1,2,3 或输入 a 全选):")
        for i, opt in enumerate(options):
            print(f"  {i+1}. {opt} ({names[opt]})")
        
        choice = input("请输入: ").strip().lower()
        if choice == 'a':
            selected_all = options.copy()
            if exclude_when_select_all:
                selected_all = [x for x in selected_all if x not in set(exclude_when_select_all)]
            return selected_all
        
        selected = []
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(',')]
            for idx in indices:
                if 0 <= idx < len(options):
                    selected.append(options[idx])
        except ValueError:
            pass
        return selected
    
    # 选择防御者（可多选）
    selected_defenders = multi_select(defenders, defender_names, "选择防御者策略")
    if not selected_defenders:
        print("未选择任何防御者，退出。")
        return
    print(f"已选防御者: {[defender_names[d] for d in selected_defenders]}")
    
    # 选择攻击者（可多选）
    selected_attackers = multi_select(
        attackers,
        attacker_names,
        "选择攻击者策略",
    )
    if not selected_attackers:
        print("未选择任何攻击者，退出。")
        return
    print(f"已选攻击者: {[attacker_names[a] for a in selected_attackers]}")

    # 设置评估回合数
    ep_input = input("\n评估回合数 (默认500): ").strip()
    episodes = int(ep_input) if ep_input.isdigit() else 500
    
    # 设置GIF数量
    gif_input = input("保存GIF数量 (默认0，不保存): ").strip()
    gif_count = int(gif_input) if gif_input.isdigit() else 0

    png_input = input("保存轨迹PNG数量 (默认0，不保存): ").strip()
    traj_png_count = int(png_input) if png_input.isdigit() else 0

    # 生成所有组合
    configs = []
    for d in selected_defenders:
        for a in selected_attackers:
            configs.append({'defender': d, 'attacker': a})

    print(f"\n========================================")
    print(f"评估配置汇总:")
    print(f"  防御者: {len(selected_defenders)}个")
    print(f"  攻击者: {len(selected_attackers)}个")
    print(f"  总组合: {len(configs)}个")
    print(f"  每组回合数: {episodes}")
    print(f"  GIF数量: {gif_count}")
    print(f"  轨迹PNG数量: {traj_png_count}")
    print(f"========================================")
    
    for i, c in enumerate(configs):
        print(f"  {i+1}. {defender_names[c['defender']]} vs {attacker_names[c['attacker']]}")
    
    confirm = input("\n确认开始评估? (y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消。")
        return
    
    print(f"\n开始评估...")
    run_suite(
        configs,
        global_episodes=episodes,
        gif_episodes=gif_count,
        save_traj_png=traj_png_count > 0,
        traj_png_count=traj_png_count,
    )


def _parse_defenders_arg(single_defender: str, defenders_arg: Optional[str]) -> List[str]:
    raw = defenders_arg if defenders_arg else single_defender
    if raw is None:
        return []
    items = [x.strip() for x in str(raw).split(',') if x.strip()]
    deduped = []
    seen = set()
    for item in items:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def main():
    parser = argparse.ArgumentParser(description="评估脚本 - 防御者 vs 攻击者")
    parser.add_argument('--suite', action='store_true', help="运行预设套件模式")
    parser.add_argument('--no-interactive', action='store_true', help="跳过交互模式，使用命令行参数")
    
    # 命令行参数（用于非交互模式）
    parser.add_argument('--defender', '-d', default='hrl', help="防御者策略")
    parser.add_argument('--defenders', type=str, default=None, help="多个防御者，用逗号分隔（如 hrl,protect）")
    parser.add_argument('--attacker', '-a', default='attacker_global', help="攻击者策略")
    parser.add_argument('--episodes', '-n', type=int, default=500, help="评估回合数")
    parser.add_argument('--gif', action='store_true', help="保存GIF")
    parser.add_argument('--gif-episodes', type=int, default=10, help="保存GIF的episode数量（需配合 --gif）")
    parser.add_argument('--checkpoint', type=str, default=None, help="模型检查点路径")
    parser.add_argument('--attacker-checkpoint', type=str, default=None, help="学习型攻击者 checkpoint 路径")
    parser.add_argument('--hrl-num-skills', type=int, choices=[2], default=HRL_EVAL_NUM_SKILLS,
                        help="HRL技能个数，固定为2：protect+chase")
    parser.add_argument('--hrl-protect-skill-path', type=str, default=HRL_EVAL_PROTECT_SKILL_PATH,
                        help="HRL Protect技能模型路径（覆盖自动查找）")
    parser.add_argument('--hrl-chase-skill-path', type=str, default=HRL_EVAL_CHASE_SKILL_PATH,
                        help="HRL chase技能模型路径（覆盖自动查找）")
    parser.add_argument('--network-type', type=str, default=None,
                        choices=['nmn', 'nmn_ctde', 'nmn_ctde_shared', 'nmn_ctde_task_shared', 'nmn_ctde_task_shared_distill', 'nmn_no_shared_radar', 'nmn_dual_gru_raw', 'nmn_dual_gru_raw_ctde', 'nmn_gru', 'mlp', 'mlp_ctde', 'mlp_gru', 'mlp_noctde', 'hrl_top', 'hrl_top_noctde', 'hrl_top_gru', 'hrl_top_dual_gru_raw'],
                        help="网络类型，不指定则自动检测")
    parser.add_argument(
        '--use-privileged-classifier',
        dest='use_privileged_classifier',
        action='store_true',
        help='仅使用 privileged classifier head 做HRL技能决策，不走actor',
    )
    parser.add_argument(
        '--use-privileged-probe',
        dest='use_privileged_classifier',
        action='store_true',
        help=argparse.SUPPRESS,
    )
    parser.add_argument('--save-stats', action='store_true', help="保存评估统计JSON")
    parser.add_argument('--serial', action='store_true', help="禁用Ray并行，使用串行评估（调试用）")
    parser.add_argument('--stats-path', type=str, default=None, help="统计JSON输出路径")
    parser.add_argument('--traj-png-count', type=int, default=0, help="保存轨迹PNG数量")
    parser.add_argument(
        '--eval-deterministic-seeds',
        action='store_true',
        help="使用固定episode seed，便于paired评估",
    )
    parser.add_argument('--eval-fixed-seed', type=int, default=SetupParameters.EVAL_FIXED_SEED)
    parser.add_argument('--eval-seed-offset', type=int, default=0)
    parser.add_argument(
        '--hrl-step-trace',
        action='store_true',
        help="记录HRL顶层逐步决策trace到stats raw.episode_hrl_step_trace",
    )
    parser.add_argument('--hrl-step-trace-max-steps', type=int, default=0)
    
    args = parser.parse_args()
    if args.hrl_step_trace:
        os.environ['VS_HRL_STEP_TRACE_ENABLE'] = '1'
        os.environ['VS_HRL_STEP_TRACE_MAX_STEPS'] = str(max(0, int(args.hrl_step_trace_max_steps)))
    traj_png_count = max(0, int(args.traj_png_count))
    cli_gif_episodes = max(0, int(args.gif_episodes))
    attacker_strategy_params = None

    # 默认进入交互式界面，除非指定 --no-interactive 或 --suite
    if args.suite:
        # Default fixed suite
        default_suite = [
            {'defender': 'chase', 'attacker': 'all'},
            {'defender': 'protect', 'attacker': 'all'},
            {'defender': 'hrl', 'attacker': 'all'},
        ]
        run_suite(
            default_suite,
            global_episodes=args.episodes,
            save_traj_png=traj_png_count > 0,
            traj_png_count=traj_png_count,
            hrl_num_skills=args.hrl_num_skills,
            hrl_protect_skill_path=args.hrl_protect_skill_path,
            hrl_chase_skill_path=args.hrl_chase_skill_path,
            use_privileged_classifier=args.use_privileged_classifier,
        )
    elif args.no_interactive:
        defenders = _parse_defenders_arg(args.defender, args.defenders)
        if not defenders:
            raise ValueError("至少需要一个防御者策略")
        supported_defenders = sorted(set(TEST_DEFENDER_STRATEGIES + ALL_RULE_DEFENDER_STRATEGIES))
        unsupported = [d for d in defenders if d not in supported_defenders]
        if unsupported:
            raise ValueError(f"当前评估支持策略: {supported_defenders}; 非法项: {unsupported}")
        ckpt_required_defenders = set(RL_DEFENDER_STRATEGIES)

        if len(defenders) > 1:
            if args.checkpoint:
                raise ValueError("多防御者模式不支持单一 --checkpoint，请去掉后使用各自默认checkpoint")
            multi_suite = []
            for d in defenders:
                cfg = {
                    'defender': d,
                    'attacker': args.attacker,
                    'episodes': args.episodes,
                    'attacker_strategy_params': attacker_strategy_params,
                }
                if args.attacker_checkpoint:
                    cfg['attacker_checkpoint'] = args.attacker_checkpoint
                ckpt = _default_model_path(d)
                if ckpt:
                    cfg['checkpoint'] = ckpt
                multi_suite.append(cfg)
            run_suite(
                multi_suite,
                global_episodes=args.episodes,
                gif_episodes=cli_gif_episodes if args.gif else 0,
                save_traj_png=traj_png_count > 0,
                traj_png_count=traj_png_count,
                hrl_num_skills=args.hrl_num_skills,
                hrl_protect_skill_path=args.hrl_protect_skill_path,
                hrl_chase_skill_path=args.hrl_chase_skill_path,
                use_privileged_classifier=args.use_privileged_classifier,
            )
            return

        defender = defenders[0]

        if args.attacker == 'all':
            ckpt_for_all = args.checkpoint or _default_model_path(defender)
            if defender in ckpt_required_defenders and not ckpt_for_all:
                raise ValueError("attacker=all 时必须通过 --checkpoint 指定模型路径")
            print(
                f"[CLI] attacker=all 将展开为{len(ALL_ATTACKER_STRATEGIES)}种策略"
                f"（不含比例切换），每种使用 --episodes"
            )
            run_suite(
                [{
                    'defender': defender,
                    'attacker': 'all',
                    'episodes': args.episodes,
                    'checkpoint': ckpt_for_all,
                    'attacker_strategy_params': attacker_strategy_params,
                    'attacker_checkpoint': args.attacker_checkpoint,
                }],
                global_episodes=args.episodes,
                gif_episodes=cli_gif_episodes if args.gif else 0,
                save_traj_png=traj_png_count > 0,
                traj_png_count=traj_png_count,
                hrl_num_skills=args.hrl_num_skills,
                hrl_protect_skill_path=args.hrl_protect_skill_path,
                hrl_chase_skill_path=args.hrl_chase_skill_path,
                use_privileged_classifier=args.use_privileged_classifier,
            )
            return

        # 单次运行模式（命令行参数）
        ckpt = args.checkpoint
        if not ckpt:
            ckpt = _default_model_path(defender)

        old_eval_use_random_seed = SetupParameters.EVAL_USE_RANDOM_SEED
        old_eval_fixed_seed = SetupParameters.EVAL_FIXED_SEED
        if args.eval_deterministic_seeds:
            SetupParameters.EVAL_USE_RANDOM_SEED = False
            SetupParameters.EVAL_FIXED_SEED = int(args.eval_fixed_seed)

        metrics, _ = run_evaluation(
            defender_strategy=defender,
            attacker_strategy=args.attacker,
            attacker_strategy_params=attacker_strategy_params,
            num_episodes=args.episodes,
            defender_checkpoint=ckpt,
            attacker_checkpoint=args.attacker_checkpoint,
            save_gif=True if args.gif else False,
            gif_episodes=cli_gif_episodes if args.gif else 0,
            save_traj_png=traj_png_count > 0,
            traj_png_count=traj_png_count,
            save_stats=args.save_stats,
            stats_path=args.stats_path,
            network_type=args.network_type,
            force_serial=args.serial,
            hrl_num_skills=args.hrl_num_skills,
            hrl_protect_skill_path=args.hrl_protect_skill_path,
            hrl_chase_skill_path=args.hrl_chase_skill_path,
            use_privileged_classifier=args.use_privileged_classifier,
            seed_offset=int(args.eval_seed_offset),
        )
        SetupParameters.EVAL_USE_RANDOM_SEED = old_eval_use_random_seed
        SetupParameters.EVAL_FIXED_SEED = old_eval_fixed_seed
        print(
            f"[EVAL RESULT] "
            f"success={_fmt_metric_ci_pm_from_row(metrics, 'success_rate', scale=100.0, decimals=1, suffix='%')}, "
            f"a_win={_fmt_metric_ci_pm_from_row(metrics, 'attacker_win_rate', scale=100.0, decimals=1, suffix='%')}, "
            f"draw={_fmt_metric_ci_pm_from_row(metrics, 'draw_rate', scale=100.0, decimals=1, suffix='%')}, "
            f"d_cap={_fmt_metric_ci_pm_from_row(metrics, 'defender_capture_rate', scale=100.0, decimals=1, suffix='%')}, "
            f"a_cap={_fmt_metric_ci_pm_from_row(metrics, 'attacker_capture_rate', scale=100.0, decimals=1, suffix='%')}, "
            f"d_col={_fmt_metric_ci_pm_from_row(metrics, 'defender_collision_rate', scale=100.0, decimals=1, suffix='%')}, "
            f"mean_len={_fmt_metric_pm_from_row(metrics, 'mean_episode_length', decimals=1)}, "
            f"path_len={_fmt_metric_pm(metrics['defender_path_length_mean'], metrics['defender_path_length_std'], decimals=1)}"
        )
        if _is_hrl_like_strategy(defender) and metrics['hrl_protect_selection_rate'] is not None:
            print(
                f"[EVAL HRL SKILL] "
                f"protect={_fmt_metric_pm_from_row(metrics, 'hrl_protect_selection_rate', scale=100.0, decimals=1, suffix='%')}, "
                f"chase={_fmt_metric_pm_from_row(metrics, 'hrl_chase_selection_rate', scale=100.0, decimals=1, suffix='%')}"
            )
        if ray.is_initialized():
            ray.shutdown()
    else:
        # 默认进入交互式界面
        interactive_suite_mode()

if __name__ == "__main__":
    main()
