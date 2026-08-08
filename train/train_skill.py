"""
TAD PPO Driver - 主训练循环

支持模式:
- 'rl': 纯强化学习
- 'mixed': 专家监督+RL混合训练，监督权重使用余弦退火
"""

import json
import os
import os.path as osp

# 关键：必须在import torch之前设置CUDA_VISIBLE_DEVICES
# 从配置文件读取GPU_ID（这里用简单方式避免循环导入）
_gpu_id_file = osp.join(osp.dirname(osp.dirname(__file__)), 'configs', 'skill_config.py')
_gpu_id = 1  # 默认值
try:
    with open(_gpu_id_file, 'r') as f:
        for line in f:
            if 'GPU_ID' in line and '=' in line and not line.strip().startswith('#'):
                _gpu_id = int(line.split('=')[1].strip().split('#')[0].strip())
                break
except:
    pass
if os.environ.get('SKILL_GPU_ID') is not None:
    _gpu_id = int(os.environ['SKILL_GPU_ID'])
os.environ['CUDA_VISIBLE_DEVICES'] = str(_gpu_id)
print(f"[GPU] CUDA_VISIBLE_DEVICES={_gpu_id} (在import torch之前设置)")

import time
import math
import numpy as np
import torch
import ray
import sys
from datetime import datetime
from typing import Dict, List, Optional

# Add parent directory to sys.path to allow running as a script
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from configs.skill_config import SetupParameters, TrainingParameters, RecordingParameters
from configs.skill_config import NetParameters
from configs import map_config
from attacker.frozen_pool import PROGRAMMATIC_ATTACKER_ALIASES
from skill.model import Model
from skill.runner import Runner, resolve_training_mode
from skill.util import (
    configure_cpu_runtime,
    get_adjusted_n_envs,
    get_device,
    get_num_gpus,
    get_ray_temp_dir,
    make_gif,
    make_trajectory_plot,
    print_device_info,
    print_ram_info,
    set_global_seeds,
    write_to_tensorboard,
)
from utils.process_info import print_training_process_info


BASELINE_ENV_CONFIG = {
    'reward_mode': 'baseline',
    'episode_len': 300,
}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == '':
        return int(default)
    return int(float(raw))


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == '':
        return float(default)
    return float(raw)


def _env_float_pair(name: str, default):
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == '':
        return default
    parts = [p.strip() for p in str(raw).replace(';', ',').split(',') if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"{name} must contain two comma-separated floats, got {raw!r}")
    return (float(parts[0]), float(parts[1]))


def _env_float_list(name: str, default=None):
    raw = os.environ.get(name)
    if raw is None:
        return default
    text = str(raw).strip()
    if not text:
        return default
    values = []
    for token in text.split(','):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    return values if values else default


def _env_float_matrix(name: str, default=None):
    raw = os.environ.get(name)
    if raw is None:
        return default
    text = str(raw).strip()
    if not text:
        return default
    rows = []
    for row_text in text.split(';'):
        row_text = row_text.strip()
        if not row_text:
            continue
        row = [float(token.strip()) for token in row_text.split(',') if token.strip()]
        if row:
            rows.append(row)
    return rows if rows else default


def _env_str_list(name: str, default=None):
    raw = os.environ.get(name)
    if raw is None:
        return default
    values = [item.strip() for item in str(raw).split(',') if item.strip()]
    return values if values else default


def _env_str(name: str, default: str):
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == '':
        return default
    return str(raw).strip()


def _env_json_object(name: str, default=None):
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return dict(default or {})
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must contain valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _apply_dual_gru_log_std_schedule(step: int):
    """Optionally anneal dual-GRU policy std cap during pure-RL experiments."""
    if not bool(getattr(NetParameters, "NMN_DUAL_GRU_LOG_STD_ANNEAL_ENABLE", False)):
        return None
    start = int(getattr(NetParameters, "NMN_DUAL_GRU_LOG_STD_ANNEAL_START", 0))
    end = int(getattr(NetParameters, "NMN_DUAL_GRU_LOG_STD_ANNEAL_END", start))
    initial_max = float(getattr(NetParameters, "_NMN_DUAL_GRU_LOG_STD_ANNEAL_INITIAL_MAX",
                                getattr(NetParameters, "NMN_DUAL_GRU_MAX_LOG_STD", 2.0)))
    final_max = float(getattr(NetParameters, "NMN_DUAL_GRU_LOG_STD_ANNEAL_FINAL_MAX", initial_max))
    if end <= start:
        progress = 1.0 if step >= end else 0.0
    else:
        progress = min(1.0, max(0.0, (float(step) - float(start)) / float(end - start)))
    max_log_std = initial_max + (final_max - initial_max) * progress
    NetParameters.NMN_DUAL_GRU_MAX_LOG_STD = float(max_log_std)
    return float(max_log_std)


def _sync_dual_gru_log_std_bounds(network, max_log_std=None):
    """Apply launch-time or scheduled dual-GRU log_std bounds to a network."""
    if not (
        hasattr(network, "log_std")
        and (hasattr(network, "min_log_std") or hasattr(network, "max_log_std"))
    ):
        return None
    min_bound = float(getattr(NetParameters, "NMN_DUAL_GRU_MIN_LOG_STD", -20.0))
    max_bound = float(
        getattr(NetParameters, "NMN_DUAL_GRU_MAX_LOG_STD", 2.0)
        if max_log_std is None else max_log_std
    )
    if min_bound > max_bound:
        min_bound, max_bound = max_bound, min_bound
    if hasattr(network, "min_log_std"):
        network.min_log_std = min_bound
    if hasattr(network, "max_log_std"):
        network.max_log_std = max_bound
    return min_bound, max_bound


def _apply_env_overrides():
    """Optional launch-time overrides for isolated skill experiments."""
    SetupParameters.SEED = _env_int("SKILL_SEED", SetupParameters.SEED)
    SetupParameters.TRAIN_USE_RANDOM_SEED = _env_bool(
        "SKILL_TRAIN_USE_RANDOM_SEED",
        SetupParameters.TRAIN_USE_RANDOM_SEED,
    )
    SetupParameters.EVAL_USE_RANDOM_SEED = _env_bool(
        "SKILL_EVAL_USE_RANDOM_SEED",
        SetupParameters.EVAL_USE_RANDOM_SEED,
    )
    SetupParameters.EVAL_FIXED_SEED = _env_int(
        "SKILL_EVAL_FIXED_SEED",
        SetupParameters.EVAL_FIXED_SEED,
    )
    SetupParameters.SKILL_MODE = os.environ.get("SKILL_MODE", SetupParameters.SKILL_MODE)
    SetupParameters.BOTTOM_NETWORK_TYPE = os.environ.get(
        "BOTTOM_NETWORK_TYPE",
        SetupParameters.BOTTOM_NETWORK_TYPE,
    )
    if os.environ.get("TRAIN_ATTACKER_STRATEGIES"):
        SetupParameters.TRAIN_ATTACKER_STRATEGIES = [
            item.strip()
            for item in os.environ["TRAIN_ATTACKER_STRATEGIES"].split(',')
            if item.strip()
        ]
    if os.environ.get("TRAIN_ATTACKER_STRATEGY_WEIGHTS") is not None:
        SetupParameters.TRAIN_ATTACKER_STRATEGY_WEIGHTS = _env_float_list(
            "TRAIN_ATTACKER_STRATEGY_WEIGHTS",
            getattr(SetupParameters, "TRAIN_ATTACKER_STRATEGY_WEIGHTS", None),
        )
    if os.environ.get("TRAIN_ATTACKER_STRATEGY_WEIGHT_SCHEDULE") is not None:
        SetupParameters.TRAIN_ATTACKER_STRATEGY_WEIGHT_SCHEDULE = _env_float_matrix(
            "TRAIN_ATTACKER_STRATEGY_WEIGHT_SCHEDULE",
            getattr(SetupParameters, "TRAIN_ATTACKER_STRATEGY_WEIGHT_SCHEDULE", None),
        )
    if os.environ.get("TRAIN_ATTACKER_STRATEGY_WEIGHT_SWITCH_STEPS") is not None:
        SetupParameters.TRAIN_ATTACKER_STRATEGY_WEIGHT_SWITCH_STEPS = _env_float_list(
            "TRAIN_ATTACKER_STRATEGY_WEIGHT_SWITCH_STEPS",
            getattr(SetupParameters, "TRAIN_ATTACKER_STRATEGY_WEIGHT_SWITCH_STEPS", None),
        )
    if os.environ.get("TRAIN_LEARNED_ATTACKER_SPECS") is not None:
        SetupParameters.TRAIN_LEARNED_ATTACKER_SPECS = _env_json_object(
            "TRAIN_LEARNED_ATTACKER_SPECS",
            getattr(SetupParameters, "TRAIN_LEARNED_ATTACKER_SPECS", {}),
        )
    SetupParameters.TRAIN_LEARNED_ATTACKER_ALIAS = _env_str(
        "TRAIN_LEARNED_ATTACKER_ALIAS",
        getattr(SetupParameters, "TRAIN_LEARNED_ATTACKER_ALIAS", "attacker_rl"),
    )

    TrainingParameters.lr = _env_float("SKILL_LR", TrainingParameters.lr)
    TrainingParameters.LR_FINAL = _env_float("SKILL_LR_FINAL", TrainingParameters.LR_FINAL)
    TrainingParameters.LR_SCHEDULE = os.environ.get("SKILL_LR_SCHEDULE", TrainingParameters.LR_SCHEDULE)
    TrainingParameters.N_ENVS = _env_int("SKILL_N_ENVS", TrainingParameters.N_ENVS)
    TrainingParameters.N_STEPS = _env_int("SKILL_N_STEPS", TrainingParameters.N_STEPS)
    TrainingParameters.N_MAX_STEPS = _env_float("SKILL_N_MAX_STEPS", TrainingParameters.N_MAX_STEPS)
    TrainingParameters.LOG_EPOCH_STEPS = _env_int("SKILL_LOG_EPOCH_STEPS", TrainingParameters.LOG_EPOCH_STEPS)
    TrainingParameters.MINIBATCH_SIZE = _env_int("SKILL_MINIBATCH_SIZE", TrainingParameters.MINIBATCH_SIZE)
    TrainingParameters.N_EPOCHS = _env_int("SKILL_N_EPOCHS", TrainingParameters.N_EPOCHS)
    TrainingParameters.TBPTT_STEPS = _env_int("SKILL_TBPTT_STEPS", TrainingParameters.TBPTT_STEPS)
    TrainingParameters.ENTROPY_COEF = _env_float("SKILL_ENTROPY_COEF", TrainingParameters.ENTROPY_COEF)
    TrainingParameters.GAMMA = _env_float("SKILL_GAMMA", TrainingParameters.GAMMA)
    TrainingParameters.LAM = _env_float("SKILL_LAM", TrainingParameters.LAM)
    TrainingParameters.CLIP_RANGE = _env_float(
        "SKILL_CLIP_RANGE",
        TrainingParameters.CLIP_RANGE,
    )
    TrainingParameters.VALUE_CLIP_RANGE = _env_float(
        "SKILL_VALUE_CLIP_RANGE",
        TrainingParameters.VALUE_CLIP_RANGE,
    )
    TrainingParameters.ADV_ACTION_BC_COEF = _env_float(
        "SKILL_ADV_ACTION_BC_COEF",
        getattr(TrainingParameters, "ADV_ACTION_BC_COEF", 0.0),
    )
    TrainingParameters.ADV_ACTION_BC_MAX_WEIGHT = _env_float(
        "SKILL_ADV_ACTION_BC_MAX_WEIGHT",
        getattr(TrainingParameters, "ADV_ACTION_BC_MAX_WEIGHT", 3.0),
    )
    TrainingParameters.EX_VALUE_COEF = _env_float("SKILL_VALUE_COEF", TrainingParameters.EX_VALUE_COEF)
    TrainingParameters.MAX_GRAD_NORM = _env_float("SKILL_MAX_GRAD_NORM", TrainingParameters.MAX_GRAD_NORM)
    TrainingParameters.BEST_MODEL_METRIC = _env_str(
        "SKILL_BEST_MODEL_METRIC",
        getattr(TrainingParameters, "BEST_MODEL_METRIC", "reward"),
    )
    TrainingParameters.REWARD_NORMALIZATION = _env_bool(
        "SKILL_REWARD_NORMALIZATION",
        TrainingParameters.REWARD_NORMALIZATION,
    )
    TrainingParameters.EARLY_STOP_ENABLED = _env_bool(
        "SKILL_EARLY_STOP_ENABLED",
        bool(getattr(TrainingParameters, "EARLY_STOP_ENABLED", False)),
    )
    TrainingParameters.EARLY_STOP_MIN_STEPS = _env_int(
        "SKILL_EARLY_STOP_MIN_STEPS",
        int(getattr(TrainingParameters, "EARLY_STOP_MIN_STEPS", 0)),
    )
    TrainingParameters.EARLY_STOP_PATIENCE = _env_int(
        "SKILL_EARLY_STOP_PATIENCE",
        int(getattr(TrainingParameters, "EARLY_STOP_PATIENCE", 5)),
    )
    TrainingParameters.EARLY_STOP_MIN_DELTA = _env_float(
        "SKILL_EARLY_STOP_MIN_DELTA",
        float(getattr(TrainingParameters, "EARLY_STOP_MIN_DELTA", 0.0)),
    )
    TrainingParameters.POLICY_ANCHOR_ENABLE = _env_bool("SKILL_POLICY_ANCHOR_ENABLE", False)
    TrainingParameters.POLICY_ANCHOR_COEF = _env_float("SKILL_POLICY_ANCHOR_COEF", 0.0)

    if os.environ.get("NMN_DUAL_GRU_INITIAL_LOG_STD") is not None:
        NetParameters.NMN_DUAL_GRU_INITIAL_LOG_STD = _env_float(
            "NMN_DUAL_GRU_INITIAL_LOG_STD",
            NetParameters.NMN_DUAL_GRU_INITIAL_LOG_STD,
        )
    if os.environ.get("NMN_DUAL_GRU_MIN_LOG_STD") is not None:
        NetParameters.NMN_DUAL_GRU_MIN_LOG_STD = _env_float(
            "NMN_DUAL_GRU_MIN_LOG_STD",
            NetParameters.NMN_DUAL_GRU_MIN_LOG_STD,
        )
    if os.environ.get("NMN_DUAL_GRU_MAX_LOG_STD") is not None:
        NetParameters.NMN_DUAL_GRU_MAX_LOG_STD = _env_float(
            "NMN_DUAL_GRU_MAX_LOG_STD",
            NetParameters.NMN_DUAL_GRU_MAX_LOG_STD,
        )
    NetParameters._NMN_DUAL_GRU_LOG_STD_ANNEAL_INITIAL_MAX = float(NetParameters.NMN_DUAL_GRU_MAX_LOG_STD)
    NetParameters.NMN_DUAL_GRU_LOG_STD_ANNEAL_ENABLE = _env_bool(
        "NMN_DUAL_GRU_LOG_STD_ANNEAL_ENABLE",
        getattr(NetParameters, "NMN_DUAL_GRU_LOG_STD_ANNEAL_ENABLE", False),
    )
    NetParameters.NMN_DUAL_GRU_LOG_STD_ANNEAL_START = _env_int(
        "NMN_DUAL_GRU_LOG_STD_ANNEAL_START",
        getattr(NetParameters, "NMN_DUAL_GRU_LOG_STD_ANNEAL_START", 0),
    )
    NetParameters.NMN_DUAL_GRU_LOG_STD_ANNEAL_END = _env_int(
        "NMN_DUAL_GRU_LOG_STD_ANNEAL_END",
        getattr(NetParameters, "NMN_DUAL_GRU_LOG_STD_ANNEAL_END", 0),
    )
    NetParameters.NMN_DUAL_GRU_LOG_STD_ANNEAL_FINAL_MAX = _env_float(
        "NMN_DUAL_GRU_LOG_STD_ANNEAL_FINAL_MAX",
        getattr(NetParameters, "NMN_DUAL_GRU_LOG_STD_ANNEAL_FINAL_MAX", NetParameters.NMN_DUAL_GRU_MAX_LOG_STD),
    )
    NetParameters.NMN_DUAL_GRU_INITIAL_MEAN_BIAS = _env_float_pair(
        "NMN_DUAL_GRU_INITIAL_MEAN_BIAS",
        NetParameters.NMN_DUAL_GRU_INITIAL_MEAN_BIAS,
    )
    if os.environ.get("NMN_DUAL_GRU_POLICY_HEAD_GAIN") is not None:
        NetParameters.NMN_DUAL_GRU_POLICY_HEAD_GAIN = _env_float(
            "NMN_DUAL_GRU_POLICY_HEAD_GAIN",
            NetParameters.NMN_DUAL_GRU_POLICY_HEAD_GAIN,
        )
    NetParameters.NMN_DUAL_GRU_RESIDUAL_WARMUP_STEPS = _env_int(
        "NMN_DUAL_GRU_RESIDUAL_WARMUP_STEPS",
        getattr(NetParameters, "NMN_DUAL_GRU_RESIDUAL_WARMUP_STEPS", 0),
    )

    if os.environ.get("SKILL_MODEL_PATH"):
        RecordingParameters.MODEL_PATH = os.environ["SKILL_MODEL_PATH"]
        RecordingParameters.SUMMARY_PATH = osp.join(RecordingParameters.MODEL_PATH, "summary")
        RecordingParameters.GIFS_PATH = osp.join(RecordingParameters.MODEL_PATH, "gifs")
    RecordingParameters.RETRAIN = _env_bool("SKILL_RETRAIN", getattr(RecordingParameters, "RETRAIN", False))
    RecordingParameters.FRESH_RETRAIN = _env_bool(
        "SKILL_FRESH_RETRAIN",
        getattr(RecordingParameters, "FRESH_RETRAIN", False),
    )
    if os.environ.get("SKILL_RESTORE_DIR"):
        RecordingParameters.RESTORE_DIR = os.environ["SKILL_RESTORE_DIR"]
    RecordingParameters.TENSORBOARD = _env_bool("SKILL_TENSORBOARD", RecordingParameters.TENSORBOARD)
    RecordingParameters.EVAL_INTERVAL = _env_int("SKILL_EVAL_INTERVAL", RecordingParameters.EVAL_INTERVAL)
    RecordingParameters.SAVE_INTERVAL = _env_int("SKILL_SAVE_INTERVAL", RecordingParameters.SAVE_INTERVAL)
    RecordingParameters.GIF_INTERVAL = _env_int("SKILL_GIF_INTERVAL", RecordingParameters.GIF_INTERVAL)
    RecordingParameters.TRAJ_INTERVAL = _env_int("SKILL_TRAJ_INTERVAL", RecordingParameters.TRAJ_INTERVAL)
    RecordingParameters.EVAL_EPISODES = _env_int("SKILL_EVAL_EPISODES", RecordingParameters.EVAL_EPISODES)
    TrainingParameters.BALANCED_EVAL_ATTACKERS = _env_str_list(
        "SKILL_BALANCED_EVAL_ATTACKERS",
        getattr(TrainingParameters, "BALANCED_EVAL_ATTACKERS", ()),
    )
    TrainingParameters.BALANCED_EVAL_EPISODES = _env_int(
        "SKILL_BALANCED_EVAL_EPISODES",
        int(getattr(TrainingParameters, "BALANCED_EVAL_EPISODES", 0)),
    )
    TrainingParameters.BALANCED_EVAL_METRIC = _env_str(
        "SKILL_BALANCED_EVAL_METRIC",
        str(getattr(TrainingParameters, "BALANCED_EVAL_METRIC", "mean_win")),
    )


def _resolve_bottom_network_type() -> str:
    network_type = str(getattr(SetupParameters, 'BOTTOM_NETWORK_TYPE', 'nmn')).strip().lower()
    if network_type not in (
        'nmn', 'nmn_ctde', 'nmn_ctde_shared', 'nmn_ctde_task_shared',
        'nmn_ctde_task_shared_distill', 'nmn_no_shared_radar', 'nmn_dual_gru_raw',
        'nmn_dual_gru_raw_ctde', 'nmn_gru', 'mlp', 'mlp_ctde', 'mlp_gru', 'mlp_noctde',
    ):
        raise ValueError(
            "BOTTOM_NETWORK_TYPE must be one of "
            "['nmn', 'nmn_ctde', 'nmn_ctde_shared', 'nmn_ctde_task_shared', "
            "'nmn_ctde_task_shared_distill', 'nmn_no_shared_radar', 'nmn_dual_gru_raw', "
            "'nmn_dual_gru_raw_ctde', 'nmn_gru', "
            "'mlp', 'mlp_ctde', 'mlp_gru', 'mlp_noctde'], "
            f"got {network_type!r}"
        )
    return 'mlp_noctde' if network_type == 'mlp' else network_type


def _sync_dual_gru_residual_training(network, step: int):
    warmup_steps = int(getattr(NetParameters, "NMN_DUAL_GRU_RESIDUAL_WARMUP_STEPS", 0) or 0)
    if warmup_steps <= 0 or not hasattr(network, "set_residual_training_enabled"):
        return None
    enabled = int(step) >= warmup_steps
    previous = getattr(network, "_residual_training_enabled", None)
    if previous is None or bool(previous) != bool(enabled):
        network.set_residual_training_enabled(enabled)
        network._residual_training_enabled = bool(enabled)
        state = "enabled" if enabled else "frozen"
        print(f"[DualGRUResidual] residual branch {state} at step {int(step):,} (warmup={warmup_steps:,})")
    return bool(enabled)


def cosine_anneal_il_weight(current_step: int) -> float:
    """
    计算当前步数对应的IL权重（余弦退火）
    
    Args:
        current_step: 当前训练步数
        
    Returns:
        IL权重，范围 [IL_FINAL_WEIGHT, IL_INITIAL_WEIGHT]
    """
    initial = TrainingParameters.IL_INITIAL_WEIGHT
    final = TrainingParameters.IL_FINAL_WEIGHT
    anneal_steps = TrainingParameters.IL_ANNEAL_STEPS
    
    if current_step >= anneal_steps:
        return final
    
    # 余弦退火: weight = final + (initial - final) * 0.5 * (1 + cos(pi * step / anneal_steps))
    progress = current_step / anneal_steps
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    weight = final + (initial - final) * cosine_decay
    
    return weight


def _resolve_learning_rate(current_step: int, total_steps: int) -> float:
    """Resolve the optimizer LR for skill PPO from TrainingParameters."""
    initial = float(getattr(TrainingParameters, 'lr', 5e-4))
    final = float(getattr(TrainingParameters, 'LR_FINAL', initial))
    schedule = str(getattr(TrainingParameters, 'LR_SCHEDULE', 'constant')).strip().lower()
    if schedule == 'constant' or total_steps <= 0:
        return initial

    progress = min(1.0, max(0.0, float(current_step) / float(total_steps)))
    if schedule == 'linear':
        return initial + (final - initial) * progress
    if schedule == 'cosine':
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final + (initial - final) * cosine_decay

    raise ValueError(
        "TrainingParameters.LR_SCHEDULE must be one of "
        "['constant', 'linear', 'cosine'], "
        f"got {schedule!r}"
    )


def main():
    _apply_env_overrides()
    _apply_dual_gru_log_std_schedule(0)
    training_mode = resolve_training_mode(TrainingParameters.TRAINING_MODE)
    TrainingParameters.TRAINING_MODE = training_mode

    skill_mode = str(getattr(SetupParameters, 'SKILL_MODE', '')).strip().lower()
    is_baseline_mode = skill_mode == 'baseline'

    model_network_type = _resolve_bottom_network_type()
    is_nmn_network = model_network_type in (
        'nmn', 'nmn_ctde', 'nmn_ctde_shared', 'nmn_ctde_task_shared',
        'nmn_ctde_task_shared_distill', 'nmn_no_shared_radar', 'nmn_dual_gru_raw',
        'nmn_dual_gru_raw_ctde', 'nmn_gru', 'mlp_gru',
    )
    print_training_process_info("train_skill")
    nmn_cl_enabled = bool(
        is_nmn_network and getattr(TrainingParameters, 'ENABLE_NMN_CL', True)
    )
    nmn_cl_stage1_steps = int(getattr(TrainingParameters, 'NMN_CL_STAGE1_STEPS', 0))
    nmn_cl_stage2_steps = int(getattr(TrainingParameters, 'NMN_CL_STAGE2_STEPS', 0))
    nmn_cl_start_stage = int(getattr(TrainingParameters, 'NMN_CL_START_STAGE', 1))
    nmn_cl_stage1_density = str(
        getattr(TrainingParameters, 'NMN_CL_STAGE1_OBSTACLE_DENSITY', 'none')
    ).strip().lower()
    nmn_cl_stage2_density = str(
        getattr(
            TrainingParameters,
            'NMN_CL_STAGE2_OBSTACLE_DENSITY',
            getattr(SetupParameters, 'OBSTACLE_DENSITY', 'dense'),
        )
    ).strip().lower()
    if nmn_cl_enabled:
        if nmn_cl_start_stage not in (1, 2):
            raise ValueError(f"NMN_CL_START_STAGE must be 1 or 2, got {nmn_cl_start_stage!r}")
        if nmn_cl_stage1_steps <= 0 or nmn_cl_stage2_steps <= 0:
            raise ValueError(
                "NMN-CL requires positive NMN_CL_STAGE1_STEPS and NMN_CL_STAGE2_STEPS."
            )

    runner_env_cfg = {
        'skill_mode': skill_mode,
        'expert_skill_mode': skill_mode,
        'training_mode': training_mode,
        'seed': int(SetupParameters.SEED),
        'train_use_random_seed': bool(SetupParameters.TRAIN_USE_RANDOM_SEED),
        'eval_use_random_seed': bool(SetupParameters.EVAL_USE_RANDOM_SEED),
        'eval_fixed_seed': int(SetupParameters.EVAL_FIXED_SEED),
    }
    if is_baseline_mode:
        runner_env_cfg.update(BASELINE_ENV_CONFIG)
    else:
        runner_env_cfg['reward_mode'] = skill_mode
    if getattr(SetupParameters, 'TRAIN_ATTACKER_STRATEGIES', None):
        runner_env_cfg['attacker_strategy_pool'] = getattr(SetupParameters, 'TRAIN_ATTACKER_STRATEGIES')
    if getattr(SetupParameters, 'TRAIN_ATTACKER_STRATEGY_WEIGHTS', None) is not None:
        runner_env_cfg['attacker_strategy_pool_weights'] = getattr(SetupParameters, 'TRAIN_ATTACKER_STRATEGY_WEIGHTS')
    runner_env_cfg['learned_attacker_alias'] = getattr(
        SetupParameters,
        'TRAIN_LEARNED_ATTACKER_ALIAS',
        'attacker_rl',
    )
    if getattr(SetupParameters, 'TRAIN_LEARNED_ATTACKER_SPECS', None):
        runner_env_cfg['learned_attacker_specs'] = getattr(
            SetupParameters,
            'TRAIN_LEARNED_ATTACKER_SPECS',
        )

    set_global_seeds(SetupParameters.SEED)
    
    # 打印设备信息
    print_device_info()
    
    # =========== RETRAIN / FRESH_RETRAIN 加载逻辑 ===========
    model_dict = None
    fresh_retrain = getattr(RecordingParameters, 'FRESH_RETRAIN', False)
    retrain = getattr(RecordingParameters, 'RETRAIN', False)
    
    if retrain or fresh_retrain:
        checkpoint_path = getattr(RecordingParameters, 'RESTORE_DIR', None)
        if checkpoint_path and osp.exists(checkpoint_path):
            model_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if fresh_retrain:
                print(f"[FRESH_RETRAIN] Loaded model weights from {checkpoint_path}, resetting training progress")
            else:
                print(f"[RETRAIN] Loaded checkpoint from {checkpoint_path}")
        else:
            print(f"[WARNING] RESTORE_DIR not found or not set: {checkpoint_path}")
    # =========================================================
    
    timestamp = datetime.now().strftime('%m-%d-%H-%M')
    run_name = f"{SetupParameters.SKILL_MODE}_{training_mode}_{timestamp}"
    
    # 直接使用 MODEL_PATH，不再嵌套子目录
    # 路径结构: models/defender_xxx/
    model_dir = RecordingParameters.MODEL_PATH
    gif_dir = osp.join(RecordingParameters.MODEL_PATH, 'gifs')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(gif_dir, exist_ok=True)

    def _existing_best_metric(filename: str, fields) -> float:
        if not retrain:
            return -float('inf')
        path = osp.join(model_dir, filename)
        if not osp.isfile(path):
            return -float('inf')
        try:
            payload = torch.load(path, map_location='cpu', weights_only=False)
            for field in fields:
                value = payload.get(field) if isinstance(payload, dict) else None
                if value is not None:
                    return float(value)
        except Exception as exc:
            print(f"[RETRAIN] Could not read existing best metric from {path}: {exc}")
        return -float('inf')

    # A resumed process must retain the selection thresholds established by
    # the earlier segment.  Otherwise the first post-resume evaluation would
    # overwrite best_win/best_reward/best_balanced even when it is worse.
    main._best_eval_win = _existing_best_metric(
        'best_win_model.pth', ('eval_win', 'reward')
    )
    main._best_eval_reward = _existing_best_metric(
        'best_reward_model.pth', ('eval_reward', 'reward')
    )
    main._best_balanced_eval_metric = _existing_best_metric(
        'best_balanced_model.pth', ('reward',)
    )
    if retrain:
        print(
            "[RETRAIN] Existing selection thresholds: "
            f"win={main._best_eval_win:.4f}, "
            f"reward={main._best_eval_reward:.4f}, "
            f"balanced={main._best_balanced_eval_metric:.4f}"
        )
    
    summary_writer = None
    if RecordingParameters.TENSORBOARD:
        from torch.utils.tensorboard import SummaryWriter
        log_dir = RecordingParameters.SUMMARY_PATH
        os.makedirs(log_dir, exist_ok=True)
        summary_writer = SummaryWriter(log_dir)
    
    # 使用安全的GPU检测
    print_ram_info()
    
    num_gpus = get_num_gpus()
    device = get_device(prefer_gpu=True)
    is_cpu_training = device.type == 'cpu'

    # 根据RAM调整并行环境数量；CPU-only机器禁用自动放大，并加资源上限保护。
    n_envs = get_adjusted_n_envs(
        TrainingParameters.N_ENVS,
        allow_scale_up=not is_cpu_training,
    )
    if is_cpu_training:
        configure_cpu_runtime(n_envs)

    if model_dict and not fresh_retrain:
        # RETRAIN: 恢复训练进度 (兼容旧格式)
        if isinstance(model_dict, dict) and 'step' in model_dict:
            global_step = int(model_dict.get('step', 0))
            best_reward = float(model_dict.get('reward', -float('inf')))
            print(f"[RETRAIN] Resuming from step {global_step:,}, best_reward {best_reward:.2f}")
        else:
            # 旧格式没有进度信息，从头开始
            global_step = 0
            best_reward = -float('inf')
            print("[RETRAIN] Old checkpoint format, starting from step 0")
    else:
        # 新训练或 FRESH_RETRAIN: 从头开始
        global_step = 0
        best_reward = -float('inf')

    nmn_cl_run_start_stage = nmn_cl_start_stage
    current_nmn_stage = None
    if nmn_cl_enabled:
        if model_dict and not fresh_retrain:
            saved_start_stage = model_dict.get('nmn_cl_start_stage', nmn_cl_start_stage)
            try:
                saved_start_stage = int(saved_start_stage)
            except (TypeError, ValueError):
                saved_start_stage = nmn_cl_start_stage
            if saved_start_stage in (1, 2):
                nmn_cl_run_start_stage = saved_start_stage

            saved_stage = model_dict.get('nmn_cl_stage', None)
            try:
                current_nmn_stage = int(saved_stage) if saved_stage is not None else None
            except (TypeError, ValueError):
                current_nmn_stage = None

        if current_nmn_stage not in (1, 2):
            if nmn_cl_run_start_stage == 1 and global_step >= nmn_cl_stage1_steps:
                current_nmn_stage = 2
            else:
                current_nmn_stage = nmn_cl_run_start_stage

        if current_nmn_stage == 1 and nmn_cl_run_start_stage == 1 and global_step >= nmn_cl_stage1_steps:
            current_nmn_stage = 2
    
    print("=" * 60)
    print(f"TAD PPO Training - {run_name}")
    print(f"Skill Mode: {SetupParameters.SKILL_MODE}")
    print(f"Training Mode: {training_mode}")
    print(f"Device: {device} (可用GPU数量: {num_gpus})")
    print(f"Network Type: {SetupParameters.BOTTOM_NETWORK_TYPE}")
    print(
        f"LR Schedule: {TrainingParameters.LR_SCHEDULE} "
        f"({TrainingParameters.lr:.2e} -> {TrainingParameters.LR_FINAL:.2e})"
    )
    if bool(getattr(NetParameters, "NMN_DUAL_GRU_LOG_STD_ANNEAL_ENABLE", False)):
        print(
            "Dual-GRU LogStd Cap Schedule: "
            f"max {float(getattr(NetParameters, '_NMN_DUAL_GRU_LOG_STD_ANNEAL_INITIAL_MAX', NetParameters.NMN_DUAL_GRU_MAX_LOG_STD)):.2f}"
            f" -> {float(getattr(NetParameters, 'NMN_DUAL_GRU_LOG_STD_ANNEAL_FINAL_MAX', NetParameters.NMN_DUAL_GRU_MAX_LOG_STD)):.2f} "
            f"from step {int(getattr(NetParameters, 'NMN_DUAL_GRU_LOG_STD_ANNEAL_START', 0)):,}"
            f" to {int(getattr(NetParameters, 'NMN_DUAL_GRU_LOG_STD_ANNEAL_END', 0)):,}"
        )
    if getattr(SetupParameters, 'TRAIN_ATTACKER_STRATEGIES', None):
        print(f"Attacker Pool: {getattr(SetupParameters, 'TRAIN_ATTACKER_STRATEGIES')}")
    if getattr(SetupParameters, 'TRAIN_ATTACKER_STRATEGY_WEIGHTS', None) is not None:
        print(f"Attacker Weights: {getattr(SetupParameters, 'TRAIN_ATTACKER_STRATEGY_WEIGHTS')}")
    if getattr(SetupParameters, 'TRAIN_LEARNED_ATTACKER_SPECS', None):
        print(
            "Learned Attacker Specs: "
            f"{getattr(SetupParameters, 'TRAIN_LEARNED_ATTACKER_SPECS')}"
        )
    balanced_eval_attackers = list(getattr(TrainingParameters, 'BALANCED_EVAL_ATTACKERS', ()) or ())
    balanced_eval_episodes = int(getattr(TrainingParameters, 'BALANCED_EVAL_EPISODES', 0) or 0)
    balanced_eval_metric_name = str(
        getattr(TrainingParameters, 'BALANCED_EVAL_METRIC', 'mean_win')
    ).strip().lower()
    early_stop_enabled = bool(getattr(TrainingParameters, 'EARLY_STOP_ENABLED', False))
    early_stop_min_steps = int(getattr(TrainingParameters, 'EARLY_STOP_MIN_STEPS', 0))
    early_stop_patience = int(getattr(TrainingParameters, 'EARLY_STOP_PATIENCE', 5))
    early_stop_min_delta = float(getattr(TrainingParameters, 'EARLY_STOP_MIN_DELTA', 0.0))
    if balanced_eval_attackers and balanced_eval_episodes > 0:
        print(
            f"Balanced Eval: attackers={balanced_eval_attackers}, "
            f"episodes_each={balanced_eval_episodes}, metric={balanced_eval_metric_name}"
        )
    if nmn_cl_enabled:
        print(
            f"NMN-CL: start_stage={nmn_cl_run_start_stage}, active_stage={current_nmn_stage}, "
            f"stage1(no obstacles,dummy obstacle input)={nmn_cl_stage1_steps:,} steps, "
            f"stage2(obstacle_density={nmn_cl_stage2_density})={nmn_cl_stage2_steps:,} steps"
        )
        if nmn_cl_run_start_stage == 2 and model_dict is None:
            print("[NMN-CL] START_STAGE=2 without restored weights: stage2 will train from scratch.")
    if is_baseline_mode:
        print(f"Reward Mode: {BASELINE_ENV_CONFIG['reward_mode']} (old dense guidance)")
    if training_mode == 'mixed':
        print(f"IL Anneal: {TrainingParameters.IL_INITIAL_WEIGHT} -> {TrainingParameters.IL_FINAL_WEIGHT} over {TrainingParameters.IL_ANNEAL_STEPS:,} steps")
    print(f"Num Runners: {n_envs}")
    print(f"Steps per Runner: {TrainingParameters.N_STEPS}")
    if retrain:
        print(f"RETRAIN: True (继续训练，恢复进度)")
    if fresh_retrain:
        print(f"FRESH_RETRAIN: True (加载权重，重置进度)")
    print("=" * 60)
    
    # 使用所有可用CPU初始化Ray，Runner不使用GPU（推理在CPU上完成）
    # GPU仅用于主进程的模型训练
    ray_tmp = get_ray_temp_dir()
    import os as _os
    ray_num_cpus = _os.cpu_count() or n_envs  # 使用全部CPU
    ray_num_gpus = 0  # Runner不需要GPU，训练由主进程在GPU上完成
    print(f"[Ray] Init with {ray_num_cpus} CPUs (system total), {n_envs} runners")
    if ray_tmp:
        ray.init(num_cpus=ray_num_cpus, num_gpus=ray_num_gpus, _temp_dir=ray_tmp)
    else:
        ray.init(num_cpus=ray_num_cpus, num_gpus=ray_num_gpus)
    
    model = Model(device=device, global_model=True, network_type=model_network_type)
    active_log_std_bounds = _sync_dual_gru_log_std_bounds(model.network)
    _sync_dual_gru_residual_training(model.network, global_step)
    if active_log_std_bounds is not None:
        print(
            "Dual-GRU LogStd Bounds: "
            f"min={active_log_std_bounds[0]:.2f}, max={active_log_std_bounds[1]:.2f}"
        )
    if nmn_cl_enabled:
        model.set_nmn_stage(current_nmn_stage)
    
    # 加载模型权重 (RETRAIN 或 FRESH_RETRAIN)
    if model_dict is not None:
        # 兼容旧格式 (直接 state_dict) 和新格式 (dict with 'model' key)
        if 'model' in model_dict:
            model.set_weights(model_dict['model'])
        else:
            # 旧格式: model_dict 本身就是 state_dict
            model.set_weights(model_dict)
        if nmn_cl_enabled:
            model.set_nmn_stage(current_nmn_stage)
        print("[INFO] Model weights loaded successfully")

    if nmn_cl_enabled:
        total_steps = int(
            nmn_cl_stage2_steps if nmn_cl_run_start_stage == 2
            else nmn_cl_stage1_steps + nmn_cl_stage2_steps
        )
    else:
        total_steps = int(TrainingParameters.N_MAX_STEPS)
    steps_per_update = n_envs * TrainingParameters.N_STEPS
    remaining_steps = max(0, total_steps - global_step)
    total_updates = int(math.ceil(remaining_steps / float(steps_per_update))) if remaining_steps > 0 else 0
    current_lr = _resolve_learning_rate(global_step, total_steps)
    model.update_learning_rate(current_lr)
    print(f"Initial optimizer LR: {current_lr:.2e} at step {global_step:,}/{total_steps:,}")

    def _apply_nmn_stage_to_env_cfg(env_cfg: Dict, nmn_stage: int = None) -> Dict:
        cfg = dict(env_cfg) if env_cfg is not None else {}
        if nmn_cl_enabled:
            stage = int(current_nmn_stage if nmn_stage is None else nmn_stage)
            density = nmn_cl_stage1_density if stage == 1 else nmn_cl_stage2_density
            cfg['nmn_cl_stage'] = stage
            cfg['obstacle_density'] = density
        return cfg

    def _checkpoint_metadata(stage_override: int = None) -> Dict:
        meta = {
            'training_role': 'defender',
            'policy_name': str(os.environ.get('DEFENDER_POLICY_NAME', skill_mode)),
            'skill_mode': str(skill_mode),
            'training_mode': str(training_mode),
            'train_seed': int(SetupParameters.SEED),
            'train_use_random_seed': bool(SetupParameters.TRAIN_USE_RANDOM_SEED),
            'eval_use_random_seed': bool(SetupParameters.EVAL_USE_RANDOM_SEED),
            'eval_fixed_seed': int(SetupParameters.EVAL_FIXED_SEED),
            'train_attacker_strategies': list(
                getattr(SetupParameters, 'TRAIN_ATTACKER_STRATEGIES', ()) or ()
            ),
            'train_attacker_strategy_weights': getattr(
                SetupParameters,
                'TRAIN_ATTACKER_STRATEGY_WEIGHTS',
                None,
            ),
            'train_attacker_strategy_weight_schedule': getattr(
                SetupParameters,
                'TRAIN_ATTACKER_STRATEGY_WEIGHT_SCHEDULE',
                None,
            ),
            'train_attacker_strategy_weight_switch_steps': getattr(
                SetupParameters,
                'TRAIN_ATTACKER_STRATEGY_WEIGHT_SWITCH_STEPS',
                None,
            ),
            'learned_attacker_alias': str(
                getattr(SetupParameters, 'TRAIN_LEARNED_ATTACKER_ALIAS', 'attacker_rl')
            ),
            'learned_attacker_specs': dict(
                getattr(SetupParameters, 'TRAIN_LEARNED_ATTACKER_SPECS', {}) or {}
            ),
            'programmatic_attacker_specs': {
                alias: {
                    'registry_policy': policy_name,
                    'source': 'attacker_heuristics.registry',
                }
                for alias, policy_name in PROGRAMMATIC_ATTACKER_ALIASES.items()
                if alias in (getattr(SetupParameters, 'TRAIN_ATTACKER_STRATEGIES', ()) or ())
            },
        }
        if nmn_cl_enabled:
            stage = int(current_nmn_stage if stage_override is None else stage_override)
            meta.update({
                'nmn_cl_stage': stage,
                'nmn_cl_start_stage': int(nmn_cl_run_start_stage),
                'nmn_cl_stage1_steps': int(nmn_cl_stage1_steps),
                'nmn_cl_stage2_steps': int(nmn_cl_stage2_steps),
                'nmn_cl_stage1_obstacle_density': str(nmn_cl_stage1_density),
                'nmn_cl_stage2_obstacle_density': str(nmn_cl_stage2_density),
            })
        return meta

    def _resolve_attacker_weight_schedule(step: int):
        schedule = getattr(SetupParameters, 'TRAIN_ATTACKER_STRATEGY_WEIGHT_SCHEDULE', None)
        switches = getattr(SetupParameters, 'TRAIN_ATTACKER_STRATEGY_WEIGHT_SWITCH_STEPS', None)
        pool = getattr(SetupParameters, 'TRAIN_ATTACKER_STRATEGIES', None)
        if not schedule or not switches or pool is None:
            return None
        if len(schedule) != len(switches):
            raise ValueError(
                "TRAIN_ATTACKER_STRATEGY_WEIGHT_SCHEDULE and TRAIN_ATTACKER_STRATEGY_WEIGHT_SWITCH_STEPS length mismatch"
            )
        if len(pool) != len(schedule[0]):
            raise ValueError("TRAIN_ATTACKER_STRATEGY_WEIGHT_SCHEDULE width must match attacker pool size")
        current = schedule[0]
        for switch_step, weights in zip(switches, schedule):
            if step >= int(switch_step):
                current = weights
        total = float(sum(current))
        if total <= 0.0:
            raise ValueError("attacker weight schedule must have positive sum in every stage")
        return [float(v) / total for v in current]

    if is_baseline_mode and runner_env_cfg is not None:
        runner_env_cfg = dict(runner_env_cfg)
        runner_env_cfg['reward_mode'] = BASELINE_ENV_CONFIG['reward_mode']

    current_runner_env_cfg = _apply_nmn_stage_to_env_cfg(runner_env_cfg, current_nmn_stage)
    current_scheduled_attacker_weights = _resolve_attacker_weight_schedule(global_step)
    if current_scheduled_attacker_weights is not None:
        current_runner_env_cfg['attacker_strategy_pool_weights'] = current_scheduled_attacker_weights
        print(f"[AttackerSchedule] initial weights={current_scheduled_attacker_weights}")

    run_config = {
        'run_name': run_name,
        'role': 'defender',
        'policy_name': str(os.environ.get('DEFENDER_POLICY_NAME', skill_mode)),
        'skill_mode': str(skill_mode),
        'training_mode': str(training_mode),
        'network_type': str(model_network_type),
        'seed': int(SetupParameters.SEED),
        'train_use_random_seed': bool(SetupParameters.TRAIN_USE_RANDOM_SEED),
        'eval_use_random_seed': bool(SetupParameters.EVAL_USE_RANDOM_SEED),
        'eval_fixed_seed': int(SetupParameters.EVAL_FIXED_SEED),
        'attacker_pool': list(getattr(SetupParameters, 'TRAIN_ATTACKER_STRATEGIES', ()) or ()),
        'attacker_weights': getattr(SetupParameters, 'TRAIN_ATTACKER_STRATEGY_WEIGHTS', None),
        'attacker_weight_schedule': getattr(
            SetupParameters,
            'TRAIN_ATTACKER_STRATEGY_WEIGHT_SCHEDULE',
            None,
        ),
        'attacker_weight_switch_steps': getattr(
            SetupParameters,
            'TRAIN_ATTACKER_STRATEGY_WEIGHT_SWITCH_STEPS',
            None,
        ),
        'learned_attacker_alias': str(
            getattr(SetupParameters, 'TRAIN_LEARNED_ATTACKER_ALIAS', 'attacker_rl')
        ),
        'learned_attacker_specs': dict(
            getattr(SetupParameters, 'TRAIN_LEARNED_ATTACKER_SPECS', {}) or {}
        ),
        'programmatic_attacker_specs': {
            alias: {
                'registry_policy': policy_name,
                'source': 'attacker_heuristics.registry',
            }
            for alias, policy_name in PROGRAMMATIC_ATTACKER_ALIASES.items()
            if alias in (getattr(SetupParameters, 'TRAIN_ATTACKER_STRATEGIES', ()) or ())
        },
        'environment': {
            'reward_mode': str(current_runner_env_cfg.get('reward_mode', skill_mode)),
            'episode_len': int(current_runner_env_cfg.get(
                'episode_len',
                getattr(map_config.EnvParameters, 'EPISODE_LEN', 0),
            )),
            'attacker_speed': float(getattr(map_config, 'attacker_speed', 2.0)),
            'defender_speed': float(getattr(map_config, 'defender_speed', 2.6)),
            'defender_collision_outcome': 'draw_in_formal_evaluation',
            'reward_contract': (
                {
                    'step_penalty': -0.04,
                    'defender_capture': 30.0,
                    'target_breach': -20.0,
                    'defender_collision': -20.0,
                    'pursuit_progress_scale': 10.0,
                    'target_progress_scale': 20.0,
                    'timeout': 'defender_win_no_terminal_bonus',
                }
                if skill_mode == 'baseline'
                else {
                    'step_penalty': -0.08,
                    'defender_capture': 20.0,
                    'target_breach': -20.0,
                    'defender_collision': -10.0,
                    'pursuit_progress_scale': 10.0,
                    'target_progress_scale': 0.0,
                    'timeout': 'task_failure_no_terminal_bonus',
                }
                if skill_mode == 'chase'
                else None
            ),
        },
        'training': {
            'total_steps': int(total_steps),
            'num_envs': int(n_envs),
            'rollout_steps': int(TrainingParameters.N_STEPS),
            'tbptt_steps': int(TrainingParameters.TBPTT_STEPS),
            'minibatch_size': int(TrainingParameters.MINIBATCH_SIZE),
            'epochs': int(TrainingParameters.N_EPOCHS),
            'learning_rate': float(TrainingParameters.lr),
            'learning_rate_final': float(TrainingParameters.LR_FINAL),
            'learning_rate_schedule': str(TrainingParameters.LR_SCHEDULE),
            'gamma': float(TrainingParameters.GAMMA),
            'gae_lambda': float(TrainingParameters.LAM),
            'clip_range': float(TrainingParameters.CLIP_RANGE),
            'value_clip_range': float(TrainingParameters.VALUE_CLIP_RANGE),
            'entropy_coef': float(TrainingParameters.ENTROPY_COEF),
            'best_model_metric': str(TrainingParameters.BEST_MODEL_METRIC),
            'reward_normalization': bool(TrainingParameters.REWARD_NORMALIZATION),
            'dual_gru_log_std': {
                'initial': float(getattr(NetParameters, 'NMN_DUAL_GRU_INITIAL_LOG_STD', 0.0)),
                'min': float(getattr(NetParameters, 'NMN_DUAL_GRU_MIN_LOG_STD', -20.0)),
                'max': float(getattr(NetParameters, 'NMN_DUAL_GRU_MAX_LOG_STD', 2.0)),
            } if model.is_recurrent else None,
            'fresh_retrain': bool(fresh_retrain),
            'retrain': bool(retrain),
            'restore_checkpoint': getattr(RecordingParameters, 'RESTORE_DIR', None),
            'early_stop_enabled': bool(early_stop_enabled),
            'early_stop_min_steps': int(early_stop_min_steps),
            'early_stop_patience': int(early_stop_patience),
            'early_stop_min_delta': float(early_stop_min_delta),
        },
        'evaluation': {
            'interval': int(RecordingParameters.EVAL_INTERVAL),
            'episodes': int(RecordingParameters.EVAL_EPISODES),
            'balanced_attackers': list(balanced_eval_attackers),
            'balanced_episodes': int(balanced_eval_episodes),
            'balanced_metric': str(balanced_eval_metric_name),
        },
    }
    with open(osp.join(model_dir, 'run_config.json'), 'w', encoding='utf-8') as handle:
        json.dump(run_config, handle, indent=2, ensure_ascii=False)

    runners = [Runner.remote(i, env_configs=current_runner_env_cfg, network_type=model_network_type) for i in range(n_envs)]
    
    print(f"\nStarting training for {total_updates} updates...")
    print(f"Target environment steps: {total_steps:,}")
    print(f"Remaining environment steps: {remaining_steps:,}")
    if nmn_cl_enabled:
        print(
            f"[NMN-CL] active_stage={current_nmn_stage}, "
            f"obstacle_density={current_runner_env_cfg.get('obstacle_density')}"
        )
    start_time = time.time()
    no_improve_evals = 0
    
    for update in range(1, total_updates + 1):
        t_start = time.time()
        recreate_runners = False
        should_stop = False

        if (
            nmn_cl_enabled
            and current_nmn_stage == 1
            and nmn_cl_run_start_stage == 1
            and global_step >= nmn_cl_stage1_steps
        ):
            stage1_path = osp.join(model_dir, 'stage1_model.pth')
            model.save(
                stage1_path,
                step=global_step,
                reward=best_reward,
                extra_metadata=_checkpoint_metadata(stage_override=1),
            )
            current_nmn_stage = 2
            model.set_nmn_stage(current_nmn_stage)
            best_reward = -float('inf')
            recreate_runners = True
            print(
                f"[NMN-CL] Saved stage1 checkpoint -> {stage1_path} and switched to "
                f"stage2 at step {global_step:,} (obstacle_density={nmn_cl_stage2_density})"
            )

        if recreate_runners:
            current_runner_env_cfg = _apply_nmn_stage_to_env_cfg(runner_env_cfg, current_nmn_stage)
            for r in runners:
                ray.kill(r)
            runners = [
                Runner.remote(i, env_configs=current_runner_env_cfg, network_type=model_network_type)
                for i in range(n_envs)
            ]

        current_lr = _resolve_learning_rate(global_step, total_steps)
        scheduled_max_log_std = _apply_dual_gru_log_std_schedule(global_step)
        active_log_std_bounds = _sync_dual_gru_log_std_bounds(
            model.network,
            max_log_std=scheduled_max_log_std,
        )
        _sync_dual_gru_residual_training(model.network, global_step)
        model.update_learning_rate(current_lr)

        scheduled_weights = _resolve_attacker_weight_schedule(global_step)
        if scheduled_weights is not None:
            current_runner_env_cfg['attacker_strategy_pool_weights'] = scheduled_weights
            if scheduled_weights != current_scheduled_attacker_weights:
                print(f"[AttackerSchedule] step={global_step:,} weights={scheduled_weights}")
                current_scheduled_attacker_weights = list(scheduled_weights)
            try:
                ray.get([r.set_attacker_strategy_pool_weights.remote(scheduled_weights) for r in runners])
            except Exception as exc:
                print(f"[AttackerSchedule] failed to apply weights: {exc}")
        
        weights = model.get_weights()
        weight_id = ray.put(weights)
        if active_log_std_bounds is not None:
            ray.get([
                r.set_dual_gru_log_std_bounds.remote(
                    active_log_std_bounds[0],
                    active_log_std_bounds[1],
                )
                for r in runners
            ])
        ray.get([r.set_weights.remote(weight_id) for r in runners])
        
        rollout_futures = [r.run.remote(TrainingParameters.N_STEPS) for r in runners]
        
        rollouts = ray.get(rollout_futures)
        
        all_perf = {'per_r': [], 'per_episode_len': [], 'win': []}
        for rollout in rollouts:
            perf = rollout['perf']
            all_perf['per_r'].extend(perf['per_r'])
            all_perf['per_episode_len'].extend(perf['per_episode_len'])
            all_perf['win'].extend(perf['win'])
        
        # 计算当前IL权重（余弦退火）
        il_weight = cosine_anneal_il_weight(global_step) if training_mode == 'mixed' else 0.0
        
        if training_mode == 'rl':
            # 纯强化学习
            obs_all = np.concatenate([r['obs'] for r in rollouts], axis=0)
            critic_obs_all = np.concatenate([r['critic_obs'] for r in rollouts], axis=0)
            actions_all = np.concatenate([r['actions'] for r in rollouts], axis=0)
            log_probs_all = np.concatenate([r['log_probs'] for r in rollouts], axis=0)
            returns_all = np.concatenate([r['returns'] for r in rollouts], axis=0)
            values_all = np.concatenate([r['values'] for r in rollouts], axis=0)
            dones_all = np.concatenate([r['dones'] for r in rollouts], axis=0)
            actor_hiddens_all = None
            critic_hiddens_all = None
            if model.is_recurrent and 'actor_hiddens' in rollouts[0]:
                actor_hiddens_all = np.concatenate([r['actor_hiddens'] for r in rollouts], axis=0)
                critic_hiddens_all = np.concatenate([r['critic_hiddens'] for r in rollouts], axis=0)
            aux_targets_all = None
            if 'aux_targets' in rollouts[0]:
                aux_targets_all = {
                    key: np.concatenate([r['aux_targets'][key] for r in rollouts], axis=0)
                    for key in rollouts[0]['aux_targets'].keys()
                }
            mb_loss = model.train(
                actor_obs=obs_all,
                critic_obs=critic_obs_all,
                actions=actions_all,
                old_log_probs=log_probs_all,
                returns=returns_all,
                values=values_all,
                actor_hiddens=actor_hiddens_all,
                critic_hiddens=critic_hiddens_all,
                aux_targets=aux_targets_all,
                dones=dones_all,
                num_envs=n_envs,
                rollout_steps=TrainingParameters.N_STEPS,
                tbptt_steps=TrainingParameters.TBPTT_STEPS,
            )
            
            il_loss = None
            
        else:
            # mixed: IL + RL 混合训练
            obs_all = np.concatenate([r['obs'] for r in rollouts], axis=0)
            critic_obs_all = np.concatenate([r['critic_obs'] for r in rollouts], axis=0)
            actions_all = np.concatenate([r['actions'] for r in rollouts], axis=0)
            log_probs_all = np.concatenate([r['log_probs'] for r in rollouts], axis=0)
            returns_all = np.concatenate([r['returns'] for r in rollouts], axis=0)
            values_all = np.concatenate([r['values'] for r in rollouts], axis=0)
            expert_actions_all = np.concatenate([r['expert_actions'] for r in rollouts], axis=0)
            dones_all = np.concatenate([r['dones'] for r in rollouts], axis=0)
            actor_hiddens_all = None
            critic_hiddens_all = None
            if model.is_recurrent and 'actor_hiddens' in rollouts[0]:
                actor_hiddens_all = np.concatenate([r['actor_hiddens'] for r in rollouts], axis=0)
                critic_hiddens_all = np.concatenate([r['critic_hiddens'] for r in rollouts], axis=0)
            aux_targets_all = None
            if 'aux_targets' in rollouts[0]:
                aux_targets_all = {
                    key: np.concatenate([r['aux_targets'][key] for r in rollouts], axis=0)
                    for key in rollouts[0]['aux_targets'].keys()
                }

            # 传入expert_actions和il_weight进行混合训练
            mb_loss = model.train_mixed(
                actor_obs=obs_all,
                critic_obs=critic_obs_all,
                actions=actions_all,
                old_log_probs=log_probs_all,
                returns=returns_all,
                values=values_all,
                expert_actions=expert_actions_all,
                il_weight=il_weight,
                actor_hiddens=actor_hiddens_all,
                critic_hiddens=critic_hiddens_all,
                aux_targets=aux_targets_all,
                dones=dones_all,
                num_envs=n_envs,
                rollout_steps=TrainingParameters.N_STEPS,
                tbptt_steps=TrainingParameters.TBPTT_STEPS,
            )
            
            il_loss = [mb_loss.get('il_loss', 0.0), 0.0]  # [loss, grad_norm]
        
        steps_this_update = n_envs * TrainingParameters.N_STEPS
        global_step += steps_this_update
        
        if (global_step // TrainingParameters.LOG_EPOCH_STEPS) > ((global_step - steps_this_update) // TrainingParameters.LOG_EPOCH_STEPS):
            mean_reward = np.mean(all_perf['per_r']) if all_perf['per_r'] else 0.0
            mean_ep_len = np.mean(all_perf['per_episode_len']) if all_perf['per_episode_len'] else 0.0
            win_rate = np.mean(all_perf['win']) if all_perf['win'] else 0.0
            speed_mean = float(np.mean([r.get('action_speed_norm_mean', 0.0) for r in rollouts])) if rollouts else 0.0
            speed_p10 = float(np.mean([r.get('action_speed_norm_p10', 0.0) for r in rollouts])) if rollouts else 0.0
            speed_p90 = float(np.mean([r.get('action_speed_norm_p90', 0.0) for r in rollouts])) if rollouts else 0.0
            abs_turn_mean = float(np.mean([r.get('action_abs_turn_mean', 0.0) for r in rollouts])) if rollouts else 0.0
            progress = min(100.0, (global_step / total_steps * 100)) if total_steps > 0 else 0.0
            
            il_info = f" | IL_w: {il_weight:.3f}" if training_mode == 'mixed' else ""
            if mb_loss and mb_loss.get('anchor_loss') is not None:
                il_info += f" | Anchor: {float(mb_loss.get('anchor_loss', 0.0)):.4f}"
            if mb_loss and mb_loss.get('adv_action_bc_loss') is not None:
                adv_bc = float(mb_loss.get('adv_action_bc_loss', 0.0))
                if adv_bc > 0.0:
                    il_info += f" | AdvBC: {adv_bc:.4f}"
            stage_info = ""
            if nmn_cl_enabled:
                active_density = nmn_cl_stage1_density if current_nmn_stage == 1 else nmn_cl_stage2_density
                stage_info = f" | NMN-CL: S{current_nmn_stage}@{active_density}"
            std_info = ""
            if hasattr(model.network, 'log_std'):
                with torch.no_grad():
                    raw_log_std = model.network.log_std.detach().float().cpu().numpy()
                    bounded_log_std = raw_log_std
                    if hasattr(model.network, '_bounded_log_std'):
                        bounded_log_std = model.network._bounded_log_std().detach().float().cpu().numpy()
                    std_info = (
                        f" | LogStd: [{raw_log_std[0]:.2f},{raw_log_std[1]:.2f}]"
                        f"->[{bounded_log_std[0]:.2f},{bounded_log_std[1]:.2f}]"
                    )
                    if scheduled_max_log_std is not None:
                        std_info += f" cap={scheduled_max_log_std:.2f}"
            print(f"Step {global_step:,} ({progress:.1f}%) | "
                  f"Reward: {mean_reward:.2f} | "
                  f"EpLen: {mean_ep_len:.1f} | "
                  f"Win: {win_rate:.2%} | "
                  f"LR: {current_lr:.2e} | "
                  f"Speed: {speed_mean:.3f}[{speed_p10:.3f},{speed_p90:.3f}] | "
                  f"AbsTurn: {abs_turn_mean:.3f}{il_info}{stage_info}{std_info}")
            
            write_to_tensorboard(
                summary_writer, global_step,
                performance_dict=all_perf,
                mb_loss=[mb_loss['losses']] if mb_loss is not None else None,
                imitation_loss=il_loss,
                evaluate=False
            )
            
            # 记录IL权重
            if summary_writer and training_mode == 'mixed':
                summary_writer.add_scalar('IL/weight', il_weight, global_step)
                if mb_loss and 'il_loss' in mb_loss and mb_loss['il_loss'] is not None:
                    summary_writer.add_scalar('IL/loss', mb_loss['il_loss'], global_step)
            if summary_writer and mb_loss and mb_loss.get('aux_loss') is not None:
                summary_writer.add_scalar('Loss/Auxiliary', mb_loss['aux_loss'], global_step)
            if summary_writer and mb_loss and mb_loss.get('anchor_loss') is not None:
                summary_writer.add_scalar('Loss/Policy_Anchor', mb_loss['anchor_loss'], global_step)
            if summary_writer and mb_loss and mb_loss.get('adv_action_bc_loss') is not None:
                summary_writer.add_scalar('Loss/Advantage_Action_BC', mb_loss['adv_action_bc_loss'], global_step)
            if summary_writer:
                summary_writer.add_scalar('Train/Learning_Rate', current_lr, global_step)
                summary_writer.add_scalar('Action/speed_norm_mean', speed_mean, global_step)
                summary_writer.add_scalar('Action/speed_norm_p10', speed_p10, global_step)
                summary_writer.add_scalar('Action/speed_norm_p90', speed_p90, global_step)
                summary_writer.add_scalar('Action/abs_turn_mean', abs_turn_mean, global_step)
            if summary_writer and nmn_cl_enabled:
                summary_writer.add_scalar('NMN_CL/stage', int(current_nmn_stage), global_step)
        
        if (global_step // RecordingParameters.EVAL_INTERVAL) > ((global_step - steps_this_update) // RecordingParameters.EVAL_INTERVAL):
            print("--- Evaluation ---")
            eval_weight_id = ray.put(model.get_weights())
            ray.get([r.set_weights.remote(eval_weight_id) for r in runners])
            eval_result = ray.get(runners[0].evaluate.remote(
                num_episodes=RecordingParameters.EVAL_EPISODES,
                greedy=True,
                record_gif=((global_step // RecordingParameters.GIF_INTERVAL) > ((global_step - steps_this_update) // RecordingParameters.GIF_INTERVAL))
            ))
            
            eval_perf = eval_result['perf']
            eval_reward = np.mean(eval_perf['per_r']) if eval_perf['per_r'] else 0.0
            eval_ep_len = np.mean(eval_perf['per_episode_len']) if eval_perf['per_episode_len'] else 0.0
            eval_win = np.mean(eval_perf['win']) if eval_perf['win'] else 0.0
            
            print(f"Eval Reward: {eval_reward:.2f} | "
                  f"Eval EpLen: {eval_ep_len:.1f} | "
                  f"Eval Win: {eval_win:.2%}")
            eval_outcome_rates = dict(eval_result.get('outcome_rates', {}) or {})
            if eval_outcome_rates:
                print(f"Eval Outcomes: {eval_outcome_rates}")
            
            write_to_tensorboard(
                summary_writer, global_step,
                performance_dict=eval_perf,
                evaluate=True,
                greedy=True
            )
            
            if eval_result.get('frames') and len(eval_result['frames']) > 0:
                gif_path = osp.join(gif_dir, f"eval_{global_step}.gif")
                make_gif(eval_result['frames'], gif_path)
            
            # Generate academic trajectory plot (controlled by TRAJ_INTERVAL)
            traj_data = eval_result.get('trajectory_data')
            if traj_data and ((global_step // RecordingParameters.TRAJ_INTERVAL) > ((global_step - steps_this_update) // RecordingParameters.TRAJ_INTERVAL)):
                traj_png = osp.join(gif_dir, f"traj_{global_step}.png")
                make_trajectory_plot(traj_data, traj_png, dpi=150)
            
            best_metric_name = str(getattr(TrainingParameters, 'BEST_MODEL_METRIC', 'reward')).strip().lower()
            eval_capture = float(
                eval_outcome_rates.get('defender_caught_attacker', 0.0)
            )
            if best_metric_name == 'capture':
                eval_save_metric = eval_capture
                best_label = 'Capture'
            elif best_metric_name == 'win':
                eval_save_metric = float(eval_win)
                best_label = 'Win'
            else:
                eval_save_metric = float(eval_reward)
                best_label = 'Reward'

            improved = eval_save_metric > best_reward + early_stop_min_delta
            if improved:
                best_reward = eval_save_metric
                no_improve_evals = 0
                best_path = osp.join(model_dir, 'best_model.pth')
                model.save(
                    best_path,
                    step=global_step,
                    reward=best_reward,
                    extra_metadata={
                        **_checkpoint_metadata(),
                        'best_metric': best_metric_name,
                        'eval_win': float(eval_win),
                        'eval_reward': float(eval_reward),
                        'eval_episodes': int(len(eval_perf.get('win', []))),
                        'eval_outcome_rates': eval_outcome_rates,
                    },
                )
                print(f"New best model saved! {best_label}: {best_reward:.4f}")

            win_best_path = osp.join(model_dir, 'best_win_model.pth')
            reward_best_path = osp.join(model_dir, 'best_reward_model.pth')
            if not hasattr(main, '_best_eval_win'):
                main._best_eval_win = -float('inf')
            if not hasattr(main, '_best_eval_reward'):
                main._best_eval_reward = -float('inf')
            if float(eval_win) > float(main._best_eval_win):
                main._best_eval_win = float(eval_win)
                model.save(
                    win_best_path,
                    step=global_step,
                    reward=float(eval_win),
                    extra_metadata={
                        **_checkpoint_metadata(),
                        'best_metric': 'win',
                        'eval_win': float(eval_win),
                        'eval_reward': float(eval_reward),
                        'eval_outcome_rates': eval_outcome_rates,
                    },
                )
                print(f"New win-best model saved! Win: {float(eval_win):.4f}")
            if float(eval_reward) > float(main._best_eval_reward):
                main._best_eval_reward = float(eval_reward)
                model.save(
                    reward_best_path,
                    step=global_step,
                    reward=float(eval_reward),
                    extra_metadata={
                        **_checkpoint_metadata(),
                        'best_metric': 'reward',
                        'eval_win': float(eval_win),
                        'eval_reward': float(eval_reward),
                        'eval_outcome_rates': eval_outcome_rates,
                    },
                )
                print(f"New reward-best model saved! Reward: {float(eval_reward):.4f}")

            if balanced_eval_attackers and balanced_eval_episodes > 0:
                balanced_rows = []
                balanced_wins = []
                balanced_captures = []
                for attacker_name in balanced_eval_attackers:
                    attacker_result = ray.get(runners[0].evaluate.remote(
                        num_episodes=balanced_eval_episodes,
                        greedy=True,
                        record_gif=False,
                        attacker_strategy=attacker_name,
                    ))
                    attacker_perf = attacker_result['perf']
                    attacker_win = float(np.mean(attacker_perf['win'])) if attacker_perf['win'] else 0.0
                    attacker_reward = float(np.mean(attacker_perf['per_r'])) if attacker_perf['per_r'] else 0.0
                    attacker_ep_len = float(np.mean(attacker_perf['per_episode_len'])) if attacker_perf['per_episode_len'] else 0.0
                    attacker_outcome_rates = dict(
                        attacker_result.get('outcome_rates', {}) or {}
                    )
                    attacker_capture = float(
                        attacker_outcome_rates.get('defender_caught_attacker', 0.0)
                    )
                    attacker_timeout = float(
                        attacker_outcome_rates.get('timeout_defender_wins', 0.0)
                    )
                    balanced_rows.append({
                        'attacker': str(attacker_name),
                        'win': attacker_win,
                        'capture': attacker_capture,
                        'timeout': attacker_timeout,
                        'reward': attacker_reward,
                        'episode_len': attacker_ep_len,
                        'outcome_rates': attacker_outcome_rates,
                    })
                    balanced_wins.append(attacker_win)
                    balanced_captures.append(attacker_capture)
                    print(
                        f"  BalancedEval[{attacker_name}]: "
                        f"Win={attacker_win:.2%} | Reward={attacker_reward:.2f} | EpLen={attacker_ep_len:.1f}"
                    )
                    if attacker_outcome_rates:
                        print(f"    Outcomes={attacker_outcome_rates}")

                balanced_mean_win = float(np.mean(balanced_wins)) if balanced_wins else 0.0
                balanced_min_win = float(np.min(balanced_wins)) if balanced_wins else 0.0
                balanced_mean_capture = float(np.mean(balanced_captures)) if balanced_captures else 0.0
                balanced_min_capture = float(np.min(balanced_captures)) if balanced_captures else 0.0
                if balanced_eval_metric_name == 'min_capture':
                    balanced_metric = balanced_min_capture
                    balanced_metric_label = 'min_capture'
                elif balanced_eval_metric_name == 'mean_capture':
                    balanced_metric = balanced_mean_capture
                    balanced_metric_label = 'mean_capture'
                elif balanced_eval_metric_name == 'min_win':
                    balanced_metric = balanced_min_win
                    balanced_metric_label = 'min_win'
                else:
                    balanced_metric = balanced_mean_win
                    balanced_metric_label = 'mean_win'
                print(
                    f"BalancedEval Summary: mean_win={balanced_mean_win:.2%} | "
                    f"min_win={balanced_min_win:.2%} | "
                    f"mean_capture={balanced_mean_capture:.2%} | "
                    f"min_capture={balanced_min_capture:.2%}"
                )

                if summary_writer:
                    summary_writer.add_scalar('BalancedEval/Mean_Win', balanced_mean_win, global_step)
                    summary_writer.add_scalar('BalancedEval/Min_Win', balanced_min_win, global_step)
                    summary_writer.add_scalar('BalancedEval/Mean_Capture', balanced_mean_capture, global_step)
                    summary_writer.add_scalar('BalancedEval/Min_Capture', balanced_min_capture, global_step)
                    for row in balanced_rows:
                        tag = str(row['attacker']).replace('/', '_')
                        summary_writer.add_scalar(f'BalancedEval/{tag}_Win', row['win'], global_step)
                        summary_writer.add_scalar(f'BalancedEval/{tag}_Reward', row['reward'], global_step)

                if not hasattr(main, '_best_balanced_eval_metric'):
                    main._best_balanced_eval_metric = -float('inf')
                if balanced_metric > float(main._best_balanced_eval_metric):
                    main._best_balanced_eval_metric = float(balanced_metric)
                    balanced_best_path = osp.join(model_dir, 'best_balanced_model.pth')
                    model.save(
                        balanced_best_path,
                        step=global_step,
                        reward=float(balanced_metric),
                        extra_metadata={
                            **_checkpoint_metadata(),
                            'best_metric': f'balanced_{balanced_metric_label}',
                            'balanced_mean_win': balanced_mean_win,
                            'balanced_min_win': balanced_min_win,
                            'balanced_mean_capture': balanced_mean_capture,
                            'balanced_min_capture': balanced_min_capture,
                            'balanced_eval_attackers': list(balanced_eval_attackers),
                            'balanced_eval_episodes': int(balanced_eval_episodes),
                            'balanced_eval_rows': balanced_rows,
                        },
                    )
                    print(
                        f"New balanced-best model saved! "
                        f"{balanced_metric_label}: {balanced_metric:.4f}"
                    )

            if early_stop_enabled and not improved and global_step >= early_stop_min_steps:
                no_improve_evals += 1
                print(
                    f"Early-stop plateau: {no_improve_evals}/{early_stop_patience} "
                    f"(best {best_metric_name}={best_reward:.4f})"
                )
                should_stop = no_improve_evals >= early_stop_patience
            
            print("------------------")
        
        if (global_step // RecordingParameters.SAVE_INTERVAL) > ((global_step - steps_this_update) // RecordingParameters.SAVE_INTERVAL):
            latest_path = osp.join(model_dir, 'latest_model.pth')
            model.save(
                latest_path,
                step=global_step,
                reward=best_reward,
                extra_metadata=_checkpoint_metadata(),
            )
            print(f"Latest model saved: {latest_path}")

        if should_stop:
            print(f"Stopping defender best-response training at step {global_step:,}.")
            break
    
    final_path = osp.join(model_dir, 'final_model.pth')
    model.save(
        final_path,
        step=global_step,
        reward=best_reward,
        extra_metadata=_checkpoint_metadata(),
    )
    print(f"\nFinal model saved: {final_path}")
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time / 3600:.2f} hours")
    print(f"Best reward: {best_reward:.2f}")
    
    if summary_writer:
        summary_writer.close()
    
    ray.shutdown()


if __name__ == '__main__':
    main()
