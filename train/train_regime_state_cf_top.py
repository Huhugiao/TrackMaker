"""State-level counterfactual top training for regime-adaptive HRL.

This is intentionally separate from PPO launchers. It trains only the top
policy by asking, at visited states, which frozen low skill has the best
short-horizon risk/efficiency utility from the exact same simulator snapshot.
"""

import json
import os
import random
import sys
import time
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.paths import CHECKPOINTS_DIR
from configs import map_config
from configs.skill_config import NetParameters
from envs.hrl_env import HRLEnv
from networks import create_network
from skill.util import build_critic_observation, set_global_seeds
from utils.path_risk import compute_path_risk_metrics
from utils.process_info import print_training_process_info
from utils.top_policy_calibration import apply_chase_logit_bias, build_two_skill_class_weights


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(default) if raw is None or str(raw).strip() == "" else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(default) if raw is None or str(raw).strip() == "" else float(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "y"}


def _split_csv(text: str) -> Tuple[str, ...]:
    return tuple(x.strip().lower() for x in str(text).split(",") if x.strip())


SEED = _env_int("SCF_SEED", 20260514)
ITERS = _env_int("SCF_ITERS", 18)
EPISODES_PER_ITER = _env_int("SCF_EPISODES_PER_ITER", 80)
VAL_EPISODES = _env_int("SCF_VAL_EPISODES", 48)
EPOCHS_PER_ITER = _env_int("SCF_EPOCHS_PER_ITER", 3)
BATCH_SIZE = _env_int("SCF_BATCH_SIZE", 192)
SEQ_LEN = _env_int("SCF_SEQ_LEN", 32)
LR = _env_float("SCF_LR", 1.2e-4)
WEIGHT_DECAY = _env_float("SCF_WEIGHT_DECAY", 7.0e-4)
HORIZON = _env_int("SCF_HORIZON", 48)
SAMPLE_INTERVAL = _env_int("SCF_SAMPLE_INTERVAL", 8)
MAX_SAMPLES_PER_EP = _env_int("SCF_MAX_SAMPLES_PER_EP", 24)
MAX_STEPS = _env_int("SCF_MAX_STEPS", 449)
REPLAY_MAX_ROWS = _env_int("SCF_REPLAY_MAX_ROWS", 60000)
ROLLOUT_RANDOM_RATE = _env_float("SCF_ROLLOUT_RANDOM_RATE", 0.22)
ROLLOUT_LABEL_RATE = _env_float("SCF_ROLLOUT_LABEL_RATE", 0.35)
LABEL_SMOOTHING = _env_float("SCF_LABEL_SMOOTHING", 0.03)
PROTECT_CLASS_WEIGHT = _env_float("SCF_PROTECT_CLASS_WEIGHT", 1.0)
CHASE_CLASS_WEIGHT = _env_float("SCF_CHASE_CLASS_WEIGHT", 1.0)
TOP_CHASE_LOGIT_BIAS = _env_float("SCF_TOP_CHASE_LOGIT_BIAS", 0.0)
ENABLE_UTILITY_GAP_LABEL = _env_bool("SCF_ENABLE_UTILITY_GAP_LABEL", False)
CHASE_UTILITY_GAP_REQ = _env_float("SCF_CHASE_UTILITY_GAP_REQ", 0.25)
NEUTRAL_CHASE_GAP_REQ = _env_float("SCF_NEUTRAL_CHASE_GAP_REQ", 0.45)
DISADV_DEFAULT_CHASE_GAP_REQ = _env_float("SCF_DISADV_DEFAULT_CHASE_GAP_REQ", 0.55)
ENABLE_SOFT_UTILITY_AUX = _env_bool("SCF_ENABLE_SOFT_UTILITY_AUX", False)
SOFT_UTILITY_AUX_WEIGHT = _env_float("SCF_SOFT_UTILITY_AUX_WEIGHT", 0.05)
SOFT_UTILITY_GAP_SCALE = _env_float("SCF_SOFT_UTILITY_GAP_SCALE", 1.0)
SOFT_UTILITY_TARGET_FLOOR = _env_float("SCF_SOFT_UTILITY_TARGET_FLOOR", 0.05)
SOFT_UTILITY_TARGET_CEIL = _env_float("SCF_SOFT_UTILITY_TARGET_CEIL", 0.95)
UTILITY_GAP_DEFAULT_ATTACKERS = tuple(
    str(x).strip().lower()
    for x in _split_csv(os.environ.get("SCF_UTILITY_GAP_DEFAULT_ATTACKERS", "default"))
)
DEVICE_NAME = os.environ.get("SCF_DEVICE", "cpu").strip().lower()
OUTPUT_DIR = os.environ.get("SCF_OUTPUT_DIR", "").strip()
RISK_METRIC = os.environ.get("SCF_RISK_METRIC", "astar").strip().lower()
ASTAR_GRID_SIZE = _env_float("SCF_ASTAR_GRID_SIZE", 8.0)
ASTAR_OBSTACLE_PADDING = _env_float("SCF_ASTAR_OBSTACLE_PADDING", 12.0)
ATTACKERS = _split_csv(
    os.environ.get("SCF_ATTACKERS", "default,evasive")
)
REGIMES = _split_csv(os.environ.get("SCF_REGIMES", "advantage,neutral,disadvantage"))
REGIME_PROBS = tuple(float(x) for x in os.environ.get("SCF_REGIME_PROBS", "0.42,0.35,0.23").split(","))
SKILL_LAYOUT = "protect_chase"
NUM_SKILLS = 2

SAFE_MARGIN_STEPS = _env_float("SCF_SAFE_MARGIN_STEPS", 34.0)
RISK_MARGIN_STEPS = _env_float("SCF_RISK_MARGIN_STEPS", 14.0)
CHASE_TOLERANCE_SAFE = _env_float("SCF_CHASE_TOLERANCE_SAFE", 0.55)
NONBASE_TOLERANCE = _env_float("SCF_NONBASE_TOLERANCE", 0.25)
DISADV_CHASE_BONUS_REQ = _env_float("SCF_DISADV_CHASE_BONUS_REQ", 0.45)
RISK_CHASE_BONUS_REQ = _env_float("SCF_RISK_CHASE_BONUS_REQ", 0.25)
BASE_CHASE_WEIGHT = _env_float("SCF_BASE_CHASE_WEIGHT", 0.8)
SAFE_CHASE_WEIGHT = _env_float("SCF_SAFE_CHASE_WEIGHT", 3.4)
BASE_SAFETY_WEIGHT = _env_float("SCF_BASE_SAFETY_WEIGHT", 0.7)
RISK_SAFETY_WEIGHT = _env_float("SCF_RISK_SAFETY_WEIGHT", 2.8)
BASE_DELAY_WEIGHT = _env_float("SCF_BASE_DELAY_WEIGHT", 0.3)
RISK_DELAY_WEIGHT = _env_float("SCF_RISK_DELAY_WEIGHT", 1.2)
PROTECT_FALLBACK_TOL = _env_float("SCF_PROTECT_FALLBACK_TOL", 0.0)
PROTECT_CHASE_CLEAR_MARGIN = _env_float("SCF_PROTECT_CHASE_CLEAR_MARGIN", 0.8)
CAPTURE_REWARD = _env_float("SCF_CAPTURE_REWARD", 9.0)
TIMEOUT_WIN_REWARD = _env_float("SCF_TIMEOUT_WIN_REWARD", 1.6)
ATTACKER_CAPTURE_PENALTY = _env_float("SCF_ATTACKER_CAPTURE_PENALTY", 12.0)
DEFENDER_COLLISION_PENALTY = _env_float("SCF_DEFENDER_COLLISION_PENALTY", 8.0)
COLLISION_STEP_PENALTY = _env_float("SCF_COLLISION_STEP_PENALTY", 2.5)
MIN_MARGIN_PENALTY = _env_float("SCF_MIN_MARGIN_PENALTY", 0.08)
RADAR_DANGER_DIST = _env_float("SCF_RADAR_DANGER_DIST", 0.04)
RADAR_DANGER_PENALTY = _env_float("SCF_RADAR_DANGER_PENALTY", 0.4)

DEFAULT_INIT_TOP_PATH = (
    CHECKPOINTS_DIR / "hrl_regime_adaptive_toponly_20260513_222519" / "best_model.pth"
)
INIT_TOP_PATH = os.environ.get(
    "SCF_INIT_TOP_PATH",
    str(DEFAULT_INIT_TOP_PATH),
).strip()
ALLOW_RANDOM_INIT = _env_bool("SCF_ALLOW_RANDOM_INIT", False)
CHASE_PATH = os.environ.get(
    "SCF_CHASE_PATH",
    str(CHECKPOINTS_DIR / "defender_chase_nmn_dual_gru_raw_dense_05-05-19-12" / "final_model.pth"),
).strip()
PROTECT_PATH = os.environ.get(
    "SCF_PROTECT_PATH",
    str(CHECKPOINTS_DIR / "defender_protect_mlp_ctde_repro_20260526" / "final_model.pth"),
).strip()


def _compat_load(path: str, device):
    if not hasattr(np, "_core"):
        sys.modules["numpy._core"] = np.core
        sys.modules["numpy._core.multiarray"] = np.core.multiarray
    return torch.load(path, map_location=device, weights_only=False)


def _configure_top_dims():
    NetParameters.HRL_NUM_SKILLS = int(NUM_SKILLS)
    NetParameters.HRL_DURATION_BINS = (1,)
    NetParameters.HRL_NUM_DURATION_BINS = 1
    NetParameters.HRL_TOP_DISCRETE_ACTION_DIM = int(NUM_SKILLS)
    NetParameters.HRL_TOP_ACTION_DIM = int(NUM_SKILLS + 1)


def _resolve_init_top_path(init_top_path: str, allow_random_init: bool):
    requested = str(init_top_path or "").strip()
    if requested:
        path = Path(requested).expanduser()
        if path.is_file():
            return path.resolve()

    if allow_random_init:
        return None

    detail = f"not found: {requested}" if requested else "SCF_INIT_TOP_PATH is empty"
    raise FileNotFoundError(
        "Chapter 2 State-CF training requires its historical top-policy initialization "
        f"checkpoint ({detail}). Set SCF_INIT_TOP_PATH to a valid checkpoint. To "
        "intentionally start a new, non-reproduction experiment from random weights, "
        "set SCF_ALLOW_RANDOM_INIT=1."
    )


def _make_top(device: torch.device, init_top_path):
    _configure_top_dims()
    net = create_network("hrl_top_dual_gru_raw").to(device)
    if init_top_path is not None:
        ckpt = _compat_load(str(init_top_path), device)
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = net.load_state_dict(state, strict=False)
        print(
            f"[InitTop] {init_top_path} missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    else:
        print(
            "[InitTop] SCF_ALLOW_RANDOM_INIT=1: starting a new non-reproduction "
            "experiment from random weights.",
            flush=True,
        )
    if hasattr(net, "set_skill_classifier_for_action"):
        net.set_skill_classifier_for_action(False)
    if hasattr(net, "set_contextual_skill_head_for_action"):
        net.set_contextual_skill_head_for_action(False)
    return net


def _make_env(attacker: str, device: torch.device):
    return HRLEnv(
        protect_model_path=PROTECT_PATH,
        chase_model_path=CHASE_PATH,
        primary_skill_name="protect",
        attacker_strategy=str(attacker),
        device=str(device),
        hold_min=1,
        hold_max=1,
        disable_hold_control=True,
        macro_duration_bins=[1],
    )


def _state_metrics(env: HRLEnv) -> Dict[str, float]:
    sim = env.env
    d = sim.defender
    a = sim.attacker
    t = sim.target
    defender_speed = max(1e-6, float(getattr(sim, "defender_speed", 2.6)))
    attacker_speed = max(1e-6, float(getattr(sim, "attacker_speed", 2.0)))
    width = float(getattr(sim, "width", 640.0))
    height = float(getattr(sim, "height", 640.0))
    risk = compute_path_risk_metrics(
        state={"defender": d, "attacker": a, "target": t},
        defender_speed=defender_speed,
        attacker_speed=attacker_speed,
        width=width,
        height=height,
        grid_size=float(ASTAR_GRID_SIZE),
        obstacle_padding=float(ASTAR_OBSTACLE_PADDING),
        obstacles=getattr(map_config, "obstacles", []),
        metric=str(RISK_METRIC),
    )
    d_da = float(risk["defender_attacker"])
    d_at = float(risk["attacker_target"])
    d_dt = float(np.hypot(float(d["x"]) - float(t["x"]), float(d["y"]) - float(t["y"])))
    t_def_att = d_da / defender_speed
    t_att_tgt = d_at / attacker_speed
    margin = float(risk["margin"])
    diag = float(np.hypot(width, height))
    return {
        "d_da": d_da,
        "d_at": d_at,
        "d_dt": d_dt,
        "t_def_att": t_def_att,
        "t_att_tgt": t_att_tgt,
        "margin": margin,
        "astar_margin": float(risk["astar_margin"]),
        "euclidean_margin": float(risk["euclidean_margin"]),
        "detour_da": float(risk["detour_defender_attacker"]),
        "detour_at": float(risk["detour_attacker_target"]),
        "diag": max(1.0, diag),
        "step": float(getattr(sim, "step_count", 0)),
    }


def _classify(info: Dict) -> Tuple[bool, bool, bool, bool, bool]:
    reason = str(info.get("reason", ""))
    d_cap = "defender_caught_attacker" in reason
    a_cap = "attacker_caught_target" in reason or "attacker_win" in reason
    d_col = bool(info.get("defender_collision", False)) or "defender_collision" in reason or "defender_out" in reason
    timeout = "timeout" in reason or "time_limit" in reason or "max_steps" in reason or "truncated" in reason
    d_win = d_cap or (timeout and not a_cap and not d_col)
    return bool(d_win), bool(d_cap), bool(d_col), bool(a_cap), bool(timeout)


def _attacker_action(env: HRLEnv, attacker_obs: np.ndarray) -> np.ndarray:
    if env.attacker_policy is None:
        return np.asarray(env._static_attacker_action, dtype=np.float32)
    if hasattr(env.attacker_policy, "get_action_with_info"):
        action, _ = env.attacker_policy.get_action_with_info(attacker_obs)
        return np.asarray(action, dtype=np.float32)
    return np.asarray(env.attacker_policy.get_action(attacker_obs), dtype=np.float32)


def _rollout_skill(env: HRLEnv, skill_idx: int, horizon: int) -> Dict:
    skill_name = env.skill_names[int(skill_idx)]
    init = _state_metrics(env)
    info = {}
    collision_steps = 0
    rewards = 0.0
    done = False
    steps = 0
    min_margin = float(init["margin"])
    min_radar = 1.0

    for _ in range(int(horizon)):
        if env.cached_obs is None or env.cached_skill_obs is None:
            env._process_observation(env.env.current_obs)
        defender_obs, attacker_obs = env.cached_skill_obs
        radar = np.asarray(defender_obs[7:71], dtype=np.float32) if np.asarray(defender_obs).size >= 71 else np.ones(64)
        if radar.size:
            min_radar = min(min_radar, float(np.min(radar)))
        net = env.skill_nets[int(skill_idx)]
        action = env._skill_action_from_net(skill_name, net, defender_obs, attacker_obs)
        next_raw_obs, reward, term, trunc, info = env.env.step(
            action=action,
            attacker_action=_attacker_action(env, attacker_obs),
        )
        rewards += float(reward)
        collision_steps += int(bool(info.get("defender_collision", False)))
        env._process_observation(next_raw_obs)
        steps += 1
        cur = _state_metrics(env)
        min_margin = min(min_margin, float(cur["margin"]))
        done = bool(term or trunc)
        if done:
            break

    final = _state_metrics(env)
    d_win, d_cap, d_col, a_cap, timeout = _classify(info or {})
    bad = bool(a_cap or d_col or collision_steps > 0)
    diag = float(init["diag"])
    progress_chase = (float(init["d_da"]) - float(final["d_da"])) / diag
    progress_safety = (float(final["margin"]) - float(init["margin"])) / 80.0
    attacker_delay = (float(final["d_at"]) - float(init["d_at"])) / diag
    time_cost = 0.0015 * float(steps)

    terminal = 0.0
    if d_cap:
        terminal += CAPTURE_REWARD + max(0.0, (float(horizon) - float(steps))) * 0.015
    if timeout and d_win:
        terminal += TIMEOUT_WIN_REWARD
    if a_cap:
        terminal -= ATTACKER_CAPTURE_PENALTY
    if d_col:
        terminal -= DEFENDER_COLLISION_PENALTY
    terminal -= COLLISION_STEP_PENALTY * float(collision_steps)

    # Risk-sensitive but not conservative-only: margin protects bad cases,
    # chase progress matters strongly once the immediate target risk is low.
    risk_gate = 1.0 / (1.0 + np.exp((float(init["margin"]) - RISK_MARGIN_STEPS) / 8.0))
    safe_gate = 1.0 / (1.0 + np.exp(-(float(init["margin"]) - SAFE_MARGIN_STEPS) / 8.0))
    utility = (
        terminal
        + (RISK_SAFETY_WEIGHT * risk_gate + BASE_SAFETY_WEIGHT) * progress_safety
        + (BASE_CHASE_WEIGHT + SAFE_CHASE_WEIGHT * safe_gate) * progress_chase
        + (RISK_DELAY_WEIGHT * risk_gate + BASE_DELAY_WEIGHT) * attacker_delay
        - time_cost
    )
    if min_margin < -4.0:
        utility -= MIN_MARGIN_PENALTY * abs(float(min_margin))
    if min_radar < RADAR_DANGER_DIST:
        utility -= RADAR_DANGER_PENALTY * (RADAR_DANGER_DIST - float(min_radar)) / max(1e-6, RADAR_DANGER_DIST)

    return {
        "skill": skill_name,
        "utility": float(utility),
        "done": bool(done),
        "bad": bool(bad),
        "defender_capture": bool(d_cap),
        "attacker_capture": bool(a_cap),
        "defender_collision": bool(d_col or collision_steps > 0),
        "timeout_win": bool(timeout and d_win),
        "steps": int(steps),
        "init": init,
        "final": final,
        "progress_chase": float(progress_chase),
        "progress_safety": float(progress_safety),
        "attacker_delay": float(attacker_delay),
        "min_margin": float(min_margin),
        "raw_reward_sum": float(rewards),
    }


def _utility_gap_required(regime: str, attacker: str | None) -> float:
    margin_regime = _margin_regime_name(regime)
    attacker_name = str(attacker or "").strip().lower()
    default_like = attacker_name in set(UTILITY_GAP_DEFAULT_ATTACKERS)
    if default_like and margin_regime == "disadvantage":
        return float(DISADV_DEFAULT_CHASE_GAP_REQ)
    if default_like and margin_regime == "neutral":
        return float(NEUTRAL_CHASE_GAP_REQ)
    return float(CHASE_UTILITY_GAP_REQ)


def _choose_label(skill_names: Tuple[str, ...], reports: List[Dict], regime: str, attacker: str | None = None) -> int:
    utilities = np.asarray([float(r["utility"]) for r in reports], dtype=np.float32)
    best = int(np.argmax(utilities))
    chase_idx = skill_names.index("chase") if "chase" in skill_names else -1
    protect_idx = skill_names.index("protect") if "protect" in skill_names else -1
    init_margin = float(reports[0]["init"]["margin"])
    safe = init_margin >= SAFE_MARGIN_STEPS
    risky = init_margin <= RISK_MARGIN_STEPS
    margin_regime = _margin_regime_name(regime)

    if (
        bool(ENABLE_UTILITY_GAP_LABEL)
        and chase_idx >= 0
        and protect_idx >= 0
        and best == chase_idx
        and not reports[protect_idx]["bad"]
    ):
        chase_gap = float(utilities[chase_idx] - utilities[protect_idx])
        if chase_gap < _utility_gap_required(regime, attacker):
            return int(protect_idx)

    # If chase is almost as good in safe states, force the aggressive label.
    if chase_idx >= 0 and safe:
        if (utilities[best] - utilities[chase_idx]) <= CHASE_TOLERANCE_SAFE and not reports[chase_idx]["bad"]:
            return int(chase_idx)

    if chase_idx >= 0 and best == chase_idx and risky:
        defensive = [idx for idx in (protect_idx,) if idx >= 0]
        if defensive:
            defensive_best = max(defensive, key=lambda idx: float(utilities[idx]))
            if (utilities[chase_idx] - utilities[defensive_best]) < RISK_CHASE_BONUS_REQ:
                return int(defensive_best)

    if protect_idx >= 0 and PROTECT_FALLBACK_TOL > 0.0 and not reports[protect_idx]["bad"]:
        protect_gap = float(utilities[best] - utilities[protect_idx])
        chase_clear = False
        if chase_idx >= 0 and not reports[chase_idx]["bad"]:
            chase_clear = (
                float(utilities[chase_idx] - utilities[protect_idx])
                >= float(PROTECT_CHASE_CLEAR_MARGIN)
            )
        if protect_gap <= float(PROTECT_FALLBACK_TOL) and not chase_clear:
            return int(protect_idx)

    # Protect should not dominate neutral/advantage unless it clearly wins.
    if protect_idx >= 0 and best == protect_idx and margin_regime != "disadvantage":
        nonbase = [idx for idx, name in enumerate(skill_names) if name != "protect"]
        nonbase_best = max(nonbase, key=lambda idx: float(utilities[idx]))
        if (utilities[protect_idx] - utilities[nonbase_best]) <= NONBASE_TOLERANCE:
            return int(nonbase_best)

    # In disadvantage, allow chase only when it is meaningfully better than
    # the best defensive option. This is the "bad cases preserve win-rate" bias.
    if chase_idx >= 0 and best == chase_idx and margin_regime == "disadvantage" and risky:
        defensive = [idx for idx in (protect_idx,) if idx >= 0]
        if defensive:
            defensive_best = max(defensive, key=lambda idx: float(utilities[idx]))
            if (utilities[chase_idx] - utilities[defensive_best]) < DISADV_CHASE_BONUS_REQ:
                return int(defensive_best)

    return int(best)


def _soft_utility_targets(rows: List[Dict], device: torch.device) -> torch.Tensor:
    gaps = []
    scale = max(1e-6, float(SOFT_UTILITY_GAP_SCALE))
    floor = float(SOFT_UTILITY_TARGET_FLOOR)
    ceil = float(SOFT_UTILITY_TARGET_CEIL)
    if floor > ceil:
        floor, ceil = ceil, floor
    for row in rows:
        utilities = row.get("utilities", [0.0, 0.0])
        protect_u = float(utilities[0]) if len(utilities) > 0 else 0.0
        chase_u = float(utilities[1]) if len(utilities) > 1 else protect_u
        gaps.append((chase_u - protect_u) / scale)
    target = torch.sigmoid(torch.as_tensor(gaps, dtype=torch.float32, device=device))
    return torch.clamp(target, min=floor, max=ceil)


def _soft_utility_aux_loss(logits: torch.Tensor, rows: List[Dict], device: torch.device) -> torch.Tensor:
    if not bool(ENABLE_SOFT_UTILITY_AUX) or float(SOFT_UTILITY_AUX_WEIGHT) <= 0.0:
        return torch.zeros((), dtype=logits.dtype, device=device)
    if logits.shape[-1] < 2 or not rows:
        return torch.zeros((), dtype=logits.dtype, device=device)
    target = _soft_utility_targets(rows, device=device).to(dtype=logits.dtype)
    chase_minus_protect = logits[:, 1] - logits[:, 0]
    return F.binary_cross_entropy_with_logits(chase_minus_protect, target) * float(SOFT_UTILITY_AUX_WEIGHT)


def _margin_regime_name(regime: str) -> str:
    text = str(regime).strip().lower()
    if ":" in text:
        _speed, margin = text.split(":", 1)
        return margin.strip()
    if "_speed__" in text and "_margin" in text:
        return text.split("_speed__", 1)[1].split("_margin", 1)[0]
    return text


def _regime_reset_options(regime: str) -> Dict[str, str]:
    text = str(regime).strip().lower()
    if ":" in text:
        speed, margin = text.split(":", 1)
        return {"speed_regime": speed.strip(), "margin_regime": margin.strip()}
    return {"regime": text}


def _pad_window(history: deque, dim: int) -> np.ndarray:
    items = list(history)[-SEQ_LEN:]
    if not items:
        return np.zeros((SEQ_LEN, dim), dtype=np.float32)
    while len(items) < SEQ_LEN:
        items.insert(0, items[0])
    return np.stack(items[-SEQ_LEN:], axis=0).astype(np.float32)


def _top_action_from_window(net, actor_window, critic_window, device: torch.device) -> int:
    net.eval()
    with torch.no_grad():
        actor = torch.as_tensor(actor_window[None, :, :], dtype=torch.float32, device=device)
        critic = torch.as_tensor(critic_window[None, :, :], dtype=torch.float32, device=device)
        mean, _value, _log_std = net(actor, critic)
        logits = mean[:, -1, :int(NUM_SKILLS)]
        logits = apply_chase_logit_bias(
            logits,
            skill_names=("protect", "chase"),
            chase_logit_bias=TOP_CHASE_LOGIT_BIAS,
        )
        return int(torch.argmax(logits, dim=-1).item())


def _sample_attacker(rng: np.random.Generator) -> str:
    return str(rng.choice(np.asarray(ATTACKERS, dtype=object)))


def _sample_regime(rng: np.random.Generator) -> str:
    probs = np.asarray(REGIME_PROBS, dtype=np.float64)
    if probs.size != len(REGIMES) or probs.sum() <= 0:
        probs = np.ones((len(REGIMES),), dtype=np.float64)
    probs = probs / probs.sum()
    return str(rng.choice(np.asarray(REGIMES, dtype=object), p=probs))


def _collect_rows(
    net,
    device: torch.device,
    rng: np.random.Generator,
    episodes: int,
    seed_base: int,
    split: str,
) -> Tuple[List[Dict], Dict]:
    rows = []
    label_counts = Counter()
    rollout_counts = Counter()
    outcome_counts = Counter()
    utility_sums = defaultdict(float)
    t0 = time.time()

    for ep in range(int(episodes)):
        attacker = _sample_attacker(rng)
        regime = _sample_regime(rng)
        os.environ["TAD_REGIME_RANDOMIZATION"] = "1"
        if ":" in str(regime):
            os.environ["TAD_REGIME_DECOUPLED"] = "1"
            speed_regime, margin_regime = str(regime).split(":", 1)
            os.environ["TAD_SPEED_REGIME"] = speed_regime.strip()
            os.environ["TAD_MARGIN_REGIME"] = margin_regime.strip()
        else:
            os.environ["TAD_REGIME_DECOUPLED"] = "0"
            os.environ.pop("TAD_SPEED_REGIME", None)
            os.environ.pop("TAD_MARGIN_REGIME", None)
            os.environ["TAD_REGIME"] = regime
        env = _make_env(attacker, device)
        obs, info = env.reset(seed=int(seed_base + ep), options=_regime_reset_options(regime))
        skill_names = tuple(env.skill_names)
        actor_hist = deque(maxlen=SEQ_LEN)
        critic_hist = deque(maxlen=SEQ_LEN)
        done = False
        steps = 0
        sampled = 0
        last_info = dict(info or {})

        while not done and steps < MAX_STEPS:
            defender_obs, attacker_obs = obs if isinstance(obs, tuple) else (obs, obs)
            actor_vec = np.asarray(defender_obs, dtype=np.float32).copy()
            critic_vec = np.asarray(build_critic_observation(defender_obs, attacker_obs), dtype=np.float32).copy()
            actor_hist.append(actor_vec)
            critic_hist.append(critic_vec)
            actor_window = _pad_window(actor_hist, NetParameters.ACTOR_RAW_LEN)
            critic_window = _pad_window(critic_hist, NetParameters.CRITIC_RAW_LEN)

            should_label = (
                (steps % max(1, int(SAMPLE_INTERVAL)) == 0 and sampled < int(MAX_SAMPLES_PER_EP))
                or (rng.random() < ROLLOUT_LABEL_RATE / max(1, int(MAX_STEPS)))
            )
            if should_label:
                root = env.snapshot_state()
                reports = []
                for skill_idx in range(len(skill_names)):
                    env.restore_state(root)
                    reports.append(_rollout_skill(env, skill_idx, HORIZON))
                env.restore_state(root)
                label = _choose_label(skill_names, reports, regime, attacker=attacker)
                rows.append({
                    "actor": actor_window,
                    "critic": critic_window,
                    "label": int(label),
                    "regime": regime,
                    "attacker": attacker,
                    "step": int(steps),
                    "margin": float(reports[0]["init"]["margin"]),
                    "utilities": [float(r["utility"]) for r in reports],
                    "bad": [bool(r["bad"]) for r in reports],
                })
                label_counts[skill_names[label]] += 1
                for idx, report in enumerate(reports):
                    utility_sums[skill_names[idx]] += float(report["utility"])
                sampled += 1

            if rng.random() < ROLLOUT_RANDOM_RATE:
                skill_idx = int(rng.integers(0, len(skill_names)))
            else:
                skill_idx = _top_action_from_window(net, actor_window, critic_window, device)
            rollout_counts[skill_names[int(skill_idx)]] += 1
            obs, _reward, term, trunc, last_info = env.step(np.asarray([skill_idx], dtype=np.float32))
            done = bool(term or trunc)
            steps += 1

        d_win, d_cap, d_col, a_cap, timeout = _classify(last_info or {})
        outcome_counts["win"] += int(d_win)
        outcome_counts["capture"] += int(d_cap)
        outcome_counts["collision"] += int(d_col)
        outcome_counts["attacker_capture"] += int(a_cap)
        outcome_counts["timeout_win"] += int(timeout and d_win)
        del env

        if (ep + 1) % 10 == 0 or (ep + 1) == int(episodes):
            elapsed = (time.time() - t0) / 60.0
            print(
                f"[{split}] ep={ep + 1}/{episodes} rows={len(rows)} "
                f"labels={dict(label_counts)} rollout={dict(rollout_counts)} elapsed={elapsed:.1f}m",
                flush=True,
            )

    n_util = max(1, sum(label_counts.values()) * max(1, len(label_counts)))
    stats = {
        "rows": len(rows),
        "labels": dict(label_counts),
        "rollout_counts": dict(rollout_counts),
        "outcomes": dict(outcome_counts),
        "avg_utilities": {k: float(v / max(1, sum(label_counts.values()))) for k, v in utility_sums.items()},
        "episodes": int(episodes),
    }
    return rows, stats


def _balanced_batches(rows: List[Dict], rng: np.random.Generator):
    by_label = defaultdict(list)
    for idx, row in enumerate(rows):
        by_label[int(row["label"])].append(int(idx))
    labels = sorted(label for label, items in by_label.items() if items)
    if not labels:
        return
    n_batches = max(1, int(np.ceil(len(rows) / max(1, int(BATCH_SIZE)))))
    per_label = max(1, int(np.ceil(int(BATCH_SIZE) / max(1, len(labels)))))
    for _ in range(n_batches):
        batch = []
        for label in labels:
            src = np.asarray(by_label[label], dtype=np.int64)
            take = rng.choice(src, size=per_label, replace=(len(src) < per_label))
            batch.extend(int(x) for x in take.tolist())
        rng.shuffle(batch)
        idxs = np.asarray(batch[:int(BATCH_SIZE)], dtype=np.int64)
        actor = np.stack([rows[int(i)]["actor"] for i in idxs], axis=0).astype(np.float32)
        critic = np.stack([rows[int(i)]["critic"] for i in idxs], axis=0).astype(np.float32)
        labels_np = np.asarray([rows[int(i)]["label"] for i in idxs], dtype=np.int64)
        batch_rows = [rows[int(i)] for i in idxs]
        yield actor, critic, labels_np, batch_rows


def _eval_rows(net, rows: List[Dict], device: torch.device) -> Dict:
    net.eval()
    total = 0
    correct = 0
    pred_counter = Counter()
    by_regime = defaultdict(lambda: [0, 0])
    by_margin = defaultdict(lambda: [0, 0])
    with torch.no_grad():
        for start in range(0, len(rows), 384):
            chunk = rows[start:start + 384]
            actor = torch.as_tensor(np.stack([r["actor"] for r in chunk]), dtype=torch.float32, device=device)
            critic = torch.as_tensor(np.stack([r["critic"] for r in chunk]), dtype=torch.float32, device=device)
            labels = np.asarray([r["label"] for r in chunk], dtype=np.int64)
            mean, _value, _log_std = net(actor, critic)
            logits = apply_chase_logit_bias(
                mean[:, -1, :int(NUM_SKILLS)],
                skill_names=("protect", "chase"),
                chase_logit_bias=TOP_CHASE_LOGIT_BIAS,
            )
            pred = torch.argmax(logits, dim=-1).detach().cpu().numpy()
            total += len(chunk)
            correct += int((pred == labels).sum())
            for p, y, row in zip(pred, labels, chunk):
                pred_counter[int(p)] += 1
                by_regime[str(row["regime"])][0] += int(p == y)
                by_regime[str(row["regime"])][1] += 1
                margin = float(row.get("margin", 0.0))
                bucket = "safe" if margin >= SAFE_MARGIN_STEPS else ("risk" if margin <= RISK_MARGIN_STEPS else "mid")
                by_margin[bucket][0] += int(p == y)
                by_margin[bucket][1] += 1
    return {
        "acc": float(correct / max(1, total)),
        "pred": dict(pred_counter),
        "by_regime": {k: float(v[0] / max(1, v[1])) for k, v in by_regime.items()},
        "by_margin": {k: float(v[0] / max(1, v[1])) for k, v in by_margin.items()},
    }


def _save(net, out_dir: Path, name: str, step: int, score: float, extra: Dict):
    payload = {
        "model": {k: v.detach().cpu() for k, v in net.state_dict().items()},
        "network_type": "hrl_top_dual_gru_raw",
        "step": int(step),
        "reward": float(score),
        "hrl_num_skills": int(NUM_SKILLS),
        "hrl_duration_bins": (1,),
        "hrl_top_discrete_action_dim": int(NUM_SKILLS),
        "regime_state_cf_top": True,
        "skill_layout": SKILL_LAYOUT,
    }
    payload.update(dict(extra or {}))
    torch.save(payload, out_dir / name)


def main():
    print_training_process_info("train_regime_state_cf_top")
    if RISK_METRIC not in {"astar", "euclidean"}:
        raise ValueError(f"SCF_RISK_METRIC must be 'astar' or 'euclidean', got {RISK_METRIC!r}")
    os.environ.setdefault("TAD_REGIME_RANDOMIZATION", "1")
    os.environ.setdefault("TAD_REGIME_PROBS", ",".join(str(x) for x in REGIME_PROBS))
    set_global_seeds(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    device = torch.device("cpu" if DEVICE_NAME == "cpu" or not torch.cuda.is_available() else "cuda")
    torch.set_num_threads(max(1, _env_int("SCF_TORCH_THREADS", 2)))
    resolved_init_top_path = _resolve_init_top_path(INIT_TOP_PATH, ALLOW_RANDOM_INIT)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(OUTPUT_DIR or CHECKPOINTS_DIR / f"hrl_regime_state_cf_top_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "seed": SEED,
        "iters": ITERS,
        "episodes_per_iter": EPISODES_PER_ITER,
        "val_episodes": VAL_EPISODES,
        "epochs_per_iter": EPOCHS_PER_ITER,
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "horizon": HORIZON,
        "sample_interval": SAMPLE_INTERVAL,
        "max_samples_per_ep": MAX_SAMPLES_PER_EP,
        "protect_class_weight": PROTECT_CLASS_WEIGHT,
        "chase_class_weight": CHASE_CLASS_WEIGHT,
        "top_chase_logit_bias": TOP_CHASE_LOGIT_BIAS,
        "enable_utility_gap_label": ENABLE_UTILITY_GAP_LABEL,
        "chase_utility_gap_req": CHASE_UTILITY_GAP_REQ,
        "neutral_chase_gap_req": NEUTRAL_CHASE_GAP_REQ,
        "disadv_default_chase_gap_req": DISADV_DEFAULT_CHASE_GAP_REQ,
        "enable_soft_utility_aux": ENABLE_SOFT_UTILITY_AUX,
        "soft_utility_aux_weight": SOFT_UTILITY_AUX_WEIGHT,
        "soft_utility_gap_scale": SOFT_UTILITY_GAP_SCALE,
        "soft_utility_target_floor": SOFT_UTILITY_TARGET_FLOOR,
        "soft_utility_target_ceil": SOFT_UTILITY_TARGET_CEIL,
        "utility_gap_default_attackers": UTILITY_GAP_DEFAULT_ATTACKERS,
        "risk_metric": RISK_METRIC,
        "astar_grid_size": ASTAR_GRID_SIZE,
        "astar_obstacle_padding": ASTAR_OBSTACLE_PADDING,
        "attackers": ATTACKERS,
        "regimes": REGIMES,
        "regime_probs": REGIME_PROBS,
        "skill_layout": SKILL_LAYOUT,
        "num_skills": NUM_SKILLS,
        "init_top_path": str(resolved_init_top_path) if resolved_init_top_path is not None else "",
        "init_top_requested_path": INIT_TOP_PATH,
        "init_mode": "checkpoint" if resolved_init_top_path is not None else "random_explicit",
        "allow_random_init": ALLOW_RANDOM_INIT,
        "chase_path": CHASE_PATH,
        "protect_path": PROTECT_PATH,
        "safe_margin_steps": SAFE_MARGIN_STEPS,
        "risk_margin_steps": RISK_MARGIN_STEPS,
        "chase_tolerance_safe": CHASE_TOLERANCE_SAFE,
        "nonbase_tolerance": NONBASE_TOLERANCE,
        "disadv_chase_bonus_req": DISADV_CHASE_BONUS_REQ,
        "risk_chase_bonus_req": RISK_CHASE_BONUS_REQ,
        "base_chase_weight": BASE_CHASE_WEIGHT,
        "safe_chase_weight": SAFE_CHASE_WEIGHT,
        "base_safety_weight": BASE_SAFETY_WEIGHT,
        "risk_safety_weight": RISK_SAFETY_WEIGHT,
        "base_delay_weight": BASE_DELAY_WEIGHT,
        "risk_delay_weight": RISK_DELAY_WEIGHT,
        "protect_fallback_tol": PROTECT_FALLBACK_TOL,
        "protect_chase_clear_margin": PROTECT_CHASE_CLEAR_MARGIN,
        "capture_reward": CAPTURE_REWARD,
        "timeout_win_reward": TIMEOUT_WIN_REWARD,
        "attacker_capture_penalty": ATTACKER_CAPTURE_PENALTY,
        "defender_collision_penalty": DEFENDER_COLLISION_PENALTY,
        "collision_step_penalty": COLLISION_STEP_PENALTY,
        "min_margin_penalty": MIN_MARGIN_PENALTY,
        "radar_danger_dist": RADAR_DANGER_DIST,
        "radar_danger_penalty": RADAR_DANGER_PENALTY,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("=" * 100)
    print("Regime State-CF Top Training")
    print(json.dumps(config, indent=2))
    print("=" * 100)

    net = _make_top(device, resolved_init_top_path)
    opt = torch.optim.AdamW(net.parameters(), lr=float(LR), weight_decay=float(WEIGHT_DECAY))
    skill_names = ("protect", "chase")
    class_weights = build_two_skill_class_weights(
        skill_names,
        protect_weight=PROTECT_CLASS_WEIGHT,
        chase_weight=CHASE_CLASS_WEIGHT,
        device=device,
    )
    replay: List[Dict] = []
    history = []
    best_score = -1.0

    val_rows, val_stats = _collect_rows(net, device, rng, VAL_EPISODES, SEED + 700_000, "val0")
    (out_dir / "val0_stats.json").write_text(json.dumps(val_stats, indent=2), encoding="utf-8")

    for it in range(1, int(ITERS) + 1):
        train_rows, train_stats = _collect_rows(net, device, rng, EPISODES_PER_ITER, SEED + it * 10_000, f"train{it}")
        replay.extend(train_rows)
        if len(replay) > int(REPLAY_MAX_ROWS):
            replay = replay[-int(REPLAY_MAX_ROWS):]

        losses = []
        ce_losses = []
        aux_losses = []
        for epoch in range(1, int(EPOCHS_PER_ITER) + 1):
            net.train()
            for actor_np, critic_np, labels_np, batch_rows in _balanced_batches(replay, rng):
                actor = torch.as_tensor(actor_np, dtype=torch.float32, device=device)
                critic = torch.as_tensor(critic_np, dtype=torch.float32, device=device)
                labels = torch.as_tensor(labels_np, dtype=torch.long, device=device)
                mean, _value, _log_std = net(actor, critic)
                logits = mean[:, -1, :int(NUM_SKILLS)]
                ce_loss = F.cross_entropy(
                    logits,
                    labels,
                    weight=class_weights,
                    label_smoothing=float(LABEL_SMOOTHING),
                )
                aux_loss = _soft_utility_aux_loss(logits, batch_rows, device=device)
                loss = ce_loss + aux_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
                losses.append(float(loss.detach().cpu()))
                ce_losses.append(float(ce_loss.detach().cpu()))
                aux_losses.append(float(aux_loss.detach().cpu()))

        train_eval = _eval_rows(net, replay[-min(len(replay), 8000):], device)
        val_eval = _eval_rows(net, val_rows, device)
        label_counter = Counter(int(row["label"]) for row in replay)
        target_rate = 1.0 / max(1, int(NUM_SKILLS))
        score = (
            float(val_eval["acc"])
            - sum(
                0.05 * abs(float(val_eval["pred"].get(i, 0)) / max(1, len(val_rows)) - target_rate)
                for i in range(int(NUM_SKILLS))
            )
        )
        row = {
            "iter": int(it),
            "loss": float(np.mean(losses)) if losses else 0.0,
            "ce_loss": float(np.mean(ce_losses)) if ce_losses else 0.0,
            "soft_utility_aux_loss": float(np.mean(aux_losses)) if aux_losses else 0.0,
            "replay_rows": int(len(replay)),
            "replay_labels": dict(label_counter),
            "train_stats": train_stats,
            "train_eval": train_eval,
            "val_eval": val_eval,
            "score": float(score),
        }
        history.append(row)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        _save(net, out_dir, "last_model.pth", it, score, {"history_tail": row})
        print(
            f"[Iter {it:02d}] loss={row['loss']:.4f} replay={len(replay)} "
            f"labels={dict(label_counter)} val_acc={val_eval['acc']:.3f} "
            f"val_pred={val_eval['pred']} by_margin={val_eval['by_margin']} score={score:.3f}",
            flush=True,
        )
        if score > best_score:
            best_score = float(score)
            _save(net, out_dir, "best_model.pth", it, best_score, {"history_tail": row})
            print(f"[Save] best score={best_score:.3f} -> {out_dir / 'best_model.pth'}", flush=True)

    print(f"[Done] saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
