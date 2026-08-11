"""Defender-opponent pool for attacker training."""

from __future__ import annotations

import glob
import json
import math
import os
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from configs import map_config
from configs.skill_config import NetParameters
from envs import env_lib
from networks import create_network
from policies.defender_global import DefenderGlobalPolicy
from policies.defender_hrl_apollonius import DefenderHRLApolloniusLabelPolicy
from policies.defender_hrl_rule import DefenderHRLRulePolicy
from policies.defender_reach_avoid import create_reach_avoid_defender_policy
from skill.model import Model
from skill.util import build_critic_observation, get_device


RULE_DEFENDER_STRATEGIES = (
    "chase",
    "protect",
    "astar_to_attacker",
    "astar_to_target",
    "cbf_qp",
    "cbf_qp_local",
    "cbf_qp_local_obs",
)
LEARNED_SKILL_DEFENDER_STRATEGIES = (
    "skill_primary",
    "skill_chase",
    "skill_protect",
)
LEARNED_HIERARCHICAL_DEFENDER_STRATEGIES = (
    "hrl",
)
HIERARCHICAL_DEFENDER_STRATEGIES = (
    "hrl_rule_geo_trend",
    "hrl_rule_apollonius_label",
)
SUPPORTED_DEFENDER_STRATEGIES = (
    RULE_DEFENDER_STRATEGIES
    + LEARNED_SKILL_DEFENDER_STRATEGIES
    + LEARNED_HIERARCHICAL_DEFENDER_STRATEGIES
    + HIERARCHICAL_DEFENDER_STRATEGIES
)

DEFENDER_POLICY_CHECKPOINT_PARAMS = {
    "skill_primary": "primary_skill_path",
    "skill_chase": "chase_skill_path",
    "skill_protect": "protect_skill_path",
    "hrl": "top_policy_path",
}

FORMAL_SKILL_MODEL_DIRS = {
    "protect": os.path.join("models", "defender_protect_mlp_ctde_frozen6_20260721_105148"),
    "chase": os.path.join("models", "defender_chase_nmn_dual_gru_raw_dense_05-05-19-12"),
}
HRL_PROTECT_MODEL_PATH = os.path.join(
    "models", "defender_protect_mlp_ctde_repro_20260526", "final_model.pth"
)


def env_json_object(name: str, default: Optional[Mapping] = None) -> Dict:
    """Read an optional environment variable as a JSON object."""
    raw = os.environ.get(str(name))
    if raw is None or not raw.strip():
        return dict(default or {})
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must contain valid JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return dict(value)


def normalize_defender_policy_specs(
    policy_specs: Optional[Mapping[str, Mapping]] = None,
) -> Dict[str, Dict]:
    """Validate named Defender policies and resolve generic checkpoint fields."""
    if policy_specs is None:
        return {}
    if not isinstance(policy_specs, Mapping):
        raise ValueError("defender_policy_specs must be a JSON object keyed by alias")

    normalized = {}
    reserved_aliases = set(SUPPORTED_DEFENDER_STRATEGIES) | {"random"}
    for raw_alias, raw_spec in policy_specs.items():
        alias = str(raw_alias).strip().lower()
        if not alias:
            raise ValueError("defender policy alias must be non-empty")
        if alias in reserved_aliases:
            raise ValueError(f"defender policy alias conflicts with a built-in strategy: {alias!r}")
        if alias in normalized:
            raise ValueError(f"duplicate defender policy alias after normalization: {alias!r}")
        if not isinstance(raw_spec, Mapping):
            raise ValueError(f"defender_policy_specs[{alias!r}] must be a JSON object")

        unknown_fields = set(raw_spec) - {"strategy", "checkpoint", "params", "controller_config"}
        if unknown_fields:
            raise ValueError(
                f"defender_policy_specs[{alias!r}] has unknown fields: {sorted(unknown_fields)}"
            )

        strategy = str(raw_spec.get("strategy", "")).strip().lower()
        if strategy not in SUPPORTED_DEFENDER_STRATEGIES:
            raise ValueError(
                f"defender_policy_specs[{alias!r}].strategy must be one of "
                f"{SUPPORTED_DEFENDER_STRATEGIES}, got {strategy!r}"
            )

        params_sources = [
            value
            for value in (raw_spec.get("params"), raw_spec.get("controller_config"))
            if value is not None
        ]
        if len(params_sources) > 1:
            raise ValueError(
                f"defender_policy_specs[{alias!r}] must use only one of "
                "'params' or 'controller_config'"
            )
        if params_sources and not isinstance(params_sources[0], Mapping):
            raise ValueError(
                f"defender_policy_specs[{alias!r}] controller parameters must be a JSON object"
            )
        params = dict(params_sources[0]) if params_sources else {}

        checkpoint = raw_spec.get("checkpoint")
        if checkpoint is not None:
            checkpoint = str(checkpoint).strip()
            if not checkpoint:
                raise ValueError(f"defender_policy_specs[{alias!r}].checkpoint must be non-empty")
            checkpoint_param = DEFENDER_POLICY_CHECKPOINT_PARAMS.get(strategy)
            if checkpoint_param is None:
                raise ValueError(
                    f"defender_policy_specs[{alias!r}].checkpoint is not supported for "
                    f"strategy {strategy!r}; use params with explicit controller paths"
                )
            configured_path = params.get(checkpoint_param)
            if configured_path not in (None, "") and str(configured_path) != checkpoint:
                raise ValueError(
                    f"defender_policy_specs[{alias!r}] defines conflicting checkpoint paths"
                )
            params[checkpoint_param] = checkpoint

        normalized[alias] = {
            "strategy": strategy,
            "controller_config": params,
        }
    return normalized


def resolve_defender_policy_spec(
    strategy: str,
    controller_config: Optional[Dict] = None,
    policy_specs: Optional[Mapping[str, Mapping]] = None,
) -> Tuple[str, Dict]:
    """Resolve an alias to a built-in strategy and merged controller config."""
    key = str(strategy).strip().lower()
    specs = normalize_defender_policy_specs(policy_specs)
    if key not in specs:
        return key, dict(controller_config or {})

    spec = specs[key]
    merged_config = dict(controller_config or {})
    merged_config.update(spec["controller_config"])
    return str(spec["strategy"]), merged_config


def _resolve_runtime_device(device: Optional[str]):
    device_spec = "auto" if device is None else str(device).strip().lower()
    if device_spec in ("", "auto"):
        return get_device(prefer_gpu=True)
    if device_spec == "cpu":
        return torch.device("cpu")
    if device_spec.startswith("cuda"):
        return get_device(prefer_gpu=True)
    return torch.device(device_spec)


def _find_latest_checkpoint(model_prefixes: Sequence[str]) -> Optional[str]:
    candidates = []
    for prefix in model_prefixes:
        patterns = [
            os.path.join("models", f"{prefix}_*", "best_model.pth"),
        ]
        for pattern in patterns:
            candidates.extend(glob.glob(pattern))
    candidates = [path for path in candidates if os.path.isfile(path)]
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def _find_latest_file(patterns: Sequence[str]) -> Optional[str]:
    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))
    candidates = [path for path in candidates if os.path.isfile(path)]
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def _find_formal_skill_checkpoint(strategy: str) -> Optional[str]:
    model_dir = FORMAL_SKILL_MODEL_DIRS.get(strategy)
    if model_dir is None:
        return None
    for filename in ("best_balanced_model.pth", "final_model.pth", "best_model.pth"):
        path = os.path.join(model_dir, filename)
        if os.path.isfile(path):
            return path
    return None


def _find_latest_hrl_checkpoint() -> Optional[str]:
    preferred = os.path.join(
        "models",
        "hrl_ch2_m1_astar_cached_top_20260606_170036",
        "best_model.pth",
    )
    if os.path.isfile(preferred):
        return preferred
    patterns = [
        os.path.join("models", "hrl_ch2_*_top_*", "best_model.pth"),
    ]
    return _find_latest_file(patterns)


def _default_model_path(strategy: str) -> Optional[str]:
    key = str(strategy).strip().lower()
    if key == "hrl":
        return _find_latest_hrl_checkpoint()

    formal_path = _find_formal_skill_checkpoint(key)
    if formal_path is not None:
        return formal_path

    prefix_map = {
        "protect": ["defender_protect_mlp_ctde_frozen6", "defender_protect_dense"],
        "chase": ["defender_chase_nmn_dual_gru_raw_dense"],
    }
    prefixes = prefix_map.get(key)
    if not prefixes:
        return None
    for prefix in prefixes:
        path = _find_latest_checkpoint((prefix,))
        if path is not None:
            return path
    return None


def _resolve_hrl_skill_paths(
    num_skills: int = 2,
    primary_path: Optional[str] = None,
    chase_path: Optional[str] = None,
) -> Tuple[str, str]:
    num_skills = int(num_skills)
    if num_skills != 2:
        raise ValueError(f"hrl_num_skills must be 2, got {num_skills!r}")

    primary_path = None if primary_path is None else str(primary_path).strip() or None
    chase_path = None if chase_path is None else str(chase_path).strip() or None
    primary_path = primary_path or HRL_PROTECT_MODEL_PATH
    chase_path = chase_path or _default_model_path("chase")

    if not primary_path or not os.path.exists(primary_path):
        raise FileNotFoundError(f"Missing primary HRL skill checkpoint: {primary_path}")
    if not chase_path or not os.path.exists(chase_path):
        raise FileNotFoundError(f"Missing chase HRL skill checkpoint: {chase_path}")
    return primary_path, chase_path


def _compat_numpy_checkpoint_load(path: str, device):
    import numpy as _np
    import sys as _sys

    if not hasattr(_np, "_core"):
        _sys.modules["numpy._core"] = _np.core
        _sys.modules["numpy._core.multiarray"] = _np.core.multiarray
    return torch.load(path, map_location=device, weights_only=False)


def _detect_network_type(state_dict) -> str:
    keys = set(state_dict.keys())
    has_hrl_top_marker = ("hrl_top_marker" in keys) or ("discrete_policy_marker" in keys)
    if "shared_tracking_branch.0.weight" in keys:
        return "nmn_ctde_task_shared"
    if "shared_radar_encoder.net.0.weight" in keys:
        return "nmn_ctde_shared"
    if "actor_tracking_gru.weight_ih_l0" in keys and "actor_obstacle_gru.weight_ih_l0" in keys:
        return "nmn_dual_gru_raw"
    has_tracking = any("tracking_branch" in key for key in keys)
    has_actor_backbone = any("actor_backbone" in key for key in keys)
    has_actor_gru = any("actor_gru" in key for key in keys)
    if has_tracking and has_actor_gru:
        return "nmn_gru"
    if has_actor_gru:
        action_dim = None
        if "log_std" in state_dict and hasattr(state_dict["log_std"], "shape"):
            action_dim = int(state_dict["log_std"].shape[0])
        elif "policy_mean.weight" in state_dict and hasattr(state_dict["policy_mean.weight"], "shape"):
            action_dim = int(state_dict["policy_mean.weight"].shape[0])
        if action_dim is not None and action_dim >= 3:
            return "hrl_top_gru"
        return "mlp_gru"
    if has_tracking:
        critic_in_dim = None
        if "critic_backbone.0.weight" in state_dict and hasattr(state_dict["critic_backbone.0.weight"], "shape"):
            critic_in_dim = int(state_dict["critic_backbone.0.weight"].shape[1])
        return "nmn_ctde" if critic_in_dim == NetParameters.CRITIC_VECTOR_LEN else "nmn"
    if has_actor_backbone:
        action_dim = None
        if "log_std" in state_dict and hasattr(state_dict["log_std"], "shape"):
            action_dim = int(state_dict["log_std"].shape[0])
        elif "policy_mean.weight" in state_dict and hasattr(state_dict["policy_mean.weight"], "shape"):
            action_dim = int(state_dict["policy_mean.weight"].shape[0])
        critic_in_dim = None
        if "critic_backbone.0.weight" in state_dict and hasattr(state_dict["critic_backbone.0.weight"], "shape"):
            critic_in_dim = int(state_dict["critic_backbone.0.weight"].shape[1])
        if has_hrl_top_marker or (action_dim is not None and action_dim >= 3):
            return "hrl_top_noctde" if critic_in_dim == NetParameters.ACTOR_VECTOR_LEN else "hrl_top"
        return "mlp_noctde" if critic_in_dim == NetParameters.ACTOR_VECTOR_LEN else "mlp_ctde"
    return "nmn"


def _load_skill_model(model_path: str, device, skill_name: str = "skill"):
    checkpoint = _compat_numpy_checkpoint_load(model_path, device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    network_type = None
    if isinstance(checkpoint, dict) and checkpoint.get("network_type"):
        network_type = str(checkpoint["network_type"]).strip().lower()
    if not network_type:
        network_type = _detect_network_type(state_dict)
    if network_type in {"hrl_top", "hrl_top_noctde", "hrl_top_gru"}:
        raise ValueError(
            f"{skill_name} checkpoint points to a top-level HRL model ({network_type}) "
            f"instead of a bottom skill model: {model_path}"
        )

    net = create_network(network_type).to(device)
    net.load_state_dict(state_dict)
    net.eval()
    return net


def _infer_hidden_dim_from_state_dict(state_dict) -> Optional[int]:
    if "actor_backbone.0.weight" in state_dict and hasattr(state_dict["actor_backbone.0.weight"], "shape"):
        return int(state_dict["actor_backbone.0.weight"].shape[0])
    if "actor_in_proj.weight" in state_dict and hasattr(state_dict["actor_in_proj.weight"], "shape"):
        return int(state_dict["actor_in_proj.weight"].shape[0])
    return None


def _infer_action_dim_from_state_dict(state_dict) -> Optional[int]:
    if "log_std" in state_dict and hasattr(state_dict["log_std"], "shape"):
        return int(state_dict["log_std"].shape[0])
    if "policy_mean.weight" in state_dict and hasattr(state_dict["policy_mean.weight"], "shape"):
        return int(state_dict["policy_mean.weight"].shape[0])
    return None


def _load_policy_model(
    model_path: str,
    device,
    policy_name: str,
    hrl_num_skills: Optional[int] = None,
):
    checkpoint = _compat_numpy_checkpoint_load(model_path, device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    network_type = None
    if isinstance(checkpoint, dict) and checkpoint.get("network_type"):
        network_type = str(checkpoint["network_type"]).strip().lower()
    if not network_type:
        network_type = _detect_network_type(state_dict)

    old_hidden_dim = int(NetParameters.HIDDEN_DIM)
    old_action_dim = int(NetParameters.ACTION_DIM)
    old_hrl_num_skills = int(getattr(NetParameters, "HRL_NUM_SKILLS", 2))
    old_hrl_duration_bins = tuple(getattr(NetParameters, "HRL_DURATION_BINS", (1,)))
    old_hrl_num_duration_bins = int(getattr(NetParameters, "HRL_NUM_DURATION_BINS", len(old_hrl_duration_bins)))
    old_hrl_top_action_dim = int(getattr(NetParameters, "HRL_TOP_ACTION_DIM", old_hrl_num_skills + 1))
    old_hrl_top_discrete_action_dim = int(
        getattr(NetParameters, "HRL_TOP_DISCRETE_ACTION_DIM", old_hrl_num_skills)
    )

    try:
        hidden_dim = _infer_hidden_dim_from_state_dict(state_dict)
        if hidden_dim is not None:
            NetParameters.HIDDEN_DIM = int(hidden_dim)

        action_dim = _infer_action_dim_from_state_dict(state_dict)
        if network_type in {"hrl_top", "hrl_top_noctde", "hrl_top_gru"}:
            resolved_num_skills = int(
                checkpoint.get("hrl_num_skills", hrl_num_skills if hrl_num_skills is not None else 2)
            ) if isinstance(checkpoint, dict) else int(hrl_num_skills if hrl_num_skills is not None else 2)
            resolved_num_skills = max(2, resolved_num_skills)
            NetParameters.HRL_NUM_SKILLS = resolved_num_skills
            NetParameters.HRL_DURATION_BINS = (1,)
            NetParameters.HRL_NUM_DURATION_BINS = 1
            NetParameters.HRL_TOP_DISCRETE_ACTION_DIM = resolved_num_skills
            NetParameters.HRL_TOP_ACTION_DIM = max(resolved_num_skills + 1, int(action_dim or (resolved_num_skills + 1)))
        elif action_dim is not None:
            NetParameters.ACTION_DIM = int(action_dim)

        model = Model(device=device, global_model=False, network_type=network_type)
        model.set_weights(state_dict)
        model.network.eval()
        return model, network_type
    except Exception as exc:
        raise RuntimeError(f"Failed to load {policy_name} model from {model_path}: {exc}") from exc
    finally:
        NetParameters.HIDDEN_DIM = old_hidden_dim
        NetParameters.ACTION_DIM = old_action_dim
        NetParameters.HRL_NUM_SKILLS = old_hrl_num_skills
        NetParameters.HRL_DURATION_BINS = old_hrl_duration_bins
        NetParameters.HRL_NUM_DURATION_BINS = old_hrl_num_duration_bins
        NetParameters.HRL_TOP_ACTION_DIM = old_hrl_top_action_dim
        NetParameters.HRL_TOP_DISCRETE_ACTION_DIM = old_hrl_top_discrete_action_dim


def _resolve_single_skill_entry(strategy: str, controller_config: Optional[Dict] = None) -> Tuple[str, str]:
    cfg = dict(controller_config or {})
    key = str(strategy).strip().lower()
    if key == "skill_protect":
        protect_path = str(cfg.get("protect_skill_path") or "").strip() or _default_model_path("protect")
        if not protect_path or not os.path.exists(protect_path):
            raise FileNotFoundError(f"Missing protect skill checkpoint: {protect_path}")
        return protect_path, "protect"

    primary_path, chase_path = _resolve_hrl_skill_paths(
        num_skills=int(cfg.get("hrl_num_skills", 2)),
        primary_path=cfg.get("primary_skill_path"),
        chase_path=cfg.get("chase_skill_path"),
    )
    if key == "skill_primary":
        return primary_path, "protect"
    if key == "skill_chase":
        return chase_path, "chase"
    raise ValueError(f"Unsupported learned skill defender strategy: {strategy}")


def _predict_skill_action(model: Model, defender_obs: np.ndarray, attacker_obs: np.ndarray) -> np.ndarray:
    critic_obs = build_critic_observation(defender_obs, attacker_obs)
    action, _pre_tanh, _value, _log_prob = model.evaluate(defender_obs, critic_obs, greedy=True)
    return np.asarray(action, dtype=np.float32)


def _decode_skill_index(top_action: np.ndarray, num_skills: int) -> int:
    arr = np.asarray(top_action, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return 0
    if arr.size == 1:
        return int(np.clip(int(np.rint(float(arr[0]))), 0, max(0, num_skills - 1)))
    return int(np.argmax(arr[:num_skills]))


class BaseDefenderController:
    """Base interface for defender opponents used during attacker training."""

    def __init__(self, strategy: str):
        self.strategy = str(strategy).strip().lower()

    def reset(self):
        pass

    def get_action(self, defender_obs: np.ndarray, attacker_obs: np.ndarray, env) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _apply_defender_hard_obstacle_mask(action: np.ndarray, env) -> np.ndarray:
        if action is None:
            return action

        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size != 2:
            return action

        base_env = env
        required = ["_control_to_physical", "_simulate_motion", "_encode_action_like_input", "_get_action_limits"]
        if not all(hasattr(base_env, name) for name in required) or not hasattr(base_env, "defender"):
            return action

        normalized_input = bool(np.all(np.abs(arr) <= 1.0 + 1e-6))
        physical = base_env._control_to_physical(arr, role="defender")
        if physical is None:
            return action

        orig_angle, orig_speed = float(physical[0]), float(physical[1])
        max_turn, max_speed, _ = base_env._get_action_limits("defender")
        ref_agent = base_env.defender
        agent_radius = float(getattr(map_config, "agent_radius", getattr(base_env, "pixel_size", 4.0) * 0.5))

        speed_scales = (1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1, 0.0)
        angle_scales = (0.0, -0.2, 0.2, -0.4, 0.4, -0.6, 0.6, -0.8, 0.8, -1.0, 1.0)
        best = None

        for s in speed_scales:
            cand_speed = float(np.clip(orig_speed * s, 0.0, max_speed))
            for a in angle_scales:
                cand_angle = float(np.clip(orig_angle + a * max_turn, -max_turn, max_turn))
                nx, ny = base_env._simulate_motion(ref_agent, cand_angle, cand_speed, role="defender")
                px = float(getattr(base_env, "pixel_size", 4.0))
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
            return base_env._encode_action_like_input(0.0, 0.0, "defender", normalized_input)
        return base_env._encode_action_like_input(best[1], best[2], "defender", normalized_input)


class PrimitiveRuleDefenderController(BaseDefenderController):
    """Single-layer rule-based defender."""

    def __init__(self, strategy: str):
        super().__init__(strategy=strategy)
        key = self.strategy
        if key in ("chase", "astar_to_attacker"):
            self.policy = DefenderGlobalPolicy(skill_mode="chase")
        elif key in ("protect", "astar_to_target"):
            self.policy = DefenderGlobalPolicy(skill_mode="protect")
        elif key in ("cbf_qp", "cbf_qp_local", "cbf_qp_local_obs"):
            self.policy = create_reach_avoid_defender_policy(key)
        else:
            raise ValueError(f"Unsupported primitive defender strategy: {strategy}")

    def reset(self):
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def get_action(self, defender_obs: np.ndarray, attacker_obs: np.ndarray, env) -> np.ndarray:
        del attacker_obs
        privileged_state = env.get_privileged_state()
        action = self.policy.get_action(defender_obs, privileged_state)
        return action


class LearnedSkillDefenderController(BaseDefenderController):
    """Single learned bottom-skill defender."""

    def __init__(self, strategy: str, controller_config: Optional[Dict] = None):
        super().__init__(strategy=strategy)
        cfg = dict(controller_config or {})
        self.device = _resolve_runtime_device(cfg.get("device", "cpu"))
        # Optional evaluation ablation: match the execution shield used by the
        # learned HRL controller without changing standalone-skill defaults.
        self.apply_controller_obstacle_mask = bool(cfg.get("apply_controller_obstacle_mask", False))
        model_path, skill_name = _resolve_single_skill_entry(self.strategy, cfg)
        self.skill_name = str(skill_name)
        self.model, self.network_type = _load_policy_model(
            model_path=model_path,
            device=self.device,
            policy_name=self.skill_name,
        )
        self.last_skill_name: Optional[str] = None
        self.skill_selection_counts = {self.skill_name: 0}

    def reset(self):
        if hasattr(self.model, "reset_gru_sequence"):
            self.model.reset_gru_sequence()
        self.last_skill_name = None
        self.skill_selection_counts = {self.skill_name: 0}

    def get_action(self, defender_obs: np.ndarray, attacker_obs: np.ndarray, env) -> np.ndarray:
        self.last_skill_name = self.skill_name
        self.skill_selection_counts[self.skill_name] += 1
        action = _predict_skill_action(self.model, defender_obs, attacker_obs)
        if self.apply_controller_obstacle_mask:
            return self._apply_defender_hard_obstacle_mask(action, env)
        return action


class HierarchicalRuleDefenderController(BaseDefenderController):
    """Hierarchical defender: rule-based top policy + learned bottom skills."""

    def __init__(
        self,
        strategy: str,
        env,
        num_skills: int = 2,
        primary_skill_path: Optional[str] = None,
        chase_skill_path: Optional[str] = None,
        device: str = "cpu",
    ):
        super().__init__(strategy=strategy)
        self.env = env
        self.device = _resolve_runtime_device(device)
        self.cached_skill_obs = None

        primary_path, chase_path = _resolve_hrl_skill_paths(
            num_skills=num_skills,
            primary_path=primary_skill_path,
            chase_path=chase_skill_path,
        )

        self.primary_skill_name = "protect"
        self.primary_net = _load_skill_model(primary_path, self.device, skill_name=self.primary_skill_name)
        self.chase_net = _load_skill_model(chase_path, self.device, skill_name="chase")

        self.skill_names = [self.primary_skill_name, "chase"]
        self.skill_nets = [self.primary_net, self.chase_net]

        self._skill_actor_hidden = {name: None for name in self.skill_names}
        self._skill_critic_hidden = {name: None for name in self.skill_names}

        if self.strategy == "hrl_rule_geo_trend":
            self.top_policy = DefenderHRLRulePolicy()
        elif self.strategy == "hrl_rule_apollonius_label":
            self.top_policy = DefenderHRLApolloniusLabelPolicy()
        else:
            raise ValueError(f"Unsupported hierarchical defender strategy: {strategy}")

    def reset(self):
        self.cached_skill_obs = None
        if hasattr(self.top_policy, "reset"):
            self.top_policy.reset()
        self._skill_actor_hidden = {name: None for name in self.skill_names}
        self._skill_critic_hidden = {name: None for name in self.skill_names}

    def _skill_action_from_net(self, skill_name: str, net, defender_obs: np.ndarray, attacker_obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_tensor = torch.as_tensor(defender_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            critic_obs = build_critic_observation(defender_obs, attacker_obs)
            critic_tensor = torch.as_tensor(critic_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            if bool(getattr(net, "is_recurrent", False)) and hasattr(net, "forward_recurrent"):
                actor_hidden = self._skill_actor_hidden.get(skill_name)
                critic_hidden = self._skill_critic_hidden.get(skill_name)
                mean, _value, _log_std, next_actor_hidden, next_critic_hidden = net.forward_recurrent(
                    obs_tensor,
                    critic_tensor,
                    actor_hidden=actor_hidden,
                    critic_hidden=critic_hidden,
                )
                self._skill_actor_hidden[skill_name] = next_actor_hidden.detach() if next_actor_hidden is not None else None
                self._skill_critic_hidden[skill_name] = (
                    next_critic_hidden.detach() if next_critic_hidden is not None else None
                )
            else:
                policy_output, _value, _log_std = net(obs_tensor, critic_tensor)
                mean = policy_output
            return torch.tanh(mean).cpu().numpy()[0]

    def _select_skill_index(self, defender_obs: np.ndarray, attacker_obs: np.ndarray) -> int:
        self.cached_skill_obs = (
            np.asarray(defender_obs, dtype=np.float32),
            np.asarray(attacker_obs, dtype=np.float32),
        )
        if self.strategy == "hrl_rule_geo_trend":
            privileged_state = self.env.get_privileged_state()
            top_action = self.top_policy.get_action(defender_obs, privileged_state, skill_names=self.skill_names)
        elif self.strategy == "hrl_rule_apollonius_label":
            top_action = self.top_policy.get_action(defender_obs, self, attacker_obs=attacker_obs)
        else:
            raise ValueError(f"Unsupported hierarchical strategy: {self.strategy}")
        action_arr = np.asarray(top_action, dtype=np.float32).reshape(-1)
        skill_idx = int(np.clip(int(np.rint(float(action_arr[0]))), 0, len(self.skill_names) - 1))
        return skill_idx

    def get_action(self, defender_obs: np.ndarray, attacker_obs: np.ndarray, env) -> np.ndarray:
        del env
        skill_idx = self._select_skill_index(defender_obs, attacker_obs)
        skill_name = self.skill_names[skill_idx]
        net = self.skill_nets[skill_idx]
        action = self._skill_action_from_net(skill_name, net, defender_obs, attacker_obs)
        return self._apply_defender_hard_obstacle_mask(action, self.env)


class LearnedHRLDefenderController(BaseDefenderController):
    """Learned HRL defender: learned top policy + learned bottom skill bank."""

    def __init__(
        self,
        strategy: str,
        env,
        num_skills: int = 2,
        top_policy_path: Optional[str] = None,
        primary_skill_path: Optional[str] = None,
        chase_skill_path: Optional[str] = None,
        device: str = "cpu",
    ):
        super().__init__(strategy=strategy)
        self.env = env
        self.device = _resolve_runtime_device(device)

        primary_path, chase_path = _resolve_hrl_skill_paths(
            num_skills=num_skills,
            primary_path=primary_skill_path,
            chase_path=chase_skill_path,
        )
        self.skill_names = ["protect", "chase"]
        skill_paths = [primary_path, chase_path]

        self.skill_models = {}
        for name, path in zip(self.skill_names, skill_paths):
            model, _network_type = _load_policy_model(
                model_path=path,
                device=self.device,
                policy_name=name,
            )
            self.skill_models[name] = model

        self.num_skills = int(len(self.skill_names))
        self.top_policy_path = top_policy_path or _default_model_path("hrl")
        if not self.top_policy_path or not os.path.exists(self.top_policy_path):
            raise FileNotFoundError(f"Missing learned HRL top checkpoint: {self.top_policy_path}")
        self.top_model, self.top_network_type = _load_policy_model(
            model_path=self.top_policy_path,
            device=self.device,
            policy_name="hrl_top",
            hrl_num_skills=self.num_skills,
        )
        self.last_skill_index: Optional[int] = None
        self.last_skill_name: Optional[str] = None
        self.skill_selection_counts = {name: 0 for name in self.skill_names}

    def reset(self):
        if hasattr(self.top_model, "reset_gru_sequence"):
            self.top_model.reset_gru_sequence()
        for model in self.skill_models.values():
            if hasattr(model, "reset_gru_sequence"):
                model.reset_gru_sequence()
        self.last_skill_index = None
        self.last_skill_name = None
        self.skill_selection_counts = {name: 0 for name in self.skill_names}

    def get_action(self, defender_obs: np.ndarray, attacker_obs: np.ndarray, env) -> np.ndarray:
        critic_obs = build_critic_observation(defender_obs, attacker_obs)
        top_action, _pre_tanh, _value, _log_prob = self.top_model.evaluate(
            defender_obs,
            critic_obs,
            greedy=True,
        )
        skill_idx = _decode_skill_index(top_action, len(self.skill_names))
        skill_name = self.skill_names[skill_idx]
        self.last_skill_index = int(skill_idx)
        self.last_skill_name = str(skill_name)
        self.skill_selection_counts[skill_name] += 1
        action = _predict_skill_action(self.skill_models[skill_name], defender_obs, attacker_obs)
        return self._apply_defender_hard_obstacle_mask(action, env)


def create_defender_controller(
    strategy: str,
    env,
    controller_config: Optional[Dict] = None,
    policy_specs: Optional[Mapping[str, Mapping]] = None,
):
    key, cfg = resolve_defender_policy_spec(
        strategy,
        controller_config=controller_config,
        policy_specs=policy_specs,
    )
    if key in RULE_DEFENDER_STRATEGIES:
        return PrimitiveRuleDefenderController(strategy=key)
    if key in LEARNED_SKILL_DEFENDER_STRATEGIES:
        return LearnedSkillDefenderController(strategy=key, controller_config=cfg)
    if key in LEARNED_HIERARCHICAL_DEFENDER_STRATEGIES:
        return LearnedHRLDefenderController(
            strategy=key,
            env=env,
            num_skills=int(cfg.get("hrl_num_skills", 2)),
            top_policy_path=cfg.get("top_policy_path"),
            primary_skill_path=cfg.get("primary_skill_path"),
            chase_skill_path=cfg.get("chase_skill_path"),
            device=str(cfg.get("device", "cpu")),
        )
    if key in HIERARCHICAL_DEFENDER_STRATEGIES:
        return HierarchicalRuleDefenderController(
            strategy=key,
            env=env,
            num_skills=int(cfg.get("hrl_num_skills", 2)),
            primary_skill_path=cfg.get("primary_skill_path"),
            chase_skill_path=cfg.get("chase_skill_path"),
            device=str(cfg.get("device", "cpu")),
        )
    raise ValueError(f"Unsupported defender strategy: {strategy}. Valid={SUPPORTED_DEFENDER_STRATEGIES}")


__all__ = [
    "RULE_DEFENDER_STRATEGIES",
    "LEARNED_SKILL_DEFENDER_STRATEGIES",
    "LEARNED_HIERARCHICAL_DEFENDER_STRATEGIES",
    "HIERARCHICAL_DEFENDER_STRATEGIES",
    "SUPPORTED_DEFENDER_STRATEGIES",
    "normalize_defender_policy_specs",
    "resolve_defender_policy_spec",
    "create_defender_controller",
]
