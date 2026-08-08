"""Adapters between CoppeliaTrackEnv and existing bottom-skill policies."""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np

from configs.map_config import EnvParameters
from skill.util import build_critic_observation as _build_critic_observation

from .track_env import CoppeliaTrackEnv, VelocityCommand


RADAR_DIM = 64
ACTOR_OBS_DIM = 71
PRIVILEGED_OBS_DIM = 72
CRITIC_OBS_DIM = ACTOR_OBS_DIM + PRIVILEGED_OBS_DIM


def normalize_angle_deg(angle: float) -> float:
    angle = float(angle) % 360.0
    if angle > 180.0:
        angle -= 360.0
    return float(angle)


def _center(agent: dict[str, float], pixel_size: float) -> np.ndarray:
    return np.asarray(
        [
            float(agent["x"]) + float(pixel_size) * 0.5,
            float(agent["y"]) + float(pixel_size) * 0.5,
        ],
        dtype=np.float32,
    )


def _state_from_transition(transition_or_state: dict[str, Any]) -> dict[str, Any]:
    if "state" in transition_or_state:
        return dict(transition_or_state.get("state") or {})
    return dict(transition_or_state or {})


def _require_agent(state: dict[str, Any], key: str) -> dict[str, float]:
    value = state.get(key)
    if not isinstance(value, dict):
        raise KeyError(f"transition state missing {key!r}")
    return {
        "x": float(value["x"]),
        "y": float(value["y"]),
        "theta": float(value.get("theta", 0.0)) % 360.0,
    }


def _normalize_radar(raw: np.ndarray, env: CoppeliaTrackEnv) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    if arr.size != RADAR_DIM:
        raise ValueError(f"radar must have shape ({RADAR_DIM},), got {arr.shape}")
    # CoppeliaTrackEnv._radar returns distances in px; test fakes may already
    # pass normalized values. Treat arrays outside [-1, 1] as distances.
    if arr.size and (float(np.nanmax(arr)) > 1.0 or float(np.nanmin(arr)) < -1.0):
        map_diagonal = float(math.hypot(float(env.width), float(env.height)))
        max_range = float(min(getattr(EnvParameters, "FOV_RANGE", 300), map_diagonal))
        arr = (arr / max(max_range, 1e-6)) * 2.0 - 1.0
    return np.nan_to_num(np.clip(arr, -1.0, 1.0), nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)


def build_critic_observation(actor_obs: np.ndarray, privileged_obs: np.ndarray) -> np.ndarray:
    critic = _build_critic_observation(
        np.asarray(actor_obs, dtype=np.float32),
        np.asarray(privileged_obs, dtype=np.float32),
    )
    critic = np.asarray(critic, dtype=np.float32).reshape(-1)
    if critic.shape != (CRITIC_OBS_DIM,):
        raise ValueError(f"critic observation must have shape ({CRITIC_OBS_DIM},), got {critic.shape}")
    return critic


def _install_numpy_core_pickle_alias() -> None:
    """Allow NumPy-2 pickles to load in NumPy-1 environments."""
    if not hasattr(np, "core"):
        return
    sys.modules.setdefault("numpy._core", np.core)
    try:
        import numpy.core.multiarray as multiarray
        import numpy.core.numeric as numeric
    except Exception:
        return
    sys.modules.setdefault("numpy._core.multiarray", multiarray)
    sys.modules.setdefault("numpy._core.numeric", numeric)


class CoppeliaSkillObservationProvider:
    """Ground-truth Coppelia observation provider for existing bottom skills."""

    def __init__(
        self,
        env: CoppeliaTrackEnv,
        radar_fn: Callable[[dict[str, float]], np.ndarray] | None = None,
        assume_attacker_visible: bool = True,
    ) -> None:
        self.env = env
        self.radar_fn = radar_fn
        self.assume_attacker_visible = bool(assume_attacker_visible)

    def observations(self, transition_or_state: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        actor = self.actor_observation(transition_or_state)
        privileged = self.privileged_observation(transition_or_state)
        critic = build_critic_observation(actor, privileged)
        return actor, privileged, critic

    def actor_observation(self, transition_or_state: dict[str, Any]) -> np.ndarray:
        state = _state_from_transition(transition_or_state)
        defender = _require_agent(state, "defender")
        attacker = _require_agent(state, "attacker")
        target = _require_agent(state, "target")

        obs = np.zeros(ACTOR_OBS_DIM, dtype=np.float32)
        map_diagonal = float(math.hypot(float(self.env.width), float(self.env.height)))
        defender_center = _center(defender, self.env.pixel_size)

        attacker_center = _center(attacker, self.env.pixel_size)
        attacker_rel = attacker_center - defender_center
        attacker_dist = float(np.linalg.norm(attacker_rel))
        attacker_bearing_abs = math.degrees(math.atan2(float(attacker_rel[1]), float(attacker_rel[0])))
        attacker_bearing = normalize_angle_deg(attacker_bearing_abs - float(defender["theta"]))
        obs[0] = float(np.clip((attacker_dist / max(map_diagonal, 1e-6)) * 2.0 - 1.0, -1.0, 1.0))
        obs[1] = float(np.clip(attacker_bearing / 180.0, -1.0, 1.0))
        fov_half = float(getattr(EnvParameters, "FOV_ANGLE", 360.0)) * 0.5
        fov_edge = min(abs(attacker_bearing + fov_half), abs(attacker_bearing - fov_half))
        obs[2] = float(np.clip((fov_edge / max(fov_half, 1e-6)) * 2.0 - 1.0, -1.0, 1.0))
        obs[3] = 1.0 if self.assume_attacker_visible else 0.0
        obs[4] = -1.0

        target_center = _center(target, self.env.pixel_size)
        target_rel = target_center - defender_center
        target_dist = float(np.linalg.norm(target_rel))
        target_bearing_abs = math.degrees(math.atan2(float(target_rel[1]), float(target_rel[0])))
        target_bearing = normalize_angle_deg(target_bearing_abs - float(defender["theta"]))
        obs[5] = float(np.clip((target_dist / max(map_diagonal, 1e-6)) * 2.0 - 1.0, -1.0, 1.0))
        obs[6] = float(np.clip(target_bearing / 180.0, -1.0, 1.0))
        obs[7:] = self._radar(defender)
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    def privileged_observation(self, transition_or_state: dict[str, Any]) -> np.ndarray:
        state = _state_from_transition(transition_or_state)
        defender = _require_agent(state, "defender")
        attacker = _require_agent(state, "attacker")
        target = _require_agent(state, "target")

        obs = np.zeros(PRIVILEGED_OBS_DIM, dtype=np.float32)
        obs[0] = float(np.clip((attacker["x"] / max(float(self.env.width), 1.0)) * 2.0 - 1.0, -1.0, 1.0))
        obs[1] = float(np.clip((attacker["y"] / max(float(self.env.height), 1.0)) * 2.0 - 1.0, -1.0, 1.0))
        obs[2] = float(np.clip((attacker["theta"] / 180.0) - 1.0, -1.0, 1.0))
        obs[3] = float(np.clip((defender["x"] / max(float(self.env.width), 1.0)) * 2.0 - 1.0, -1.0, 1.0))
        obs[4] = float(np.clip((defender["y"] / max(float(self.env.height), 1.0)) * 2.0 - 1.0, -1.0, 1.0))
        obs[5] = float(np.clip((defender["theta"] / 180.0) - 1.0, -1.0, 1.0))
        obs[6] = float(np.clip((target["x"] / max(float(self.env.width), 1.0)) * 2.0 - 1.0, -1.0, 1.0))
        obs[7] = float(np.clip((target["y"] / max(float(self.env.height), 1.0)) * 2.0 - 1.0, -1.0, 1.0))
        obs[8:] = self._radar(attacker)
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    def _radar(self, agent: dict[str, float]) -> np.ndarray:
        if self.radar_fn is not None:
            return _normalize_radar(self.radar_fn(agent), self.env)
        if hasattr(self.env, "_radar"):
            return _normalize_radar(self.env._radar(agent), self.env)
        return np.zeros(RADAR_DIM, dtype=np.float32)


class NormalizedCmdVelAdapter:
    """Convert existing normalized skill actions to velocity commands."""

    def __init__(self, env: CoppeliaTrackEnv, allow_reverse: bool = False) -> None:
        self.env = env
        self.allow_reverse = bool(allow_reverse)

    def defender_command(self, action: np.ndarray | list[float] | tuple[float, float]) -> VelocityCommand:
        return self.command(
            action,
            max_v=float(self.env.defender_max_v),
            max_w=float(self.env.defender_max_w),
        )

    def attacker_command(self, action: np.ndarray | list[float] | tuple[float, float]) -> VelocityCommand:
        return self.command(
            action,
            max_v=float(self.env.attacker_max_v),
            max_w=float(self.env.attacker_max_w),
        )

    def command(
        self,
        action: np.ndarray | list[float] | tuple[float, float],
        max_v: float,
        max_w: float,
    ) -> VelocityCommand:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        turn_norm = float(arr[0]) if arr.size >= 1 else 0.0
        speed_norm = float(arr[1]) if arr.size >= 2 else 0.0
        turn_norm = float(np.clip(turn_norm, -1.0, 1.0))
        speed_norm = float(np.clip(speed_norm, -1.0, 1.0))
        if self.allow_reverse:
            linear = speed_norm * float(max_v)
        else:
            linear = (speed_norm + 1.0) * 0.5 * float(max_v)
        angular = turn_norm * float(max_w)
        return VelocityCommand(linear_mps=float(linear), angular_radps=float(angular))


def serialize_recurrent_hidden(model: Any, role: str) -> np.ndarray | None:
    hidden = getattr(model, f"_{role}_hidden", None)
    network = getattr(model, "network", None)
    if network is None:
        return None
    if hasattr(network, "recurrent_hidden_spec"):
        num_layers, hidden_size = network.recurrent_hidden_spec(role)
        if hidden is None:
            return np.zeros((int(num_layers), int(hidden_size)), dtype=np.float32)
        hidden_cpu = hidden.detach().to(device="cpu", dtype=getattr(hidden, "dtype", None)).float()
        if hidden_cpu.dim() == 3:
            hidden_cpu = hidden_cpu[:, 0, :]
        return hidden_cpu.numpy().copy()
    gru = getattr(network, f"{role}_gru", None)
    if gru is None:
        return None
    if hidden is None:
        return np.zeros((gru.num_layers, gru.hidden_size), dtype=np.float32)
    hidden_cpu = hidden.detach().to(device="cpu", dtype=getattr(hidden, "dtype", None)).float()
    if hidden_cpu.dim() == 3:
        hidden_cpu = hidden_cpu[:, 0, :]
    return hidden_cpu.numpy().copy()


def compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    dones = np.asarray(dones, dtype=np.float32).reshape(-1)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if not (rewards.shape == dones.shape == values.shape):
        raise ValueError(
            f"rewards, dones, and values must have same shape, got {rewards.shape}, {dones.shape}, {values.shape}"
        )
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(rewards.shape[0])):
        next_value = float(last_value) if t == rewards.shape[0] - 1 else float(values[t + 1])
        next_non_terminal = 1.0 - float(dones[t])
        delta = float(rewards[t]) + float(gamma) * next_value * next_non_terminal - float(values[t])
        last_gae = delta + float(gamma) * float(gae_lambda) * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)


def load_skill_model(
    checkpoint: str | Path,
    network_type: str,
    device: str | Any,
    global_model: bool = False,
):
    from skill.model import Model

    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"skill checkpoint not found: {path}")
    _install_numpy_core_pickle_alias()
    model = Model(device=device, global_model=global_model, network_type=str(network_type))
    model.load(str(path))
    return model
