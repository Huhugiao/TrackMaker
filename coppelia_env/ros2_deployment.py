"""ROS 2 deployment math shared by CoppeliaSim and future TurtleBot4 runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np

from coppelia_env.skill_adapter import (
    build_actor_observation,
    build_critic_observation,
    build_privileged_observation,
)


@dataclass(frozen=True)
class DeploymentConfig:
    width_px: int = 640
    height_px: int = 640
    pixel_size_px: float = 4.0
    scale_m_per_px: float = 0.01
    radar_rays: int = 64
    radar_range_m: float = 3.0
    decision_hz: float = 9.0
    command_hz: float = 20.0
    input_timeout_s: float = 0.5
    startup_timeout_s: float = 30.0
    max_steps: int = 449
    defender_max_linear_mps: float = 0.234
    attacker_max_linear_mps: float = 0.180
    defender_max_angular_radps: float = math.radians(54.0)
    attacker_max_angular_radps: float = math.radians(108.0)
    attacker_max_accel_mps2: float = 0.486
    capture_distance_m: float = 0.45
    capture_sector_deg: float = 30.0
    target_distance_m: float = 0.35


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float = 0.0


@dataclass(frozen=True)
class VelocityCommand:
    linear_mps: float = 0.0
    angular_radps: float = 0.0


@dataclass(frozen=True)
class TerminalResult:
    reason: str
    outcome: str
    step: int


@dataclass(frozen=True)
class CheckpointSpec:
    path: Path
    network_type: str
    sha256: str


class DefenderObservationMemory:
    """Mirror TADEnv's last-visible attacker state across policy steps."""

    def __init__(self) -> None:
        self.last_observed_attacker: dict[str, float] | None = None
        self.steps_since_observed = 0

    def reset(self) -> None:
        self.last_observed_attacker = None
        self.steps_since_observed = 0

    def update(
        self,
        state: dict[str, dict[str, float]],
        obstacles: Iterable[dict[str, Any]],
    ) -> bool:
        visible = not line_of_sight_blocked(state["defender"], state["attacker"], obstacles)
        if visible:
            self.last_observed_attacker = dict(state["attacker"])
            self.steps_since_observed = 0
        else:
            self.steps_since_observed += 1
        return visible


def _distance_point_to_segment(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    dx, dy = x2 - x1, y2 - y1
    denominator = dx * dx + dy * dy
    if denominator <= 1e-12:
        return math.hypot(px - x1, py - y1)
    fraction = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / denominator))
    return math.hypot(px - (x1 + fraction * dx), py - (y1 + fraction * dy))


def _point_hits_obstacle(px: float, py: float, radius: float, obstacle: dict[str, Any]) -> bool:
    kind = obstacle.get("type")
    if kind == "rect":
        x1, y1 = float(obstacle["x"]), float(obstacle["y"])
        x2, y2 = x1 + float(obstacle["w"]), y1 + float(obstacle["h"])
        return math.hypot(px - max(x1, min(px, x2)), py - max(y1, min(py, y2))) < radius
    if kind == "circle":
        return math.hypot(px - float(obstacle["cx"]), py - float(obstacle["cy"])) < (
            radius + float(obstacle["r"])
        )
    if kind == "segment":
        distance = _distance_point_to_segment(
            px,
            py,
            float(obstacle["x1"]),
            float(obstacle["y1"]),
            float(obstacle["x2"]),
            float(obstacle["y2"]),
        )
        return distance < radius + 0.5 * float(obstacle.get("thick", 8.0))
    return False


def _ray_circle_distance(ox: float, oy: float, dx: float, dy: float, cx: float, cy: float, radius: float) -> float:
    fx, fy = ox - cx, oy - cy
    projection = dx * fx + dy * fy
    discriminant = projection * projection - (fx * fx + fy * fy - radius * radius)
    if discriminant < 0.0:
        return math.inf
    root = math.sqrt(discriminant)
    near, far = -projection - root, -projection + root
    return near if near > 0.0 else (far if far >= 0.0 else math.inf)


def _ray_segment_distance(
    ox: float, oy: float, dx: float, dy: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    sx, sy = x2 - x1, y2 - y1
    cross = dx * sy - dy * sx
    if abs(cross) < 1e-9:
        return math.inf
    qx, qy = x1 - ox, y1 - oy
    distance = (qx * sy - qy * sx) / cross
    fraction = (qx * dy - qy * dx) / cross
    return distance if distance >= 0.0 and 0.0 <= fraction <= 1.0 else math.inf


def _ray_obstacle_distance(ox: float, oy: float, dx: float, dy: float, obstacle: dict[str, Any]) -> float:
    kind = obstacle.get("type")
    if kind == "rect":
        rx, ry = float(obstacle["x"]), float(obstacle["y"])
        rw, rh = float(obstacle["w"]), float(obstacle["h"])
        distances = []
        if abs(dx) > 1e-9:
            for x in (rx, rx + rw):
                distance = (x - ox) / dx
                y = oy + distance * dy
                if distance >= 0.0 and ry <= y <= ry + rh:
                    distances.append(distance)
        if abs(dy) > 1e-9:
            for y in (ry, ry + rh):
                distance = (y - oy) / dy
                x = ox + distance * dx
                if distance >= 0.0 and rx <= x <= rx + rw:
                    distances.append(distance)
        return min(distances, default=math.inf)
    if kind == "circle":
        return _ray_circle_distance(
            ox, oy, dx, dy, float(obstacle["cx"]), float(obstacle["cy"]), float(obstacle["r"])
        )
    if kind == "segment":
        x1, y1 = float(obstacle["x1"]), float(obstacle["y1"])
        x2, y2 = float(obstacle["x2"]), float(obstacle["y2"])
        radius = 0.5 * float(obstacle.get("thick", 8.0))
        length = math.hypot(x2 - x1, y2 - y1)
        distances = [
            _ray_circle_distance(ox, oy, dx, dy, x1, y1, radius),
            _ray_circle_distance(ox, oy, dx, dy, x2, y2, radius),
        ]
        if length > 1e-9:
            nx, ny = -(y2 - y1) / length * radius, (x2 - x1) / length * radius
            distances.extend(
                (
                    _ray_segment_distance(ox, oy, dx, dy, x1 + nx, y1 + ny, x2 + nx, y2 + ny),
                    _ray_segment_distance(ox, oy, dx, dy, x1 - nx, y1 - ny, x2 - nx, y2 - ny),
                )
            )
        return min(distances)
    return math.inf


def line_of_sight_blocked(
    first: dict[str, float],
    second: dict[str, float],
    obstacles: Iterable[dict[str, Any]],
    pixel_size: float = 4.0,
) -> bool:
    """Return the Gym-equivalent 2-D map occlusion between agent centers."""

    half_size = 0.5 * float(pixel_size)
    x1, y1 = float(first["x"]) + half_size, float(first["y"]) + half_size
    x2, y2 = float(second["x"]) + half_size, float(second["y"]) + half_size
    angle = math.atan2(y2 - y1, x2 - x1)
    distance = math.hypot(x2 - x1, y2 - y1)
    dx, dy = math.cos(angle), math.sin(angle)
    nearest = min((_ray_obstacle_distance(x1, y1, dx, dy, obstacle) for obstacle in obstacles), default=math.inf)
    return nearest < distance - 1e-3


def sample_fixed_spawn(
    manifest_path: Path | str,
    seed: int,
    *,
    robot_clearance_m: float = 0.24,
    target_clearance_m: float = 0.20,
) -> dict[str, Pose2D]:
    """Sample a deterministic, physically clear paired-seed start state."""

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    map_data = manifest["map"]
    width, height = float(map_data["width_px"]), float(map_data["height_px"])
    scale = float(map_data["scale_m_per_px"])
    obstacles = list(manifest["obstacles"])
    rng = random.Random(int(seed))
    robot_radius = max(0.0, float(robot_clearance_m)) / scale
    target_radius = max(0.0, float(target_clearance_m)) / scale

    def clear(point: tuple[float, float], radius: float) -> bool:
        x, y = point
        return (
            radius <= x <= width - radius
            and radius <= y <= height - radius
            and not any(_point_hits_obstacle(x, y, radius, obstacle) for obstacle in obstacles)
        )

    def point(radius: float, margin: float) -> tuple[float, float]:
        padding = max(radius, margin)
        for _ in range(10000):
            candidate = (rng.uniform(padding, width - padding), rng.uniform(padding, height - padding))
            if clear(candidate, radius):
                return candidate
        raise RuntimeError(f"unable to sample a clear spawn for seed {seed}")

    for _ in range(500):
        defender = point(robot_radius, 40.0)
        attacker = point(robot_radius, 40.0)
        if math.dist(defender, attacker) < 150.0:
            continue
        target = point(target_radius, 55.0)
        d_target = math.dist(defender, target)
        a_target = math.dist(attacker, target)
        if min(d_target, a_target) < 80.0:
            continue
        # Preserve the original feasible-game spawn constraint for A=2.0, D=2.6.
        if a_target * 2.6 <= d_target * 2.0:
            continue

        def world(position: tuple[float, float], yaw: float) -> Pose2D:
            return Pose2D(
                x=(position[0] - width * 0.5) * scale,
                y=(height * 0.5 - position[1]) * scale,
                yaw=yaw,
            )

        return {
            "defender": world(defender, rng.uniform(-math.pi, math.pi)),
            "attacker": world(attacker, rng.uniform(-math.pi, math.pi)),
            "target": world(target, 0.0),
        }
    raise RuntimeError(f"unable to sample a valid paired start state for seed {seed}")


def _finite(value: float, name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Return planar yaw from a ROS quaternion."""

    values = [_finite(v, "quaternion") for v in (x, y, z, w)]
    norm = math.sqrt(sum(v * v for v in values))
    if norm <= 1e-9:
        raise ValueError("quaternion norm must be positive")
    x, y, z, w = (v / norm for v in values)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * _finite(yaw, "yaw")
    return 0.0, 0.0, math.sin(half), math.cos(half)


def world_pose_to_training(pose: Pose2D, config: DeploymentConfig) -> dict[str, float]:
    """Convert ROS map coordinates into the original 640x640 image convention."""

    scale = max(float(config.scale_m_per_px), 1e-9)
    cx = _finite(pose.x, "pose.x") / scale + float(config.width_px) * 0.5
    cy = float(config.height_px) * 0.5 - _finite(pose.y, "pose.y") / scale
    return {
        "x": cx - float(config.pixel_size_px) * 0.5,
        "y": cy - float(config.pixel_size_px) * 0.5,
        "theta": (-math.degrees(_finite(pose.yaw, "pose.yaw"))) % 360.0,
    }


def build_training_state(poses: dict[str, Pose2D], config: DeploymentConfig) -> dict[str, dict[str, float]]:
    missing = [role for role in ("defender", "attacker", "target") if role not in poses]
    if missing:
        raise KeyError(f"missing map poses: {missing}")
    return {role: world_pose_to_training(poses[role], config) for role in ("defender", "attacker", "target")}


def normalize_laser_scan(
    ranges: Iterable[float],
    *,
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    config: DeploymentConfig,
) -> np.ndarray:
    """Resample a ROS LaserScan into the training radar's clockwise 64 rays."""

    raw = np.asarray(list(ranges), dtype=np.float64).reshape(-1)
    if raw.size < 2:
        raise ValueError("LaserScan must contain at least two ranges")
    increment = _finite(angle_increment, "angle_increment")
    if abs(increment) <= 1e-12:
        raise ValueError("angle_increment must be non-zero")

    lo = max(0.0, _finite(range_min, "range_min"))
    configured_max = max(lo + 1e-6, float(config.radar_range_m))
    message_max = _finite(range_max, "range_max")
    usable_max = min(configured_max, message_max) if message_max > lo else configured_max
    raw = np.where(np.isfinite(raw), raw, usable_max)
    raw = np.clip(raw, lo, usable_max)

    angles = _finite(angle_min, "angle_min") + np.arange(raw.size, dtype=np.float64) * increment
    angles = np.arctan2(np.sin(angles), np.cos(angles))
    order = np.argsort(angles)
    angles = angles[order]
    raw = raw[order]
    extended_angles = np.concatenate((angles - 2.0 * math.pi, angles, angles + 2.0 * math.pi))
    extended_ranges = np.concatenate((raw, raw, raw))

    # Gym used image coordinates (y down), so ray index increases clockwise.
    targets = -2.0 * math.pi * np.arange(int(config.radar_rays), dtype=np.float64) / float(config.radar_rays)
    targets = np.arctan2(np.sin(targets), np.cos(targets))
    sampled = np.interp(targets, extended_angles, extended_ranges)
    normalized = sampled / max(configured_max, 1e-9) * 2.0 - 1.0
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def build_policy_observations(
    poses: dict[str, Pose2D],
    defender_radar: np.ndarray,
    attacker_radar: np.ndarray,
    config: DeploymentConfig,
    *,
    defender_memory: DefenderObservationMemory | None = None,
    obstacles: Iterable[dict[str, Any]] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = build_training_state(poses, config)
    visible = True if defender_memory is None else defender_memory.update(state, obstacles)
    actor = build_actor_observation(
        state,
        width=config.width_px,
        height=config.height_px,
        pixel_size=config.pixel_size_px,
        defender_radar=defender_radar,
        assume_attacker_visible=visible,
        last_observed_attacker=None if defender_memory is None else defender_memory.last_observed_attacker,
        steps_since_observed=0 if defender_memory is None else defender_memory.steps_since_observed,
    )
    privileged = build_privileged_observation(
        state,
        width=config.width_px,
        height=config.height_px,
        attacker_radar=attacker_radar,
    )
    return actor, privileged, build_critic_observation(actor, privileged)


def normalized_action_to_command(
    action: np.ndarray | list[float] | tuple[float, float],
    *,
    max_linear_mps: float,
    max_angular_radps: float,
) -> VelocityCommand:
    arr = np.asarray(action, dtype=np.float64).reshape(-1)
    if arr.shape != (2,) or not np.all(np.isfinite(arr)):
        raise ValueError("policy action must contain exactly two finite values")
    turn, speed = np.clip(arr, -1.0, 1.0)
    return VelocityCommand(
        linear_mps=float((speed + 1.0) * 0.5 * max(0.0, float(max_linear_mps))),
        # Gym theta grows clockwise in image coordinates; ROS yaw grows CCW.
        angular_radps=float(-turn * abs(float(max_angular_radps))),
    )


def clamp_command(command: VelocityCommand, max_linear_mps: float, max_angular_radps: float) -> VelocityCommand:
    linear = _finite(command.linear_mps, "linear_mps")
    angular = _finite(command.angular_radps, "angular_radps")
    return VelocityCommand(
        linear_mps=float(np.clip(linear, 0.0, max(0.0, float(max_linear_mps)))),
        angular_radps=float(np.clip(angular, -abs(float(max_angular_radps)), abs(float(max_angular_radps)))),
    )


class LinearSlewLimiter:
    def __init__(self, max_accel_mps2: float) -> None:
        self.max_accel_mps2 = max(0.0, float(max_accel_mps2))
        self.value = 0.0

    def reset(self) -> None:
        self.value = 0.0

    def apply(self, requested_mps: float, dt_s: float) -> float:
        requested = _finite(requested_mps, "requested_mps")
        delta = self.max_accel_mps2 * max(0.0, _finite(dt_s, "dt_s"))
        self.value += float(np.clip(requested - self.value, -delta, delta))
        return float(self.value)


class EpisodeMonitor:
    def __init__(self, config: DeploymentConfig) -> None:
        self.config = config
        self.step = 0
        self.done = False

    def reset(self) -> None:
        self.step = 0
        self.done = False

    def update(
        self,
        poses: dict[str, Pose2D],
        *,
        defender_collision: bool = False,
        attacker_collision: bool = False,
    ) -> TerminalResult | None:
        if self.done:
            return None
        self.step += 1
        result = self._terminal(poses, bool(defender_collision), bool(attacker_collision))
        if result is not None:
            self.done = True
        return result

    def _terminal(
        self,
        poses: dict[str, Pose2D],
        defender_collision: bool,
        attacker_collision: bool,
    ) -> TerminalResult | None:
        if defender_collision:
            return TerminalResult("defender_collision", "draw", self.step)
        # Match the Gym/eval contract: attacker contacts are recorded as
        # collision events, while only defender_collision is a terminal draw.

        defender, attacker, target = (poses[name] for name in ("defender", "attacker", "target"))
        dx = attacker.x - defender.x
        dy = attacker.y - defender.y
        capture_distance = math.hypot(dx, dy)
        capture_bearing = math.degrees(math.atan2(dy, dx) - defender.yaw)
        capture_bearing = (capture_bearing + 180.0) % 360.0 - 180.0
        if (
            capture_distance <= float(self.config.capture_distance_m)
            and abs(capture_bearing) <= float(self.config.capture_sector_deg) * 0.5
        ):
            return TerminalResult("defender_caught_attacker", "defender_success", self.step)
        if math.hypot(attacker.x - target.x, attacker.y - target.y) <= float(self.config.target_distance_m):
            return TerminalResult("attacker_caught_target", "attacker_success", self.step)
        if self.step >= int(self.config.max_steps):
            return TerminalResult("timeout_defender_wins", "defender_success", self.step)
        return None


def inputs_are_fresh(stamps: Iterable[float], now: float, timeout_s: float) -> bool:
    values = [float(v) for v in stamps]
    if not values or not math.isfinite(float(now)) or not all(math.isfinite(v) for v in values):
        return False
    timeout = max(0.0, float(timeout_s))
    return all(0.0 <= float(now) - value <= timeout for value in values)


def validate_checkpoint(spec: CheckpointSpec) -> dict[str, Any]:
    path = Path(spec.path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != str(spec.sha256):
        raise ValueError(f"checkpoint checksum mismatch for {path}: {digest}")
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    network_type = str(checkpoint.get("network_type", "")) if isinstance(checkpoint, dict) else ""
    if network_type != str(spec.network_type):
        raise ValueError(f"checkpoint network_type mismatch for {path}: {network_type!r}")
    return {"path": str(path), "network_type": network_type, "sha256": digest}


def config_dict(config: DeploymentConfig) -> dict[str, Any]:
    return asdict(config)
