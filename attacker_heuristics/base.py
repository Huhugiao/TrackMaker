"""Shared state representation, navigation, and diagnostics for heuristics."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, Optional, Tuple

import numpy as np

from configs import map_config
from policies.attacker_global import AttackerGlobalPolicy


DIAGNOSTIC_KEYS = (
    "mode",
    "state",
    "subgoal",
    "risk",
    "switch_reason",
    "replan_count",
    "mechanism",
    "details",
)


def normalize_angle_deg(value: float) -> float:
    """Normalize an angle to ``[-180, 180]``."""
    value = float(value) % 360.0
    if value > 180.0:
        value -= 360.0
    return value


def unit(vector: np.ndarray, fallback: Tuple[float, float] = (1.0, 0.0)) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-9:
        return np.asarray(fallback, dtype=np.float64)
    return arr / norm


def clip_point(point: np.ndarray, margin: float = 24.0) -> np.ndarray:
    arr = np.asarray(point, dtype=np.float64).copy()
    arr[0] = np.clip(arr[0], margin, float(map_config.width) - margin)
    arr[1] = np.clip(arr[1], margin, float(map_config.height) - margin)
    return arr


def cross2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def line_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    denom = float(np.dot(segment, segment))
    if denom <= 1e-9:
        return float(np.linalg.norm(np.asarray(point) - np.asarray(start)))
    u = float(np.clip(np.dot(np.asarray(point) - np.asarray(start), segment) / denom, 0.0, 1.0))
    projection = np.asarray(start) + u * segment
    return float(np.linalg.norm(np.asarray(point) - projection))


@dataclass(frozen=True)
class WorldState:
    attacker: np.ndarray
    defender: np.ndarray
    target: np.ndarray
    attacker_heading: float
    defender_heading: float
    attacker_velocity: np.ndarray
    defender_velocity: np.ndarray
    step: int

    @property
    def attacker_target_distance(self) -> float:
        return float(np.linalg.norm(self.target - self.attacker))

    @property
    def attacker_defender_distance(self) -> float:
        return float(np.linalg.norm(self.defender - self.attacker))

    @property
    def defender_target_distance(self) -> float:
        return float(np.linalg.norm(self.target - self.defender))


@dataclass
class Decision:
    subgoal: np.ndarray
    state: str
    risk: float
    switch_reason: str
    speed_fraction: float = 1.0
    avoid_radius: float = 0.0
    details: Dict[str, object] = field(default_factory=dict)
    direct_action: Optional[np.ndarray] = None


def decode_world(obs: np.ndarray, previous: Optional[WorldState], step: int) -> WorldState:
    """Decode the repository's 72-D privileged Attacker observation."""
    arr = np.asarray(obs, dtype=np.float64).reshape(-1)
    if arr.size < 8:
        raise ValueError(f"Attacker observation requires at least 8 values, got {arr.size}")

    width = float(map_config.width)
    height = float(map_config.height)
    attacker = np.array([0.5 * (arr[0] + 1.0) * width, 0.5 * (arr[1] + 1.0) * height])
    defender = np.array([0.5 * (arr[3] + 1.0) * width, 0.5 * (arr[4] + 1.0) * height])
    target = np.array([0.5 * (arr[6] + 1.0) * width, 0.5 * (arr[7] + 1.0) * height])
    if previous is None:
        attacker_velocity = np.zeros(2, dtype=np.float64)
        defender_velocity = np.zeros(2, dtype=np.float64)
    else:
        attacker_velocity = attacker - previous.attacker
        defender_velocity = defender - previous.defender
    return WorldState(
        attacker=attacker,
        defender=defender,
        target=target,
        attacker_heading=float((arr[2] + 1.0) * 180.0),
        defender_heading=float((arr[5] + 1.0) * 180.0),
        attacker_velocity=attacker_velocity,
        defender_velocity=defender_velocity,
        step=int(step),
    )


def interception_risk(state: WorldState, point: Optional[np.ndarray] = None) -> float:
    """Bounded arrival-time risk for an intended point.

    Risk is high when the faster Defender can reach the point before the
    Attacker and when it already lies near the A--point segment.
    """
    point = state.target if point is None else np.asarray(point, dtype=np.float64)
    a_eta = float(np.linalg.norm(point - state.attacker)) / max(float(map_config.attacker_speed), 1e-6)
    d_eta = float(np.linalg.norm(point - state.defender)) / max(float(map_config.defender_speed), 1e-6)
    margin = a_eta - d_eta
    geometry = math.exp(-line_distance(state.defender, state.attacker, point) / 75.0)
    timing = 1.0 / (1.0 + math.exp(-float(np.clip(margin / 12.0, -30.0, 30.0))))
    proximity = math.exp(-state.attacker_defender_distance / 80.0)
    return float(np.clip(0.55 * timing + 0.30 * geometry + 0.15 * proximity, 0.0, 1.0))


class GoalNavigator:
    """A* waypoint follower shared by high-level mechanisms."""

    def __init__(self, replan_interval: int = 14):
        self.planner = AttackerGlobalPolicy(strategy="default")
        self.replan_interval = max(1, int(replan_interval))
        self.path = []
        self.path_index = 0
        self.last_goal: Optional[np.ndarray] = None
        self.last_defender: Optional[np.ndarray] = None
        self.last_plan_step = -10**9
        self.replan_count = 0

    def reset(self) -> None:
        self.planner.reset()
        self.path = []
        self.path_index = 0
        self.last_goal = None
        self.last_defender = None
        self.last_plan_step = -10**9
        self.replan_count = 0

    def _needs_replan(self, state: WorldState, goal: np.ndarray, avoid_radius: float) -> bool:
        if not self.path or self.last_goal is None:
            return True
        if float(np.linalg.norm(goal - self.last_goal)) > 18.0:
            return True
        if state.step - self.last_plan_step >= self.replan_interval:
            if avoid_radius <= 0.0 or self.last_defender is None:
                return True
            return bool(np.linalg.norm(state.defender - self.last_defender) > 8.0)
        return False

    def action(
        self,
        state: WorldState,
        goal: np.ndarray,
        *,
        speed_fraction: float = 1.0,
        avoid_radius: float = 0.0,
    ) -> np.ndarray:
        goal = clip_point(goal)
        if self._needs_replan(state, goal, avoid_radius):
            old_radius = float(self.planner.defender_avoid_radius)
            self.planner.defender_avoid_radius = max(0.0, float(avoid_radius))
            try:
                self.path = self.planner.plan_path(
                    state.attacker,
                    goal,
                    state.defender if avoid_radius > 0.0 else None,
                )
            finally:
                self.planner.defender_avoid_radius = old_radius
            self.path_index = 0
            self.last_goal = goal.copy()
            self.last_defender = state.defender.copy()
            self.last_plan_step = state.step
            self.replan_count += 1

        while self.path_index < len(self.path):
            if np.linalg.norm(state.attacker - self.path[self.path_index]) >= 9.0:
                break
            self.path_index += 1
        waypoint = goal if self.path_index >= len(self.path) else np.asarray(self.path[self.path_index])
        desired_heading = math.degrees(math.atan2(
            waypoint[1] - state.attacker[1], waypoint[0] - state.attacker[0]
        ))
        error = normalize_angle_deg(desired_heading - state.attacker_heading)
        turn = float(np.clip(0.8 * error / float(map_config.attacker_max_angular_speed), -1.0, 1.0))

        fraction = float(np.clip(speed_fraction, 0.0, 1.0))
        # Turning at full speed causes wide arcs and repeated obstacle contacts.
        if abs(error) > 65.0:
            fraction = min(fraction, 0.35)
        elif abs(error) > 35.0:
            fraction = min(fraction, 0.65)
        speed = 2.0 * fraction - 1.0
        return np.array([turn, speed], dtype=np.float32)


class HeuristicPolicy:
    """Stable interface for stateful programmatic Attacker policies."""

    mechanism = "unspecified"

    def __init__(self, *, seed: int = 0):
        self.base_seed = int(seed)
        self.rng = np.random.default_rng(self.base_seed)
        self.navigator = GoalNavigator()
        self._previous_world: Optional[WorldState] = None
        self._step = 0
        self._last_diag = self._empty_diagnostics()
        self.reset(seed=seed)

    def _empty_diagnostics(self) -> Dict[str, object]:
        return {
            "mode": self.__class__.__name__,
            "state": "reset",
            "subgoal": [0.0, 0.0],
            "risk": 0.0,
            "switch_reason": "reset",
            "replan_count": 0,
            "mechanism": str(self.mechanism),
            "details": {},
        }

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        self.navigator.reset()
        self._previous_world = None
        self._step = 0
        self._last_diag = self._empty_diagnostics()
        self._reset_internal()

    def _reset_internal(self) -> None:
        pass

    def _decide(self, state: WorldState) -> Decision:
        raise NotImplementedError

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        state = decode_world(obs, self._previous_world, self._step)
        decision = self._decide(state)
        if decision.direct_action is None:
            action = self.navigator.action(
                state,
                decision.subgoal,
                speed_fraction=decision.speed_fraction,
                avoid_radius=decision.avoid_radius,
            )
        else:
            action = np.asarray(decision.direct_action, dtype=np.float32).reshape(2)
        action = np.clip(action, -1.0, 1.0).astype(np.float32, copy=False)
        details = dict(decision.details)
        self._last_diag = {
            "mode": self.__class__.__name__,
            "state": str(decision.state),
            "subgoal": [float(decision.subgoal[0]), float(decision.subgoal[1])],
            "risk": float(np.clip(decision.risk, 0.0, 1.0)),
            "switch_reason": str(decision.switch_reason),
            "replan_count": int(self.navigator.replan_count),
            "mechanism": str(self.mechanism),
            "details": details,
        }
        self._previous_world = state
        self._step += 1
        return action

    def get_action_with_info(self, obs: np.ndarray):
        action = self.get_action(obs)
        return action, dict(self._last_diag)

    @property
    def diagnostics(self) -> Dict[str, object]:
        return dict(self._last_diag)

