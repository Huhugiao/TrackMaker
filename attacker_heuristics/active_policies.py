"""Programmatic Attacker policies retained in the frozen pool."""

from __future__ import annotations

from typing import Optional

import numpy as np

from configs import map_config
from envs import env_lib
from attacker_heuristics.base import (
    Decision,
    HeuristicPolicy,
    WorldState,
    clip_point,
    interception_risk,
    line_distance,
    unit,
)


def _perpendicular(direction: np.ndarray) -> np.ndarray:
    direction = unit(direction)
    return np.array([-direction[1], direction[0]], dtype=np.float64)


def _flank_waypoint(
    state: WorldState,
    side_sign: float,
    forward: float,
    lateral: float,
) -> np.ndarray:
    to_target = state.target - state.attacker
    return clip_point(
        state.attacker
        + forward * to_target
        + float(side_sign) * lateral * _perpendicular(to_target)
    )


def _segment_projection(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    denominator = max(float(np.dot(segment, segment)), 1e-9)
    return float(np.dot(point - start, segment) / denominator)


class GeometryGatedFeintV3Policy(HeuristicPolicy):
    mechanism = "episode-level corridor geometry gate activates one feint; otherwise immediate direct attack"

    def _reset_internal(self) -> None:
        self.phase = "feint"
        self.fake_side: Optional[float] = None
        self.initial_defender: Optional[np.ndarray] = None
        self.feint_goal: Optional[np.ndarray] = None
        self.commit_goal: Optional[np.ndarray] = None
        self.switches = 0
        self.feint_enabled: Optional[bool] = None
        self.initial_corridor_distance = 0.0
        self.initial_projection = 0.0

    def _decide(self, state: WorldState) -> Decision:
        if self.feint_enabled is None:
            self.initial_corridor_distance = line_distance(
                state.defender, state.attacker, state.target
            )
            self.initial_projection = _segment_projection(
                state.defender, state.attacker, state.target
            )
            self.feint_enabled = bool(
                0.05 < self.initial_projection < 1.15
                and self.initial_corridor_distance < 105.0
                and state.defender_target_distance < 1.25 * state.attacker_target_distance
            )
        if not self.feint_enabled:
            return Decision(
                state.target,
                "geometry_direct",
                interception_risk(state),
                "initial_corridor_not_blocked",
                speed_fraction=1.0,
                avoid_radius=6.0,
                details={
                    "feint_enabled": False,
                    "initial_corridor_distance": float(self.initial_corridor_distance),
                    "initial_projection": float(self.initial_projection),
                },
            )

        risk = interception_risk(state)
        if self.fake_side is None:
            direction = state.target - state.attacker
            side = _perpendicular(direction)
            self.fake_side = 1.0 if np.dot(state.defender - state.attacker, side) > 0.0 else -1.0
            self.initial_defender = state.defender.copy()
            self.feint_goal = _flank_waypoint(state, self.fake_side, 0.34, 105.0)
            self.commit_goal = _flank_waypoint(state, -self.fake_side, 0.68, 115.0)

        defender_shift = float(np.linalg.norm(state.defender - self.initial_defender))
        if self.phase == "feint" and (
            defender_shift > 48.0
            or state.step >= 44
            or state.attacker_defender_distance < 72.0
        ):
            self.phase = "reverse_commit"
            self.switches += 1
            reason = "defender_displaced" if defender_shift > 48.0 else "feint_budget_exhausted"
        else:
            reason = "maintain_false_intent"

        if self.phase == "reverse_commit" and (
            np.linalg.norm(state.attacker - self.commit_goal) < 42.0
            or state.attacker_target_distance < 88.0
            or state.step > 145
        ):
            self.phase = "target_commit"
            self.switches += 1
            reason = "opposite_channel_established"

        goal = self.feint_goal if self.phase == "feint" else self.commit_goal
        if self.phase == "target_commit":
            goal = state.target
        return Decision(
            goal,
            self.phase,
            risk,
            reason,
            speed_fraction=0.88 if self.phase == "feint" else 1.0,
            avoid_radius=14.0,
            details={
                "fake_side": float(self.fake_side),
                "defender_displacement": defender_shift,
                "phase_switches": int(self.switches),
                "feint_enabled": True,
                "initial_corridor_distance": float(self.initial_corridor_distance),
                "initial_projection": float(self.initial_projection),
            },
        )


def _obstacle_center(obstacle) -> Optional[np.ndarray]:
    kind = obstacle.get("type")
    if kind == "rect":
        return np.array([
            obstacle["x"] + 0.5 * obstacle["w"],
            obstacle["y"] + 0.5 * obstacle["h"],
        ])
    if kind == "circle":
        return np.array([obstacle["cx"], obstacle["cy"]])
    if kind == "segment":
        return np.array([
            0.5 * (obstacle["x1"] + obstacle["x2"]),
            0.5 * (obstacle["y1"] + obstacle["y2"]),
        ])
    return None


def _sample_line_blocked(start: np.ndarray, end: np.ndarray) -> bool:
    for alpha in np.linspace(0.08, 0.92, 28):
        point = (1.0 - alpha) * start + alpha * end
        if env_lib.is_point_blocked(float(point[0]), float(point[1]), padding=0.0):
            return True
    return False


class OcclusionDashV2Policy(HeuristicPolicy):
    mechanism = "single bounded obstacle-shadow staging phase followed by an irreversible target dash"

    def _reset_internal(self) -> None:
        self.shadow_goal: Optional[np.ndarray] = None
        self.shadow_uses = 0
        self.phase = "select_shadow"
        self.staging_deadline = 0

    @staticmethod
    def _find_shadow(state: WorldState) -> Optional[np.ndarray]:
        candidates = []
        for obstacle in getattr(map_config, "obstacles", []):
            center = _obstacle_center(obstacle)
            if center is None:
                continue
            if (
                center[0] < 12
                or center[0] > map_config.width - 12
                or center[1] < 12
                or center[1] > map_config.height - 12
            ):
                continue
            away = unit(center - state.defender)
            for distance in (34.0, 48.0):
                point = clip_point(center + distance * away)
                if env_lib.is_point_blocked(
                    point[0],
                    point[1],
                    padding=float(map_config.agent_radius) + 3.0,
                ):
                    continue
                if not _sample_line_blocked(state.defender, point):
                    continue
                route_cost = (
                    np.linalg.norm(point - state.attacker)
                    + 0.8 * np.linalg.norm(state.target - point)
                )
                candidates.append((float(route_cost), point))
        return min(candidates, key=lambda item: item[0])[1] if candidates else None

    def _decide(self, state: WorldState) -> Decision:
        risk = interception_risk(state)
        reason = "shadow_stage_active"
        if self.phase == "select_shadow":
            candidate = self._find_shadow(state)
            direct = max(state.attacker_target_distance, 1.0)
            if candidate is not None:
                route = (
                    np.linalg.norm(candidate - state.attacker)
                    + np.linalg.norm(state.target - candidate)
                )
                if route > 1.42 * direct:
                    candidate = None
            self.shadow_goal = candidate
            self.staging_deadline = state.step + 72
            self.phase = "shadow_stage" if candidate is not None else "target_dash"
            reason = "bounded_shadow_selected" if candidate is not None else "no_efficient_shadow"
        if self.phase == "shadow_stage" and (
            np.linalg.norm(state.attacker - self.shadow_goal) < 34.0
            or state.step >= self.staging_deadline
            or state.attacker_target_distance < 92.0
        ):
            reached = np.linalg.norm(state.attacker - self.shadow_goal) < 34.0
            self.phase = "target_dash"
            self.shadow_uses += 1
            reason = "shadow_reached" if reached else "shadow_deadline"
        goal = self.shadow_goal if self.phase == "shadow_stage" else state.target
        return Decision(
            goal,
            self.phase,
            risk,
            reason,
            speed_fraction=0.92 if self.phase == "shadow_stage" else 1.0,
            avoid_radius=10.0,
            details={
                "shadow_available": self.shadow_goal is not None,
                "shadow_uses": int(self.shadow_uses),
                "staging_steps_left": int(max(0, self.staging_deadline - state.step)),
            },
        )


__all__ = ["GeometryGatedFeintV3Policy", "OcclusionDashV2Policy"]
