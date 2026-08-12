"""Local-observation steering filter for learned Defender policies."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np


class RadarSafetyFilter:
    """Keep policy speed unchanged and minimally project unsafe steering."""

    def __init__(
        self,
        *,
        radar_rays: int = 64,
        radar_max_range: float = 300.0,
        agent_radius: float = 8.0,
        safety_margin: float = 4.0,
        immediate_margin: float = 4.0,
        horizon_steps: int = 6,
        turn_samples: int = 33,
        selection_mode: str = "closest",
        latch_steering: bool = True,
    ):
        self.radar_rays = int(max(4, radar_rays))
        self.radar_max_range = float(max(1.0, radar_max_range))
        self.agent_radius = float(max(0.0, agent_radius))
        self.clearance_radius = float(max(0.0, self.agent_radius + safety_margin))
        self.immediate_clearance_radius = float(max(0.0, self.agent_radius + immediate_margin))
        self.horizon_steps = int(max(1, horizon_steps))
        self.turn_samples = int(max(3, turn_samples))
        self.selection_mode = str(selection_mode).strip().lower()
        if self.selection_mode not in {"closest", "max_clearance"}:
            raise ValueError(f"Unknown radar safety selection_mode={selection_mode!r}")
        self.latch_steering = bool(latch_steering)
        self._latched_turn_sign = 0.0

    def reset(self) -> None:
        self._latched_turn_sign = 0.0

    def get_state(self) -> Dict[str, float]:
        return {"latched_turn_sign": float(self._latched_turn_sign)}

    def set_state(self, state: Dict[str, float] | None) -> None:
        self._latched_turn_sign = float((state or {}).get("latched_turn_sign", 0.0))

    def _obstacle_points(self, defender_obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(defender_obs, dtype=np.float32).reshape(-1)
        radar = np.asarray(obs[7:7 + self.radar_rays], dtype=np.float64)
        if radar.size != self.radar_rays:
            raise ValueError(
                f"Defender observation requires {self.radar_rays} radar values, got {radar.size}"
            )

        distances = (np.clip(radar, -1.0, 1.0) + 1.0) * 0.5 * self.radar_max_range
        angles = np.arange(self.radar_rays, dtype=np.float64) * (2.0 * math.pi / self.radar_rays)

        # Conservative midpoint samples close angular gaps between adjacent rays.
        midpoint_distances = np.minimum(distances, np.roll(distances, -1))
        midpoint_angles = angles + math.pi / self.radar_rays
        all_distances = np.concatenate((distances, midpoint_distances))
        all_angles = np.concatenate((angles, midpoint_angles))
        valid = all_distances < self.radar_max_range - 1e-6
        if not np.any(valid):
            return np.empty((0, 2), dtype=np.float64)
        return np.stack(
            (
                all_distances[valid] * np.cos(all_angles[valid]),
                all_distances[valid] * np.sin(all_angles[valid]),
            ),
            axis=1,
        )

    @staticmethod
    def _point_segment_clearance(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
        if points.size == 0:
            return math.inf
        segment = end - start
        denom = float(np.dot(segment, segment))
        if denom <= 1e-12:
            return float(np.linalg.norm(points - start, axis=1).min())
        projection = np.clip(((points - start) @ segment) / denom, 0.0, 1.0)
        closest = start + projection[:, None] * segment
        return float(np.linalg.norm(points - closest, axis=1).min())

    def _rollout_clearance(
        self,
        points: np.ndarray,
        turn_deg: float,
        speed: float,
        steps: int | None = None,
    ) -> float:
        if points.size == 0:
            return math.inf
        position = np.zeros(2, dtype=np.float64)
        heading = 0.0
        turn_rad = math.radians(float(turn_deg))
        min_clearance = math.inf
        for _ in range(self.horizon_steps if steps is None else int(max(1, steps))):
            heading += turn_rad
            next_position = position + float(speed) * np.array(
                [math.cos(heading), math.sin(heading)], dtype=np.float64
            )
            min_clearance = min(
                min_clearance,
                self._point_segment_clearance(points, position, next_position),
            )
            position = next_position
        return float(min_clearance)

    @staticmethod
    def _decode_action(action: np.ndarray, max_turn: float, max_speed: float) -> Tuple[float, float, bool]:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size != 2:
            raise ValueError("Defender action must contain exactly two elements")
        normalized = bool(np.all(np.abs(arr) <= 1.0 + 1e-6))
        if normalized:
            turn = float(np.clip(arr[0], -1.0, 1.0) * max_turn)
            speed = float((np.clip(arr[1], -1.0, 1.0) + 1.0) * 0.5 * max_speed)
            return turn, speed, True
        return float(arr[0]), float(arr[1]), False

    @staticmethod
    def _encode_action(turn: float, speed: float, max_turn: float, max_speed: float, normalized: bool):
        if not normalized:
            return np.asarray([turn, speed], dtype=np.float32)
        turn_norm = 0.0 if max_turn <= 1e-9 else float(np.clip(turn / max_turn, -1.0, 1.0))
        speed_norm = -1.0 if max_speed <= 1e-9 else float(np.clip(2.0 * speed / max_speed - 1.0, -1.0, 1.0))
        return np.asarray([turn_norm, speed_norm], dtype=np.float32)

    def filter_action(
        self,
        defender_obs: np.ndarray,
        action: np.ndarray,
        *,
        max_turn: float,
        max_speed: float,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        raw = np.asarray(action, dtype=np.float32).reshape(-1)
        turn, speed, normalized = self._decode_action(raw, max_turn, max_speed)
        points = self._obstacle_points(defender_obs)
        latched = bool(self.latch_steering and self._latched_turn_sign != 0.0)
        low_speed_latch = bool(latched and speed < 0.25 * max_speed)
        planning_speed = float(speed)
        raw_clearance = self._rollout_clearance(points, turn, planning_speed)
        raw_immediate_clearance = self._rollout_clearance(points, turn, planning_speed, steps=1)
        diagnostics: Dict[str, object] = {
            "enabled": True,
            "intervened": False,
            "mode": "pass",
            "raw_turn": float(turn),
            "executed_turn": float(turn),
            "speed": float(speed),
            "raw_clearance": float(raw_clearance),
            "raw_immediate_clearance": float(raw_immediate_clearance),
            "executed_clearance": float(raw_clearance),
            "executed_immediate_clearance": float(raw_immediate_clearance),
            "action_delta": 0.0,
        }
        raw_is_safe = (
            raw_clearance > self.clearance_radius
            and raw_immediate_clearance > self.immediate_clearance_radius
        )
        if raw_is_safe:
            # Preserve only the previous side preference across recurrent
            # low-speed pulses.  The actually safe policy action still passes
            # through unchanged; the latch may only break a later steering tie.
            if not low_speed_latch:
                self._latched_turn_sign = 0.0
            diagnostics["latched_turn_sign"] = float(self._latched_turn_sign)
            return action, diagnostics

        candidate_turns = np.linspace(-max_turn, max_turn, self.turn_samples, dtype=np.float64)
        candidate_turns = np.unique(np.append(candidate_turns, turn))
        candidates = [
            (
                self._rollout_clearance(points, float(candidate), planning_speed),
                self._rollout_clearance(points, float(candidate), planning_speed, steps=1),
                float(candidate),
            )
            for candidate in candidate_turns
        ]
        safe = [
            item for item in candidates
            if item[0] > self.clearance_radius and item[1] > self.immediate_clearance_radius
        ]
        if safe:
            if self.selection_mode == "max_clearance":
                clearance, immediate_clearance, selected_turn = max(
                    safe,
                    key=lambda item: (
                        item[0],
                        item[1],
                        -abs(item[2] - turn),
                        item[2] * self._latched_turn_sign,
                    ),
                )
            else:
                clearance, immediate_clearance, selected_turn = min(
                    safe,
                    key=lambda item: (
                        abs(item[2] - turn),
                        0 if item[2] * self._latched_turn_sign > 0.0 else 1,
                        -item[0],
                        -item[1],
                    ),
                )
            mode = "steer"
        else:
            clearance, immediate_clearance, selected_turn = max(
                candidates,
                key=lambda item: (
                    item[1],
                    item[0],
                    -abs(item[2] - turn),
                    item[2] * self._latched_turn_sign,
                ),
            )
            mode = "unavoidable"
        if self.latch_steering and selected_turn != 0.0:
            self._latched_turn_sign = float(np.sign(selected_turn))

        applied = self._encode_action(selected_turn, speed, max_turn, max_speed, normalized)
        # The safety layer owns steering only; preserve the policy speed bit-for-bit.
        applied[1] = raw[1]
        diagnostics.update(
            {
                "intervened": not bool(np.allclose(applied, raw, atol=1e-6)),
                "mode": mode,
                "executed_turn": float(selected_turn),
                "executed_clearance": float(clearance),
                "executed_immediate_clearance": float(immediate_clearance),
                "action_delta": float(np.linalg.norm(np.asarray(applied) - raw)),
                "latched_turn_sign": float(self._latched_turn_sign),
            }
        )
        return applied, diagnostics
