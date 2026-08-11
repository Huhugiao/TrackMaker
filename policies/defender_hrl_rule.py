"""
Rule-based top-level HRL defender policies.

Current policy:
- Geo-trend top rule for 2-skill protect/chase HRL.
- Uses privileged geometry for fast policy prototyping in vs.py.
"""

from typing import Iterable, Optional, Tuple

import numpy as np
from configs import map_config


class DefenderHRLRulePolicy:
    """Geometry + trend rule for selecting protect/chase top skills."""

    def __init__(
        self,
        defender_speed: Optional[float] = None,
        attacker_speed: Optional[float] = None,
        threat_distance_target: float = 200.0,
        safe_target_distance_for_chase: float = 165.0,
        chase_distance_defender: float = 70.0,
        guard_margin: float = 14.0,
        chase_margin: float = 2.0,
        trend_epsilon: float = 0.20,
        threat_weight: float = 1.15,
        chase_weight: float = 1.0,
    ):
        self.defender_speed = float(
            map_config.defender_speed if defender_speed is None else defender_speed
        )
        self.attacker_speed = float(
            map_config.attacker_speed if attacker_speed is None else attacker_speed
        )
        self.threat_distance_target = float(threat_distance_target)
        self.safe_target_distance_for_chase = float(safe_target_distance_for_chase)
        self.chase_distance_defender = float(chase_distance_defender)
        self.guard_margin = float(guard_margin)
        self.chase_margin = float(chase_margin)
        self.trend_epsilon = float(trend_epsilon)
        self.threat_weight = float(threat_weight)
        self.chase_weight = float(chase_weight)

        self.attacker_target_capture_radius = float(
            getattr(map_config, 'agent_radius', 8.0) + getattr(map_config, 'target_radius', 16.0)
        )
        self.defender_target_reach_radius = float(
            getattr(map_config, 'agent_radius', 8.0) + getattr(map_config, 'target_radius', 16.0)
        )
        self.defender_attacker_capture_radius = float(
            getattr(map_config, 'capture_radius', 20.0)
        )
        self.prev_attacker_target_distance = None
        self.prev_attacker_pos = None
        self.last_debug = {}

    def reset(self):
        self.prev_attacker_target_distance = None
        self.prev_attacker_pos = None
        self.last_debug = {}

    @staticmethod
    def _extract_point(privileged_state: dict, role: str) -> np.ndarray:
        item = privileged_state[role]
        return np.asarray([item['center_x'], item['center_y']], dtype=np.float32)

    @staticmethod
    def _time_to_reach(distance: float, radius: float, speed: float) -> float:
        return max(0.0, float(distance) - float(radius)) / max(float(speed), 1e-6)

    @staticmethod
    def _resolve_skill_indices(skill_names: Optional[Iterable[str]]) -> Tuple[int, int]:
        if not skill_names:
            return 0, 1
        normalized = [str(name).strip().lower() for name in skill_names]
        chase_idx = normalized.index('chase') if 'chase' in normalized else 1
        primary_idx = 0
        for idx, name in enumerate(normalized):
            if name != 'chase':
                primary_idx = idx
                break
        return int(primary_idx), int(chase_idx)

    def select_skill(
        self,
        privileged_state: dict,
        skill_names: Optional[Iterable[str]] = None,
    ) -> int:
        defender_pos = self._extract_point(privileged_state, 'defender')
        attacker_pos = self._extract_point(privileged_state, 'attacker')
        target_pos = self._extract_point(privileged_state, 'target')

        d_at = float(np.linalg.norm(attacker_pos - target_pos))
        d_dt = float(np.linalg.norm(defender_pos - target_pos))
        d_da = float(np.linalg.norm(defender_pos - attacker_pos))

        t_at = self._time_to_reach(
            d_at,
            self.attacker_target_capture_radius,
            self.attacker_speed,
        )
        t_dt = self._time_to_reach(
            d_dt,
            self.defender_target_reach_radius,
            self.defender_speed,
        )
        t_da = self._time_to_reach(
            d_da,
            self.defender_attacker_capture_radius,
            self.defender_speed,
        )

        delta_d_at = None
        if self.prev_attacker_target_distance is not None:
            delta_d_at = float(d_at - self.prev_attacker_target_distance)

        to_target = target_pos - attacker_pos
        to_target_norm = float(np.linalg.norm(to_target))
        progress_to_target = None
        if self.prev_attacker_pos is not None and to_target_norm > 1e-6:
            attacker_delta = attacker_pos - self.prev_attacker_pos
            progress_to_target = float(np.dot(attacker_delta, to_target / to_target_norm))

        attacker_theta_deg = float(privileged_state['attacker'].get('theta', 0.0))
        theta_rad = np.deg2rad(attacker_theta_deg)
        attacker_forward = np.asarray([np.cos(theta_rad), np.sin(theta_rad)], dtype=np.float32)
        heading_to_target = 0.0
        if to_target_norm > 1e-6:
            heading_to_target = float(np.dot(attacker_forward, to_target / to_target_norm))

        self.prev_attacker_target_distance = float(d_at)
        self.prev_attacker_pos = attacker_pos.copy()

        primary_idx, chase_idx = self._resolve_skill_indices(skill_names)
        decision = primary_idx
        reason = 'target_guard'

        attacker_advancing = (
            (delta_d_at is not None and delta_d_at <= -self.trend_epsilon)
            or (progress_to_target is not None and progress_to_target >= self.trend_epsilon)
            or heading_to_target >= 0.55
        )
        attacker_retreating = (
            (delta_d_at is not None and delta_d_at >= self.trend_epsilon)
            or (progress_to_target is not None and progress_to_target <= -self.trend_epsilon)
            or heading_to_target <= -0.25
        )
        target_threat = (
            d_at <= self.threat_distance_target
            or t_at <= (t_dt + self.guard_margin)
            or (attacker_advancing and d_at <= (self.threat_distance_target + 70.0))
        )
        chase_window = (
            d_at >= self.safe_target_distance_for_chase
            and (
                d_da <= self.chase_distance_defender
                or (t_da + self.chase_margin) <= t_at
                or attacker_retreating
            )
        )

        if target_threat:
            decision = primary_idx
            reason = 'target_threat'
        elif chase_window:
            decision = chase_idx
            reason = 'chase_window'
        elif attacker_advancing:
            decision = primary_idx
            reason = 'attacker_advancing'
        elif attacker_retreating:
            decision = chase_idx
            reason = 'attacker_retreating'
        else:
            threat_score = t_dt - t_at
            chase_score = t_at - t_da
            if self.threat_weight * threat_score >= self.chase_weight * chase_score:
                decision = primary_idx
                reason = 'threat_score'
            else:
                decision = chase_idx
                reason = 'chase_score'

        self.last_debug = {
            'd_at': d_at,
            'd_dt': d_dt,
            'd_da': d_da,
            't_at': t_at,
            't_dt': t_dt,
            't_da': t_da,
            'delta_d_at': delta_d_at,
            'progress_to_target': progress_to_target,
            'heading_to_target': heading_to_target,
            'attacker_advancing': bool(attacker_advancing),
            'attacker_retreating': bool(attacker_retreating),
            'target_threat': bool(target_threat),
            'chase_window': bool(chase_window),
            'decision': int(decision),
            'reason': reason,
        }
        return int(decision)

    def get_action(
        self,
        obs: np.ndarray,
        privileged_state: dict,
        skill_names: Optional[Iterable[str]] = None,
    ) -> np.ndarray:
        del obs
        skill_idx = self.select_skill(privileged_state, skill_names=skill_names)
        return np.asarray([skill_idx], dtype=np.float32)
