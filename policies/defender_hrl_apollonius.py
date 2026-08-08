"""
Apollonius-style top-level HRL rule policy.

At each step:
- compute an obstacle-agnostic nominal defender direction from pursuit-evasion geometry
- infer current baseline/chase primitive directions
- select the skill whose direction is closer to the nominal direction
"""

from typing import Iterable, Optional, Tuple

import math
import numpy as np
import torch

from configs import map_config
from skill.util import build_critic_observation


def _normalize_angle(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


class DefenderHRLApolloniusLabelPolicy:
    """Match bottom-skill action direction to an Apollonius nominal direction."""

    def __init__(
        self,
        defender_speed: Optional[float] = None,
        attacker_speed: Optional[float] = None,
        apollonius_time_slack: float = 1.25,
        guard_trigger_dist: float = 44.0,
        guard_ring_radius: float = 38.0,
    ):
        self.defender_speed = float(
            map_config.defender_speed if defender_speed is None else defender_speed
        )
        self.attacker_speed = float(
            map_config.attacker_speed if attacker_speed is None else attacker_speed
        )
        self.apollonius_time_slack = float(max(1.0, apollonius_time_slack))
        self.guard_trigger_dist = float(guard_trigger_dist)
        self.guard_ring_radius = float(guard_ring_radius)
        self.prev_attacker_pos = None
        self.last_debug = {}

    def reset(self):
        self.prev_attacker_pos = None
        self.last_debug = {}

    @staticmethod
    def _extract_state(privileged_state: dict):
        defender = privileged_state['defender']
        attacker = privileged_state['attacker']
        target = privileged_state['target']
        defender_pos = np.asarray([defender['center_x'], defender['center_y']], dtype=np.float64)
        attacker_pos = np.asarray([attacker['center_x'], attacker['center_y']], dtype=np.float64)
        target_pos = np.asarray([target['center_x'], target['center_y']], dtype=np.float64)
        defender_heading = float(defender.get('theta', 0.0))
        attacker_heading = float(attacker.get('theta', 0.0))
        return defender_pos, defender_heading, attacker_pos, attacker_heading, target_pos

    @staticmethod
    def _dist(p: np.ndarray, q: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(p, dtype=np.float64) - np.asarray(q, dtype=np.float64)))

    @staticmethod
    def _bearing_deg(src: np.ndarray, dst: np.ndarray) -> float:
        vec = np.asarray(dst, dtype=np.float64) - np.asarray(src, dtype=np.float64)
        return math.degrees(math.atan2(float(vec[1]), float(vec[0])))

    @staticmethod
    def _angle_diff_deg(a: float, b: float) -> float:
        return abs(_normalize_angle(float(a) - float(b)))

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

    def _guard_anchor(self, attacker_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        vec = np.asarray(target_pos, dtype=np.float64) - np.asarray(attacker_pos, dtype=np.float64)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            return np.asarray(target_pos, dtype=np.float64) + np.asarray(
                [-self.guard_ring_radius, 0.0], dtype=np.float64
            )
        return np.asarray(target_pos, dtype=np.float64) - (vec / norm) * self.guard_ring_radius

    def _estimate_attacker_velocity(
        self,
        attacker_pos: np.ndarray,
        attacker_heading: float,
        target_pos: np.ndarray,
    ) -> np.ndarray:
        if self.prev_attacker_pos is not None:
            measured = np.asarray(attacker_pos, dtype=np.float64) - np.asarray(self.prev_attacker_pos, dtype=np.float64)
            measured_speed = float(np.linalg.norm(measured))
            if measured_speed > 1e-4:
                if measured_speed > self.attacker_speed * 1.25:
                    measured = measured * (self.attacker_speed / measured_speed)
                return measured

        heading_vec = np.asarray(
            [math.cos(math.radians(attacker_heading)), math.sin(math.radians(attacker_heading))],
            dtype=np.float64,
        )
        if float(np.linalg.norm(heading_vec)) < 1e-6:
            target_vec = np.asarray(target_pos, dtype=np.float64) - np.asarray(attacker_pos, dtype=np.float64)
            target_norm = float(np.linalg.norm(target_vec))
            if target_norm > 1e-6:
                heading_vec = target_vec / target_norm
            else:
                heading_vec = np.asarray([1.0, 0.0], dtype=np.float64)
        return heading_vec * self.attacker_speed

    def _solve_apollonius_intercept(
        self,
        defender_pos: np.ndarray,
        attacker_pos: np.ndarray,
        attacker_vel: np.ndarray,
    ):
        rel = np.asarray(attacker_pos, dtype=np.float64) - np.asarray(defender_pos, dtype=np.float64)
        vel = np.asarray(attacker_vel, dtype=np.float64)
        a = float(np.dot(vel, vel) - self.defender_speed * self.defender_speed)
        b = float(2.0 * np.dot(rel, vel))
        c = float(np.dot(rel, rel))

        roots = []
        if abs(a) < 1e-8:
            if abs(b) > 1e-8:
                roots.append(-c / b)
        else:
            disc = b * b - 4.0 * a * c
            if disc >= 0.0:
                sqrt_disc = math.sqrt(max(0.0, disc))
                roots.extend([(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)])

        roots = [float(r) for r in roots if np.isfinite(r) and r > 1e-4]
        if not roots:
            return None, None
        t_int = min(roots)
        intercept = np.asarray(attacker_pos, dtype=np.float64) + np.asarray(attacker_vel, dtype=np.float64) * t_int
        return float(t_int), intercept

    def _select_nominal_goal(
        self,
        defender_pos: np.ndarray,
        attacker_pos: np.ndarray,
        attacker_heading: float,
        target_pos: np.ndarray,
    ):
        attacker_vel = self._estimate_attacker_velocity(attacker_pos, attacker_heading, target_pos)
        attacker_speed = max(float(np.linalg.norm(attacker_vel)), 1e-6)
        if self._dist(defender_pos, attacker_pos) <= 96.0:
            return np.asarray(attacker_pos, dtype=np.float64), 'direct_chase'

        guard_goal = self._guard_anchor(attacker_pos, target_pos)
        t_guard = self._dist(defender_pos, guard_goal) / max(self.defender_speed, 1e-6)
        t_att_target = self._dist(attacker_pos, target_pos) / attacker_speed
        t_intercept, intercept_goal = self._solve_apollonius_intercept(defender_pos, attacker_pos, attacker_vel)

        if intercept_goal is None or t_intercept is None:
            return np.asarray(attacker_pos, dtype=np.float64), 'fallback_chase'

        attacker_near_target = self._dist(attacker_pos, target_pos) <= self.guard_trigger_dist
        intercept_timely = t_intercept <= self.apollonius_time_slack * t_att_target
        if intercept_timely or (not attacker_near_target):
            return np.asarray(intercept_goal, dtype=np.float64), 'apollonius_intercept'
        if t_intercept < t_guard * 0.92:
            return np.asarray(intercept_goal, dtype=np.float64), 'apollonius_intercept_fast'
        return np.asarray(guard_goal, dtype=np.float64), 'guard_anchor'

    @staticmethod
    def _infer_skill_action(env, skill_name: str, net, defender_obs: np.ndarray, attacker_obs: np.ndarray) -> np.ndarray:
        device = env.device
        obs_tensor = torch.as_tensor(defender_obs, dtype=torch.float32, device=device).unsqueeze(0)
        critic_obs = build_critic_observation(defender_obs, attacker_obs)
        critic_tensor = torch.as_tensor(critic_obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            if bool(getattr(net, 'is_recurrent', False)) and hasattr(net, 'forward_recurrent'):
                actor_hidden = env._skill_actor_hidden.get(skill_name)
                critic_hidden = env._skill_critic_hidden.get(skill_name)
                mean, _value, _log_std, _next_actor_hidden, _next_critic_hidden = net.forward_recurrent(
                    obs_tensor,
                    critic_tensor,
                    actor_hidden=actor_hidden,
                    critic_hidden=critic_hidden,
                )
            else:
                mean, _value, _log_std = net(obs_tensor, critic_tensor)
            action = torch.tanh(mean).cpu().numpy()[0]
        return np.asarray(action, dtype=np.float32)

    @staticmethod
    def _action_heading_deg(base_env, action: np.ndarray) -> float:
        physical = base_env._control_to_physical(action, 'defender')
        if physical is None:
            return float(base_env.defender.get('theta', 0.0))
        angle_delta, speed = physical
        if abs(float(speed)) <= 1e-6:
            return float(base_env.defender.get('theta', 0.0))
        return float(base_env.defender.get('theta', 0.0) + float(angle_delta))

    def get_action(self, obs: np.ndarray, env, attacker_obs: np.ndarray = None) -> np.ndarray:
        base_env = env.env if hasattr(env, 'env') else env
        privileged_state = base_env.get_privileged_state()
        defender_pos, defender_heading, attacker_pos, attacker_heading, target_pos = self._extract_state(privileged_state)

        nominal_goal, nominal_mode = self._select_nominal_goal(
            defender_pos,
            attacker_pos,
            attacker_heading,
            target_pos,
        )
        nominal_heading = self._bearing_deg(defender_pos, nominal_goal)

        if attacker_obs is None:
            if hasattr(env, 'cached_skill_obs') and env.cached_skill_obs is not None:
                defender_obs, attacker_obs = env.cached_skill_obs
            else:
                raise ValueError('HRL Apollonius label policy requires attacker observation.')
        else:
            defender_obs = np.asarray(obs, dtype=np.float32)
            attacker_obs = np.asarray(attacker_obs, dtype=np.float32)

        skill_names = list(getattr(env, 'skill_names', ['baseline', 'chase']))
        primary_idx, chase_idx = self._resolve_skill_indices(skill_names)
        primary_name = skill_names[primary_idx]
        chase_name = skill_names[chase_idx]
        primary_net = env.skill_nets[primary_idx]
        chase_net = env.skill_nets[chase_idx]

        primary_action = self._infer_skill_action(env, primary_name, primary_net, defender_obs, attacker_obs)
        chase_action = self._infer_skill_action(env, chase_name, chase_net, defender_obs, attacker_obs)

        primary_heading = self._action_heading_deg(base_env, primary_action)
        chase_heading = self._action_heading_deg(base_env, chase_action)
        primary_err = self._angle_diff_deg(primary_heading, nominal_heading)
        chase_err = self._angle_diff_deg(chase_heading, nominal_heading)

        if primary_err <= chase_err:
            decision = primary_idx
            reason = 'primary_closer'
        else:
            decision = chase_idx
            reason = 'chase_closer'

        self.prev_attacker_pos = np.asarray(attacker_pos, dtype=np.float64).copy()
        self.last_debug = {
            'nominal_mode': nominal_mode,
            'nominal_heading': float(nominal_heading),
            'primary_heading': float(primary_heading),
            'chase_heading': float(chase_heading),
            'primary_err': float(primary_err),
            'chase_err': float(chase_err),
            'decision': int(decision),
            'decision_reason': reason,
            'defender_heading': float(defender_heading),
        }
        return np.asarray([decision], dtype=np.float32)
