"""
CBF-QP defender baseline for TAD.

Implemented policy:
- cbf_qp: Apollonius-style interception nominal control + CBF-QP safety filter
- cbf_qp_local: weaker local chase/guard nominal control + obstacle-only CBF filter
- cbf_qp_local_obs: local-observation Apollonius interception + radar-CBF filtering
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Tuple

import cvxpy as cp
import numpy as np

from envs import env_lib
from configs import map_config


def _normalize_angle(angle_deg: float) -> float:
    """Normalize angle to [-180, 180]."""
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def _bearing_deg(src: np.ndarray, dst: np.ndarray) -> float:
    vec = np.asarray(dst, dtype=np.float64) - np.asarray(src, dtype=np.float64)
    return math.degrees(math.atan2(float(vec[1]), float(vec[0])))


def _point_segment_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    vx = float(x2 - x1)
    vy = float(y2 - y1)
    wx = float(px - x1)
    wy = float(py - y1)
    seg_len2 = vx * vx + vy * vy
    if seg_len2 <= 1e-9:
        return math.hypot(px - x1, py - y1)
    ratio = max(0.0, min(1.0, (wx * vx + wy * vy) / seg_len2))
    proj_x = x1 + ratio * vx
    proj_y = y1 + ratio * vy
    return math.hypot(px - proj_x, py - proj_y)


class _ReachAvoidDefenderBase:
    """Common geometry helpers shared by defender rule baselines."""

    def __init__(
        self,
        defender_speed: Optional[float] = None,
        defender_max_turn: Optional[float] = None,
        attacker_speed: Optional[float] = None,
        obstacle_padding: Optional[float] = None,
        grid_size: float = 8.0,
        waypoint_reach: float = 10.0,
        replan_interval: int = 4,
    ):
        self.max_speed = float(
            getattr(map_config, "defender_speed", 2.6) if defender_speed is None else defender_speed
        )
        self.max_turn = float(
            getattr(map_config, "defender_max_angular_speed", 6.0)
            if defender_max_turn is None
            else defender_max_turn
        )
        self.max_turn_rad = math.radians(self.max_turn)
        self.attacker_speed = float(
            getattr(map_config, "attacker_speed", 2.0) if attacker_speed is None else attacker_speed
        )
        self.agent_radius = float(getattr(map_config, "agent_radius", 8.0))
        if obstacle_padding is None:
            obstacle_padding = self.agent_radius + 2.0
        self.obstacle_padding = float(obstacle_padding)
        self.grid_size = float(grid_size)
        self.waypoint_reach = float(waypoint_reach)
        self.replan_interval = int(max(1, replan_interval))

        self.map_w = float(getattr(map_config, "width", 640.0))
        self.map_h = float(getattr(map_config, "height", 640.0))

        self.prev_attacker_pos: Optional[np.ndarray] = None
        self.prev_control = np.zeros(2, dtype=np.float64)
        self.path: List[np.ndarray] = []
        self.current_path_index = 0
        self.last_goal: Optional[np.ndarray] = None
        self.step_count = 0

        if getattr(env_lib, "_RECT_OBS", None) is None:
            env_lib.build_occupancy(
                width=map_config.width,
                height=map_config.height,
                cell=map_config.pixel_size,
                obstacles=getattr(map_config, "obstacles", []),
            )

    def reset(self):
        self.prev_attacker_pos = None
        self.prev_control = np.zeros(2, dtype=np.float64)
        self.path = []
        self.current_path_index = 0
        self.last_goal = None
        self.step_count = 0

    def _extract_state(
        self, privileged_state: Dict
    ) -> Tuple[np.ndarray, float, np.ndarray, float, np.ndarray]:
        d = privileged_state["defender"]
        a = privileged_state["attacker"]
        t = privileged_state["target"]
        defender_pos = np.array([d["center_x"], d["center_y"]], dtype=np.float64)
        attacker_pos = np.array([a["center_x"], a["center_y"]], dtype=np.float64)
        target_pos = np.array([t["center_x"], t["center_y"]], dtype=np.float64)
        defender_heading = float(d.get("theta", 0.0))
        attacker_heading = float(a.get("theta", 0.0))
        return defender_pos, defender_heading, attacker_pos, attacker_heading, target_pos

    @staticmethod
    def _dist(p: np.ndarray, q: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(p, dtype=np.float64) - np.asarray(q, dtype=np.float64)))

    def _clip_point(self, p: np.ndarray) -> np.ndarray:
        q = np.asarray(p, dtype=np.float64).copy()
        q[0] = float(np.clip(q[0], 1.0, self.map_w - 1.0))
        q[1] = float(np.clip(q[1], 1.0, self.map_h - 1.0))
        return q

    def _is_blocked(self, p: np.ndarray, padding: Optional[float] = None) -> bool:
        pad = self.obstacle_padding if padding is None else float(padding)
        return bool(env_lib.is_point_blocked(float(p[0]), float(p[1]), padding=pad))

    def _nearest_free_point(self, p: np.ndarray) -> np.ndarray:
        q = self._clip_point(p)
        if not self._is_blocked(q):
            return q
        for radius in (4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 40.0, 56.0):
            for ang in range(0, 360, 20):
                c = q + np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))], dtype=np.float64) * radius
                c = self._clip_point(c)
                if not self._is_blocked(c):
                    return c
        return q

    def _obstacle_sdf(self, point: np.ndarray, padding: Optional[float] = None) -> float:
        px = float(point[0])
        py = float(point[1])
        pad = self.obstacle_padding if padding is None else float(padding)
        best = float("inf")

        rects = getattr(env_lib, "_RECT_OBS", None)
        if rects is not None:
            for x, y, w, h in rects:
                cx = float(x + 0.5 * w)
                cy = float(y + 0.5 * h)
                half_x = float(0.5 * w + pad)
                half_y = float(0.5 * h + pad)
                qx = abs(px - cx) - half_x
                qy = abs(py - cy) - half_y
                outside = math.hypot(max(qx, 0.0), max(qy, 0.0))
                inside = min(max(qx, qy), 0.0)
                best = min(best, outside + inside)

        circles = getattr(env_lib, "_CIRCLE_OBS", None)
        if circles is not None:
            for cx, cy, r in circles:
                best = min(best, math.hypot(px - cx, py - cy) - float(r + pad))

        segments = getattr(env_lib, "_SEGMENT_OBS", None)
        if segments is not None:
            for x1, y1, x2, y2, thick in segments:
                capsule_r = float(0.5 * thick + pad)
                best = min(best, _point_segment_distance(px, py, x1, y1, x2, y2) - capsule_r)

        if not np.isfinite(best):
            best = min(px, self.map_w - px, py, self.map_h - py) - pad
        return float(best)

    def _segment_blocked(self, src: np.ndarray, dst: np.ndarray, padding: float = 0.0) -> bool:
        seg = np.asarray(dst, dtype=np.float64) - np.asarray(src, dtype=np.float64)
        length = float(np.linalg.norm(seg))
        if length <= 1e-6:
            return False
        ang = math.atan2(float(seg[1]), float(seg[0]))
        free = float(env_lib.ray_distance_grid(src, ang, length, padding=padding))
        return bool(free < length - 1e-3)

    def _plan_path(self, start_pos: np.ndarray, goal_pos: np.ndarray) -> List[np.ndarray]:
        start_grid = (int(start_pos[0] / self.grid_size), int(start_pos[1] / self.grid_size))
        goal_grid = (int(goal_pos[0] / self.grid_size), int(goal_pos[1] / self.grid_size))

        open_set = []
        heapq.heappush(open_set, (0.0, start_grid))
        came_from = {}
        g_score = {start_grid: 0.0}

        max_x = max(1, int(self.map_w / self.grid_size))
        max_y = max(1, int(self.map_h / self.grid_size))

        while open_set:
            _cost, current = heapq.heappop(open_set)
            if current == goal_grid:
                path = []
                while current in came_from:
                    pos = np.array(
                        [current[0] * self.grid_size + self.grid_size * 0.5, current[1] * self.grid_size + self.grid_size * 0.5],
                        dtype=np.float64,
                    )
                    path.append(pos)
                    current = came_from[current]
                path.append(start_pos.copy())
                path.reverse()
                path[-1] = goal_pos.copy()
                return path

            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx = current[0] + dx
                    ny = current[1] + dy
                    if nx < 0 or ny < 0 or nx >= max_x or ny >= max_y:
                        continue
                    neighbor = (nx, ny)
                    neighbor_pos = np.array(
                        [nx * self.grid_size + self.grid_size * 0.5, ny * self.grid_size + self.grid_size * 0.5],
                        dtype=np.float64,
                    )
                    if self._is_blocked(neighbor_pos):
                        continue

                    move_cost = 1.414 if (dx != 0 and dy != 0) else 1.0
                    tentative = g_score[current] + move_cost
                    if tentative < g_score.get(neighbor, float("inf")):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative
                        heuristic = math.hypot(nx - goal_grid[0], ny - goal_grid[1])
                        heapq.heappush(open_set, (tentative + heuristic, neighbor))

        return [start_pos.copy(), goal_pos.copy()]

    def _select_navigation_waypoint(self, start_pos: np.ndarray, goal_pos: np.ndarray) -> np.ndarray:
        goal = self._nearest_free_point(self._clip_point(goal_pos))
        self.step_count += 1

        if not self._segment_blocked(start_pos, goal, padding=max(0.0, self.agent_radius - 1.0)):
            self.path = []
            self.current_path_index = 0
            self.last_goal = goal.copy()
            return goal

        need_replan = False
        if not self.path:
            need_replan = True
        elif self.last_goal is None:
            need_replan = True
        elif self._dist(goal, self.last_goal) > self.grid_size * 2.0:
            need_replan = True
        elif self.step_count % self.replan_interval == 0:
            need_replan = True

        if need_replan:
            self.path = self._plan_path(start_pos, goal)
            self.current_path_index = 0
            self.last_goal = goal.copy()

        while self.current_path_index + 1 < len(self.path):
            waypoint = self.path[self.current_path_index + 1]
            if self._dist(start_pos, waypoint) <= self.waypoint_reach:
                self.current_path_index += 1
            else:
                break

        if self.current_path_index + 1 < len(self.path):
            return self.path[self.current_path_index + 1].copy()
        return goal

    def _to_env_action(self, control: np.ndarray) -> np.ndarray:
        u = np.asarray(control, dtype=np.float64).reshape(2)
        speed = float(np.clip(u[0], 0.0, self.max_speed))
        turn_deg = float(np.clip(math.degrees(u[1]), -self.max_turn, self.max_turn))
        angle_delta_norm = float(np.clip(turn_deg / max(self.max_turn, 1e-6), -1.0, 1.0))
        speed_ratio = float(np.clip(speed / max(self.max_speed, 1e-6), 0.0, 1.0))
        speed_norm = speed_ratio * 2.0 - 1.0
        return np.array([angle_delta_norm, speed_norm], dtype=np.float32)


class DefenderCBFQPPolicy(_ReachAvoidDefenderBase):
    """Apollonius nominal interception plus CBF-QP safety filtering."""

    def __init__(
        self,
        apollonius_time_slack: float = 1.25,
        guard_trigger_dist: float = 44.0,
        guard_ring_radius: float = 38.0,
        nominal_turn_gain: float = 1.05,
        visibility_margin: float = 1.0,
        visibility_probe_extra: float = 24.0,
        obstacle_margin: float = 0.0,
        obstacle_slack_weight: float = 220.0,
        visibility_slack_weight: float = 4.0,
        smooth_weight: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.apollonius_time_slack = float(max(1.0, apollonius_time_slack))
        self.guard_trigger_dist = float(guard_trigger_dist)
        self.guard_ring_radius = float(guard_ring_radius)
        self.nominal_turn_gain = float(nominal_turn_gain)
        self.visibility_margin = float(visibility_margin)
        self.visibility_probe_extra = float(visibility_probe_extra)
        self.obstacle_margin = float(obstacle_margin)
        self.obstacle_slack_weight = float(obstacle_slack_weight)
        self.visibility_slack_weight = float(visibility_slack_weight)
        self.smooth_weight = float(smooth_weight)

    def _guard_anchor(self, attacker_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        vec = target_pos - attacker_pos
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            anchor = target_pos + np.array([-self.guard_ring_radius, 0.0], dtype=np.float64)
        else:
            anchor = target_pos - (vec / norm) * self.guard_ring_radius
        return self._nearest_free_point(self._clip_point(anchor))

    def _estimate_attacker_velocity(
        self, attacker_pos: np.ndarray, attacker_heading: float, target_pos: np.ndarray
    ) -> np.ndarray:
        if self.prev_attacker_pos is not None:
            measured = attacker_pos - self.prev_attacker_pos
            measured_speed = float(np.linalg.norm(measured))
            if measured_speed > 1e-4:
                if measured_speed > self.attacker_speed * 1.25:
                    measured *= self.attacker_speed / measured_speed
                return measured

        heading_vec = np.array(
            [math.cos(math.radians(attacker_heading)), math.sin(math.radians(attacker_heading))],
            dtype=np.float64,
        )
        if float(np.linalg.norm(heading_vec)) < 1e-6:
            target_vec = target_pos - attacker_pos
            target_norm = float(np.linalg.norm(target_vec))
            if target_norm > 1e-6:
                heading_vec = target_vec / target_norm
            else:
                heading_vec = np.array([1.0, 0.0], dtype=np.float64)
        return heading_vec * self.attacker_speed

    def _solve_apollonius_intercept(
        self, defender_pos: np.ndarray, attacker_pos: np.ndarray, attacker_vel: np.ndarray
    ) -> Tuple[Optional[float], Optional[np.ndarray]]:
        rel = np.asarray(attacker_pos, dtype=np.float64) - np.asarray(defender_pos, dtype=np.float64)
        vel = np.asarray(attacker_vel, dtype=np.float64)
        a = float(np.dot(vel, vel) - self.max_speed * self.max_speed)
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
        intercept = self._clip_point(attacker_pos + attacker_vel * t_int)
        intercept = self._nearest_free_point(intercept)
        return t_int, intercept

    def _select_nominal_goal(
        self,
        defender_pos: np.ndarray,
        attacker_pos: np.ndarray,
        attacker_heading: float,
        target_pos: np.ndarray,
    ) -> np.ndarray:
        attacker_vel = self._estimate_attacker_velocity(attacker_pos, attacker_heading, target_pos)
        attacker_speed = max(float(np.linalg.norm(attacker_vel)), 1e-6)
        if self._dist(defender_pos, attacker_pos) <= 96.0:
            return self._nearest_free_point(attacker_pos)

        guard_goal = self._guard_anchor(attacker_pos, target_pos)
        t_guard = self._dist(defender_pos, guard_goal) / max(self.max_speed, 1e-6)
        t_att_target = self._dist(attacker_pos, target_pos) / attacker_speed
        t_intercept, intercept_goal = self._solve_apollonius_intercept(defender_pos, attacker_pos, attacker_vel)

        if intercept_goal is None or t_intercept is None:
            return self._nearest_free_point(attacker_pos)

        attacker_near_target = self._dist(attacker_pos, target_pos) <= self.guard_trigger_dist
        intercept_timely = t_intercept <= self.apollonius_time_slack * t_att_target
        if intercept_timely or not attacker_near_target:
            return intercept_goal
        if t_intercept < t_guard * 0.92:
            return intercept_goal
        return guard_goal

    def _nominal_control(
        self,
        defender_pos: np.ndarray,
        defender_heading: float,
        attacker_pos: np.ndarray,
        attacker_heading: float,
        target_pos: np.ndarray,
    ) -> np.ndarray:
        strategic_goal = self._select_nominal_goal(defender_pos, attacker_pos, attacker_heading, target_pos)
        waypoint = self._select_navigation_waypoint(defender_pos, strategic_goal)

        desired_heading = _bearing_deg(defender_pos, waypoint)
        heading_err = _normalize_angle(desired_heading - defender_heading)
        turn_cmd_deg = float(np.clip(self.nominal_turn_gain * heading_err, -self.max_turn, self.max_turn))

        speed = self.max_speed
        abs_err = abs(heading_err)
        if abs_err > 90.0:
            speed *= 0.35
        elif abs_err > 60.0:
            speed *= 0.60
        elif abs_err > 35.0:
            speed *= 0.82

        waypoint_dist = self._dist(defender_pos, waypoint)
        if waypoint_dist < 18.0:
            speed *= 0.72

        goal_dist = self._dist(defender_pos, strategic_goal)
        if goal_dist < 36.0:
            speed *= 0.82

        return np.array([speed, math.radians(turn_cmd_deg)], dtype=np.float64)

    def _clip_control(self, control: np.ndarray) -> np.ndarray:
        u = np.asarray(control, dtype=np.float64).reshape(2).copy()
        u[0] = float(np.clip(u[0], 0.0, self.max_speed))
        u[1] = float(np.clip(u[1], -self.max_turn_rad, self.max_turn_rad))
        return u

    def _predict_defender_step(
        self, defender_pos: np.ndarray, defender_heading: float, control: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        u = self._clip_control(control)
        new_heading = _normalize_angle(defender_heading + math.degrees(u[1]))
        heading_rad = math.radians(new_heading)
        move = np.array([math.cos(heading_rad), math.sin(heading_rad)], dtype=np.float64) * u[0]
        next_pos = self._clip_point(defender_pos + move)
        return next_pos, new_heading

    def _predict_attacker_step(
        self,
        attacker_pos: np.ndarray,
        attacker_heading: float,
        target_pos: np.ndarray,
    ) -> np.ndarray:
        attacker_vel = self._estimate_attacker_velocity(attacker_pos, attacker_heading, target_pos)
        speed = float(np.linalg.norm(attacker_vel))
        if speed < 1e-6:
            direction = target_pos - attacker_pos
            norm = float(np.linalg.norm(direction))
            if norm > 1e-6:
                attacker_vel = direction / norm * self.attacker_speed
        elif speed > self.attacker_speed:
            attacker_vel = attacker_vel * (self.attacker_speed / speed)
        next_pos = self._clip_point(attacker_pos + attacker_vel)
        if self._is_blocked(next_pos, padding=max(2.0, self.agent_radius + 2.0)):
            return attacker_pos.copy()
        return next_pos

    def _visibility_barrier(
        self,
        defender_pos: np.ndarray,
        defender_heading: float,
        attacker_pos: np.ndarray,
        attacker_heading: float,
        target_pos: np.ndarray,
        control: np.ndarray,
    ) -> float:
        dist_att_target = self._dist(attacker_pos, target_pos)
        dist_def_att = self._dist(defender_pos, attacker_pos)
        if dist_att_target > 96.0 or dist_def_att > 144.0:
            return 1.0
        if dist_att_target > max(56.0, 1.25 * self.guard_trigger_dist):
            if self._segment_blocked(defender_pos, attacker_pos, padding=0.0):
                return 1.0

        next_defender, _ = self._predict_defender_step(defender_pos, defender_heading, control)
        next_attacker = self._predict_attacker_step(attacker_pos, attacker_heading, target_pos)
        los_vec = next_attacker - next_defender
        los_dist = float(np.linalg.norm(los_vec))
        if los_dist < 1e-6:
            return 1.0
        los_ang = math.atan2(float(los_vec[1]), float(los_vec[0]))
        ray_max = los_dist + self.visibility_margin + self.visibility_probe_extra
        free_dist = float(env_lib.ray_distance_grid(next_defender, los_ang, ray_max, padding=0.0))
        return free_dist - los_dist - self.visibility_margin

    def _obstacle_barrier(
        self,
        defender_pos: np.ndarray,
        defender_heading: float,
        control: np.ndarray,
    ) -> float:
        next_defender, _ = self._predict_defender_step(defender_pos, defender_heading, control)
        return self._obstacle_sdf(next_defender, padding=self.obstacle_padding) - self.obstacle_margin

    def _control_jacobian(self, fun, ref: np.ndarray) -> np.ndarray:
        ref = self._clip_control(ref)
        eps = np.array([0.15, math.radians(0.8)], dtype=np.float64)
        jac = np.zeros(2, dtype=np.float64)
        for i in range(2):
            step = np.zeros(2, dtype=np.float64)
            step[i] = eps[i]
            up = self._clip_control(ref + step)
            um = self._clip_control(ref - step)
            fp = float(fun(up))
            fm = float(fun(um))
            denom = float(up[i] - um[i])
            if abs(denom) < 1e-8:
                jac[i] = 0.0
            else:
                jac[i] = (fp - fm) / denom
        return jac

    def _solve_cbf_qp(
        self,
        defender_pos: np.ndarray,
        defender_heading: float,
        attacker_pos: np.ndarray,
        attacker_heading: float,
        target_pos: np.ndarray,
        ref: np.ndarray,
    ) -> np.ndarray:
        ref = self._clip_control(ref)

        obs_fun = lambda u: self._obstacle_barrier(defender_pos, defender_heading, u)
        vis_fun = lambda u: self._visibility_barrier(
            defender_pos, defender_heading, attacker_pos, attacker_heading, target_pos, u
        )

        h_obs = float(obs_fun(ref))
        h_vis = float(vis_fun(ref))
        j_obs = self._control_jacobian(obs_fun, ref)
        j_vis = self._control_jacobian(vis_fun, ref)

        u = cp.Variable(2)
        s_obs = cp.Variable(nonneg=True)
        s_vis = cp.Variable(nonneg=True)

        scale = np.array([1.0 / max(self.max_speed, 1e-6), 1.0 / max(self.max_turn_rad, 1e-6)], dtype=np.float64)
        obj = cp.sum_squares(cp.multiply(scale, u - ref))
        if np.any(self.prev_control):
            obj += self.smooth_weight * cp.sum_squares(cp.multiply(scale, u - self.prev_control))
        obj += self.obstacle_slack_weight * s_obs + self.visibility_slack_weight * s_vis

        constraints = [
            u[0] >= 0.0,
            u[0] <= self.max_speed,
            u[1] >= -self.max_turn_rad,
            u[1] <= self.max_turn_rad,
            h_obs + j_obs @ (u - ref) + s_obs >= 0.0,
            h_vis + j_vis @ (u - ref) + s_vis >= 0.0,
        ]

        prob = cp.Problem(cp.Minimize(obj), constraints)
        solution = None
        for solver in (cp.OSQP, cp.ECOS, cp.SCS):
            try:
                prob.solve(solver=solver, warm_start=True, verbose=False)
            except (cp.SolverError, ValueError):
                continue
            if u.value is not None:
                solution = np.asarray(u.value, dtype=np.float64).reshape(2)
                break

        if solution is None:
            return ref
        return self._clip_control(solution)

    def get_action(self, obs: np.ndarray, privileged_state: Dict) -> np.ndarray:
        defender_pos, defender_heading, attacker_pos, attacker_heading, target_pos = self._extract_state(privileged_state)
        ref = self._nominal_control(defender_pos, defender_heading, attacker_pos, attacker_heading, target_pos)
        control = self._solve_cbf_qp(
            defender_pos=defender_pos,
            defender_heading=defender_heading,
            attacker_pos=attacker_pos,
            attacker_heading=attacker_heading,
            target_pos=target_pos,
            ref=ref,
        )

        self.prev_attacker_pos = attacker_pos.copy()
        self.prev_control = control.copy()
        return self._to_env_action(control)


class DefenderCBFQPLocalPolicy(_ReachAvoidDefenderBase):
    """Weaker local CBF defender without global interception or path planning."""

    def __init__(
        self,
        guard_trigger_dist: float = 40.0,
        guard_ring_radius: float = 34.0,
        nominal_turn_gain: float = 0.9,
        obstacle_margin: float = 1.0,
        obstacle_slack_weight: float = 90.0,
        smooth_weight: float = 0.015,
        cruise_speed_ratio: float = 0.88,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.guard_trigger_dist = float(guard_trigger_dist)
        self.guard_ring_radius = float(guard_ring_radius)
        self.nominal_turn_gain = float(nominal_turn_gain)
        self.obstacle_margin = float(obstacle_margin)
        self.obstacle_slack_weight = float(obstacle_slack_weight)
        self.smooth_weight = float(smooth_weight)
        self.cruise_speed_ratio = float(np.clip(cruise_speed_ratio, 0.2, 1.0))

    def _guard_anchor(self, attacker_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        vec = target_pos - attacker_pos
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            anchor = target_pos + np.array([-self.guard_ring_radius, 0.0], dtype=np.float64)
        else:
            anchor = target_pos - (vec / norm) * self.guard_ring_radius
        return self._nearest_free_point(self._clip_point(anchor))

    def _select_nominal_goal(
        self,
        defender_pos: np.ndarray,
        attacker_pos: np.ndarray,
        target_pos: np.ndarray,
    ) -> np.ndarray:
        dist_att_target = self._dist(attacker_pos, target_pos)
        dist_def_att = self._dist(defender_pos, attacker_pos)

        if dist_att_target <= self.guard_trigger_dist:
            return self._guard_anchor(attacker_pos, target_pos)
        if dist_def_att <= 88.0:
            return self._nearest_free_point(attacker_pos)

        to_attacker = attacker_pos - defender_pos
        to_attacker_norm = float(np.linalg.norm(to_attacker))
        if to_attacker_norm < 1e-6:
            return self._nearest_free_point(attacker_pos)
        pull = defender_pos + (to_attacker / to_attacker_norm) * min(56.0, 0.7 * dist_def_att)
        blend = 0.72 * pull + 0.28 * self._guard_anchor(attacker_pos, target_pos)
        return self._nearest_free_point(self._clip_point(blend))

    def _nominal_control(
        self,
        defender_pos: np.ndarray,
        defender_heading: float,
        attacker_pos: np.ndarray,
        target_pos: np.ndarray,
    ) -> np.ndarray:
        strategic_goal = self._select_nominal_goal(defender_pos, attacker_pos, target_pos)
        desired_heading = _bearing_deg(defender_pos, strategic_goal)
        heading_err = _normalize_angle(desired_heading - defender_heading)
        turn_cmd_deg = float(np.clip(self.nominal_turn_gain * heading_err, -self.max_turn, self.max_turn))

        speed = self.max_speed * self.cruise_speed_ratio
        abs_err = abs(heading_err)
        if abs_err > 90.0:
            speed *= 0.28
        elif abs_err > 60.0:
            speed *= 0.50
        elif abs_err > 35.0:
            speed *= 0.72

        goal_dist = self._dist(defender_pos, strategic_goal)
        if goal_dist < 42.0:
            speed *= 0.72
        elif goal_dist < 80.0:
            speed *= 0.86

        return np.array([speed, math.radians(turn_cmd_deg)], dtype=np.float64)

    def _clip_control(self, control: np.ndarray) -> np.ndarray:
        u = np.asarray(control, dtype=np.float64).reshape(2).copy()
        u[0] = float(np.clip(u[0], 0.0, self.max_speed))
        u[1] = float(np.clip(u[1], -self.max_turn_rad, self.max_turn_rad))
        return u

    def _predict_defender_step(
        self, defender_pos: np.ndarray, defender_heading: float, control: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        u = self._clip_control(control)
        new_heading = _normalize_angle(defender_heading + math.degrees(u[1]))
        heading_rad = math.radians(new_heading)
        move = np.array([math.cos(heading_rad), math.sin(heading_rad)], dtype=np.float64) * u[0]
        next_pos = self._clip_point(defender_pos + move)
        return next_pos, new_heading

    def _obstacle_barrier(
        self,
        defender_pos: np.ndarray,
        defender_heading: float,
        control: np.ndarray,
    ) -> float:
        next_defender, _ = self._predict_defender_step(defender_pos, defender_heading, control)
        return self._obstacle_sdf(next_defender, padding=self.obstacle_padding) - self.obstacle_margin

    def _control_jacobian(self, fun, ref: np.ndarray) -> np.ndarray:
        ref = self._clip_control(ref)
        eps = np.array([0.15, math.radians(0.8)], dtype=np.float64)
        jac = np.zeros(2, dtype=np.float64)
        for i in range(2):
            step = np.zeros(2, dtype=np.float64)
            step[i] = eps[i]
            up = self._clip_control(ref + step)
            um = self._clip_control(ref - step)
            fp = float(fun(up))
            fm = float(fun(um))
            denom = float(up[i] - um[i])
            if abs(denom) < 1e-8:
                jac[i] = 0.0
            else:
                jac[i] = (fp - fm) / denom
        return jac

    def _solve_cbf_qp(
        self,
        defender_pos: np.ndarray,
        defender_heading: float,
        ref: np.ndarray,
    ) -> np.ndarray:
        ref = self._clip_control(ref)
        obs_fun = lambda u: self._obstacle_barrier(defender_pos, defender_heading, u)

        h_obs = float(obs_fun(ref))
        j_obs = self._control_jacobian(obs_fun, ref)

        u = cp.Variable(2)
        s_obs = cp.Variable(nonneg=True)

        scale = np.array([1.0 / max(self.max_speed, 1e-6), 1.0 / max(self.max_turn_rad, 1e-6)], dtype=np.float64)
        obj = cp.sum_squares(cp.multiply(scale, u - ref))
        if np.any(self.prev_control):
            obj += self.smooth_weight * cp.sum_squares(cp.multiply(scale, u - self.prev_control))
        obj += self.obstacle_slack_weight * s_obs

        constraints = [
            u[0] >= 0.0,
            u[0] <= self.max_speed,
            u[1] >= -self.max_turn_rad,
            u[1] <= self.max_turn_rad,
            h_obs + j_obs @ (u - ref) + s_obs >= 0.0,
        ]

        prob = cp.Problem(cp.Minimize(obj), constraints)
        solution = None
        for solver in (cp.OSQP, cp.ECOS, cp.SCS):
            try:
                prob.solve(solver=solver, warm_start=True, verbose=False)
            except (cp.SolverError, ValueError):
                continue
            if u.value is not None:
                solution = np.asarray(u.value, dtype=np.float64).reshape(2)
                break

        if solution is None:
            return ref
        return self._clip_control(solution)

    def get_action(self, obs: np.ndarray, privileged_state: Dict) -> np.ndarray:
        del obs
        defender_pos, defender_heading, attacker_pos, _, target_pos = self._extract_state(privileged_state)
        ref = self._nominal_control(defender_pos, defender_heading, attacker_pos, target_pos)
        control = self._solve_cbf_qp(
            defender_pos=defender_pos,
            defender_heading=defender_heading,
            ref=ref,
        )
        self.prev_control = control.copy()
        return self._to_env_action(control)


class DefenderCBFQPLocalObsPolicy(_ReachAvoidDefenderBase):
    """Local-observation Apollonius interception with radar-based obstacle CBF."""

    def __init__(
        self,
        apollonius_time_slack: float = 1.2,
        guard_trigger_dist: float = 44.0,
        guard_ring_radius: float = 38.0,
        nominal_turn_gain: float = 1.0,
        obstacle_margin: float = 10.0,
        obstacle_slack_weight: float = 160.0,
        smooth_weight: float = 0.02,
        radar_beam_half_width_deg: float = 12.0,
        cruise_speed_ratio: float = 0.94,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.apollonius_time_slack = float(max(1.0, apollonius_time_slack))
        self.guard_trigger_dist = float(guard_trigger_dist)
        self.guard_ring_radius = float(guard_ring_radius)
        self.nominal_turn_gain = float(nominal_turn_gain)
        self.obstacle_margin = float(obstacle_margin)
        self.obstacle_slack_weight = float(obstacle_slack_weight)
        self.smooth_weight = float(smooth_weight)
        self.radar_beam_half_width_deg = float(max(0.0, radar_beam_half_width_deg))
        self.cruise_speed_ratio = float(np.clip(cruise_speed_ratio, 0.2, 1.0))
        self.map_diagonal = math.hypot(self.map_w, self.map_h)
        self.radar_max_range = float(min(getattr(map_config, "FOV_RANGE", 300.0), self.map_diagonal))
        self.radar_rays = int(getattr(map_config, "RADAR_RAYS", 64))
        self.body_collision_radius = float(self.agent_radius + self.obstacle_padding * 0.2)

    def _denorm_distance(self, value: float) -> float:
        src = float(np.clip(value, -1.0, 1.0))
        return ((src + 1.0) * 0.5) * self.map_diagonal

    @staticmethod
    def _denorm_bearing_rad(value: float) -> float:
        src = float(np.clip(value, -1.0, 1.0))
        return math.radians(src * 180.0)

    def _decode_relative_point(self, distance_norm: float, bearing_norm: float) -> np.ndarray:
        dist = self._denorm_distance(distance_norm)
        ang = self._denorm_bearing_rad(bearing_norm)
        return np.array([dist * math.cos(ang), dist * math.sin(ang)], dtype=np.float64)

    def _extract_local_state(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, float, np.ndarray]:
        src = np.asarray(obs, dtype=np.float32).reshape(-1)
        if src.shape[0] < 71:
            raise ValueError(f"cbf_qp_local_obs expects defender obs dim >= 71, got {src.shape[0]}")
        attacker_pos = self._decode_relative_point(src[0], src[1])
        target_pos = self._decode_relative_point(src[5], src[6])
        is_visible = bool(float(src[3]) > 0.5)
        unobserved_steps = ((float(np.clip(src[4], -1.0, 1.0)) + 1.0) * 0.5) * float(
            getattr(map_config.EnvParameters, "MAX_UNOBSERVED_STEPS", 1.0)
        )
        radar = np.asarray(src[7:71], dtype=np.float64)
        return attacker_pos, target_pos, is_visible, unobserved_steps, radar

    def _estimate_attacker_velocity_local(
        self,
        attacker_pos: np.ndarray,
        target_pos: np.ndarray,
        is_visible: bool,
        unobserved_steps: float,
    ) -> np.ndarray:
        direction = target_pos - attacker_pos
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            direction = attacker_pos.copy()
            norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            direction = np.array([1.0, 0.0], dtype=np.float64)
            norm = 1.0
        direction = direction / norm

        speed_scale = 1.0
        if not is_visible:
            speed_scale *= 0.92
        if unobserved_steps > 0.0:
            decay = min(0.18, 0.006 * float(unobserved_steps))
            speed_scale *= max(0.74, 1.0 - decay)
        return direction * (self.attacker_speed * speed_scale)

    def _guard_anchor_local(self, attacker_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        vec = target_pos - attacker_pos
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            return target_pos + np.array([-self.guard_ring_radius, 0.0], dtype=np.float64)
        return target_pos - (vec / norm) * self.guard_ring_radius

    def _solve_apollonius_intercept_local(
        self, attacker_pos: np.ndarray, attacker_vel: np.ndarray
    ) -> Tuple[Optional[float], Optional[np.ndarray]]:
        rel = np.asarray(attacker_pos, dtype=np.float64)
        vel = np.asarray(attacker_vel, dtype=np.float64)
        a = float(np.dot(vel, vel) - self.max_speed * self.max_speed)
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
        return t_int, attacker_pos + attacker_vel * t_int

    def _select_nominal_goal_local(
        self,
        attacker_pos: np.ndarray,
        target_pos: np.ndarray,
        is_visible: bool,
        unobserved_steps: float,
    ) -> np.ndarray:
        attacker_vel = self._estimate_attacker_velocity_local(attacker_pos, target_pos, is_visible, unobserved_steps)
        attacker_speed = max(float(np.linalg.norm(attacker_vel)), 1e-6)

        if float(np.linalg.norm(attacker_pos)) <= 96.0:
            return attacker_pos.copy()

        guard_goal = self._guard_anchor_local(attacker_pos, target_pos)
        t_guard = float(np.linalg.norm(guard_goal)) / max(self.max_speed, 1e-6)
        t_att_target = float(np.linalg.norm(target_pos - attacker_pos)) / attacker_speed
        t_intercept, intercept_goal = self._solve_apollonius_intercept_local(attacker_pos, attacker_vel)

        if intercept_goal is None or t_intercept is None:
            return attacker_pos.copy()

        attacker_near_target = float(np.linalg.norm(target_pos - attacker_pos)) <= self.guard_trigger_dist
        intercept_timely = t_intercept <= self.apollonius_time_slack * t_att_target
        if intercept_timely or not attacker_near_target:
            return intercept_goal
        if t_intercept < t_guard * 0.92:
            return intercept_goal
        return guard_goal

    def _nominal_control_local(
        self,
        attacker_pos: np.ndarray,
        target_pos: np.ndarray,
        is_visible: bool,
        unobserved_steps: float,
        radar: np.ndarray,
    ) -> np.ndarray:
        strategic_goal = self._select_nominal_goal_local(attacker_pos, target_pos, is_visible, unobserved_steps)
        desired_heading = math.degrees(math.atan2(float(strategic_goal[1]), float(strategic_goal[0])))
        heading_err = _normalize_angle(desired_heading)
        turn_cmd_deg = float(np.clip(self.nominal_turn_gain * heading_err, -self.max_turn, self.max_turn))

        speed = self.max_speed * self.cruise_speed_ratio
        abs_err = abs(heading_err)
        if abs_err > 90.0:
            speed *= 0.35
        elif abs_err > 60.0:
            speed *= 0.58
        elif abs_err > 35.0:
            speed *= 0.80

        goal_dist = float(np.linalg.norm(strategic_goal))
        if goal_dist < 18.0:
            speed *= 0.68
        elif goal_dist < 36.0:
            speed *= 0.82

        front_clearance = self._directional_clearance(radar, math.radians(turn_cmd_deg))
        speed = min(speed, max(0.0, front_clearance - self.obstacle_margin))
        return np.array([speed, math.radians(turn_cmd_deg)], dtype=np.float64)

    def _clip_control(self, control: np.ndarray) -> np.ndarray:
        u = np.asarray(control, dtype=np.float64).reshape(2).copy()
        u[0] = float(np.clip(u[0], 0.0, self.max_speed))
        u[1] = float(np.clip(u[1], -self.max_turn_rad, self.max_turn_rad))
        return u

    def _angle_to_radar_index(self, angle_rad: float) -> float:
        angle = (float(angle_rad) % (2.0 * math.pi)) / (2.0 * math.pi)
        return angle * self.radar_rays

    def _interp_radar_distance(self, radar: np.ndarray, angle_rad: float) -> float:
        if radar.size == 0:
            return self.radar_max_range
        idx = self._angle_to_radar_index(angle_rad)
        i0 = int(math.floor(idx)) % radar.size
        i1 = (i0 + 1) % radar.size
        frac = idx - math.floor(idx)
        d0 = ((float(np.clip(radar[i0], -1.0, 1.0)) + 1.0) * 0.5) * self.radar_max_range
        d1 = ((float(np.clip(radar[i1], -1.0, 1.0)) + 1.0) * 0.5) * self.radar_max_range
        return (1.0 - frac) * d0 + frac * d1

    def _directional_clearance(self, radar: np.ndarray, angle_rad: float) -> float:
        offsets_deg = (-self.radar_beam_half_width_deg, 0.0, self.radar_beam_half_width_deg)
        clearances = [
            self._interp_radar_distance(radar, angle_rad + math.radians(offset_deg))
            for offset_deg in offsets_deg
        ]
        return max(0.0, min(clearances) - self.body_collision_radius)

    def _obstacle_barrier_local(self, radar: np.ndarray, control: np.ndarray) -> float:
        u = self._clip_control(control)
        move_heading = float(u[1])
        directional_clearance = self._directional_clearance(radar, move_heading)
        return directional_clearance - u[0] - self.obstacle_margin

    def _control_jacobian(self, fun, ref: np.ndarray) -> np.ndarray:
        ref = self._clip_control(ref)
        eps = np.array([0.15, math.radians(0.8)], dtype=np.float64)
        jac = np.zeros(2, dtype=np.float64)
        for i in range(2):
            step = np.zeros(2, dtype=np.float64)
            step[i] = eps[i]
            up = self._clip_control(ref + step)
            um = self._clip_control(ref - step)
            fp = float(fun(up))
            fm = float(fun(um))
            denom = float(up[i] - um[i])
            if abs(denom) < 1e-8:
                jac[i] = 0.0
            else:
                jac[i] = (fp - fm) / denom
        return jac

    def _solve_cbf_qp_local(self, radar: np.ndarray, ref: np.ndarray) -> np.ndarray:
        ref = self._clip_control(ref)
        obs_fun = lambda u: self._obstacle_barrier_local(radar, u)
        h_obs = float(obs_fun(ref))
        j_obs = self._control_jacobian(obs_fun, ref)

        u = cp.Variable(2)
        s_obs = cp.Variable(nonneg=True)
        scale = np.array([1.0 / max(self.max_speed, 1e-6), 1.0 / max(self.max_turn_rad, 1e-6)], dtype=np.float64)
        obj = cp.sum_squares(cp.multiply(scale, u - ref))
        if np.any(self.prev_control):
            obj += self.smooth_weight * cp.sum_squares(cp.multiply(scale, u - self.prev_control))
        obj += self.obstacle_slack_weight * s_obs

        constraints = [
            u[0] >= 0.0,
            u[0] <= self.max_speed,
            u[1] >= -self.max_turn_rad,
            u[1] <= self.max_turn_rad,
            h_obs + j_obs @ (u - ref) + s_obs >= 0.0,
        ]

        prob = cp.Problem(cp.Minimize(obj), constraints)
        solution = None
        for solver in (cp.OSQP, cp.ECOS, cp.SCS):
            try:
                prob.solve(solver=solver, warm_start=True, verbose=False)
            except (cp.SolverError, ValueError):
                continue
            if u.value is not None:
                solution = np.asarray(u.value, dtype=np.float64).reshape(2)
                break

        if solution is None:
            return ref
        return self._clip_control(solution)

    def get_action(self, obs: np.ndarray, privileged_state: Optional[Dict] = None) -> np.ndarray:
        del privileged_state
        attacker_pos, target_pos, is_visible, unobserved_steps, radar = self._extract_local_state(obs)
        ref = self._nominal_control_local(
            attacker_pos=attacker_pos,
            target_pos=target_pos,
            is_visible=is_visible,
            unobserved_steps=unobserved_steps,
            radar=radar,
        )
        control = self._solve_cbf_qp_local(radar, ref)
        self.prev_control = control.copy()
        return self._to_env_action(control)


REACH_AVOID_DEFENDER_POLICIES = {
    "cbf_qp": DefenderCBFQPPolicy,
    "cbf_qp_local": DefenderCBFQPLocalPolicy,
    "cbf_qp_local_obs": DefenderCBFQPLocalObsPolicy,
}


def create_reach_avoid_defender_policy(policy_name: str, **kwargs):
    name = str(policy_name).strip().lower()
    cls = REACH_AVOID_DEFENDER_POLICIES.get(name)
    if cls is None:
        valid = ", ".join(sorted(REACH_AVOID_DEFENDER_POLICIES.keys()))
        raise ValueError(f"Unknown reach-avoid defender policy '{policy_name}'. Valid: {valid}")
    return cls(**kwargs)
