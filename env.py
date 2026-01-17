import os
import sys
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import env_lib, map_config
from map_config import EnvParameters

# Initialize pygame in headless mode for Ray workers
# This prevents hanging when creating surfaces in distributed environments
if not pygame.get_init():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("DISPLAY", "")
    pygame.init()
    pygame.display.set_mode((1, 1))  # Minimal display surface

class TADEnv(gym.Env):
    Metadata = {'render_modes': ['rgb_array'], 'render_fps': 40}

    def __init__(self, spawn_outside_fov=False, reward_mode='standard'):
        super().__init__()
        self.spawn_outside_fov = bool(spawn_outside_fov)
        self.reward_mode = reward_mode  # 'standard', 'protect', 'chase'
        self.mask_flag = getattr(map_config, 'mask_flag', False)
        self.width = map_config.width
        self.height = map_config.height
        self.pixel_size = map_config.pixel_size
        self.attacker_speed = map_config.attacker_speed
        self.defender_speed = map_config.defender_speed

        self.defender = None
        self.attacker = None
        self.target = None
        self._render_surface = None
        self.defender_trajectory = []
        self.attacker_trajectory = []
        self.step_count = 0
        self.prev_defender_pos = None
        self.last_defender_pos = None
        self.prev_attacker_pos = None
        self.last_attacker_pos = None

        self.fov_angle = EnvParameters.FOV_ANGLE
        self.fov_range = EnvParameters.FOV_RANGE
        self.radar_rays = EnvParameters.RADAR_RAYS

        self.capture_radius = float(getattr(map_config, 'capture_radius', 20.0))
        self.capture_sector_angle_deg = float(getattr(map_config, 'capture_sector_angle_deg', 30.0))
        self.capture_required_steps = int(getattr(map_config, 'capture_required_steps', 1))
        self._capture_counter_defender = 0
        self._capture_counter_attacker = 0

        self.last_observed_attacker_pos = None
        self.steps_since_observed = 0
        self._best_distance_attacker = None
        self._best_distance_target = None

        obs_dim = 5 + 64 + 2  # 71维: attacker_info(5) + radar(64) + target_info(2)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        self.current_obs = None
        self._fov_cache = None
        self._fov_cache_valid = False

    def _get_obs_features(self):
        defender_obs = self._get_defender_observation()
        attacker_obs = self._get_attacker_observation()
        return defender_obs, attacker_obs

    def get_normalized_attacker_info(self):
        """
        Get normalized attacker relative position and visibility flag for GRU training.

        Returns:
            rel_x_norm: Normalized relative x in [0, 1]
            rel_y_norm: Normalized relative y in [0, 1]
            is_visible: Boolean indicating if attacker is visible (in FOV and not occluded)
        """
        true_rel_vec, true_dist = self._get_relative_position(self.defender, self.attacker)
        absolute_angle = math.atan2(true_rel_vec[1], true_rel_vec[0])
        true_rel_angle_deg = self._normalize_angle(math.degrees(absolute_angle) - self.defender['theta'])
        fov_half = self.fov_angle * 0.5

        in_fov, occluded = self._update_visibility(true_rel_angle_deg, true_dist, fov_half)
        is_visible = (in_fov > 0.5 and occluded < 0.5)

        # Calculate relative position in [0, 1] range
        map_diagonal = math.hypot(self.width, self.height)
        normalized_distance = np.clip((true_dist / map_diagonal) * 2.0 - 1.0, -1.0, 1.0)

        abs_ang = math.atan2(true_rel_vec[1], true_rel_vec[0])
        rel_ang = self._normalize_angle(math.degrees(abs_ang) - self.defender['theta'])

        # Convert to normalized [0, 1] coordinates
        # Map [-1, 1] to [0, 1]
        rel_x_norm = (normalized_distance + 1.0) / 2.0
        rel_y_norm = (rel_ang / 180.0 + 1.0) / 2.0

        return float(rel_x_norm), float(rel_y_norm), bool(is_visible)

    def _get_defender_observation(self, use_privileged=False, gru_prediction=None):
        """
        Get defender observation.

        Args:
            use_privileged: If True, use true attacker position even when occluded
            gru_prediction: Optional tuple (pred_x_norm, pred_y_norm) from GRU in [0, 1] range
                          Used when attacker is occluded and use_privileged is False
        """
        obs = np.zeros(71, dtype=np.float32)

        true_rel_vec, true_dist = self._get_relative_position(self.defender, self.attacker)
        absolute_angle = math.atan2(true_rel_vec[1], true_rel_vec[0])
        true_rel_angle_deg = self._normalize_angle(math.degrees(absolute_angle) - self.defender['theta'])
        fov_half = self.fov_angle * 0.5

        in_fov, occluded = self._update_visibility(true_rel_angle_deg, true_dist, fov_half)

        obs_attacker_state = None
        is_visible = (in_fov > 0.5 and occluded < 0.5)

        if is_visible or use_privileged:
            obs_attacker_state = self.attacker
        elif gru_prediction is not None:
            # Use GRU prediction: convert normalized [0, 1] back to position
            pred_x_norm, pred_y_norm = gru_prediction
            # Convert [0, 1] back to [-1, 1]
            pred_dist_norm = pred_x_norm * 2.0 - 1.0
            pred_bearing_norm = pred_y_norm * 2.0 - 1.0
            # Convert [-1, 1] to actual values
            map_diagonal = math.hypot(self.width, self.height)
            pred_dist = ((pred_dist_norm + 1.0) / 2.0) * map_diagonal
            pred_bearing = pred_bearing_norm * 180.0
            # Calculate position from distance and bearing
            pred_abs_angle = math.radians(self._normalize_angle(pred_bearing + self.defender['theta']))
            pred_rel_x = pred_dist * math.cos(pred_abs_angle)
            pred_rel_y = pred_dist * math.sin(pred_abs_angle)
            # Create attacker state from prediction
            defender_center = np.array([self.defender['x'] + self.pixel_size * 0.5,
                                       self.defender['y'] + self.pixel_size * 0.5])
            pred_center = defender_center + np.array([pred_rel_x, pred_rel_y])
            obs_attacker_state = {
                'x': pred_center[0] - self.pixel_size * 0.5,
                'y': pred_center[1] - self.pixel_size * 0.5,
                'theta': 0.0
            }
        elif self.last_observed_attacker_pos is not None:
            obs_attacker_state = {
                'x': self.last_observed_attacker_pos[0] - self.pixel_size * 0.5,
                'y': self.last_observed_attacker_pos[1] - self.pixel_size * 0.5,
                'theta': 0.0
            }
        else:
            obs_attacker_state = None

        if obs_attacker_state is not None:
            rel_vec, distance = self._get_relative_position(self.defender, obs_attacker_state)
            map_diagonal = math.hypot(self.width, self.height)
            normalized_distance = np.clip((distance / map_diagonal) * 2.0 - 1.0, -1.0, 1.0)

            abs_ang = math.atan2(rel_vec[1], rel_vec[0])
            rel_ang = self._normalize_angle(math.degrees(abs_ang) - self.defender['theta'])
            normalized_bearing = np.clip(rel_ang / 180.0, -1.0, 1.0)

            fov_edge_angle = min(abs(rel_ang + fov_half), abs(rel_ang - fov_half))
            normalized_fov_edge = np.clip((fov_edge_angle / fov_half) * 2.0 - 1.0, -1.0, 1.0) if fov_half > 0 else 0.0
        else:
            normalized_distance = 1.0
            normalized_bearing = 0.0
            normalized_fov_edge = 1.0

        obs[0] = normalized_distance
        obs[1] = normalized_bearing
        obs[2] = normalized_fov_edge

        # 合并 in_fov 和 occluded 为单一 is_visible (在FOV内且未被遮挡)
        obs[3] = 1.0 if is_visible else 0.0

        max_unobserved = float(EnvParameters.MAX_UNOBSERVED_STEPS)
        normalized_unobserved = np.clip((self.steps_since_observed / max_unobserved) * 2.0 - 1.0, -1.0, 1.0)
        obs[4] = normalized_unobserved

        obs[5:5+64] = self._sense_agent_radar(self.defender, num_rays=self.radar_rays, full_circle=True)

        target_rel_vec, target_dist = self._get_relative_position(self.defender, self.target)
        target_map_diagonal = math.hypot(self.width, self.height)
        target_normalized_dist = np.clip((target_dist / target_map_diagonal) * 2.0 - 1.0, -1.0, 1.0)

        target_abs_ang = math.atan2(target_rel_vec[1], target_rel_vec[0])
        target_rel_ang = self._normalize_angle(math.degrees(target_abs_ang) - self.defender['theta'])
        target_normalized_bearing = np.clip(target_rel_ang / 180.0, -1.0, 1.0)

        # Store distance and bearing only (not rel_x, rel_y)
        obs[69] = target_normalized_dist
        obs[70] = target_normalized_bearing

        return obs

    def _get_velocity(self, agent, prev_pos):
        if prev_pos is not None:
            dx = agent['x'] - prev_pos['x']
            dy = agent['y'] - prev_pos['y']
            return np.array([dx, dy], dtype=np.float32)
        return np.zeros(2, dtype=np.float32)

    def _get_angular_velocity(self, agent, prev_pos, max_ang_speed):
        if prev_pos is not None:
            prev_heading = prev_pos.get('theta', 0.0)
            angle_change = self._normalize_angle(agent['theta'] - prev_heading)
            return np.clip(angle_change / (max_ang_speed + 1e-6), -1.0, 1.0)
        return 0.0

    def _get_relative_position(self, from_agent, to_agent):
        from_center = np.array([from_agent['x'] + self.pixel_size * 0.5, from_agent['y'] + self.pixel_size * 0.5], dtype=np.float32)
        to_center = np.array([to_agent['x'] + self.pixel_size * 0.5, to_agent['y'] + self.pixel_size * 0.5], dtype=np.float32)
        relative_vec = to_center - from_center
        distance = float(np.linalg.norm(relative_vec))
        return relative_vec, distance

    def _update_visibility(self, relative_angle_deg, distance, fov_half_deg):
        in_angle = abs(relative_angle_deg) <= fov_half_deg
        in_range = True
        in_fov = 1.0 if (in_angle and in_range) else 0.0
        occluded = 0.0

        if in_fov > 0.5:
            if self._is_line_blocked(self.defender, self.attacker):
                occluded = 1.0
                self.steps_since_observed += 1
            else:
                attacker_center = np.array([self.attacker['x'] + self.pixel_size * 0.5, self.attacker['y'] + self.pixel_size * 0.5], dtype=np.float32)
                self.last_observed_attacker_pos = attacker_center.copy()
                self.steps_since_observed = 0
        else:
            self.steps_since_observed += 1

        return in_fov, occluded

    def _get_attacker_observation(self):
        obs = np.zeros(71, dtype=np.float32)

        obs[0] = (self.attacker['x'] / self.width) * 2.0 - 1.0
        obs[1] = (self.attacker['y'] / self.height) * 2.0 - 1.0
        obs[2] = (self.attacker['theta'] / 180.0) - 1.0
        obs[3] = (self.defender['x'] / self.width) * 2.0 - 1.0
        obs[4] = (self.defender['y'] / self.height) * 2.0 - 1.0

        obs[5:5+64] = self._sense_agent_radar(self.attacker, num_rays=self.radar_rays, full_circle=True)

        # Target global normalized coordinates (no bearing needed)
        obs[69] = (self.target['x'] / self.width) * 2.0 - 1.0
        obs[70] = (self.target['y'] / self.height) * 2.0 - 1.0

        return obs

    def _normalize_angle(self, angle_deg):
        angle_deg = angle_deg % 360.0
        if angle_deg > 180.0:
            angle_deg -= 360.0
        return float(angle_deg)

    def _sense_agent_radar(self, agent, num_rays=10, full_circle=False):
        center = np.array([agent['x'] + self.pixel_size * 0.5, agent['y'] + self.pixel_size * 0.5], dtype=float)
        heading = math.radians(agent.get('theta', 0.0))
        if full_circle:
            angles = [heading + 2 * math.pi * i / num_rays for i in range(num_rays)]
        else:
            angle_range = math.pi
            angles = [heading + (i / (num_rays - 1) - 0.5) * angle_range for i in range(num_rays)]
        max_radar_range = float(min(EnvParameters.FOV_RANGE, math.hypot(self.width, self.height)))
        # 不使用padding，雷达探测点应该在障碍物表面
        pad = 0.0
        dists = env_lib.ray_distances_multi(center, angles, max_radar_range, padding=pad)
        readings = (np.asarray(dists, dtype=np.float32) / max_radar_range) * 2.0 - 1.0
        return readings

    def _is_line_blocked(self, agent1, agent2, padding=0.0):
        x1 = agent1['x'] + self.pixel_size * 0.5
        y1 = agent1['y'] + self.pixel_size * 0.5
        x2 = agent2['x'] + self.pixel_size * 0.5
        y2 = agent2['y'] + self.pixel_size * 0.5
        dx, dy = (x2 - x1), (y2 - y1)
        line_len = math.hypot(dx, dy)
        if line_len <= 1e-6:
            return False

        angle = math.atan2(dy, dx)
        check_len = line_len
        dist = env_lib.ray_distance_grid((x1, y1), angle, check_len, padding=padding)
        return bool(dist < check_len - 1e-3)

    def _parse_actions(self, action, target_action=None):
        # 如果 action 是元组且包含两个元素，则解包为 (defender_action, attacker_action)
        if isinstance(action, (tuple, list)) and len(action) == 2 and target_action is None:
            # 检查第一个元素是否是动作（2个元素的数组）
            first_elem = np.asarray(action[0], dtype=np.float32).reshape(-1)
            if first_elem.size == 2:
                return action[0], action[1]
        return action, target_action

    def _control_to_physical(self, action, role):
        if action is None:
            return None
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size != 2:
            raise ValueError("action must contain exactly two elements")
        if np.all(np.abs(arr) <= 1.0 + 1e-6):
            if role == 'defender':
                max_turn = float(getattr(map_config, 'defender_max_angular_speed', 6.0))
                max_speed = float(getattr(map_config, 'defender_speed', 2.6))
            elif role == 'attacker':
                max_turn = float(getattr(map_config, 'attacker_max_angular_speed', 12.0))
                max_speed = float(getattr(map_config, 'attacker_speed', 2.0))
            else:
                max_turn = 6.0
                max_speed = 2.0

            angle_delta = float(np.clip(arr[0], -1.0, 1.0) * max_turn)
            speed_factor = float(np.clip(arr[1], -1.0, 1.0))
            speed = (speed_factor + 1.0) * 0.5 * max_speed

            return angle_delta, speed
        return float(arr[0]), float(arr[1])

    def step(self, action=None, attacker_action=None):
        self.step_count += 1
        defender_action, attacker_action = self._parse_actions(action, attacker_action)

        if defender_action is not None:
            defender_phys = self._control_to_physical(defender_action, 'defender')
            if defender_phys is not None:
                angle_delta, speed = defender_phys
                self.defender = env_lib.agent_move_velocity(self.defender, angle_delta, speed, self.defender_speed, role='defender')

        if attacker_action is not None:
            attacker_phys = self._control_to_physical(attacker_action, 'attacker')
            if attacker_phys is not None:
                angle_delta, speed = attacker_phys
                self.attacker = env_lib.agent_move_velocity(self.attacker, angle_delta, speed, self.attacker_speed, role='attacker')

        self._fov_cache_valid = False
        self.defender_trajectory.append((self.defender['x'] + self.pixel_size / 2.0, self.defender['y'] + self.pixel_size / 2.0))
        self.attacker_trajectory.append((self.attacker['x'] + self.pixel_size / 2.0, self.attacker['y'] + self.pixel_size / 2.0))
        max_len = getattr(map_config, 'trail_max_len', 600)
        if len(self.defender_trajectory) > max_len:
            self.defender_trajectory = self.defender_trajectory[-max_len:]
        if len(self.attacker_trajectory) > max_len:
            self.attacker_trajectory = self.attacker_trajectory[-max_len:]

        if self.last_defender_pos is not None:
            self.prev_defender_pos = self.last_defender_pos.copy()
        self.last_defender_pos = self.defender.copy()
        if self.last_attacker_pos is not None:
            self.prev_attacker_pos = self.last_attacker_pos.copy()
        self.last_attacker_pos = self.attacker.copy()

        agent_radius = float(getattr(map_config, 'agent_radius', self.pixel_size * 0.5))
        defender_center_x = self.defender['x'] + self.pixel_size * 0.5
        defender_center_y = self.defender['y'] + self.pixel_size * 0.5
        defender_blocked = env_lib.is_point_blocked(defender_center_x, defender_center_y, padding=agent_radius)
        attacker_center_x = self.attacker['x'] + self.pixel_size * 0.5
        attacker_center_y = self.attacker['y'] + self.pixel_size * 0.5
        attacker_blocked = env_lib.is_point_blocked(attacker_center_x, attacker_center_y, padding=agent_radius)

        defender_captures_attacker = self._is_defender_capturing_attacker()
        if defender_captures_attacker:
            self._capture_counter_defender = min(self._capture_counter_defender + 1, self.capture_required_steps)
        else:
            self._capture_counter_defender = 0

        attacker_captures_target = self._is_attacker_capturing_target()
        if attacker_captures_target:
            self._capture_counter_attacker = min(self._capture_counter_attacker + 1, self.capture_required_steps)
        else:
            self._capture_counter_attacker = 0

        defender_radar = self._sense_agent_radar(self.defender, num_rays=self.radar_rays, full_circle=True)

        # Calculate reward based on reward_mode
        if self.reward_mode == 'protect':
            reward, terminated, truncated, info = env_lib.reward_calculate_protect(
                self.defender, self.attacker, self.target,
                prev_defender=self.prev_defender_pos,
                prev_attacker=self.prev_attacker_pos,
                defender_collision=bool(defender_blocked),
                attacker_collision=bool(attacker_blocked),
                defender_captured=bool(self._capture_counter_defender >= self.capture_required_steps),
                attacker_captured=bool(self._capture_counter_attacker >= self.capture_required_steps),
                capture_progress_defender=int(self._capture_counter_defender),
                capture_progress_attacker=int(self._capture_counter_attacker),
                capture_required_steps=int(self.capture_required_steps),
                radar=defender_radar
            )
        elif self.reward_mode == 'chase':
            reward, terminated, truncated, info = env_lib.reward_calculate_chase(
                self.defender, self.attacker, self.target,
                prev_defender=self.prev_defender_pos,
                prev_attacker=self.prev_attacker_pos,
                defender_collision=bool(defender_blocked),
                attacker_collision=bool(attacker_blocked),
                defender_captured=bool(self._capture_counter_defender >= self.capture_required_steps),
                attacker_captured=bool(self._capture_counter_attacker >= self.capture_required_steps),
                capture_progress_defender=int(self._capture_counter_defender),
                capture_progress_attacker=int(self._capture_counter_attacker),
                capture_required_steps=int(self.capture_required_steps),
                radar=defender_radar
            )
        else:  # 'standard'
            reward, terminated, truncated, info = env_lib.reward_calculate_tad(
                self.defender, self.attacker, self.target,
                prev_defender=self.prev_defender_pos,
                prev_attacker=self.prev_attacker_pos,
                defender_collision=bool(defender_blocked),
                attacker_collision=bool(attacker_blocked),
                defender_captured=bool(self._capture_counter_defender >= self.capture_required_steps),
                attacker_captured=bool(self._capture_counter_attacker >= self.capture_required_steps),
                capture_progress_defender=int(self._capture_counter_defender),
                capture_progress_attacker=int(self._capture_counter_attacker),
                capture_required_steps=int(self.capture_required_steps),
                radar=defender_radar
            )

        cur_dist_defender = float(math.hypot(self.defender['x'] - self.attacker['x'], self.defender['y'] - self.attacker['y']))
        if self._best_distance_attacker is None or cur_dist_defender < (self._best_distance_attacker - 1e-6):
            self._best_distance_attacker = cur_dist_defender

        cur_dist_target = float(math.hypot(self.attacker['x'] - self.target['x'], self.attacker['y'] - self.target['y']))
        if self._best_distance_target is None or cur_dist_target < (self._best_distance_target - 1e-6):
            self._best_distance_target = cur_dist_target

        info['closest_attacker_record_value'] = float(self._best_distance_attacker if self._best_distance_attacker is not None else cur_dist_defender)
        info['closest_target_record_value'] = float(self._best_distance_target if self._best_distance_target is not None else cur_dist_target)

        self.current_obs = self._get_obs_features()
        if self.step_count >= EnvParameters.EPISODE_LEN and not terminated:
            truncated = True

        return self.current_obs, float(reward), bool(terminated), bool(truncated), info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 先构建一次障碍物（用于初始位置采样）
        density_level = getattr(map_config, 'current_obstacle_density', None)
        if hasattr(map_config, "regenerate_obstacles"):
            map_config.regenerate_obstacles(density_level=density_level, target_pos=None)
        env_lib.build_occupancy(
            width=self.width,
            height=self.height,
            cell=getattr(map_config, 'occ_cell', getattr(map_config, 'pixel_size', self.pixel_size)),
            obstacles=getattr(map_config, 'obstacles', [])
        )

        # 采样 Defender、Attacker、Target 位置
        # Target 采样时会自动检查时间优势约束
        for _ in range(100):
            self.defender = self._sample_defender_state()
            self.attacker = self._sample_attacker_state()
            self.target = self._sample_target_state()
            # 不再需要额外的位置检查，_sample_target_state 已经确保了时间优势
            break

        # 根据采样的target位置重新生成障碍物（过滤与target重叠的障碍物）
        target_pos = {'x': self.target['x'], 'y': self.target['y'], 'r': getattr(map_config, 'target_radius', 16)}
        if hasattr(map_config, "regenerate_obstacles"):
            map_config.regenerate_obstacles(density_level=density_level, target_pos=target_pos)
        env_lib.build_occupancy(
            width=self.width,
            height=self.height,
            cell=getattr(map_config, 'occ_cell', getattr(map_config, 'pixel_size', self.pixel_size)),
            obstacles=getattr(map_config, 'obstacles', [])
        )

        self.step_count = 0
        self.defender_trajectory = []
        self.attacker_trajectory = []

        self.prev_defender_pos = self.defender.copy()
        self.last_defender_pos = self.defender.copy()
        self.prev_attacker_pos = self.attacker.copy()
        self.last_attacker_pos = self.attacker.copy()

        self.last_observed_attacker_pos = None
        self.steps_since_observed = 0
        self._capture_counter_defender = 0
        self._capture_counter_attacker = 0
        self._best_distance_attacker = float(math.hypot(self.defender['x'] - self.attacker['x'], self.defender['y'] - self.attacker['y']))
        self._best_distance_target = float(math.hypot(self.attacker['x'] - self.target['x'], self.attacker['y'] - self.target['y']))

        self._fov_cache = None
        self._fov_cache_valid = False
        self.current_obs = self._get_obs_features()
        return self.current_obs, {}

    def _check_initial_positions(self):
        """简化后的位置检查 - 只检查基本的最小距离"""
        dx = (self.defender['x'] + self.pixel_size * 0.5) - (self.attacker['x'] + self.pixel_size * 0.5)
        dy = (self.defender['y'] + self.pixel_size * 0.5) - (self.attacker['y'] + self.pixel_size * 0.5)
        dist = math.hypot(dx, dy)
        min_gap = float(getattr(map_config, 'agent_spawn_min_gap', 100.0))
        return dist >= min_gap

    def _sample_defender_state(self):
        """在地图任意位置随机生成Defender（只需避开障碍物）"""
        margin = 30
        pad = float(getattr(map_config, 'agent_radius', self.pixel_size * 0.5))

        for attempt in range(512):
            x = float(self.np_random.uniform(margin, self.width - margin - self.pixel_size))
            y = float(self.np_random.uniform(margin, self.height - margin - self.pixel_size))

            center_x = x + self.pixel_size * 0.5
            center_y = y + self.pixel_size * 0.5

            if env_lib.is_point_blocked(center_x, center_y, padding=pad):
                continue

            # 随机朝向
            theta = float(self.np_random.uniform(0.0, 360.0))
            return {'x': x, 'y': y, 'theta': theta}

        print("[WARNING] Failed to spawn defender, using fallback position")
        return {'x': float(margin), 'y': float(margin), 'theta': 0.0}

    def _sample_attacker_state(self):
        """在地图任意位置随机生成Attacker（只需避开障碍物、与Defender保持最小距离）"""
        margin = 30.0
        min_dist_to_defender = float(getattr(map_config, 'agent_spawn_min_gap', 100.0))
        pad = float(getattr(map_config, 'agent_radius', self.pixel_size * 0.5))
        
        defender_cx = self.defender['x'] + self.pixel_size * 0.5
        defender_cy = self.defender['y'] + self.pixel_size * 0.5

        for attempt in range(512):
            x = float(self.np_random.uniform(margin, self.width - margin - self.pixel_size))
            y = float(self.np_random.uniform(margin, self.height - margin - self.pixel_size))

            cx = x + self.pixel_size * 0.5
            cy = y + self.pixel_size * 0.5

            # 避开障碍物
            if env_lib.is_point_blocked(cx, cy, padding=pad):
                continue

            # 与Defender保持最小距离
            dist_to_defender = math.hypot(cx - defender_cx, cy - defender_cy)
            if dist_to_defender < min_dist_to_defender:
                continue

            # 随机朝向
            theta = float(self.np_random.uniform(0.0, 360.0))
            return {'x': x, 'y': y, 'theta': theta}

        print("[WARNING] Failed to spawn attacker, using fallback position")
        return {'x': float(self.width - margin - self.pixel_size), 
                'y': float(self.height - margin - self.pixel_size), 
                'theta': 0.0}

    def _sample_target_state(self):
        """
        在地图任意位置随机生成Target
        
        核心约束: 保证 Defender 有时间拦截 Attacker
        条件: dist(T,A) * defender_speed > dist(T,D) * attacker_speed
        即: Attacker 到达 Target 的时间 > Defender 拦截的时间
        """
        margin = 50
        target_radius = getattr(map_config, 'target_radius', 16)
        min_dist = 80.0  # 与任何Agent的最小距离
        
        defender_cx = self.defender['x'] + self.pixel_size * 0.5
        defender_cy = self.defender['y'] + self.pixel_size * 0.5
        attacker_cx = self.attacker['x'] + self.pixel_size * 0.5
        attacker_cy = self.attacker['y'] + self.pixel_size * 0.5
        
        defender_speed = float(getattr(map_config, 'defender_speed', 2.6))
        attacker_speed = float(getattr(map_config, 'attacker_speed', 2.0))

        for attempt in range(500):
            x = float(self.np_random.uniform(margin, self.width - margin - self.pixel_size))
            y = float(self.np_random.uniform(margin, self.height - margin - self.pixel_size))

            cx = x + self.pixel_size * 0.5
            cy = y + self.pixel_size * 0.5

            # 避开障碍物
            if env_lib.is_point_blocked(cx, cy, padding=target_radius):
                continue

            # 计算距离
            dist_to_defender = math.hypot(cx - defender_cx, cy - defender_cy)
            dist_to_attacker = math.hypot(cx - attacker_cx, cy - attacker_cy)
            
            # 最小距离约束
            if dist_to_defender < min_dist or dist_to_attacker < min_dist:
                continue
            
            # 时间优势约束: dist(T,A) * defender_speed > dist(T,D) * attacker_speed
            # Defender 有足够时间拦截 Attacker
            if dist_to_attacker * defender_speed > dist_to_defender * attacker_speed:
                return {'x': x, 'y': y, 'theta': 0.0}

        # Fallback: 生成在Defender附近但远离Attacker的位置
        print("[WARNING] Failed to find target position with time advantage, using fallback")
        x = float(self.width / 2)
        y = float(self.height / 2)
        return {'x': x, 'y': y, 'theta': 0.0}

    def _get_fov_points(self, force_recompute=False):
        if self._fov_cache_valid and self._fov_cache is not None and not force_recompute:
            return self._fov_cache

        ss = getattr(map_config, 'ssaa', 1)
        cx_world = self.defender['x'] + map_config.pixel_size * 0.5
        cy_world = self.defender['y'] + map_config.pixel_size * 0.5
        cx = cx_world * ss
        cy = cy_world * ss

        heading_rad = math.radians(self.defender.get('theta', 0.0))
        fov_half = math.radians(EnvParameters.FOV_ANGLE / 2.0)
        max_range = min(EnvParameters.FOV_RANGE if EnvParameters.FOV_RANGE != float('inf') else 500.0, 500.0)

        num_rays = 64
        angles = np.linspace(heading_rad - fov_half, heading_rad + fov_half, num_rays)
        dists = env_lib.ray_distances_multi((cx_world, cy_world), angles, max_range, padding=0.0)

        pts = [(cx, cy)]
        for i in range(num_rays):
            dist = dists[i]
            angle = angles[i]
            px = cx + dist * ss * math.cos(angle)
            py = cy + dist * ss * math.sin(angle)
            pts.append((px, py))

        self._fov_cache = pts
        self._fov_cache_valid = True
        return pts

    def render(self, mode='rgb_array', collision_info=None):
        if pygame is not None and self._render_surface is None:
            ss = getattr(map_config, 'ssaa', 1)
            self._render_surface = pygame.Surface((self.width * ss, self.height * ss), flags=pygame.SRCALPHA)
        fov_points = self._get_fov_points()
        canvas = env_lib.get_canvas_tad(
            self.target, self.defender, self.attacker,
            self.defender_trajectory, self.attacker_trajectory,
            surface=self._render_surface,
            fov_points=fov_points,
            collision_info=collision_info
        )
        return canvas

    def close(self):
        self._render_surface = None

    def _is_defender_capturing_attacker(self):
        tx = self.defender['x'] + self.pixel_size * 0.5
        ty = self.defender['y'] + self.pixel_size * 0.5
        gx = self.attacker['x'] + self.pixel_size * 0.5
        gy = self.attacker['y'] + self.pixel_size * 0.5
        dx, dy = (gx - tx), (gy - ty)
        dist = math.hypot(dx, dy)
        if dist > self.capture_radius:
            return False

        defender_heading = float(self.defender.get('theta', 0.0))
        angle_to_attacker = math.degrees(math.atan2(dy, dx))
        rel = self._normalize_angle(angle_to_attacker - defender_heading)
        half_sector = self.capture_sector_angle_deg * 0.5
        if abs(rel) > half_sector:
            return False

        fov_half = self.fov_angle * 0.5
        in_fov = (abs(rel) <= fov_half)
        if not in_fov:
            return False

        if self._is_line_blocked(self.defender, self.attacker):
            return False
        return True

    def _is_attacker_capturing_target(self):
        """
        检查Attacker是否捕获Target

        捕获条件：Attacker和Target边缘相碰
        即：两者中心距离 <= attacker_radius + target_radius
        """
        ax = self.attacker['x'] + self.pixel_size * 0.5
        ay = self.attacker['y'] + self.pixel_size * 0.5
        tx = self.target['x'] + self.pixel_size * 0.5
        ty = self.target['y'] + self.pixel_size * 0.5
        dx, dy = (tx - ax), (ty - ay)
        dist = math.hypot(dx, dy)

        # 获取半径
        attacker_radius = float(getattr(map_config, 'agent_radius', 8))
        target_radius = float(getattr(map_config, 'target_radius', 16))

        # 捕获条件：边缘相碰
        # 中心距离 <= attacker_radius + target_radius
        return dist <= (attacker_radius + target_radius)

    def get_privileged_state(self):
        return {
            'defender': {
                'x': float(self.defender['x']),
                'y': float(self.defender['y']),
                'theta': float(self.defender['theta']),
                'center_x': float(self.defender['x'] + self.pixel_size * 0.5),
                'center_y': float(self.defender['y'] + self.pixel_size * 0.5)
            },
            'attacker': {
                'x': float(self.attacker['x']),
                'y': float(self.attacker['y']),
                'theta': float(self.attacker['theta']),
                'center_x': float(self.attacker['x'] + self.pixel_size * 0.5),
                'center_y': float(self.attacker['y'] + self.pixel_size * 0.5)
            },
            'target': {
                'x': float(self.target['x']),
                'y': float(self.target['y']),
                'theta': float(self.target['theta']),
                'center_x': float(self.target['x'] + self.pixel_size * 0.5),
                'center_y': float(self.target['y'] + self.pixel_size * 0.5)
            },
            'map': {
                'width': float(self.width),
                'height': float(self.height)
            }
        }


TrackingEnv = TADEnv
