"""
Attacker Global Pathfinding Policy

Strategy: 使用全局障碍物信息进行路径规划，直接导航到Target
支持多种策略模式:
- 'default': 默认模式，A*寻路 + 适度避开Defender
- 'aggressive': 激进模式，直冲目标，无视Defender
- 'evasive': 规避模式，最大化与Defender距离并避开其视野
- 'flank': 侧翼包抄，绕行到Target侧翼进攻
- 'orbit': 轨道等待，绕Target运动寻找最佳进攻时机

Observation (72维):
- obs[0]: Attacker 全局 X 坐标（归一化）
- obs[1]: Attacker 全局 Y 坐标（归一化）
- obs[2]: Attacker 朝向（归一化）
- obs[3]: Defender 全局 X 坐标（归一化）
- obs[4]: Defender 全局 Y 坐标（归一化）
- obs[5]: Defender 朝向（归一化）
- obs[6]: Target 全局 X 坐标（归一化）
- obs[7]: Target 全局 Y 坐标（归一化）
- obs[8:72]: 雷达数据（64维）

Action: [angle_delta, speed_normalized]
"""

import numpy as np
import math
import heapq
from typing import Tuple, List, Optional
import map_config
import env_lib

KP_TURN = 0.8
GRID_SIZE = 8.0
OBSTACLE_PADDING = 12.0
DEFENDER_AVOID_RADIUS = 40.0
DEFENDER_VIEW_DISTANCE = 250.0  # Defender视野距离
DEFENDER_VIEW_ANGLE = 360.0     # Defender视野角度（360°全向）

STRATEGY_CONFIGS = {
    'default':   {'response': 0.6, 'speed_mult': 1.0, 'avoid_view': False},
    'aggressive':{'response': 0.0, 'speed_mult': 1.0, 'avoid_view': False},
    'evasive':   {'response': 1.5, 'speed_mult': 0.75, 'avoid_view': True},
    'flank':     {'response': 0.7, 'speed_mult': 0.95, 'avoid_view': False},
    'orbit':     {'response': 0.8, 'speed_mult': 0.85, 'avoid_view': False},
}

ALL_STRATEGIES = list(STRATEGY_CONFIGS.keys())


# ---- Precomputed padded obstacle grid (class-level, built once) ----
_STATIC_GRID = None          # np.ndarray bool, shape (ny, nx)
_STATIC_GRID_CELL = None     # float
_STATIC_GRID_NX = 0
_STATIC_GRID_NY = 0

def _ensure_static_grid(grid_size: float, padding: float, env_width: float, env_height: float):
    """Build a padded boolean grid once; subsequent calls are no-ops."""
    global _STATIC_GRID, _STATIC_GRID_CELL, _STATIC_GRID_NX, _STATIC_GRID_NY
    if _STATIC_GRID is not None and _STATIC_GRID_CELL == grid_size:
        return
    # Ensure obstacle arrays are compiled (build_occupancy must have been called)
    if not getattr(env_lib, '_OBS_COMPILED', False):
        env_lib.build_occupancy()
    nx = int(math.ceil(env_width / grid_size))
    ny = int(math.ceil(env_height / grid_size))
    grid = np.zeros((ny, nx), dtype=np.bool_)
    for iy in range(ny):
        cy = iy * grid_size + grid_size * 0.5
        for ix in range(nx):
            cx = ix * grid_size + grid_size * 0.5
            if env_lib.is_point_blocked(cx, cy, padding=padding):
                grid[iy, ix] = True
    _STATIC_GRID = grid
    _STATIC_GRID_CELL = grid_size
    _STATIC_GRID_NX = nx
    _STATIC_GRID_NY = ny


class AttackerGlobalPolicy:
    """
    Attacker全局路径规划策略
    """

    def __init__(
        self,
        env_width: float = 640,
        env_height: float = 640,
        attacker_speed: float = 2.0,
        attacker_max_turn: float = 12.0,
        kp_turn: float = KP_TURN,
        grid_size: float = GRID_SIZE,
        obstacle_padding: float = OBSTACLE_PADDING,
        defender_avoid_radius: float = DEFENDER_AVOID_RADIUS,
        defender_view_distance: float = DEFENDER_VIEW_DISTANCE,
        defender_view_angle: float = DEFENDER_VIEW_ANGLE,
        strategy: str = 'default',
        strategy_params: Optional[dict] = None,
    ):
        """
        初始化Attacker全局策略

        Args:
            env_width: 环境宽度
            env_height: 环境高度
            attacker_speed: Attacker最大速度
            attacker_max_turn: Attacker最大转向角速度（度/步）
            kp_turn: 转向比例系数
            grid_size: 路径规划网格大小
            obstacle_padding: 障碍物膨胀距离
            defender_avoid_radius: 规避defender的安全半径
            defender_view_distance: defender视野距离
            defender_view_angle: defender视野角度
            strategy: 策略名称 ('default', 'aggressive', 'evasive', 'flank', 'orbit')
            strategy_params: 策略参数字典
        """
        self.env_width = env_width
        self.env_height = env_height
        self.max_speed = attacker_speed
        self.max_turn = attacker_max_turn

        # 策略参数
        self.kp_turn = kp_turn
        self.grid_size = grid_size
        self.obstacle_padding = obstacle_padding
        self.base_defender_avoid_radius = defender_avoid_radius
        self.defender_avoid_radius = defender_avoid_radius
        self.defender_view_distance = defender_view_distance
        self.defender_view_angle = defender_view_angle
        self.strategy = strategy
        self.strategy_params = strategy_params or {}

        # 路径规划相关
        self.path = []
        self.current_path_index = 0
        self.last_planned_pos = None
        self.replan_interval = 20
        self.step_count = 0
        
        # Orbit state
        self.orbit_angle = 0.0
        self.orbit_attacking = False
        
        # Flank state
        self.flank_waypoint = None
        self.flank_reached = False
        
        # Speed multiplier from config
        cfg = STRATEGY_CONFIGS.get(strategy, STRATEGY_CONFIGS['default'])
        self.speed_mult = cfg.get('speed_mult', 1.0)
        self.response_intensity = cfg.get('response', 0.6)
        self.avoid_view = cfg.get('avoid_view', False)
        
        self._apply_strategy_config()

        # Build static obstacle grid once (shared across all instances)
        _ensure_static_grid(self.grid_size, self.obstacle_padding,
                            self.env_width, self.env_height)
        self._grid = _STATIC_GRID
        self._grid_nx = _STATIC_GRID_NX
        self._grid_ny = _STATIC_GRID_NY

    def _apply_strategy_config(self):
        """根据策略调整初始参数"""
        self.defender_avoid_radius = self.base_defender_avoid_radius * self.response_intensity
        
        if self.strategy == 'aggressive':
            self.defender_avoid_radius = 0.0
            self.replan_interval = 8
        elif self.strategy == 'evasive':
            self.defender_avoid_radius = self.base_defender_avoid_radius * 2.0
            self.replan_interval = 10
        elif self.strategy == 'flank':
            # 使用defender视野角度计算侧翼偏移量（默认为视野角度的一半）
            flank_angle_deg = self.defender_view_angle / 2
            self.flank_offset = self.strategy_params.get('offset', flank_angle_deg)
            self.replan_interval = 15
        elif self.strategy == 'orbit':
            self.orbit_radius = self.strategy_params.get('radius', 100.0)
            self.orbit_speed = self.strategy_params.get('orbit_speed', 0.03)
            self.orbit_attack_threshold = self.strategy_params.get('attack_threshold', 150.0)
            self.replan_interval = 5

    def reset(self):
        """重置策略状态"""
        self.path = []
        self.current_path_index = 0
        self.last_planned_pos = None
        self.last_defender_pos = None
        self.step_count = 0
        self.orbit_angle = 0.0
        self.orbit_attacking = False
        self.flank_waypoint = None
        self.flank_reached = False
        
        self._apply_strategy_config()

    def denormalize_pos(self, norm_x: float, norm_y: float) -> np.ndarray:
        """反归一化位置"""
        x = ((norm_x + 1.0) / 2.0) * self.env_width
        y = ((norm_y + 1.0) / 2.0) * self.env_height
        return np.array([x, y], dtype=np.float32)

    def denormalize_heading(self, norm_heading: float) -> float:
        """
        反归一化朝向（度）
        """
        return (norm_heading + 1.0) * 180.0

    def normalize_angle(self, angle: float) -> float:
        """
        将角度归一化到 [-180, 180]
        """
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def is_pos_blocked(self, x: float, y: float, defender_pos: np.ndarray = None) -> bool:
        """
        检查位置是否被障碍物阻挡
        """
        # 检查静态障碍物
        if env_lib.is_point_blocked(x, y, padding=self.obstacle_padding):
            return True

        # 检查是否太靠近defender
        if defender_pos is not None and self.defender_avoid_radius > 0:
            dist_to_defender = math.hypot(x - defender_pos[0], y - defender_pos[1])
            if dist_to_defender < self.defender_avoid_radius:
                return True

        return False

    def is_in_defender_view(self, pos: np.ndarray, defender_pos: np.ndarray, defender_heading: float = None) -> bool:
        """
        检查某位置是否在defender的视野范围内
        
        Args:
            pos: 待检查的位置
            defender_pos: defender位置
            defender_heading: defender朝向（度），360°视野时不需要
            
        Returns:
            bool: 是否在视野内
        """
        if defender_pos is None:
            return False
            
        dist_to_defender = np.linalg.norm(pos - defender_pos)
        
        # 检查距离
        if dist_to_defender > self.defender_view_distance:
            return False
            
        # 360°全向视野，只需检查距离
        if self.defender_view_angle >= 360:
            return True
            
        # 非全向视野时检查角度
        if defender_heading is not None:
            angle_to_pos = math.degrees(math.atan2(
                pos[1] - defender_pos[1],
                pos[0] - defender_pos[0]
            ))
            angle_diff = abs(self.normalize_angle(angle_to_pos - defender_heading))
            half_view = self.defender_view_angle / 2
            return angle_diff <= half_view
            
        return True

    def get_evasive_target(self, attacker_pos: np.ndarray, defender_pos: np.ndarray, 
                          target_pos: np.ndarray) -> np.ndarray:
        """
        获取规避视野的导航目标点
        优先选择远离defender视野的路径
        """
        dist_to_target = np.linalg.norm(attacker_pos - target_pos)
        if dist_to_target < 50:
            return target_pos
            
        # 如果当前位置在视野外，直接向目标移动
        if not self.is_in_defender_view(attacker_pos, defender_pos):
            return target_pos
            
        # 计算远离defender的方向
        away_dir = attacker_pos - defender_pos
        dist_to_defender = np.linalg.norm(away_dir)
        
        if dist_to_defender < 1e-6:
            # 如果重合，随机选择一个方向
            away_dir = np.array([1.0, 0.0])
        else:
            away_dir = away_dir / dist_to_defender
            
        # 向远离defender的方向移动，超出视野距离
        safe_distance = self.defender_view_distance + 30
        if dist_to_defender < safe_distance:
            evasive_point = defender_pos + away_dir * safe_distance
            evasive_point[0] = np.clip(evasive_point[0], 30, self.env_width - 30)
            evasive_point[1] = np.clip(evasive_point[1], 30, self.env_height - 30)
            return evasive_point
            
        return target_pos

    def plan_path(self, start_pos: np.ndarray, goal_pos: np.ndarray, defender_pos: np.ndarray = None) -> List[np.ndarray]:
        """
        使用A*算法规划从起点到终点的路径
        优化: 使用预计算的静态障碍物网格进行 O(1) 碰撞检测
        """
        gs = self.grid_size
        grid = self._grid
        gnx = self._grid_nx
        gny = self._grid_ny

        start_grid = (int(start_pos[0] / gs), int(start_pos[1] / gs))
        goal_grid = (int(goal_pos[0] / gs), int(goal_pos[1] / gs))

        # Defender avoidance: precompute grid-coord range
        avoid_r2 = 0.0
        def_gx = 0.0
        def_gy = 0.0
        has_defender = defender_pos is not None and self.defender_avoid_radius > 0
        if has_defender:
            avoid_r2 = self.defender_avoid_radius * self.defender_avoid_radius
            def_gx = defender_pos[0]
            def_gy = defender_pos[1]

        open_set = []
        heapq.heappush(open_set, (0, start_grid))
        came_from = {}
        g_score = {start_grid: 0}

        max_steps = 2000
        steps = 0
        half_gs = gs * 0.5

        while open_set:
            steps += 1
            if steps > max_steps:
                break

            current_cost, current = heapq.heappop(open_set)

            if current == goal_grid:
                path = []
                while current in came_from:
                    pos = np.array([
                        current[0] * gs + half_gs,
                        current[1] * gs + half_gs
                    ], dtype=np.float32)
                    path.append(pos)
                    current = came_from[current]
                start_pos_continuous = np.array([
                    start_grid[0] * gs + half_gs,
                    start_grid[1] * gs + half_gs
                ], dtype=np.float32)
                path.append(start_pos_continuous)
                path.reverse()
                path[-1] = goal_pos.copy()
                return path

            cx0, cy0 = current
            for dx in (-1, 0, 1):
                nx_ = cx0 + dx
                if nx_ < 0 or nx_ >= gnx:
                    continue
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    ny_ = cy0 + dy
                    if ny_ < 0 or ny_ >= gny:
                        continue

                    # O(1) static obstacle check via precomputed grid
                    if grid[ny_, nx_]:
                        continue

                    # Dynamic defender avoidance
                    if has_defender:
                        px = nx_ * gs + half_gs
                        py = ny_ * gs + half_gs
                        ddx = px - def_gx
                        ddy = py - def_gy
                        if ddx * ddx + ddy * ddy < avoid_r2:
                            continue

                    neighbor = (nx_, ny_)
                    move_cost = 1.414 if (dx != 0 and dy != 0) else 1.0
                    tentative_g = g_score[current] + move_cost

                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score = tentative_g + math.hypot(
                            nx_ - goal_grid[0],
                            ny_ - goal_grid[1]
                        )
                        heapq.heappush(open_set, (f_score, neighbor))

        return [start_pos.copy(), goal_pos.copy()]

    def _get_flank_target(self, attacker_pos, defender_pos, target_pos):
        """侧翼包抄策略：绕行到目标侧翼"""
        dist_to_target = np.linalg.norm(attacker_pos - target_pos)
        if dist_to_target < 40 or self.flank_reached:
            self.flank_reached = True
            return target_pos
        if self.flank_waypoint is None:
            def_to_target = target_pos - defender_pos
            def_dist = np.linalg.norm(def_to_target)
            if def_dist < 1e-6:
                perp = np.array([1.0, 0.0])
            else:
                perp = np.array([-def_to_target[1], def_to_target[0]])
                perp = perp / (np.linalg.norm(perp) + 1e-6)
            att_to_target = target_pos - attacker_pos
            side = np.sign(np.dot(perp, att_to_target)) if np.abs(np.dot(perp, att_to_target)) > 1e-6 else 1
            waypoint = target_pos + perp * self.flank_offset * side
            waypoint[0] = np.clip(waypoint[0], 30, self.env_width - 30)
            waypoint[1] = np.clip(waypoint[1], 30, self.env_height - 30)
            self.flank_waypoint = waypoint
        dist_to_waypoint = np.linalg.norm(attacker_pos - self.flank_waypoint)
        if dist_to_waypoint < 30:
            self.flank_reached = True
            return target_pos
        return self.flank_waypoint

    def _get_orbit_target(self, attacker_pos, defender_pos, target_pos):
        """轨道等待策略：绕目标运动，寻找最佳进攻时机"""
        dist_to_target = np.linalg.norm(attacker_pos - target_pos)
        defender_dist_to_target = np.linalg.norm(defender_pos - target_pos)
        
        # 如果已经在攻击状态或接近目标，直接进攻
        if self.orbit_attacking or dist_to_target < 30:
            return target_pos
            
        # 如果defender远离目标，开始进攻
        if defender_dist_to_target > self.orbit_attack_threshold:
            self.orbit_attacking = True
            return target_pos
            
        # 更新轨道角度
        self.orbit_angle += self.orbit_speed
        
        # 计算轨道点
        orbit_x = target_pos[0] + self.orbit_radius * math.cos(self.orbit_angle)
        orbit_y = target_pos[1] + self.orbit_radius * math.sin(self.orbit_angle)
        orbit_point = np.array([orbit_x, orbit_y], dtype=np.float32)
        orbit_point[0] = np.clip(orbit_point[0], 30, self.env_width - 30)
        orbit_point[1] = np.clip(orbit_point[1], 30, self.env_height - 30)
        
        # 检查当前轨道位置是否处于defender对面（好的进攻位置）
        defender_angle = math.atan2(defender_pos[1] - target_pos[1], defender_pos[0] - target_pos[0])
        angle_diff = abs(self._normalize_angle_rad(self.orbit_angle - defender_angle))
        
        # 如果在defender对面（角度差大于120度），开始进攻
        if angle_diff > math.pi * 2 / 3:
            self.orbit_attacking = True
            return target_pos
            
        return orbit_point

    def _normalize_angle_rad(self, angle):
        """将弧度归一化到 [-π, π]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        根据观测计算动作

        Args:
            obs: Attacker观测 (71维)

        Returns:
            action: [angle_delta, speed_normalized]
        """
        # 解析观测
        attacker_pos = self.denormalize_pos(obs[0], obs[1])
        attacker_heading = self.denormalize_heading(obs[2])
        defender_pos = self.denormalize_pos(obs[3], obs[4])
        target_pos = self.denormalize_pos(obs[6], obs[7])

        self.step_count += 1
        
        # 1. 确定导航目标 Goal
        nav_goal = target_pos.copy()
        
        # 根据策略调整导航目标
        if self.strategy == 'flank':
            nav_goal = self._get_flank_target(attacker_pos, defender_pos, target_pos)
        elif self.strategy == 'orbit':
            nav_goal = self._get_orbit_target(attacker_pos, defender_pos, target_pos)
        elif self.strategy == 'evasive':
            nav_goal = self.get_evasive_target(attacker_pos, defender_pos, target_pos)

        # 2. 检查是否需要重新规划路径
        need_replan = False
        if len(self.path) == 0:
            need_replan = True
        elif self.last_planned_pos is None:
            need_replan = True
        elif np.linalg.norm(attacker_pos - self.last_planned_pos) > self.grid_size * 2:
            need_replan = True
        elif self.step_count % self.replan_interval == 0:
            # Only replan on timer if defender moved significantly since last plan
            if self.last_defender_pos is None:
                need_replan = True
            else:
                defender_moved = np.linalg.norm(defender_pos - self.last_defender_pos)
                if defender_moved > self.grid_size:
                    need_replan = True

        if need_replan:
            # 多级避让策略
            dist_defender_to_target = np.linalg.norm(defender_pos - target_pos)
            target_radius = float(getattr(map_config, 'target_radius', 16.0))
            agent_radius = float(getattr(map_config, 'agent_radius', 8.0))
            
            # 基础避让半径
            if self.strategy == 'aggressive':
                base_avoid_radius = 0.0
            else:
                max_dist_for_full_avoid = 60.0
                if dist_defender_to_target >= max_dist_for_full_avoid:
                    base_avoid_radius = self.defender_avoid_radius
                else:
                    ratio = dist_defender_to_target / max_dist_for_full_avoid
                    base_avoid_radius = agent_radius + ratio * (self.defender_avoid_radius - agent_radius)
            
            # 生成尝试列表
            check_radii = [base_avoid_radius]
            if self.strategy != 'aggressive': 
                check_radii.extend([base_avoid_radius * 0.5, agent_radius, 0])
            else:
                # Aggressive模式也尝试更小半径，但主要是不避让
                check_radii.append(0)

            original_radius = self.defender_avoid_radius
            self.path = None
            
            for try_radius in check_radii:
                self.defender_avoid_radius = try_radius
                test_path = self.plan_path(
                    attacker_pos, nav_goal, 
                    defender_pos if try_radius > 0 else None
                )
                if len(test_path) > 2:
                    self.path = test_path
                    break
            
            if self.path is None:
                self.path = test_path
            
            self.defender_avoid_radius = original_radius
            self.current_path_index = 0
            self.last_planned_pos = attacker_pos.copy()
            self.last_defender_pos = defender_pos.copy()

        # 3. 寻找当前路径点
        if len(self.path) > 0:
            while (self.current_path_index < len(self.path) and
                   np.linalg.norm(attacker_pos - self.path[self.current_path_index]) < self.grid_size):
                self.current_path_index += 1

            if self.current_path_index >= len(self.path):
                # 已到达终点
                current_goal = nav_goal
            else:
                current_goal = self.path[self.current_path_index]
        else:
            current_goal = nav_goal

        # 4. 执行控制
        # 计算到当前目标点的期望航向
        goal_angle = math.degrees(math.atan2(
            current_goal[1] - attacker_pos[1],
            current_goal[0] - attacker_pos[0]
        ))

        # 计算转向误差
        heading_error = self.normalize_angle(goal_angle - attacker_heading)

        # P控制转向
        turn_delta = heading_error * self.kp_turn

        # 速度：使用策略配置的倍率
        speed = self.max_speed * self.speed_mult
        
        # Orbit策略在等待阶段降低速度
        if self.strategy == 'orbit' and not self.orbit_attacking:
            speed = self.max_speed * 0.7

        # 限制转向量
        turn_delta = np.clip(turn_delta, -self.max_turn, self.max_turn)
        # 限制速度
        speed = np.clip(speed, 0, self.max_speed)

        # 归一化到动作空间
        angle_delta_norm = turn_delta / self.max_turn
        speed_norm = (speed / self.max_speed) * 2.0 - 1.0

        return np.array([angle_delta_norm, speed_norm], dtype=np.float32)

    def get_action_with_info(self, obs: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        获取动作并返回调试信息

        Args:
            obs: Attacker观测

        Returns:
            action: 动作
            info: 调试信息字典
        """
        
        attacker_pos = self.denormalize_pos(obs[0], obs[1])
        defender_pos = self.denormalize_pos(obs[3], obs[4])
        target_pos = self.denormalize_pos(obs[6], obs[7])
        
        action = self.get_action(obs)

        # 调试信息
        info = {
            'mode': f'global_{self.strategy}',
            'attacker_pos': attacker_pos,
            'defender_pos': defender_pos,
            'target_pos': target_pos,
            'path_length': len(self.path),
            'strategy': self.strategy,
            'orbit_attacking': self.orbit_attacking if self.strategy == 'orbit' else None,
            'in_defender_view': self.is_in_defender_view(attacker_pos, defender_pos)
        }

        return action, info


def create_attacker_global_policy(**kwargs) -> AttackerGlobalPolicy:
    """
    创建Attacker全局策略的便捷函数

    Args:
        **kwargs: 传递给AttackerGlobalPolicy的参数

    Returns:
        AttackerGlobalPolicy实例
    """
    return AttackerGlobalPolicy(**kwargs)
