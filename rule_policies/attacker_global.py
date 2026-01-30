"""
Attacker Global Pathfinding Policy

Strategy: 使用全局障碍物信息进行路径规划，直接导航到Target
支持多种策略模式:
- 'default': 默认模式，A*寻路 + 避开Defender
- 'direct': 直接进攻，忽略Defender
- 'curve': 按正弦曲线轨迹接近Target
- 'wait_and_attack': 先导航到Target附近等待，倒计时结束后发起进攻
- 'conservative': 保守模式，避让半径更大
- 'zigzag': 锯齿形进攻，左右摆动增加不可预测性
- 'flank': 绕行到target侧翼进攻
- 'orbit': 绕target运动等待最佳进攻时机
- 'aggressive': 高速直冲，完全无视defender
- 'stealth': 隐蔽进攻，最大化与defender距离

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

STRATEGY_CONFIGS = {
    'default':        {'response': 0.6, 'speed_mult': 1.0},
    'direct':         {'response': 0.0, 'speed_mult': 1.0},
    'curve':          {'response': 0.5, 'speed_mult': 1.0},
    'wait_and_attack':{'response': 0.4, 'speed_mult': 0.8},
    'conservative':   {'response': 1.0, 'speed_mult': 0.9},
    'zigzag':         {'response': 0.5, 'speed_mult': 1.0},
    'flank':          {'response': 0.7, 'speed_mult': 0.95},
    'orbit':          {'response': 0.8, 'speed_mult': 0.85},
    'aggressive':     {'response': 0.0, 'speed_mult': 1.0},
    'stealth':        {'response': 1.5, 'speed_mult': 0.75},
}

ALL_STRATEGIES = list(STRATEGY_CONFIGS.keys())


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
            strategy: 策略名称 ('default', 'direct', 'curve', 'wait_and_attack', 'conservative')
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
        self.strategy = strategy
        self.strategy_params = strategy_params or {}

        # 路径规划相关
        self.path = []
        self.current_path_index = 0
        self.last_planned_pos = None
        self.replan_interval = 20
        self.step_count = 0
        
        # Curve/Zigzag state
        self.curve_phase = 0.0
        self.zigzag_direction = 1
        
        # Wait strategy state
        self.wait_timer = 0
        self.is_attacking = False
        
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
        
        self._apply_strategy_config()

    def _apply_strategy_config(self):
        """根据策略调整初始参数"""
        self.defender_avoid_radius = self.base_defender_avoid_radius * self.response_intensity
        
        if self.strategy == 'direct' or self.strategy == 'aggressive':
            self.defender_avoid_radius = 0.0
            self.replan_interval = 8
        elif self.strategy == 'conservative':
            self.defender_avoid_radius = self.base_defender_avoid_radius * 1.5
        elif self.strategy == 'curve':
            self.curve_amp = self.strategy_params.get('amp', 60.0)
            self.curve_freq = self.strategy_params.get('freq', 0.05)
        elif self.strategy == 'wait_and_attack':
            self.wait_dist = self.strategy_params.get('wait_dist', 150.0)
            self.wait_time = self.strategy_params.get('wait_time', 200)
        elif self.strategy == 'zigzag':
            self.zigzag_amp = self.strategy_params.get('amp', 50.0)
            self.zigzag_period = self.strategy_params.get('period', 40)
            self.replan_interval = 8
        elif self.strategy == 'flank':
            self.flank_offset = self.strategy_params.get('offset', 120.0)
            self.replan_interval = 15
        elif self.strategy == 'orbit':
            self.orbit_radius = self.strategy_params.get('radius', 100.0)
            self.orbit_speed = self.strategy_params.get('orbit_speed', 0.03)
            self.orbit_attack_window = self.strategy_params.get('attack_window', 60)
            self.replan_interval = 5
        elif self.strategy == 'stealth':
            self.defender_avoid_radius = self.base_defender_avoid_radius * 2.5
            self.replan_interval = 12

    def reset(self):
        """重置策略状态"""
        self.path = []
        self.current_path_index = 0
        self.last_planned_pos = None
        self.step_count = 0
        self.curve_phase = 0.0
        self.zigzag_direction = 1
        self.orbit_angle = 0.0
        self.orbit_attacking = False
        self.flank_waypoint = None
        self.flank_reached = False
        
        if self.strategy == 'wait_and_attack':
            self.wait_timer = self.wait_time
            self.is_attacking = False
        
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

    def plan_path(self, start_pos: np.ndarray, goal_pos: np.ndarray, defender_pos: np.ndarray = None) -> List[np.ndarray]:
        """
        使用A*算法规划从起点到终点的路径
        """
        # 将连续坐标转换为网格坐标
        start_grid = (int(start_pos[0] / self.grid_size), int(start_pos[1] / self.grid_size))
        goal_grid = (int(goal_pos[0] / self.grid_size), int(goal_pos[1] / self.grid_size))

        # A*算法
        open_set = []
        heapq.heappush(open_set, (0, start_grid))
        came_from = {}
        g_score = {start_grid: 0}

        max_steps = 2000
        steps = 0

        while open_set:
            steps += 1
            if steps > max_steps:
                break

            current_cost, current = heapq.heappop(open_set)

            if current == goal_grid:
                # 重建路径
                path = []
                while current in came_from:
                    # 将网格坐标转换回连续坐标
                    pos = np.array([
                        current[0] * self.grid_size + self.grid_size / 2,
                        current[1] * self.grid_size + self.grid_size / 2
                    ], dtype=np.float32)
                    path.append(pos)
                    current = came_from[current]
                # 添加起点
                start_pos_continuous = np.array([
                    start_grid[0] * self.grid_size + self.grid_size / 2,
                    start_grid[1] * self.grid_size + self.grid_size / 2
                ], dtype=np.float32)
                path.append(start_pos_continuous)
                path.reverse()
                # 确保终点是实际的目标位置
                path[-1] = goal_pos.copy()
                return path

            # 8方向移动
            neighbors = [
                (current[0] + 1, current[1]),
                (current[0] - 1, current[1]),
                (current[0], current[1] + 1),
                (current[0], current[1] - 1),
                (current[0] + 1, current[1] + 1),
                (current[0] + 1, current[1] - 1),
                (current[0] - 1, current[1] + 1),
                (current[0] - 1, current[1] - 1),
            ]

            for neighbor in neighbors:
                # 检查边界
                if (neighbor[0] < 0 or neighbor[0] >= self.env_width / self.grid_size or
                    neighbor[1] < 0 or neighbor[1] >= self.env_height / self.grid_size):
                    continue

                # 检查障碍物
                neighbor_pos = np.array([
                    neighbor[0] * self.grid_size + self.grid_size / 2,
                    neighbor[1] * self.grid_size + self.grid_size / 2
                ], dtype=np.float32)

                if self.is_pos_blocked(neighbor_pos[0], neighbor_pos[1], defender_pos):
                    continue

                # 计算新的g值
                move_cost = 1.414 if (neighbor[0] != current[0] and neighbor[1] != current[1]) else 1.0
                tentative_g_score = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + math.hypot(
                        neighbor[0] - goal_grid[0],
                        neighbor[1] - goal_grid[1]
                    )
                    heapq.heappush(open_set, (f_score, neighbor))

        # 如果没有找到路径，返回直接路径（让attacker尝试穿越）
        return [start_pos.copy(), goal_pos.copy()]
    
    def _get_curve_target(self, attacker_pos, target_pos):
        vec_to_target = target_pos - attacker_pos
        dist = np.linalg.norm(vec_to_target)
        if dist < 20:
            return target_pos
        self.curve_phase += 0.1
        perp_vec = np.array([-vec_to_target[1], vec_to_target[0]])
        perp_vec = perp_vec / (np.linalg.norm(perp_vec) + 1e-6)
        offset = math.sin(self.curve_phase * self.curve_freq) * self.curve_amp
        lookahead_dist = min(60.0, dist)
        forward_point = attacker_pos + (vec_to_target / (dist + 1e-6)) * lookahead_dist
        curve_point = forward_point + perp_vec * offset
        curve_point[0] = np.clip(curve_point[0], 20, self.env_width - 20)
        curve_point[1] = np.clip(curve_point[1], 20, self.env_height - 20)
        return curve_point

    def _get_zigzag_target(self, attacker_pos, target_pos):
        vec_to_target = target_pos - attacker_pos
        dist = np.linalg.norm(vec_to_target)
        if dist < 30:
            return target_pos
        if self.step_count % self.zigzag_period == 0:
            self.zigzag_direction *= -1
        perp_vec = np.array([-vec_to_target[1], vec_to_target[0]])
        perp_vec = perp_vec / (np.linalg.norm(perp_vec) + 1e-6)
        offset = self.zigzag_direction * self.zigzag_amp
        lookahead = min(80.0, dist * 0.6)
        forward = attacker_pos + (vec_to_target / (dist + 1e-6)) * lookahead
        zigzag_point = forward + perp_vec * offset
        zigzag_point[0] = np.clip(zigzag_point[0], 20, self.env_width - 20)
        zigzag_point[1] = np.clip(zigzag_point[1], 20, self.env_height - 20)
        return zigzag_point

    def _get_flank_target(self, attacker_pos, defender_pos, target_pos):
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
        dist_to_target = np.linalg.norm(attacker_pos - target_pos)
        defender_to_target = np.linalg.norm(defender_pos - target_pos)
        if self.orbit_attacking or dist_to_target < 30:
            return target_pos
        if defender_to_target > self.orbit_radius * 1.5:
            self.orbit_attacking = True
            return target_pos
        self.orbit_angle += self.orbit_speed
        orbit_x = target_pos[0] + self.orbit_radius * math.cos(self.orbit_angle)
        orbit_y = target_pos[1] + self.orbit_radius * math.sin(self.orbit_angle)
        orbit_point = np.array([orbit_x, orbit_y], dtype=np.float32)
        orbit_point[0] = np.clip(orbit_point[0], 30, self.env_width - 30)
        orbit_point[1] = np.clip(orbit_point[1], 30, self.env_height - 30)
        defender_angle = math.atan2(defender_pos[1] - target_pos[1], defender_pos[0] - target_pos[0])
        attacker_orbit_angle = self.orbit_angle
        angle_diff = abs(self._normalize_angle_rad(attacker_orbit_angle - defender_angle))
        if angle_diff > math.pi * 0.7:
            self.orbit_attacking = True
            return target_pos
        return orbit_point

    def _normalize_angle_rad(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def _get_stealth_target(self, attacker_pos, defender_pos, target_pos):
        dist_to_target = np.linalg.norm(attacker_pos - target_pos)
        if dist_to_target < 50:
            return target_pos
        att_to_target = target_pos - attacker_pos
        att_dist = np.linalg.norm(att_to_target)
        def_to_att = attacker_pos - defender_pos
        def_dist = np.linalg.norm(def_to_att)
        if def_dist < 60:
            flee_dir = def_to_att / (def_dist + 1e-6)
            flee_point = attacker_pos + flee_dir * 40
            flee_point[0] = np.clip(flee_point[0], 30, self.env_width - 30)
            flee_point[1] = np.clip(flee_point[1], 30, self.env_height - 30)
            return flee_point
        return target_pos


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
        
        if self.strategy == 'curve':
            nav_goal = self._get_curve_target(attacker_pos, target_pos)
        elif self.strategy == 'zigzag':
            nav_goal = self._get_zigzag_target(attacker_pos, target_pos)
        elif self.strategy == 'flank':
            nav_goal = self._get_flank_target(attacker_pos, defender_pos, target_pos)
        elif self.strategy == 'orbit':
            nav_goal = self._get_orbit_target(attacker_pos, defender_pos, target_pos)
        elif self.strategy == 'stealth':
            nav_goal = self._get_stealth_target(attacker_pos, defender_pos, target_pos)
        elif self.strategy == 'wait_and_attack':
            dist_to_target = np.linalg.norm(attacker_pos - target_pos)
            if not self.is_attacking:
                if self.wait_timer > 0:
                    self.wait_timer -= 1
                    if dist_to_target < self.wait_dist:
                        nav_goal = attacker_pos
                        if dist_to_target < self.wait_dist * 0.8:
                            vec = attacker_pos - target_pos
                            nav_goal = target_pos + vec / (np.linalg.norm(vec) + 1e-6) * self.wait_dist
                    else:
                        vec = attacker_pos - target_pos
                        nav_goal = target_pos + vec / (np.linalg.norm(vec) + 1e-6) * self.wait_dist
                else:
                    self.is_attacking = True
            if self.is_attacking:
                nav_goal = target_pos

        # 2. 检查是否需要重新规划路径
        need_replan = False
        if len(self.path) == 0:
            need_replan = True
        elif self.last_planned_pos is None:
            need_replan = True
        elif np.linalg.norm(attacker_pos - self.last_planned_pos) > self.grid_size * 2:
            need_replan = True
        elif self.step_count % self.replan_interval == 0:
            need_replan = True

        if need_replan:
            # 多级避让策略
            dist_defender_to_target = np.linalg.norm(defender_pos - target_pos)
            target_radius = float(getattr(map_config, 'target_radius', 16.0))
            agent_radius = float(getattr(map_config, 'agent_radius', 8.0))
            
            # 基础避让半径
            if self.strategy == 'direct':
                 base_avoid_radius = self.defender_avoid_radius
            else:
                max_dist_for_full_avoid = 60.0
                if dist_defender_to_target >= max_dist_for_full_avoid:
                    base_avoid_radius = self.defender_avoid_radius
                else:
                    ratio = dist_defender_to_target / max_dist_for_full_avoid
                    base_avoid_radius = agent_radius + ratio * (self.defender_avoid_radius - agent_radius)
            
            # 生成尝试列表
            check_radii = [base_avoid_radius]
            if self.strategy != 'direct': 
                check_radii.extend([base_avoid_radius * 0.5, agent_radius, 0])
            else:
                 # Direct模式也尝试更小半径，但主要是不避让
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
        
        if self.strategy == 'wait_and_attack' and not self.is_attacking:
            dist_to_nav_goal = np.linalg.norm(attacker_pos - nav_goal)
            if dist_to_nav_goal < 20:
                speed = 0.0
            else:
                speed = self.max_speed * 0.5
        elif self.strategy == 'orbit' and not self.orbit_attacking:
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
            'curve_phase': self.curve_phase,
            'wait_timer': self.wait_timer
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
