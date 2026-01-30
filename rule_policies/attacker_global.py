"""
Attacker Global Pathfinding Policy

Strategy: 使用全局障碍物信息进行路径规划，直接导航到Target

Observation (72维):
- obs[0]: Attacker 全局 X 坐标（归一化）
- obs[1]: Attacker 全局 Y 坐标（归一化）
- obs[2]: Attacker 朝向（归一化）
- obs[3]: Defender 全局 X 坐标（归一化）
- obs[4]: Defender 全局 Y 坐标（归一化）
- obs[5]: Defender 朝向（归一化）
- obs[6]: Target 全局 X 坐标（归一化）
- obs[7]: Target 全局 Y 坐标（归一化）
- obs[8:72]: 雷达数据（64维）- 本策略不使用

Action: [angle_delta, speed_normalized]
- angle_delta: 角度变化（归一化到[-1, 1]）
- speed_normalized: 速度（归一化到[-1, 1]）
"""

import numpy as np
import math
import heapq
from typing import Tuple, List, Optional
import map_config
import env_lib


# =============================================================================
# 可调参数配置
# =============================================================================

# 转向控制参数
KP_TURN = 0.8              # 转向比例系数 (0.6~1.2) - 越大响应越快，但可能震荡

# 路径规划参数
GRID_SIZE = 8.0            # 路径规划网格大小（像素）
OBSTACLE_PADDING = 12.0    # 障碍物膨胀距离（像素），确保安全距离
DEFENDER_AVOID_RADIUS = 40.0  # 规避defender的安全半径（像素）


class AttackerGlobalPolicy:
    """
    Attacker全局路径规划策略

    核心逻辑：
    1. 使用全局障碍物信息规划从当前位置到target的路径
    2. 路径规划时把defender视为动态障碍物，主动绕开
    3. 跟随规划的路径点移动
    4. 动态更新路径（每N步或当路径受阻时）
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
        """
        self.env_width = env_width
        self.env_height = env_height
        self.max_speed = attacker_speed
        self.max_turn = attacker_max_turn

        # 策略参数
        self.kp_turn = kp_turn
        self.grid_size = grid_size
        self.obstacle_padding = obstacle_padding
        self.defender_avoid_radius = defender_avoid_radius

        # 路径规划相关
        self.path = []
        self.current_path_index = 0
        self.last_planned_pos = None
        self.replan_interval = 20  # 每N步重新规划一次
        self.step_count = 0

    def reset(self):
        """重置策略状态"""
        self.path = []
        self.current_path_index = 0
        self.last_planned_pos = None
        self.step_count = 0

    def denormalize_pos(self, norm_x: float, norm_y: float) -> np.ndarray:
        """反归一化位置"""
        x = ((norm_x + 1.0) / 2.0) * self.env_width
        y = ((norm_y + 1.0) / 2.0) * self.env_height
        return np.array([x, y], dtype=np.float32)

    def denormalize_heading(self, norm_heading: float) -> float:
        """
        反归一化朝向（度）

        环境归一化: obs[2] = (theta / 180.0) - 1.0
        其中 theta ∈ [0, 360)

        反归一化: theta = (norm_heading + 1.0) * 180.0
        """
        return (norm_heading + 1.0) * 180.0

    def normalize_angle(self, angle: float) -> float:
        """
        将角度归一化到 [-180, 180]

        Args:
            angle: 角度（度）

        Returns:
            归一化后的角度
        """
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def is_pos_blocked(self, x: float, y: float, defender_pos: np.ndarray = None) -> bool:
        """
        检查位置是否被障碍物阻挡

        Args:
            x: X坐标
            y: Y坐标
            defender_pos: Defender位置（可选），如果提供则视为障碍物

        Returns:
            True if blocked, False otherwise
        """
        # 检查静态障碍物
        if env_lib.is_point_blocked(x, y, padding=self.obstacle_padding):
            return True

        # 检查是否太靠近defender
        if defender_pos is not None:
            dist_to_defender = math.hypot(x - defender_pos[0], y - defender_pos[1])
            if dist_to_defender < self.defender_avoid_radius:
                return True

        return False

    def plan_path(self, start_pos: np.ndarray, goal_pos: np.ndarray, defender_pos: np.ndarray = None) -> List[np.ndarray]:
        """
        使用A*算法规划从起点到终点的路径

        Args:
            start_pos: 起点位置 [x, y]
            goal_pos: 终点位置 [x, y]
            defender_pos: Defender位置（可选），如果提供则视为动态障碍物

        Returns:
            路径点列表（包括起点和终点）
        """
        # 将连续坐标转换为网格坐标
        start_grid = (int(start_pos[0] / self.grid_size), int(start_pos[1] / self.grid_size))
        goal_grid = (int(goal_pos[0] / self.grid_size), int(goal_pos[1] / self.grid_size))

        # A*算法
        open_set = []
        heapq.heappush(open_set, (0, start_grid))
        came_from = {}
        g_score = {start_grid: 0}

        while open_set:
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

        # 检查是否需要重新规划路径
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
            # 多级避让策略：优先尝试避让，找不到路径则逐步降低避让半径
            dist_defender_to_target = np.linalg.norm(defender_pos - target_pos)
            target_radius = float(getattr(map_config, 'target_radius', 16.0))
            agent_radius = float(getattr(map_config, 'agent_radius', 8.0))
            
            # 根据 defender 与 target 的距离，动态调整避让半径
            # defender 越靠近 target，避让半径越小（这样 attacker 仍会尝试躲避）
            max_dist_for_full_avoid = 60.0
            
            if dist_defender_to_target >= max_dist_for_full_avoid:
                # 远离 target 时，使用完整避让
                base_avoid_radius = self.defender_avoid_radius
            else:
                # 靠近 target 时，线性减小（但保持一定避让）
                ratio = dist_defender_to_target / max_dist_for_full_avoid
                base_avoid_radius = agent_radius + ratio * (self.defender_avoid_radius - agent_radius)
            
            # 尝试多个避让半径，直到找到有效路径
            original_avoid_radius = self.defender_avoid_radius
            self.path = None
            
            # 从基础避让半径开始，逐步减小直到找到路径
            for try_radius in [base_avoid_radius, base_avoid_radius * 0.5, agent_radius, 0]:
                self.defender_avoid_radius = try_radius
                test_path = self.plan_path(
                    attacker_pos, target_pos, 
                    defender_pos if try_radius > 0 else None
                )
                # 如果找到有效路径（长度 > 2），使用它
                if len(test_path) > 2:
                    self.path = test_path
                    break
            
            # 如果所有尝试都失败，使用最后一次的路径
            if self.path is None:
                self.path = test_path
            
            self.defender_avoid_radius = original_avoid_radius
            self.current_path_index = 0
            self.last_planned_pos = attacker_pos.copy()

        # 获取当前目标点
        if len(self.path) > 0:
            # 找到路径中最近的未到达点
            while (self.current_path_index < len(self.path) and
                   np.linalg.norm(attacker_pos - self.path[self.current_path_index]) < self.grid_size):
                self.current_path_index += 1

            if self.current_path_index >= len(self.path):
                # 已到达终点，直接朝向target
                current_goal = target_pos
            else:
                current_goal = self.path[self.current_path_index]
        else:
            # 没有路径，直接朝向target
            current_goal = target_pos

        # 计算到当前目标点的期望航向
        goal_angle = math.degrees(math.atan2(
            current_goal[1] - attacker_pos[1],
            current_goal[0] - attacker_pos[0]
        ))

        # 计算转向误差
        heading_error = self.normalize_angle(goal_angle - attacker_heading)

        # P控制转向
        turn_delta = heading_error * self.kp_turn

        # 速度：始终保持最大速度
        speed = self.max_speed

        # 限制转向量
        turn_delta = np.clip(turn_delta, -self.max_turn, self.max_turn)

        # 限制速度
        speed = np.clip(speed, 0, self.max_speed)

        # 归一化到动作空间
        # angle_delta: [-max_turn, max_turn] -> [-1, 1]
        angle_delta_norm = turn_delta / self.max_turn

        # speed: [0, max_speed] -> [-1, 1]
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
        # 解析观测
        attacker_pos = self.denormalize_pos(obs[0], obs[1])
        attacker_heading = self.denormalize_heading(obs[2])
        defender_pos = self.denormalize_pos(obs[3], obs[4])
        target_pos = self.denormalize_pos(obs[6], obs[7])

        # 计算距离
        dist_to_defender = np.linalg.norm(attacker_pos - defender_pos)
        dist_to_target = np.linalg.norm(attacker_pos - target_pos)

        # 计算动作
        action = self.get_action(obs)

        # 调试信息
        info = {
            'mode': 'global_pathfinding',
            'attacker_pos': attacker_pos,
            'attacker_heading': attacker_heading,
            'defender_pos': defender_pos,
            'target_pos': target_pos,
            'dist_to_defender': dist_to_defender,
            'dist_to_target': dist_to_target,
            'path_length': len(self.path),
            'current_path_index': self.current_path_index,
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
