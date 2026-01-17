"""
Defender Global Pathfinding Policy

Strategy: 使用全局信息进行路径规划，导航到Target（protect模式）或Attacker（chase模式）

此策略用于模仿学习，帮助RL策略快速收敛。
使用get_action时需要传入privileged state（全局位置信息）。

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

KP_TURN = 0.8              # 转向比例系数
GRID_SIZE = 8.0            # 路径规划网格大小（像素）
OBSTACLE_PADDING = 12.0    # 障碍物膨胀距离（像素）


class DefenderGlobalPolicy:
    """
    Defender全局路径规划策略

    核心逻辑：
    1. 使用全局障碍物信息规划从当前位置到目标的路径
    2. protect模式: 导航到Target位置守护
    3. chase模式: 导航到Attacker位置追击
    4. 跟随规划的路径点移动
    """

    def __init__(
        self,
        env_width: float = 640,
        env_height: float = 640,
        defender_speed: float = 2.0,
        defender_max_turn: float = 6.0,
        skill_mode: str = 'protect',
        kp_turn: float = KP_TURN,
        grid_size: float = GRID_SIZE,
        obstacle_padding: float = OBSTACLE_PADDING,
    ):
        """
        初始化Defender全局策略

        Args:
            env_width: 环境宽度
            env_height: 环境高度
            defender_speed: Defender最大速度
            defender_max_turn: Defender最大转向角速度（度/步）
            skill_mode: 技能模式 ('protect' 或 'chase')
            kp_turn: 转向比例系数
            grid_size: 路径规划网格大小
            obstacle_padding: 障碍物膨胀距离
        """
        self.env_width = env_width
        self.env_height = env_height
        self.max_speed = defender_speed
        self.max_turn = defender_max_turn
        self.skill_mode = skill_mode

        # 策略参数
        self.kp_turn = kp_turn
        self.grid_size = grid_size
        self.obstacle_padding = obstacle_padding

        # 路径规划相关
        self.path = []
        self.current_path_index = 0
        self.last_planned_pos = None
        self.replan_interval = 15  # 每N步重新规划一次
        self.step_count = 0

    def reset(self):
        """重置策略状态"""
        self.path = []
        self.current_path_index = 0
        self.last_planned_pos = None
        self.step_count = 0

    def normalize_angle(self, angle: float) -> float:
        """将角度归一化到 [-180, 180]"""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def is_pos_blocked(self, x: float, y: float) -> bool:
        """检查位置是否被障碍物阻挡"""
        return env_lib.is_point_blocked(x, y, padding=self.obstacle_padding)

    def plan_path(self, start_pos: np.ndarray, goal_pos: np.ndarray) -> List[np.ndarray]:
        """
        使用A*算法规划从起点到终点的路径

        Args:
            start_pos: 起点位置 [x, y]
            goal_pos: 终点位置 [x, y]

        Returns:
            路径点列表
        """
        start_grid = (int(start_pos[0] / self.grid_size), int(start_pos[1] / self.grid_size))
        goal_grid = (int(goal_pos[0] / self.grid_size), int(goal_pos[1] / self.grid_size))

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
                    pos = np.array([
                        current[0] * self.grid_size + self.grid_size / 2,
                        current[1] * self.grid_size + self.grid_size / 2
                    ], dtype=np.float32)
                    path.append(pos)
                    current = came_from[current]
                start_pos_continuous = np.array([
                    start_grid[0] * self.grid_size + self.grid_size / 2,
                    start_grid[1] * self.grid_size + self.grid_size / 2
                ], dtype=np.float32)
                path.append(start_pos_continuous)
                path.reverse()
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
                if (neighbor[0] < 0 or neighbor[0] >= self.env_width / self.grid_size or
                    neighbor[1] < 0 or neighbor[1] >= self.env_height / self.grid_size):
                    continue

                neighbor_pos = np.array([
                    neighbor[0] * self.grid_size + self.grid_size / 2,
                    neighbor[1] * self.grid_size + self.grid_size / 2
                ], dtype=np.float32)

                if self.is_pos_blocked(neighbor_pos[0], neighbor_pos[1]):
                    continue

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

        # 没有找到路径，返回直接路径
        return [start_pos.copy(), goal_pos.copy()]

    def get_action(self, obs: np.ndarray, privileged_state: dict) -> np.ndarray:
        """
        根据观测和特权状态计算动作

        Args:
            obs: Defender观测 (71维) - 仅用于兼容接口，实际使用privileged_state
            privileged_state: 特权状态字典，包含:
                - defender: {x, y, theta}
                - attacker: {x, y, theta}
                - target: {x, y}

        Returns:
            action: [angle_delta_norm, speed_norm]
        """
        pixel_size = getattr(map_config, 'pixel_size', 16)
        
        # 从特权状态获取位置
        defender_pos = np.array([
            privileged_state['defender']['center_x'],
            privileged_state['defender']['center_y']
        ], dtype=np.float32)
        defender_heading = privileged_state['defender']['theta']
        
        attacker_pos = np.array([
            privileged_state['attacker']['center_x'],
            privileged_state['attacker']['center_y']
        ], dtype=np.float32)
        
        target_pos = np.array([
            privileged_state['target']['center_x'],
            privileged_state['target']['center_y']
        ], dtype=np.float32)

        # 根据技能模式选择目标
        if self.skill_mode == 'protect':
            goal_pos = target_pos
        else:  # chase
            goal_pos = attacker_pos

        self.step_count += 1

        # 检查是否需要重新规划路径
        need_replan = False
        if len(self.path) == 0:
            need_replan = True
        elif self.last_planned_pos is None:
            need_replan = True
        elif np.linalg.norm(defender_pos - self.last_planned_pos) > self.grid_size * 2:
            need_replan = True
        elif self.step_count % self.replan_interval == 0:
            need_replan = True

        if need_replan:
            self.path = self.plan_path(defender_pos, goal_pos)
            self.current_path_index = 0
            self.last_planned_pos = defender_pos.copy()

        # 获取当前目标点
        if len(self.path) > 0:
            while (self.current_path_index < len(self.path) and
                   np.linalg.norm(defender_pos - self.path[self.current_path_index]) < self.grid_size):
                self.current_path_index += 1

            if self.current_path_index >= len(self.path):
                current_goal = goal_pos
            else:
                current_goal = self.path[self.current_path_index]
        else:
            current_goal = goal_pos

        # 计算到当前目标点的期望航向
        goal_angle = math.degrees(math.atan2(
            current_goal[1] - defender_pos[1],
            current_goal[0] - defender_pos[0]
        ))

        # 计算转向误差
        heading_error = self.normalize_angle(goal_angle - defender_heading)

        # P控制转向
        turn_delta = heading_error * self.kp_turn
        turn_delta = np.clip(turn_delta, -self.max_turn, self.max_turn)

        # 速度控制
        speed = self.max_speed

        # 归一化到动作空间
        angle_delta_norm = turn_delta / self.max_turn
        speed_norm = (speed / self.max_speed) * 2.0 - 1.0

        return np.array([angle_delta_norm, speed_norm], dtype=np.float32)


def create_defender_global_policy(**kwargs) -> DefenderGlobalPolicy:
    """创建DefenderGlobalPolicy的便捷函数"""
    return DefenderGlobalPolicy(**kwargs)
