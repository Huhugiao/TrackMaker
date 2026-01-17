"""
基础APF（人工势场法）算法实现
基于Bacteria优化策略的路径规划算法

核心思想：
- 目标产生吸引力势场（负势能）
- 障碍物产生排斥力势场（正势能）
- 细菌优化：在当前位置周围生成候选点，选择势能最小的点
"""

import numpy as np
from typing import Tuple, List, Optional


class APFPolicy:
    """
    基础APF策略类

    参数设计：
    - alpha: 势场强度系数
    - mu: 势场衰减系数
    - step_size: 细菌点距离（候选点的半径）
    - bacteria_no: 细菌点数量
    - potential_limits: 势场计算的距离范围
    """

    def __init__(self,
                 env_width=640,
                 env_height=640,
                 bacteria_no=60,
                 step_size_ratio=0.05,
                 sensor_range=700,
                 obstacle_alpha=10.0,
                 obstacle_mu=50.0,
                 target_alpha=100000,
                 target_mu=0.5,
                 potential_lower_limit=0.4,
                 potential_upper_limit=4.5,
                 safety_margin=20.0):
        """
        初始化APF策略

        Args:
            env_width: 环境宽度
            env_height: 环境高度
            bacteria_no: 细菌点数量（候选位置数）
            step_size_ratio: 步长相对于传感器范围的比例
            sensor_range: 传感器探测范围
            obstacle_alpha: 障碍物势场强度
            obstacle_mu: 障碍物势场衰减系数
            target_alpha: 目标势场强度
            target_mu: 目标势场衰减系数
            potential_lower_limit: 势场计算距离下限（归一化）
            potential_upper_limit: 势场计算距离上限（归一化）
            safety_margin: 安全距离（像素），如果到障碍物 < 此距离，势能=无穷大
        """
        self.env_width = env_width
        self.env_height = env_height
        self.bacteria_no = bacteria_no
        self.sensor_range = sensor_range
        self.step_size = step_size_ratio * sensor_range

        # 障碍物势场参数
        self.obstacle_alpha = obstacle_alpha
        self.obstacle_mu = obstacle_mu

        # 目标势场参数
        self.target_alpha = target_alpha
        self.target_mu = target_mu

        # 势场计算参数（需要归一化）
        self.map_diagonal = np.hypot(env_width, env_height)
        self.potential_lower_limit = potential_lower_limit * self.map_diagonal
        self.potential_upper_limit = potential_upper_limit * self.map_diagonal
        self.safety_margin = safety_margin  # 直接使用像素值，不再乘以对角线

        # 细菌点角度分布
        self.bacteria_angles = np.linspace(
            0, 360, bacteria_no, endpoint=False
        )

    def normalize_distance(self, distance: float) -> float:
        """将实际距离归一化到[0, 1]"""
        return np.clip(distance / self.map_diagonal, 0, 1)

    def compute_goal_potential(self, pos: np.ndarray, target: np.ndarray) -> float:
        """
        计算目标势场（吸引力）

        J_goal = -alpha * exp(-mu * d^2)

        Args:
            pos: 当前位置 [x, y]
            target: 目标位置 [x, y]

        Returns:
            势能值（负数，表示吸引力）
        """
        dx = pos[0] - target[0]
        dy = pos[1] - target[1]
        distance_sq = dx**2 + dy**2

        # 归一化距离平方
        distance_sq_norm = distance_sq / (self.map_diagonal**2)

        potential = -self.target_alpha * np.exp(-self.target_mu * distance_sq_norm)
        return float(potential)

    def compute_obstacle_potential(self, pos: np.ndarray, obstacles: np.ndarray,
                                   agent_radius: float = 0.0) -> float:
        """
        计算障碍物势场（排斥力）

        修复版本：使用连续的势场函数，避免距离范围间隙

        J_obst = alpha / (d + epsilon)^2  (反比函数，所有距离都有斥力)
        J_obst = inf                      (如果距离小于safety_margin)

        Args:
            pos: 当前位置 [x, y]
            obstacles: 障碍物数组，每行 [x, y, radius]
            agent_radius: 智能体自身的半径（需要从距离中减去）

        Returns:
            总势能值（正数，表示排斥力）
        """
        total_potential = 0.0

        if len(obstacles) == 0:
            return total_potential

        for obs in obstacles:
            obs_pos = obs[:2]
            obs_radius = obs[2] if len(obs) > 2 else 0

            # 计算到障碍物中心的距离
            dx = pos[0] - obs_pos[0]
            dy = pos[1] - obs_pos[1]
            distance = np.hypot(dx, dy)

            # 考虑障碍物半径 + 智能体自身半径
            effective_distance = max(0, distance - obs_radius - agent_radius)

            if effective_distance < self.safety_margin:
                # 距离太近，势能为无穷大
                return float('inf')

            # 使用反比函数：所有距离都有斥力
            # 势能 = alpha / (1 + (distance/scale)^2)
            # 这样近距离斥力大，远距离斥力小
            distance_norm = effective_distance / self.map_diagonal

            # 修改为反比函数，避免间隙
            potential = self.obstacle_alpha / (1.0 + (self.obstacle_mu * distance_norm)**2)
            total_potential += potential

        return float(total_potential)

    def compute_total_potential(self, pos: np.ndarray, target: np.ndarray,
                               obstacles: np.ndarray, agent_radius: float = 0.0) -> float:
        """
        计算总势能

        J_total = J_goal + J_obst

        Args:
            pos: 位置 [x, y]
            target: 目标 [x, y]
            obstacles: 障碍物数组
            agent_radius: 智能体自身的半径

        Returns:
            总势能
        """
        goal_pot = self.compute_goal_potential(pos, target)
        obst_pot = self.compute_obstacle_potential(pos, obstacles, agent_radius)
        return goal_pot + obst_pot

    def generate_bacteria_points(self, current_pos: np.ndarray,
                                current_heading: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成细菌点（候选位置）

        在当前位置周围生成圆形分布的候选点

        Args:
            current_pos: 当前位置 [x, y]
            current_heading: 当前朝向（度）

        Returns:
            bacteria_x: 细菌点x坐标数组
            bacteria_y: 细菌点y坐标数组
        """
        bacteria_x = np.zeros(self.bacteria_no)
        bacteria_y = np.zeros(self.bacteria_no)

        heading_rad = np.radians(current_heading)

        for i in range(self.bacteria_no):
            # 细菌点角度 = 当前朝向 + 偏移角度
            angle = heading_rad + np.radians(self.bacteria_angles[i])

            bacteria_x[i] = current_pos[0] + self.step_size * np.cos(angle)
            bacteria_y[i] = current_pos[1] + self.step_size * np.sin(angle)

        return bacteria_x, bacteria_y

    def select_best_position(self, current_pos: np.ndarray,
                            current_heading: float,
                            target: np.ndarray,
                            obstacles: np.ndarray,
                            agent_radius: float = 0.0,
                            current_potential: Optional[float] = None) -> Tuple[np.ndarray, float]:
        """
        选择最佳下一个位置（细菌优化策略）

        策略：
        1. 生成细菌点
        2. 计算每个细菌点的势能
        3. 计算每个细菌点到目标的距离
        4. 选择距离目标最近且势能低于当前势能的点

        Args:
            current_pos: 当前位置 [x, y]
            current_heading: 当前朝向（度）
            target: 目标位置 [x, y]
            obstacles: 障碍物数组
            agent_radius: 智能体自身的半径
            current_potential: 当前位置势能（如果为None则计算）

        Returns:
            best_pos: 最佳位置 [x, y]
            best_potential: 最佳位置势能
        """
        if current_potential is None:
            current_potential = self.compute_total_potential(
                current_pos, target, obstacles, agent_radius
            )

        # 生成细菌点
        bacteria_x, bacteria_y = self.generate_bacteria_points(
            current_pos, current_heading
        )

        # 计算每个细菌点的势能和到目标的距离
        bacteria_potentials = np.zeros(self.bacteria_no)
        bacteria_distances = np.zeros(self.bacteria_no)

        for i in range(self.bacteria_no):
            pos = np.array([bacteria_x[i], bacteria_y[i]])
            bacteria_potentials[i] = self.compute_total_potential(
                pos, target, obstacles, agent_radius
            )

            dx = pos[0] - target[0]
            dy = pos[1] - target[1]
            bacteria_distances[i] = np.hypot(dx, dy)

        current_dist = np.hypot(
            current_pos[0] - target[0],
            current_pos[1] - target[1]
        )

        # 选择策略：优先选择最安全的路径（势能最低），而不是最接近目标的路径
        # 这样可以避免Defender为了接近目标而冒险穿越障碍物
        distance_improvements = current_dist - bacteria_distances

        # 处理inf值：将inf替换为一个很大的数
        safe_current_potential = current_potential if not np.isinf(current_potential) else 1e10
        safe_bacteria_potentials = np.array([
            p if not np.isinf(p) else 1e10 for p in bacteria_potentials
        ])

        potential_improvements = safe_current_potential - safe_bacteria_potentials

        # 优先选择势能改善最大的点（最安全的路径）
        # 如果所有点势能都更高，则选择势能增加最少的点
        valid_mask = safe_bacteria_potentials < safe_current_potential

        if np.any(valid_mask):
            # 有势能更低的点，在其中选择势能最低的（最安全的）
            # 修改：从选择"距离改善最大"改为选择"势能最低"
            valid_indices = np.where(valid_mask)[0]
            best_idx = valid_indices[np.argmin(safe_bacteria_potentials[valid_indices])]
        else:
            # 所有点势能都更高，选择势能增加最少的
            best_idx = np.argmin(safe_bacteria_potentials)

        best_pos = np.array([bacteria_x[best_idx], bacteria_y[best_idx]])
        best_potential = bacteria_potentials[best_idx]

        return best_pos, best_potential

    def check_safety(self, pos: np.ndarray, obstacles: np.ndarray) -> bool:
        """
        检查位置是否安全

        Args:
            pos: 位置 [x, y]
            obstacles: 障碍物数组

        Returns:
            True if safe, False otherwise
        """
        if len(obstacles) == 0:
            return True

        for obs in obstacles:
            obs_pos = obs[:2]
            obs_radius = obs[2] if len(obs) > 2 else 0

            dx = pos[0] - obs_pos[0]
            dy = pos[1] - obs_pos[1]
            distance = np.hypot(dx, dy)

            # 考虑障碍物半径
            effective_distance = max(0, distance - obs_radius)

            if effective_distance < self.safety_margin:
                return False

        return True


class APFNavigator:
    """
    APF导航器 - 封装完整的导航逻辑

    用法：
        1. reset() - 重置导航状态
        2. navigate() - 计算下一步动作
    """

    def __init__(self, env_width=640, env_height=640, **kwargs):
        """
        初始化导航器

        Args:
            env_width: 环境宽度
            env_height: 环境高度
            **kwargs: 传递给APFPolicy的参数
        """
        self.apf = APFPolicy(env_width, env_height, **kwargs)
        self.env_width = env_width
        self.env_height = env_height

    def reset(self):
        """重置导航状态"""
        pass

    def compute_action_to_target(self, current_pos: np.ndarray,
                                current_heading: float,
                                target: np.ndarray,
                                obstacles: np.ndarray,
                                max_speed: float = 2.0,
                                max_turn_rate: float = 6.0,
                                agent_radius: float = 0.0) -> np.ndarray:
        """
        计算朝向目标的动作

        Args:
            current_pos: 当前位置 [x, y]
            current_heading: 当前朝向（度）
            target: 目标位置 [x, y]
            obstacles: 障碍物数组 [x, y, radius]
            max_speed: 最大速度
            max_turn_rate: 最大转向角速度（度/步）
            agent_radius: 智能体自身的半径

        Returns:
            action: [angle_delta_normalized, speed_normalized]
                - angle_delta_normalized: 归一化角度变化（-1 到 1，环境会乘以 max_turn_rate）
                - speed_normalized: 归一化速度（-1 到 1，环境映射到 0 到 max_speed）
        """
        # 选择最佳位置
        best_pos, _ = self.apf.select_best_position(
            current_pos, current_heading, target, obstacles, agent_radius
        )

        # 计算方向
        dx = best_pos[0] - current_pos[0]
        dy = best_pos[1] - current_pos[1]

        if np.abs(dx) < 1e-6 and np.abs(dy) < 1e-6:
            # 位置没有变化，保持当前朝向
            target_heading = current_heading
        else:
            target_heading = np.degrees(np.arctan2(dy, dx))

        # 计算角度差
        angle_diff = target_heading - current_heading

        # 归一化到 [-180, 180]
        angle_diff = ((angle_diff + 180) % 360) - 180

        # 限制转向角速度
        angle_delta = np.clip(angle_diff, -max_turn_rate, max_turn_rate)

        # 归一化角度变化到 [-1, 1]，环境期望归一化的值
        # 环境中: angle_delta_actual = np.clip(action[0], -1.0, 1.0) * max_turn_rate
        angle_delta_normalized = angle_delta / max_turn_rate if max_turn_rate > 0 else 0.0

        # 计算速度
        distance = np.hypot(dx, dy)

        # 距离越远速度越快，但不超过最大速度
        if distance < self.apf.step_size:
            # 已经很接近目标，减速
            speed = max_speed * (distance / self.apf.step_size)
        else:
            speed = max_speed

        # 归一化速度到 [-1, 1]
        # 环境中: speed = (action[1] + 1.0) * 0.5 * max_speed
        speed_normalized = (speed / max_speed) * 2.0 - 1.0

        return np.array([angle_delta_normalized, speed_normalized], dtype=np.float32)
