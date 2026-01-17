"""
Target APF策略 - 用于Attacker（或任何需要到达静态目标的智能体）

状态空间（Attacker观测，71维）：
- obs[0]: Attacker 全局 X 坐标（归一化）
- obs[1]: Attacker 全局 Y 坐标（归一化）
- obs[2]: Attacker 朝向（归一化）
- obs[3]: Defender 全局 X 坐标（归一化）
- obs[4]: Defender 全局 Y 坐标（归一化）
- obs[5:69]: 雷达数据（64维）
- obs[69]: Target 全局 X 坐标（归一化）
- obs[70]: Target 全局 Y 坐标（归一化）

动作空间：
- action[0]: 角度变化（归一化）
- action[1]: 速度（归一化）
"""

import numpy as np
import math
from typing import Optional, Tuple
from .apf import APFNavigator


# =============================================================================
# Attacker APF 参数配置
# 可以在这里调整势场参数来改变 Attacker 的行为
# =============================================================================

# 静态障碍物参数
ATTACKER_OBSTACLE_ALPHA = 100.0    # 静态障碍物斥力强度（降低，避免过于保守）
ATTACKER_OBSTACLE_MU = 3.0         # 静态障碍物衰减系数（降低，扩大避障范围）

# Target（目标）吸引力参数
ATTACKER_TARGET_ALPHA = 1100.0     # Target 吸引力强度（降低，与障碍物斥力平衡）
ATTACKER_TARGET_MU = 1.2           # Target 吸引力衰减系数（降低，保持远距离追踪）

# Defender（动态障碍物）斥力参数
ATTACKER_CONSIDER_DEFENDER = True  # 是否将 Defender 视为障碍物
ATTACKER_DEFENDER_RADIUS = 8       # Defender 基础半径（像素）
ATTACKER_DEFENDER_RADIUS_MULT = 1  # Defender 半径倍数（实际斥力场 = 基础半径 * 倍数）

# 其他参数
ATTACKER_AGENT_RADIUS = 8.0        # Attacker 自身半径（用于 APF 计算）
ATTACKER_SAFETY_MARGIN = 25        # 安全距离（像素）：如果到障碍物 < 25px，势能=无穷大，绝对禁止靠近

# 雷达障碍物检测参数
ATTACKER_MAX_OBSTACLES = 8         # 从雷达数据中提取的最多障碍物数量（增加以提高安全性）
ATTACKER_DETECTION_THRESHOLD = 0.5 # 雷达检测阈值（降低：0.9会检测到地图边界，改为0.5只检测近距离障碍物）

# 说明：
# 1. obstacle_alpha 障碍物斥力：越大越避障，太大会导致不敢前进
# 2. obstacle_mu 斥力衰减：越大斥力范围越小，太小会导致远距离也怕障碍物
# 3. target_alpha 目标吸引力：越大越想到达目标，需要与障碍物斥力平衡
# 4. target_mu 吸引力衰减：越大远距离吸引力越弱
# 5. defender_radius_mult Defender 斥力场：越大越怕 Defender，可以设为 0 忽略 Defender
# 6. max_obstacles 雷达检测障碍物数量：越多越安全，但计算量增加
# 7. detection_threshold 雷达检测阈值：越低检测范围越大
# =============================================================================


class AttackerAPFPolicy:
    """
    Attacker APF策略 - 用于需要到达静态目标的智能体

    特点：
    - 目标：静态Target位置
    - 避障：避开环境中的障碍物
    - 将Defender视为移动障碍物进行避障
    """

    def __init__(self,
                 env_width=640,
                 env_height=640,
                 attacker_speed=2.0,
                 attacker_max_turn=12.0,
                 consider_defender_as_obstacle=True,
                 defender_obstacle_radius=50,
                 agent_radius=8.0,
                 max_obstacles=None,
                 detection_threshold=None,
                 **apf_kwargs):
        """
        初始化Attacker APF策略

        Args:
            env_width: 环境宽度
            env_height: 环境高度
            attacker_speed: Attacker最大速度
            attacker_max_turn: Attacker最大转向角速度（度/步）
            consider_defender_as_obstacle: 是否将Defender视为障碍物
            defender_obstacle_radius: Defender障碍物半径（像素）
            agent_radius: Attacker自身半径（像素）
            max_obstacles: 雷达检测的最多障碍物数量（None则使用默认值）
            detection_threshold: 雷达检测阈值（None则使用默认值）
            **apf_kwargs: 传递给APFNavigator的参数
        """
        self.env_width = env_width
        self.env_height = env_height
        self.attacker_speed = attacker_speed
        self.attacker_max_turn = attacker_max_turn
        self.consider_defender = ATTACKER_CONSIDER_DEFENDER
        self.defender_radius = ATTACKER_DEFENDER_RADIUS
        self.agent_radius = ATTACKER_AGENT_RADIUS
        self.max_obstacles = max_obstacles if max_obstacles is not None else ATTACKER_MAX_OBSTACLES
        self.detection_threshold = detection_threshold if detection_threshold is not None else ATTACKER_DETECTION_THRESHOLD

        # Attacker 专用 APF 参数
        attacker_apf_params = {
            'obstacle_alpha': ATTACKER_OBSTACLE_ALPHA,
            'obstacle_mu': ATTACKER_OBSTACLE_MU,
            'target_alpha': ATTACKER_TARGET_ALPHA,
            'target_mu': ATTACKER_TARGET_MU,
            'safety_margin': ATTACKER_SAFETY_MARGIN,  # 直接使用像素值，不是比例
        }
        attacker_apf_params.update(apf_kwargs)  # 允许用户覆盖

        # 创建导航器
        self.navigator = APFNavigator(env_width, env_height, **attacker_apf_params)

        # 雷达参数
        self.radar_rays = 64
        self.fov_range = float('inf')  # 无限视距

    def reset(self):
        """重置策略状态"""
        self.navigator.reset()

    def denormalize_pos(self, norm_x: float, norm_y: float) -> np.ndarray:
        """反归一化位置"""
        x = ((norm_x + 1.0) / 2.0) * self.env_width
        y = ((norm_y + 1.0) / 2.0) * self.env_height
        return np.array([x, y], dtype=np.float32)

    def denormalize_heading(self, norm_heading: float) -> float:
        """
        反归一化朝向

        环境归一化: obs[2] = (theta / 180.0) - 1.0
        其中 theta ∈ [0, 360)

        反归一化: theta = (norm_heading + 1.0) * 180.0
        """
        return (norm_heading + 1.0) * 180.0

    def radar_to_obstacles(self, radar_data: np.ndarray,
                         agent_pos: np.ndarray,
                         agent_heading: float) -> np.ndarray:
        """
        将雷达数据转换为障碍物坐标数组

        改进策略：返回多个最近的障碍物，让APF同时考虑多个障碍物的斥力

        Args:
            radar_data: 雷达距离读数（64维，归一化到[-1, 1]）
            agent_pos: 智能体位置 [x, y]
            agent_heading: 智能体朝向（度）

        Returns:
            obstacles: 障碍物数组 [x, y, radius=0]
                       最多包含max_obstacles个障碍物（按距离从近到远排序）
        """
        max_range = min(self.fov_range, math.hypot(self.env_width, self.env_height))
        norm_distances = (radar_data + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        distances = norm_distances * max_range

        angle_step = 360.0 / self.radar_rays
        heading_rad = math.radians(agent_heading)

        # 收集所有检测到的障碍物点
        detected_obstacles = []

        for i in range(self.radar_rays):
            distance = distances[i]

            # 只处理探测到的障碍物（在检测阈值内）
            if distance < max_range * self.detection_threshold:
                angle = heading_rad + math.radians(i * angle_step)
                x = agent_pos[0] + distance * math.cos(angle)
                y = agent_pos[1] + distance * np.sin(angle)

                # 过滤掉超出地图边界的障碍物（雷达检测到的边界虚假障碍物）
                if 0 <= x <= self.env_width and 0 <= y <= self.env_height:
                                    detected_obstacles.append((distance, x, y))

        # 如果没有检测到障碍物，返回空数组
        if len(detected_obstacles) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        # 按距离排序，取最近的max_obstacles个
        detected_obstacles.sort(key=lambda obs: obs[0])
        top_obstacles = detected_obstacles[:self.max_obstacles]

        # 转换为numpy数组格式 [x, y, radius=0]
        obstacles = np.array(
            [[obs[1], obs[2], 0.0] for obs in top_obstacles],
            dtype=np.float32
        )

        return obstacles

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
        target_pos = self.denormalize_pos(obs[69], obs[70])
        radar_data = obs[5:69]

        # 调试信息（第一次调用时打印）
        if not hasattr(self, '_debug_printed'):
            print(f"[Attacker APF Debug]")
            print(f"  Attacker pos: {attacker_pos}")
            print(f"  Attacker heading: {attacker_heading:.1f}°")
            print(f"  Target pos: {target_pos}")
            print(f"  Distance to target: {np.hypot(target_pos[0]-attacker_pos[0], target_pos[1]-attacker_pos[1]):.1f}px")
            print(f"  Defender pos: {defender_pos}")
            print(f"  Target observation values: obs[69]={obs[69]:.3f}, obs[70]={obs[70]:.3f}")
            self._debug_printed = True

        # 转换雷达数据为障碍物
        obstacles = self.radar_to_obstacles(radar_data, attacker_pos, attacker_heading)

        # 如果考虑Defender作为障碍物
        if self.consider_defender:
            # Defender 作为动态障碍物需要有更强的斥力
            # 通过增大半径来实现更强的斥力场
            effective_defender_radius = self.defender_radius * ATTACKER_DEFENDER_RADIUS_MULT
            defender_obstacle = np.array([
                defender_pos[0],
                defender_pos[1],
                effective_defender_radius
            ]).reshape(1, 3)
            if len(obstacles) == 0:
                obstacles = defender_obstacle
            else:
                obstacles = np.vstack([obstacles, defender_obstacle])

        # 计算动作
        action = self.navigator.compute_action_to_target(
            current_pos=attacker_pos,
            current_heading=attacker_heading,
            target=target_pos,
            obstacles=obstacles,
            max_speed=self.attacker_speed,
            max_turn_rate=self.attacker_max_turn,
            agent_radius=self.agent_radius
        )

        return action

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
        target_pos = self.denormalize_pos(obs[69], obs[70])

        # 计算动作
        action = self.get_action(obs)

        # 调试信息
        info = {
            'attacker_pos': attacker_pos,
            'attacker_heading': attacker_heading,
            'target_pos': target_pos,
            'distance_to_target': np.hypot(
                attacker_pos[0] - target_pos[0],
                attacker_pos[1] - target_pos[1]
            ),
            'obstacles_detected': len(self.radar_to_obstacles(
                obs[5:69], attacker_pos, attacker_heading
            ))
        }

        return action, info


# 便捷函数
def create_attacker_apf_policy(**kwargs) -> AttackerAPFPolicy:
    """
    创建Attacker APF策略的便捷函数

    Args:
        **kwargs: 传递给AttackerAPFPolicy的参数

    Returns:
        AttackerAPFPolicy实例
    """
    return AttackerAPFPolicy(**kwargs)
