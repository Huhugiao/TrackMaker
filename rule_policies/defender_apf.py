"""
Tracker APF策略 - 用于Defender（或任何需要追踪动态目标的智能体）

状态空间（Defender观测，72维）：
- obs[0]: 到Attacker的归一化距离
- obs[1]: 到Attacker的归一化方位角
- obs[2]: FOV边缘距离
- obs[3]: Attacker是否在FOV内
- obs[4]: Attacker是否被遮挡
- obs[5]: 未观测到的步数（归一化）
- obs[6:70]: 全向雷达数据（64维）
- obs[70]: 到Target的归一化距离
- obs[71]: 到Target的归一化方位角

特点：
- 支持双目标追踪：可以追踪Attacker或Target
- 考虑遮挡情况：当目标被遮挡时使用预测位置
- 使用雷达数据检测障碍物
- 适应FOV限制
"""

import numpy as np
import math
from typing import Optional, Tuple, Literal
from .apf import APFNavigator


# =============================================================================
# Defender APF 参数配置
# 可以在这里调整势场参数来改变 Defender 的行为
# =============================================================================

# 静态障碍物参数
DEFENDER_OBSTACLE_ALPHA = 200    # 静态障碍物斥力强度（增加以提高避障能力）
DEFENDER_OBSTACLE_MU = 4        # 静态障碍物衰减系数（降低以扩大斥力范围）

# 目标吸引力参数（追踪 Attacker 或 Target）
DEFENDER_TARGET_ALPHA = 1800.0      # 目标吸引力强度（越大，越想追踪目标）
DEFENDER_TARGET_MU = 2           # 目标吸引力衰减系数（越大，远距离吸引力越弱）

# 其他参数
DEFENDER_AGENT_RADIUS = 8.0        # Defender 自身半径（用于 APF 计算）
DEFENDER_SAFETY_MARGIN = 20      # 安全距离（像素）

# 说明：
# 修复后的势场函数使用反比：alpha / (1 + (mu * d_norm)^2)
# - 所有距离都有斥力，避免了间隙问题
# - 调整 alpha 和 mu 可以控制斥力的强度和范围
# =============================================================================


class DefenderAPFPolicy:
    """
    Defender APF策略 - 用于需要追踪动态目标的智能体

    特点：
    - 目标：可以是Attacker或Target（动态或静态）
    - 支持FOV和遮挡情况
    - 雷达数据用于障碍物检测
    - 可以用于模仿学习的冷启动
    """

    def __init__(self,
                 env_width=640,
                 env_height=640,
                 defender_speed=2.6,
                 defender_max_turn=6.0,
                 tracking_target: Literal['attacker', 'target'] = 'attacker',
                 use_privileged_info=False,
                 agent_radius=8.0,
                 **apf_kwargs):
        """
        初始化Defender APF策略

        Args:
            env_width: 环境宽度
            env_height: 环境高度
            defender_speed: Defender最大速度
            defender_max_turn: Defender最大转向角速度（度/步）
            tracking_target: 追踪目标 ('attacker' 或 'target')
            use_privileged_info: 是否使用特权信息（真实位置）
            agent_radius: Defender自身半径（像素）
            **apf_kwargs: 传递给APFNavigator的参数
        """
        self.env_width = env_width
        self.env_height = env_height
        self.defender_speed = defender_speed
        self.defender_max_turn = defender_max_turn
        self.tracking_target = tracking_target
        self.use_privileged = use_privileged_info
        self.agent_radius = DEFENDER_AGENT_RADIUS

        # Defender 专用 APF 参数 - 目标吸引力 > 障碍物斥力
        defender_apf_params = {
            'obstacle_alpha': DEFENDER_OBSTACLE_ALPHA,
            'obstacle_mu': DEFENDER_OBSTACLE_MU,
            'target_alpha': DEFENDER_TARGET_ALPHA,
            'target_mu': DEFENDER_TARGET_MU,
            'safety_margin': DEFENDER_SAFETY_MARGIN,  # 直接使用像素值，不是比例
        }
        defender_apf_params.update(apf_kwargs)  # 允许用户覆盖

        # 创建导航器
        self.navigator = APFNavigator(env_width, env_height, **defender_apf_params)

        # 雷达参数
        self.radar_rays = 64
        self.fov_range = float('inf')

        # 存储最后的观测位置（用于遮挡情况）
        self.last_observed_target_pos = None
        self.last_observed_target_heading = None

    def reset(self):
        """重置策略状态"""
        self.navigator.reset()
        self.last_observed_target_pos = None
        self.last_observed_target_heading = None

    def set_tracking_target(self, target: Literal['attacker', 'target']):
        """设置追踪目标"""
        self.tracking_target = target

    def estimate_target_position(self, obs: np.ndarray,
                                defender_pos: np.ndarray,
                                defender_heading: float) -> Tuple[np.ndarray, float]:
        """
        估计目标位置

        当目标被遮挡或不在FOV内时，使用最后观测位置预测

        Args:
            obs: Defender观测
            defender_pos: Defender位置 [x, y]
            defender_heading: Defender朝向（度）

        Returns:
            estimated_pos: 估计的目标位置 [x, y]
            estimated_heading: 估计的目标朝向（度）
        """
        # 解析观测
        target_distance_norm = obs[0]  # 如果追踪attacker
        target_bearing_norm = obs[1]
        in_fov = obs[3] > 0.5
        occluded = obs[4] > 0.5
        is_visible = in_fov and not occluded

        map_diagonal = math.hypot(self.env_width, self.env_height)

        # 反归一化距离
        target_distance = ((target_distance_norm + 1.0) / 2.0) * map_diagonal

        # 反归一化方位角
        target_bearing = target_bearing_norm * 180.0

        # 计算目标绝对位置
        abs_angle = math.radians(target_bearing + defender_heading)
        rel_x = target_distance * math.cos(abs_angle)
        rel_y = target_distance * math.sin(abs_angle)

        target_pos = defender_pos + np.array([rel_x, rel_y])

        # 估计目标朝向（假设朝向Defender或随机）
        if self.last_observed_target_heading is not None:
            target_heading = self.last_observed_target_heading
        else:
            # 默认朝向Defender
            dx = defender_pos[0] - target_pos[0]
            dy = defender_pos[1] - target_pos[1]
            target_heading = math.degrees(math.atan2(dy, dx))

        # 更新最后观测位置
        if is_visible:
            self.last_observed_target_pos = target_pos.copy()
            self.last_observed_target_heading = target_heading
        elif self.last_observed_target_pos is None:
            # 从未观测过，使用当前位置
            self.last_observed_target_pos = target_pos.copy()
            self.last_observed_target_heading = target_heading

        return target_pos, target_heading

    def estimate_static_target_position(self, obs: np.ndarray,
                                       defender_pos: np.ndarray) -> np.ndarray:
        """
        估计静态Target位置（用于追踪静态Target）

        Args:
            obs: Defender观测
            defender_pos: Defender位置 [x, y]

        Returns:
            target_pos: Target位置 [x, y]
        """
        # 解析观测
        target_distance_norm = obs[70]
        target_bearing_norm = obs[71]
        defender_heading = 0.0  # 不需要，因为观测已经归一化

        map_diagonal = math.hypot(self.env_width, self.env_height)

        # 反归一化距离和角度
        target_distance = ((target_distance_norm + 1.0) / 2.0) * map_diagonal
        target_bearing = target_bearing_norm * 180.0

        # 需要Defender的朝向来计算绝对位置
        # 但观测中没有Defender的朝向，所以假设从观测中推断
        # 这里简化：假设bearing是相对于某个固定方向
        abs_angle = math.radians(target_bearing)
        rel_x = target_distance * math.cos(abs_angle)
        rel_y = target_distance * math.sin(abs_angle)

        target_pos = defender_pos + np.array([rel_x, rel_y])

        return target_pos

    def radar_to_obstacles(self, radar_data: np.ndarray,
                          agent_pos: np.ndarray,
                          agent_heading: float,
                          max_obstacles: int = 8,
                          detection_threshold: float = 0.8) -> np.ndarray:
        """
        将雷达数据转换为障碍物坐标数组

        改进策略：返回多个最近的障碍物，而不是只返回一个
        这样可以让APF同时考虑多个障碍物的斥力

        Args:
            radar_data: 雷达距离读数（64维，归一化到[-1, 1]）
            agent_pos: 智能体位置 [x, y]
            agent_heading: 智能体朝向（度）
            max_obstacles: 最多返回的障碍物数量（默认8个）
            detection_threshold: 检测阈值，只返回距离 < max_range * threshold 的障碍物

        Returns:
            obstacles: 障碍物数组 [x, y, radius=0]
                       最多包含max_obstacles个障碍物（按距离从近到远排序）
        """
        max_range = min(self.fov_range, math.hypot(self.env_width, self.env_height))
        norm_distances = (radar_data + 1.0) / 2.0
        distances = norm_distances * max_range

        angle_step = 360.0 / self.radar_rays
        heading_rad = math.radians(agent_heading)

        # 收集所有检测到的障碍物点
        detected_obstacles = []

        for i in range(self.radar_rays):
            distance = distances[i]

            # 只处理探测到的障碍物（在检测阈值内）
            if distance < max_range * detection_threshold:
                angle = heading_rad + math.radians(i * angle_step)
                x = agent_pos[0] + distance * math.cos(angle)
                y = agent_pos[1] + distance * math.sin(angle)

                detected_obstacles.append((distance, x, y))

        # 如果没有检测到障碍物，返回空数组
        if len(detected_obstacles) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        # 按距离排序，取最近的max_obstacles个
        detected_obstacles.sort(key=lambda obs: obs[0])
        top_obstacles = detected_obstacles[:max_obstacles]

        # 转换为numpy数组格式 [x, y, radius=0]
        obstacles = np.array(
            [[obs[1], obs[2], 0.0] for obs in top_obstacles],
            dtype=np.float32
        )

        return obstacles

    def get_defender_state_from_obs(self, obs: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        从观测中推断Defender状态

        注意：Defender的观测中没有包含自己的全局位置和朝向
        这需要从环境或外部获取

        Args:
            obs: Defender观测

        Returns:
            defender_pos: Defender位置 [x, y]
            defender_heading: Defender朝向（度）
        """
        # 由于观测中没有Defender的全局位置和朝向
        # 这里返回None，需要从外部提供
        # 或者使用默认值
        default_pos = np.array([self.env_width / 2, self.env_height / 2])
        default_heading = 0.0

        return default_pos, default_heading

    def get_action(self, obs: np.ndarray,
                  defender_pos: Optional[np.ndarray] = None,
                  defender_heading: Optional[float] = None) -> np.ndarray:
        """
        根据观测计算动作

        Args:
            obs: Defender观测 (72维)
            defender_pos: Defender全局位置 [x, y] (如果为None则使用默认值)
            defender_heading: Defender全局朝向（度）(如果为None则使用默认值)

        Returns:
            action: [angle_delta, speed_normalized]
        """
        # 获取Defender状态
        if defender_pos is None or defender_heading is None:
            defender_pos, defender_heading = self.get_defender_state_from_obs(obs)

        # 转换雷达数据为障碍物
        radar_data = obs[6:70]
        obstacles = self.radar_to_obstacles(radar_data, defender_pos, defender_heading)

        # 根据追踪目标确定目标位置
        if self.tracking_target == 'attacker':
            target_pos, _ = self.estimate_target_position(
                obs, defender_pos, defender_heading
            )
        else:  # 'target'
            target_pos = self.estimate_static_target_position(
                obs, defender_pos
            )

        # 计算动作
        action = self.navigator.compute_action_to_target(
            current_pos=defender_pos,
            current_heading=defender_heading,
            target=target_pos,
            obstacles=obstacles,
            max_speed=self.defender_speed,
            max_turn_rate=self.defender_max_turn,
            agent_radius=self.agent_radius
        )

        return action

    def get_action_with_info(self, obs: np.ndarray,
                           defender_pos: Optional[np.ndarray] = None,
                           defender_heading: Optional[float] = None) -> Tuple[np.ndarray, dict]:
        """
        获取动作并返回调试信息

        Args:
            obs: Defender观测
            defender_pos: Defender全局位置
            defender_heading: Defender全局朝向

        Returns:
            action: 动作
            info: 调试信息字典
        """
        # 获取Defender状态
        if defender_pos is None or defender_heading is None:
            defender_pos, defender_heading = self.get_defender_state_from_obs(obs)

        # 获取目标位置
        if self.tracking_target == 'attacker':
            target_pos, target_heading = self.estimate_target_position(
                obs, defender_pos, defender_heading
            )
        else:
            target_pos = self.estimate_static_target_position(obs, defender_pos)
            target_heading = 0.0

        # 计算动作
        action = self.get_action(obs, defender_pos, defender_heading)

        # 调试信息
        distance_to_target = np.hypot(
            defender_pos[0] - target_pos[0],
            defender_pos[1] - target_pos[1]
        )

        info = {
            'defender_pos': defender_pos,
            'defender_heading': defender_heading,
            'target_pos': target_pos,
            'target_heading': target_heading,
            'distance_to_target': distance_to_target,
            'tracking_target': self.tracking_target,
            'in_fov': obs[3] > 0.5,
            'occluded': obs[4] > 0.5,
            'obstacles_detected': len(self.radar_to_obstacles(
                obs[6:70], defender_pos, defender_heading
            ))
        }

        return action, info


class DualTargetDefenderAPFPolicy:
    """
    双目标追踪APF策略 - 用于Defender

    特点：
    - 同时维护追踪Attacker和Target两个策略
    - 可以根据需要切换追踪目标
    - 适用于训练两个网络的场景
    """

    def __init__(self,
                 env_width=640,
                 env_height=640,
                 defender_speed=2.6,
                 defender_max_turn=6.0,
                 **apf_kwargs):
        """
        初始化双目标追踪策略

        Args:
            env_width: 环境宽度
            env_height: 环境高度
            defender_speed: Defender最大速度
            defender_max_turn: Defender最大转向角速度
            **apf_kwargs: 传递给APFNavigator的参数
        """
        self.env_width = env_width
        self.env_height = env_height
        self.defender_speed = defender_speed
        self.defender_max_turn = defender_max_turn

        # 创建两个策略实例
        self.attacker_tracker = DefenderAPFPolicy(
            env_width=env_width,
            env_height=env_height,
            defender_speed=defender_speed,
            defender_max_turn=defender_max_turn,
            tracking_target='attacker',
            **apf_kwargs
        )

        self.target_tracker = DefenderAPFPolicy(
            env_width=env_width,
            env_height=env_height,
            defender_speed=defender_speed,
            defender_max_turn=defender_max_turn,
            tracking_target='target',
            **apf_kwargs
        )

    def reset(self):
        """重置两个策略的状态"""
        self.attacker_tracker.reset()
        self.target_tracker.reset()

    def get_action_tracking_attacker(self, obs: np.ndarray,
                                   defender_pos: Optional[np.ndarray] = None,
                                   defender_heading: Optional[float] = None) -> np.ndarray:
        """获取追踪Attacker的动作"""
        return self.attacker_tracker.get_action(obs, defender_pos, defender_heading)

    def get_action_tracking_target(self, obs: np.ndarray,
                                  defender_pos: Optional[np.ndarray] = None,
                                  defender_heading: Optional[float] = None) -> np.ndarray:
        """获取追踪Target的动作"""
        return self.target_tracker.get_action(obs, defender_pos, defender_heading)


# 便捷函数
def create_defender_apf_policy(tracking_target='attacker', **kwargs) -> DefenderAPFPolicy:
    """创建Defender APF策略的便捷函数"""
    return DefenderAPFPolicy(tracking_target=tracking_target, **kwargs)


def create_dual_defender_apf_policy(**kwargs) -> DualTargetDefenderAPFPolicy:
    """创建双目标追踪APF策略的便捷函数"""
    return DualTargetDefenderAPFPolicy(**kwargs)
