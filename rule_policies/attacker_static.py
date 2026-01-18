"""
静止的Attacker策略 - 用于测试RL训练效果
Attacker不移动，只输出零动作
"""

import numpy as np


class AttackerStaticPolicy:
    """
    静止策略：Attacker完全不移动
    用于测试环境：验证Defender的RL训练是否有效
    """
    
    def __init__(self, env_width=None, env_height=None, **kwargs):
        """
        初始化静止策略
        
        Args:
            env_width: 环境宽度（未使用，保持接口一致）
            env_height: 环境高度（未使用，保持接口一致）
            **kwargs: 其他参数（未使用，保持接口一致）
        """
        self.env_width = env_width
        self.env_height = env_height
        # print("[AttackerStaticPolicy] Initialized - Attacker will remain stationary")  # 减少日志输出
    
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        返回零动作（不转向，不移动）
        
        Args:
            obs: 观测（未使用）
            
        Returns:
            零动作 [0.0, -1.0] (无转向，负速度强制减速到0)
        """
        # 返回 [0.0, -1.0] 强制减速
        # 由于attacker有max_acc限制，几步之后速度会降为0
        return np.array([0.0, -1.0], dtype=np.float32)
        """
        返回零动作（不转向，不移动）
        
        Args:
            obs: 观测（未使用）
            
        Returns:
            零动作 [0.0, 0.0] (无转向，无前进)
        """
        return np.array([0.0, 0.0], dtype=np.float32)
    
    def reset(self):
        """重置策略状态（无状态，不做任何事）"""
        pass


if __name__ == "__main__":
    # 简单测试
    policy = AttackerStaticPolicy(env_width=800, env_height=600)
    
    # 测试动作生成
    dummy_obs = np.zeros(71, dtype=np.float32)
    action = policy.get_action(dummy_obs)
    
    print(f"Action: {action}")
    print(f"Action shape: {action.shape}")
    print(f"Action dtype: {action.dtype}")
    assert np.allclose(action, [0.0, 0.0]), "Action should be [0.0, 0.0]"
    print("✓ Test passed!")
