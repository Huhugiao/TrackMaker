"""
Rule Policies模块

提供基于规则的策略用于强化学习的：
1. 对手策略（opponent policies）
2. 模仿学习冷启动策略（imitation learning warm-start）
3. 基线对比策略（baseline comparison）

主要类：
- AttackerGlobalPolicy: 用于Attacker的全局路径规划策略
- DefenderGlobalPolicy: 用于Defender的全局路径规划策略（模仿学习目标）
"""

from .attacker_global import AttackerGlobalPolicy, create_attacker_global_policy
from .attacker_static import AttackerStaticPolicy
from .defender_global import DefenderGlobalPolicy, create_defender_global_policy

__all__ = [
    # Attacker策略
    'AttackerGlobalPolicy',
    'create_attacker_global_policy',
    'AttackerStaticPolicy',
    # Defender策略
    'DefenderGlobalPolicy',
    'create_defender_global_policy',
]
