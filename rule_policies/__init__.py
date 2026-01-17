"""
Rule Policies模块

提供基于规则的策略用于强化学习的：
1. 对手策略（opponent policies）
2. 模仿学习冷启动策略（imitation learning warm-start）
3. 基线对比策略（baseline comparison）

主要类：
- APFPolicy: 基础人工势场法策略
- AttackerAPFPolicy: 用于Attacker的APF策略（追踪静态Target）
- AttackerGlobalPolicy: 用于Attacker的全局路径规划策略
- DefenderAPFPolicy: 用于Defender的APF策略（追踪动态目标）
- DefenderGlobalPolicy: 用于Defender的全局路径规划策略（模仿学习目标）
"""

from .apf import APFPolicy, APFNavigator
from .attacker_apf import AttackerAPFPolicy, create_attacker_apf_policy
from .attacker_global import AttackerGlobalPolicy, create_attacker_global_policy
from .defender_apf import (
    DefenderAPFPolicy,
    DualTargetDefenderAPFPolicy,
    create_defender_apf_policy,
    create_dual_defender_apf_policy
)
from .defender_global import DefenderGlobalPolicy, create_defender_global_policy

__all__ = [
    # 基础类
    'APFPolicy',
    'APFNavigator',
    # Attacker策略
    'AttackerAPFPolicy',
    'create_attacker_apf_policy',
    'AttackerGlobalPolicy',
    'create_attacker_global_policy',
    # Defender策略
    'DefenderAPFPolicy',
    'DualTargetDefenderAPFPolicy',
    'create_defender_apf_policy',
    'create_dual_defender_apf_policy',
    'DefenderGlobalPolicy',
    'create_defender_global_policy',
]
