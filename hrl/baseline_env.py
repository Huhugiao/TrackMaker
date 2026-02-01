"""
Baseline Environment for End-to-End PPO Training

与HRL环境使用相同的设置(reward_mode='standard'和相同的attacker策略),
但直接输出defender的底层动作,不经过技能混合.
用于作为HRL分层策略的baseline比较.
"""

import gymnasium as gym
import numpy as np
import random
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import TADEnv
from rule_policies.attacker_global import AttackerGlobalPolicy, ALL_STRATEGIES


class BaselineEnv(gym.Env):
    """
    Baseline环境 - 端到端PPO训练
    
    与HRL环境保持一致:
    - reward_mode = 'standard'
    - 使用相同的attacker策略池
    
    区别:
    - Action直接是defender的底层动作 [angle_delta, speed]
    - 不经过任何技能混合
    """
    def __init__(self, attacker_strategy='random'):
        super().__init__()
        self.env = TADEnv(reward_mode='standard')
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        self.attacker_strategy_mode = attacker_strategy
        
        if attacker_strategy == 'random':
            init_strat = random.choice(ALL_STRATEGIES)
        else:
            init_strat = attacker_strategy
        self.attacker_policy = AttackerGlobalPolicy(strategy=init_strat)
        
        self.step_count = 0
        
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        
        if self.attacker_strategy_mode == 'random':
            new_strat = random.choice(ALL_STRATEGIES)
            self.attacker_policy = AttackerGlobalPolicy(strategy=new_strat)
        else:
            self.attacker_policy.reset()
            
        self.step_count = 0
        return obs, info
        
    def step(self, action):
        """
        直接执行defender动作,不经过技能混合
        
        Args:
            action: [angle_delta, speed] - defender的底层动作
        """
        self.step_count += 1
        
        defender_obs, attacker_obs = self.env.current_obs
        
        # 获取attacker动作
        a_attacker, _ = self.attacker_policy.get_action_with_info(attacker_obs)
        
        # 直接执行defender动作
        obs, reward, terminated, truncated, info = self.env.step(
            action=action, 
            attacker_action=a_attacker
        )
        
        return obs, reward, terminated, truncated, info
    
    def render(self, mode='rgb_array'):
        return self.env.render(mode=mode)
    
    def close(self):
        self.env.close()
