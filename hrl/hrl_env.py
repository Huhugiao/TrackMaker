
import gymnasium as gym
import numpy as np
import torch
import math
import sys
import os
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import TADEnv
from ppo.nets import DefenderNetMLP
from ppo.alg_parameters import NetParameters
from rule_policies.defender_global import DefenderGlobalPolicy
from rule_policies.attacker_global import AttackerGlobalPolicy, ALL_STRATEGIES

class HRLEnv(gym.Env):
    """
    Hierarchical RL Environment.
    
    Upper Level Action: Continuous (2 dims), interpreted as weights for [Protect, Chase].
    Lower Level Skills:
      - Protect: Pre-trained MLP
      - Chase: Rule-based DefenderGlobalPolicy (chase mode)
    """
    def __init__(self, protect_model_path, attacker_strategy='random', device='cpu'):
        super().__init__()
        self.env = TADEnv(reward_mode='standard')
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space 
        
        self.device = device
        self.attacker_strategy_mode = attacker_strategy
        
        self.protect_net = DefenderNetMLP().to(device)
        self.protect_net.eval()
        try:
            checkpoint = torch.load(protect_model_path, map_location=device)
            if 'model' in checkpoint:
                self.protect_net.load_state_dict(checkpoint['model'])
            else:
                self.protect_net.load_state_dict(checkpoint)
            print(f"[HRLEnv] Loaded protect model from {protect_model_path}")
        except Exception as e:
            print(f"[HRLEnv] Error loading protect model: {e}")
            raise e
            
        self.chase_policy = DefenderGlobalPolicy(skill_mode='chase')
        
        if attacker_strategy == 'random':
            init_strat = random.choice(ALL_STRATEGIES)
        else:
            init_strat = attacker_strategy
        self.attacker_policy = AttackerGlobalPolicy(strategy=init_strat)
        
        self.step_count = 0
        
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.chase_policy.reset()
        
        if self.attacker_strategy_mode == 'random':
            new_strat = random.choice(ALL_STRATEGIES)
            self.attacker_policy = AttackerGlobalPolicy(strategy=new_strat)
        else:
            self.attacker_policy.reset()
            
        self.step_count = 0
        return obs, info
        
    def _get_start_protect_action(self, defender_obs, attacker_obs):
        from ppo.util import build_critic_observation
        with torch.no_grad():
            obs_tensor = torch.as_tensor(defender_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            critic_obs = build_critic_observation(defender_obs, attacker_obs)
            critic_tensor = torch.as_tensor(critic_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            mean, _, _ = self.protect_net(obs_tensor, critic_tensor)
            action = torch.tanh(mean).cpu().numpy()[0]
            return action

    def step(self, action):
        self.step_count += 1
        
        weights = np.exp(action) / np.sum(np.exp(action))
        w_protect = weights[0]
        w_chase = weights[1]
        
        defender_obs, attacker_obs = self.env.current_obs
        
        a_protect = self._get_start_protect_action(defender_obs, attacker_obs)
        
        priv_state = self.env.get_privileged_state()
        a_chase = self.chase_policy.get_action(defender_obs, priv_state)
        
        final_action = w_protect * a_protect + w_chase * a_chase
        
        a_attacker, _ = self.attacker_policy.get_action_with_info(attacker_obs)
        
        obs, reward, terminated, truncated, info = self.env.step(action=final_action, attacker_action=a_attacker)
        
        info['weights'] = weights
        info['a_protect'] = a_protect
        info['a_chase'] = a_chase
        
        return obs, reward, terminated, truncated, info

