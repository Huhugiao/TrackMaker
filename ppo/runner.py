"""
TAD PPO Runner - Ray分布式采样Worker
"""

import os
import ray
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from ppo.alg_parameters import SetupParameters, TrainingParameters, NetParameters, RecordingParameters
from ppo.nets import DefenderNetMLP
from ppo.util import build_critic_observation, update_perf

import map_config
from map_config import EnvParameters
from env import TADEnv
from rule_policies import AttackerAPFPolicy, AttackerGlobalPolicy, AttackerStaticPolicy
from rule_policies.defender_global import DefenderGlobalPolicy


ATTACKER_POLICY_REGISTRY = {
    'attacker_apf': AttackerAPFPolicy,
    'attacker_global': AttackerGlobalPolicy,
    'attacker_static': AttackerStaticPolicy
}


@ray.remote(num_cpus=1, num_gpus=SetupParameters.NUM_GPU / TrainingParameters.N_ENVS if SetupParameters.NUM_GPU > 0 else 0)
class Runner:
    def __init__(self, meta_agent_id: int, env_configs: Dict = None):
        self.meta_agent_id = meta_agent_id
        self.env_configs = env_configs or {}
        
        import torch
        self.device = torch.device('cuda' if torch.cuda.is_available() and SetupParameters.USE_GPU_LOCAL else 'cpu')
        
        self.local_network = DefenderNetMLP().to(self.device)
        self.local_network.eval()
        
        self._init_env()
        
        self.opponent_policies = self._create_opponent_policies()
        self.current_opponent_key = None
        
        self._reset()
    
    def _init_env(self):
        self.env = TADEnv(reward_mode=SetupParameters.SKILL_MODE)
        
        # Create expert policy for imitation learning
        self.expert_policy = DefenderGlobalPolicy(
            env_width=self.env.width,
            env_height=self.env.height,
            defender_speed=self.env.defender_speed,
            defender_max_turn=getattr(map_config, 'defender_max_angular_speed', 6.0),
            skill_mode=SetupParameters.SKILL_MODE
        )
    
    def _create_opponent_policies(self) -> Dict[str, Any]:
        policies = {}
        # 始终包含 protect1/protect2 所需的策略
        skill_mode = SetupParameters.SKILL_MODE
        if skill_mode == 'protect1':
            policies['attacker_static'] = ATTACKER_POLICY_REGISTRY['attacker_static']
        elif skill_mode == 'protect2':
            policies['attacker_global'] = ATTACKER_POLICY_REGISTRY['attacker_global']
        else:
            # 其他模式：根据配置创建
            weights_cfg = TrainingParameters.RANDOM_OPPONENT_WEIGHTS.get('attacker', {})
            for key in weights_cfg.keys():
                if key in ATTACKER_POLICY_REGISTRY:
                    policy_cls = ATTACKER_POLICY_REGISTRY[key]
                    policies[key] = policy_cls
        return policies
    
    def _sample_opponent_policy(self) -> str:
        # protect1/protect2 强制使用指定对手策略
        skill_mode = SetupParameters.SKILL_MODE
        if skill_mode == 'protect1':
            return 'attacker_static'  # 阶段1: 静止对手
        elif skill_mode == 'protect2':
            return 'attacker_global'  # 阶段2: 导航对手
        
        # 其他模式按配置的权重随机采样
        weights_cfg = TrainingParameters.RANDOM_OPPONENT_WEIGHTS.get('attacker', {})
        keys = list(weights_cfg.keys())
        probs = np.array([weights_cfg[k] for k in keys], dtype=np.float64)
        probs = probs / probs.sum()
        chosen = np.random.choice(keys, p=probs)
        return chosen
    
    def _reset(self, for_eval: bool = False, episode_idx: int = 0):
        """
        重置环境和对手策略
        
        Args:
            for_eval: 是否为评估模式（使用评估种子设置）
            episode_idx: 当前episode索引（用于固定种子时区分不同episode）
        """
        self.current_opponent_key = self._sample_opponent_policy()
        policy_cls = self.opponent_policies.get(self.current_opponent_key)
        
        # Instantiate policy
        self.attacker_policy = policy_cls(
            env_width=self.env.width,
            env_height=self.env.height,
            attacker_speed=self.env.attacker_speed,
            attacker_max_turn=getattr(map_config, 'attacker_max_angular_speed', 12.0)
        )
        self.attacker_policy.reset()
        
        # Reset environment with random or fixed seed
        if for_eval:
            # 评估模式
            if SetupParameters.EVAL_USE_RANDOM_SEED:
                reset_seed = None  # 随机种子
            else:
                reset_seed = SetupParameters.EVAL_FIXED_SEED + episode_idx  # 固定种子
        else:
            # 训练模式
            if SetupParameters.TRAIN_USE_RANDOM_SEED:
                reset_seed = None  # 随机种子
            else:
                reset_seed = SetupParameters.SEED + self.meta_agent_id * 100  # 固定种子
        obs, _ = self.env.reset(seed=reset_seed)
        self.defender_obs, self.attacker_obs = obs
        
        self.done = False
        self.episode_reward = 0.0
        self.episode_len = 0
    
    def set_weights(self, weights):
        import torch
        state_dict = {}
        for k, v in weights.items():
            if isinstance(v, torch.Tensor):
                state_dict[k] = v.clone().detach().to(self.device)
            else:
                state_dict[k] = torch.as_tensor(v, device=self.device)
        self.local_network.load_state_dict(state_dict)
        self.local_network.eval()
    
    def run(self, num_steps: int) -> Dict[str, np.ndarray]:
        import torch
        
        mb_obs = []
        mb_critic_obs = []
        mb_actions = []
        mb_log_probs = []
        mb_values = []
        mb_rewards = []
        mb_dones = []
        mb_expert_actions = []
        
        perf = {'per_r': [], 'per_episode_len': [], 'win': []}
        
        for _ in range(num_steps):
            critic_obs = build_critic_observation(self.defender_obs, self.attacker_obs)
            
            obs_t = torch.tensor(self.defender_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            critic_obs_t = torch.tensor(critic_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            with torch.no_grad():
                actions, log_probs, pre_tanh, values = self.local_network.act(obs_t, critic_obs_t)
            tanh_action = actions.cpu().numpy().flatten()
            pre_tanh_action = pre_tanh.cpu().numpy().flatten()
            log_prob = log_probs.cpu().numpy().item()
            value = values.cpu().numpy().item()
            
            mb_obs.append(self.defender_obs.copy())
            mb_critic_obs.append(critic_obs.copy())
            mb_actions.append(pre_tanh_action)
            mb_log_probs.append(log_prob)
            mb_values.append(value)
            
            # Get expert action for IL
            priv_state = self.env.get_privileged_state()
            expert_action = self.expert_policy.get_action(self.defender_obs, priv_state)
            mb_expert_actions.append(expert_action)
            
            # Get attacker action
            attacker_action = self.attacker_policy.get_action(self.attacker_obs)
            
            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(tanh_action, attacker_action)
            done = terminated or truncated
            
            self.defender_obs, self.attacker_obs = obs
            self.done = done
            self.episode_reward += reward
            self.episode_len += 1
            
            mb_rewards.append(reward)
            mb_dones.append(done)
            
            if done:
                one_ep = {
                    'episode_reward': self.episode_reward,
                    'num_step': self.episode_len,
                    'win': info.get('win', False)
                }
                update_perf(one_ep, perf)
                perf['win'].append(one_ep['win'])
                self._reset()
        
        last_critic_obs = build_critic_observation(self.defender_obs, self.attacker_obs)
        obs_t = torch.tensor(self.defender_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        critic_obs_t = torch.tensor(last_critic_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            last_value = self.local_network.critic_value(critic_obs_t).cpu().numpy().item()
        
        mb_obs = np.array(mb_obs, dtype=np.float32)
        mb_critic_obs = np.array(mb_critic_obs, dtype=np.float32)
        mb_actions = np.array(mb_actions, dtype=np.float32)
        mb_log_probs = np.array(mb_log_probs, dtype=np.float32)
        mb_values = np.array(mb_values, dtype=np.float32)
        mb_rewards = np.array(mb_rewards, dtype=np.float32)
        mb_dones = np.array(mb_dones, dtype=np.float32)
        mb_expert_actions = np.array(mb_expert_actions, dtype=np.float32)
        
        mb_advs = np.zeros_like(mb_rewards)
        mb_returns = np.zeros_like(mb_rewards)
        lastgaelam = 0.0
        
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_value = last_value
            else:
                next_value = mb_values[t + 1]

            done_t = mb_dones[t]
            delta = mb_rewards[t] + TrainingParameters.GAMMA * next_value * (1.0 - done_t) - mb_values[t]
            lastgaelam = delta + TrainingParameters.GAMMA * TrainingParameters.LAM * (1.0 - done_t) * lastgaelam
            mb_advs[t] = lastgaelam
        
        mb_returns = mb_advs + mb_values
        
        return {
            'obs': mb_obs,
            'critic_obs': mb_critic_obs,
            'actions': mb_actions,
            'log_probs': mb_log_probs,
            'values': mb_values,
            'returns': mb_returns,
            'advs': mb_advs,
            'dones': mb_dones,
            'expert_actions': mb_expert_actions,
            'perf': perf
        }
    
    def imitation(self, num_steps: int) -> Dict[str, np.ndarray]:
        import torch
        from rule_policies import DefenderAPFPolicy
        
        if not hasattr(self, 'expert_policy'):
            self.expert_policy = DefenderAPFPolicy(
                env_width=self.env.width,
                env_height=self.env.height,
                defender_speed=self.env.defender_speed,
                defender_max_turn=getattr(map_config, 'defender_max_angular_speed', 6.0),
                tracking_target='attacker' if SetupParameters.SKILL_MODE == 'protect' else 'target'
            )
        
        mb_obs = []
        mb_critic_obs = []
        mb_expert_actions = []
        
        perf = {'per_r': [], 'per_episode_len': [], 'win': []}
        
        for _ in range(num_steps):
            priv = self.env.get_privileged_state()
            defender_pos = np.array([priv['defender']['center_x'], priv['defender']['center_y']])
            defender_heading = priv['defender']['theta']
            expert_action = self.expert_policy.get_action(self.defender_obs, defender_pos, defender_heading)
            
            critic_obs = build_critic_observation(self.defender_obs, self.attacker_obs)
            
            mb_obs.append(self.defender_obs.copy())
            mb_critic_obs.append(critic_obs.copy())
            mb_expert_actions.append(expert_action)
            
            obs_t = torch.tensor(self.defender_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            critic_obs_t = torch.tensor(critic_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            with torch.no_grad():
                actions, _, _, _ = self.local_network.act(obs_t, critic_obs_t)
            
            action = actions.cpu().numpy().flatten()
            
            # Get attacker action
            attacker_action = self.attacker_policy.get_action(self.attacker_obs)
            
            obs, reward, terminated, truncated, info = self.env.step(action, attacker_action)
            done = terminated or truncated
            
            self.defender_obs, self.attacker_obs = obs
            self.done = done
            self.episode_reward += reward
            self.episode_len += 1
            
            if done:
                one_ep = {
                    'episode_reward': self.episode_reward,
                    'num_step': self.episode_len,
                    'win': info.get('win', False)
                }
                update_perf(one_ep, perf)
                perf['win'].append(one_ep['win'])
                self._reset()
        
        mb_obs = np.array(mb_obs, dtype=np.float32)
        mb_critic_obs = np.array(mb_critic_obs, dtype=np.float32)
        mb_expert_actions = np.array(mb_expert_actions, dtype=np.float32)
        
        return {
            'obs': mb_obs,
            'critic_obs': mb_critic_obs,
            'expert_actions': mb_expert_actions,
            'perf': perf
        }
    
    def evaluate(self, num_episodes: int = 5, greedy: bool = True, record_gif: bool = False) -> Dict:
        import torch
        
        perf = {'per_r': [], 'per_episode_len': [], 'win': []}
        frames = [] if record_gif else None
        
        for ep_idx in range(num_episodes):
            self._reset(for_eval=True, episode_idx=ep_idx)
            ep_reward = 0.0
            ep_len = 0
            ep_frames = []
            
            while not self.done:
                if record_gif and ep_idx == 0:
                    frame = self.env.render(mode='rgb_array')
                    if frame is not None:
                        ep_frames.append(frame)
                
                critic_obs = build_critic_observation(self.defender_obs, self.attacker_obs)
                obs_t = torch.tensor(self.defender_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                critic_obs_t = torch.tensor(critic_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                with torch.no_grad():
                    if greedy:
                        # 使用forward获取mean，然后tanh
                        mean, _, _ = self.local_network(obs_t, critic_obs_t)
                        action = torch.tanh(mean).cpu().numpy().flatten()
                    else:
                        actions, _, _, _ = self.local_network.act(obs_t, critic_obs_t)
                        action = actions.cpu().numpy().flatten()
                
                attacker_action = self.attacker_policy.get_action(self.attacker_obs)
                obs, reward, terminated, truncated, info = self.env.step(action, attacker_action)
                done = terminated or truncated
                
                self.defender_obs, self.attacker_obs = obs
                self.done = done
                ep_reward += reward
                ep_len += 1
                
                if ep_len >= EnvParameters.EPISODE_LEN:
                    break
            
            one_ep = {
                'episode_reward': ep_reward,
                'num_step': ep_len,
                'win': info.get('win', False)
            }
            update_perf(one_ep, perf)
            perf['win'].append(one_ep['win'])
            
            if record_gif and ep_idx == 0 and ep_frames:
                frames = ep_frames
        
        return {
            'perf': perf,
            'frames': frames
        }
