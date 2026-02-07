
import os
import ray
import numpy as np
import torch
from typing import Dict, List, Any, Optional

from ppo.alg_parameters import SetupParameters, TrainingParameters, NetParameters, RecordingParameters
from ppo.nets import DefenderNetMLP
from ppo.util import build_critic_observation, update_perf, get_device, get_num_gpus, get_adjusted_n_envs
from hrl.hrl_env import HRLEnv
from map_config import EnvParameters


def _compute_runner_gpu_fraction():
    """计算每个Runner应该分配的GPU比例
    
    所有Runner共享同一个GPU（由SetupParameters.GPU_ID指定）
    每个Runner分配 1/n_envs 的GPU资源份额
    """
    from ppo.util import is_gpu_available
    n_envs = get_adjusted_n_envs(TrainingParameters.N_ENVS)
    if is_gpu_available() and n_envs > 0:
        # 所有Runner共享1个GPU，每个分配 1/n_envs 份额
        return 1.0 / n_envs
    return 0


@ray.remote(num_cpus=1, num_gpus=_compute_runner_gpu_fraction())
class HRLRunner:
    def __init__(self, meta_agent_id: int, env_configs: Dict = None):
        self.meta_agent_id = meta_agent_id
        self.env_configs = env_configs or {}
        
        # 使用安全的设备检测
        self.device = get_device(prefer_gpu=True)
        
        # Top-Level Network (Same logic as DefenderNetMLP)
        self.local_network = DefenderNetMLP().to(self.device)
        self.local_network.eval()
        
        self.protect_model_path = "/home/cyq/miniconda3/envs/lnenv/TrackMaker/models/defender_protect2_dense_01-29-10-05/best_model.pth"
        
        self._init_env()
        self._reset()
    
    def _init_env(self):
        # We can pass attacker strategy via env_configs if we want
        attacker_strat = self.env_configs.get('attacker_strategy', 'default')
        self.env = HRLEnv(
            protect_model_path=self.protect_model_path,
            attacker_strategy=attacker_strat,
            device=str(self.device)
        )
        # Note: HRLEnv handles loading Protect Model and Chasing Policy
        
    def _reset(self, for_eval: bool = False, episode_idx: int = 0):
        if for_eval:
             reset_seed = SetupParameters.EVAL_FIXED_SEED + episode_idx if not SetupParameters.EVAL_USE_RANDOM_SEED else None
        else:
             reset_seed = SetupParameters.SEED + self.meta_agent_id * 100 if not SetupParameters.TRAIN_USE_RANDOM_SEED else None
             
        obs, _ = self.env.reset(seed=reset_seed)
        self.defender_obs, self.attacker_obs = obs
        
        self.done = False
        self.episode_reward = 0.0
        self.episode_len = 0
    
    def set_weights(self, weights):
        state_dict = {}
        for k, v in weights.items():
            if isinstance(v, torch.Tensor):
                state_dict[k] = v.clone().detach().to(self.device)
            else:
                state_dict[k] = torch.as_tensor(v, device=self.device)
        self.local_network.load_state_dict(state_dict)
        self.local_network.eval()
    
    def run(self, num_steps: int) -> Dict[str, np.ndarray]:
        # Collects rollouts for Top Level Policy
        mb_obs = []
        mb_critic_obs = []
        mb_actions = []
        mb_log_probs = []
        mb_values = []
        mb_rewards = []
        mb_dones = []
        
        perf = {'per_r': [], 'per_episode_len': [], 'win': []}
        
        for _ in range(num_steps):
            critic_obs = build_critic_observation(self.defender_obs, self.attacker_obs)
            
            obs_t = torch.tensor(self.defender_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            critic_obs_t = torch.tensor(critic_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            with torch.no_grad():
                # Act returns: action (tanh), log_prob, pre_tanh, value
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
            
            # Step environment handles skills internally
            # Action passed is 2 dims (tanh output)
            obs, reward, terminated, truncated, info = self.env.step(tanh_action)
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
        
        # Calculate advantages (GAE)
        last_critic_obs = build_critic_observation(self.defender_obs, self.attacker_obs)
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
            'expert_actions': None, # No IL for top level yet
            'perf': perf
        }
    
    def evaluate(self, num_episodes: int = 5, greedy: bool = True, record_gif: bool = False) -> Dict:
        perf = {'per_r': [], 'per_episode_len': [], 'win': []}
        frames = []
        
        for ep_idx in range(num_episodes):
            self._reset(for_eval=True, episode_idx=ep_idx)
            ep_reward = 0.0
            ep_len = 0
            ep_frames = []
            
            while not self.done:
                if record_gif and ep_idx == 0:
                    frame = self.env.env.render(mode='rgb_array') # Call inner env render
                    if frame is not None:
                        ep_frames.append(frame)
                        
                critic_obs = build_critic_observation(self.defender_obs, self.attacker_obs)
                obs_t = torch.tensor(self.defender_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                critic_obs_t = torch.tensor(critic_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                 
                with torch.no_grad():
                     mean, _, _ = self.local_network(obs_t, critic_obs_t)
                     if greedy:
                         action = torch.tanh(mean).cpu().numpy().flatten()
                     else:
                         # Sample for eval if not greedy
                         std = torch.exp(self.local_network.log_std)
                         dist = torch.distributions.Normal(mean, std)
                         action = torch.tanh(dist.sample()).cpu().numpy().flatten()
                
                obs, reward, terminated, truncated, info = self.env.step(action)
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
