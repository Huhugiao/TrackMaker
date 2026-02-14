"""
TAD PPO Runner - Ray分布式采样Worker
"""

import os
import time
import ray
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from ppo.alg_parameters import SetupParameters, TrainingParameters, NetParameters, RecordingParameters
from ppo.nets import DefenderNetMLP, create_network
from ppo.util import build_critic_observation, update_perf, get_adjusted_n_envs

import map_config
from map_config import EnvParameters
from env import TADEnv
from rule_policies import AttackerGlobalPolicy, AttackerStaticPolicy
from rule_policies.attacker_global import SUPPORTED_STRATEGIES, TRAINING_STRATEGIES
from rule_policies.defender_global import DefenderGlobalPolicy


ATTACKER_POLICY_REGISTRY = {
    'attacker_global': AttackerGlobalPolicy,
    'attacker_static': AttackerStaticPolicy
}


class RewardNormalizer:
    """
    Running Return Normalization (OpenAI baselines style)
    
    维护一个折扣回报 G_t 的 running variance，用 1/sqrt(var + eps) 缩放 reward。
    只做 scale，不做 shift（不减均值），避免改变最优策略。
    
    原理：
    - 在训练前期，reward 的方差可能很大（如 protect1 的 -100 累积惩罚）
    - 通过除以 sqrt(var) 将 reward 缩放到稳定范围
    - running mean/var 使用 Welford 在线算法更新
    
    warmup_steps: 前 N 步不标准化，用于收集足够的统计数据
    clip_range: 标准化后的 reward 裁剪范围，防止极端值
    """
    def __init__(self, gamma=0.99, epsilon=1e-8, warmup_steps=100, clip_range=10.0):
        self.gamma = gamma
        self.epsilon = epsilon
        self.warmup_steps = warmup_steps
        self.clip_range = clip_range
        # Running statistics (Welford online algorithm)
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        # 折扣回报的 running estimate
        self.ret = 0.0  # 当前 episode 的折扣回报 G_t
    
    def update(self, reward, done):
        """
        更新 running statistics 并返回标准化后的 reward
        
        Args:
            reward: 原始 reward
            done: episode 是否结束
        Returns:
            normalized reward (reward / sqrt(var + eps)), clipped to [-clip_range, clip_range]
        """
        # 更新折扣回报 (当 done 时重置)
        self.ret = self.ret * self.gamma * (1.0 - float(done)) + reward
        
        # Welford online update for mean and variance
        self.count += 1
        delta = self.ret - self.mean
        self.mean += delta / self.count
        delta2 = self.ret - self.mean
        # Numerically stable variance update
        if self.count > 1:
            self.var += (delta * delta2 - self.var) / self.count
        else:
            self.var = 0.0
        # 确保 var >= 0
        self.var = max(self.var, 0.0)
        
        # Warmup: 统计量不稳定时直接返回原始 reward
        if self.count < self.warmup_steps:
            return reward
        
        # Normalize: 只 scale，不 shift
        std = max(self.var ** 0.5, self.epsilon)
        normalized = reward / std
        
        # Clip to prevent extreme values
        normalized = max(-self.clip_range, min(self.clip_range, normalized))
        return normalized
    
    def reset_ret(self):
        """Reset discounted return tracker (called on episode reset)"""
        self.ret = 0.0


@ray.remote(num_cpus=1, num_gpus=0)
class Runner:
    """Runner uses CPU for inference — small MLP is faster on CPU than
    serializing on a contended shared GPU.  GPU is reserved for training only."""
    def __init__(self, meta_agent_id: int, env_configs: Dict = None, network_type: str = 'nmn'):
        self.meta_agent_id = meta_agent_id
        self.env_configs = env_configs or {}
        self.fixed_attacker_strategy = self.env_configs.get('attacker_strategy')
        
        import torch
        # Runner 强制使用 CPU 推理，避免 GPU 竞争
        self.device = torch.device('cpu')
        
        self.local_network = create_network(network_type).to(self.device)
        self.local_network.eval()
        
        self._init_env()
        
        self.opponent_policies = self._create_opponent_policies()
        self.current_opponent_key = None
        
        self._reset()
        
        # 奖励标准化器
        self.reward_normalizer = RewardNormalizer(
            gamma=TrainingParameters.GAMMA
        ) if TrainingParameters.REWARD_NORMALIZATION else None
    
    def _init_env(self):
        if 'episode_len' in self.env_configs and self.env_configs.get('episode_len') is not None:
            EnvParameters.EPISODE_LEN = int(self.env_configs.get('episode_len'))

        reward_mode = str(self.env_configs.get('reward_mode', SetupParameters.SKILL_MODE))
        self.env = TADEnv(reward_mode=reward_mode)

        expert_skill_mode = str(self.env_configs.get('expert_skill_mode', SetupParameters.SKILL_MODE))
        if expert_skill_mode not in ('protect', 'protect1', 'protect2', 'chase'):
            expert_skill_mode = 'chase'
        
        # Create expert policy for imitation learning
        self.expert_policy = DefenderGlobalPolicy(
            env_width=self.env.width,
            env_height=self.env.height,
            defender_speed=self.env.defender_speed,
            defender_max_turn=getattr(map_config, 'defender_max_angular_speed', 6.0),
            skill_mode=expert_skill_mode
        )
    
    def _create_opponent_policies(self) -> Dict[str, Any]:
        """创建对手策略池"""
        # 始终注册两类策略，便于按配置动态切换。
        policies = {
            'attacker_global': ATTACKER_POLICY_REGISTRY['attacker_global'],
            'attacker_static': ATTACKER_POLICY_REGISTRY['attacker_static'],
        }
        return policies
    
    def _sample_opponent_policy(self) -> Tuple[str, Optional[str]]:
        """
        采样对手策略
        
        Returns:
            (policy_key, strategy): policy_key 是 ATTACKER_POLICY_REGISTRY 中的键，
                                   strategy 是 AttackerGlobalPolicy 的具体策略（如 'default', 'zigzag' 等）
        """
        skill_mode = SetupParameters.SKILL_MODE
        
        forced_strategy = self.fixed_attacker_strategy
        if forced_strategy is not None:
            forced = str(forced_strategy).lower()
            if forced == 'static':
                return 'attacker_static', None
            if forced == 'random':
                strategy = np.random.choice(TRAINING_STRATEGIES)
                return 'attacker_global', strategy
            if forced in SUPPORTED_STRATEGIES:
                return 'attacker_global', forced

        if skill_mode == 'protect1':
            return 'attacker_static', None  # 阶段1: 静止对手
        else:
            # protect2, chase, 其他模式: 从支持策略集中随机选择（含switch_random）
            strategy = np.random.choice(TRAINING_STRATEGIES)
            return 'attacker_global', strategy
    
    def _reset(self, for_eval: bool = False, episode_idx: int = 0):
        """
        重置环境和对手策略
        
        Args:
            for_eval: 是否为评估模式（使用评估种子设置）
            episode_idx: 当前episode索引（用于固定种子时区分不同episode）
        """
        policy_key, strategy = self._sample_opponent_policy()
        self.current_opponent_key = policy_key
        policy_cls = self.opponent_policies.get(policy_key)
        
        # Instantiate policy with strategy if applicable
        if policy_key == 'attacker_global' and strategy is not None:
            self.attacker_policy = policy_cls(
                env_width=self.env.width,
                env_height=self.env.height,
                attacker_speed=self.env.attacker_speed,
                attacker_max_turn=getattr(map_config, 'attacker_max_angular_speed', 12.0),
                strategy=strategy
            )
        else:
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
        # 重置标准化器的折扣回报追踪
        if hasattr(self, 'reward_normalizer') and self.reward_normalizer is not None:
            self.reward_normalizer.reset_ret()
    
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
    
    def run(self, num_steps: int, profile: bool = False) -> Dict[str, np.ndarray]:
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

        timings = None
        profiled_keys = (
            'critic_obs', 'tensorize', 'policy_inference', 'expert_policy',
            'opponent_policy', 'env_step', 'reward_norm', 'buffer_ops',
            'episode_reset', 'bootstrap_value', 'gae_compute', 'pack_numpy'
        )
        if profile:
            timings = {k: 0.0 for k in profiled_keys}
            rollout_start = time.perf_counter()
            finished_episodes = 0

        for _ in range(num_steps):
            if profile:
                t0 = time.perf_counter()
            critic_obs = build_critic_observation(self.defender_obs, self.attacker_obs)
            if profile:
                timings['critic_obs'] += time.perf_counter() - t0

            if profile:
                t0 = time.perf_counter()
            obs_t = torch.tensor(self.defender_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            critic_obs_t = torch.tensor(critic_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            if profile:
                timings['tensorize'] += time.perf_counter() - t0

            if profile:
                t0 = time.perf_counter()
            with torch.no_grad():
                actions, log_probs, pre_tanh, values = self.local_network.act(obs_t, critic_obs_t)
            if profile:
                timings['policy_inference'] += time.perf_counter() - t0

            if profile:
                t0 = time.perf_counter()
            tanh_action = actions.cpu().numpy().flatten()
            pre_tanh_action = pre_tanh.cpu().numpy().flatten()
            log_prob = log_probs.cpu().numpy().item()
            value = values.cpu().numpy().item()
            mb_obs.append(self.defender_obs.copy())
            mb_critic_obs.append(critic_obs.copy())
            mb_actions.append(pre_tanh_action)
            mb_log_probs.append(log_prob)
            mb_values.append(value)
            if profile:
                timings['buffer_ops'] += time.perf_counter() - t0

            # Get expert action for IL (skip in pure rl mode)
            if profile:
                t0 = time.perf_counter()
            if TrainingParameters.TRAINING_MODE != 'rl':
                priv_state = self.env.get_privileged_state()
                expert_action = self.expert_policy.get_action(self.defender_obs, priv_state)
            else:
                expert_action = np.zeros(NetParameters.ACTION_DIM, dtype=np.float32)
            mb_expert_actions.append(expert_action)
            if profile:
                timings['expert_policy'] += time.perf_counter() - t0

            # Get attacker action
            if profile:
                t0 = time.perf_counter()
            attacker_action = self.attacker_policy.get_action(self.attacker_obs)
            if profile:
                timings['opponent_policy'] += time.perf_counter() - t0

            # Step environment
            if profile:
                t0 = time.perf_counter()
            obs, reward, terminated, truncated, info = self.env.step(tanh_action, attacker_action)
            if profile:
                timings['env_step'] += time.perf_counter() - t0
            done = terminated or truncated

            self.defender_obs, self.attacker_obs = obs
            self.done = done
            self.episode_reward += reward
            self.episode_len += 1

            # 奖励标准化: reward / sqrt(running_var(G_t))
            if profile:
                t0 = time.perf_counter()
            if self.reward_normalizer is not None:
                norm_reward = self.reward_normalizer.update(reward, done)
            else:
                norm_reward = reward
            if profile:
                timings['reward_norm'] += time.perf_counter() - t0
            mb_rewards.append(norm_reward)
            mb_dones.append(done)

            if done:
                one_ep = {
                    'episode_reward': self.episode_reward,
                    'num_step': self.episode_len,
                    'win': info.get('win', False)
                }
                update_perf(one_ep, perf)
                perf['win'].append(one_ep['win'])
                if profile:
                    t0 = time.perf_counter()
                self._reset()
                if profile:
                    timings['episode_reset'] += time.perf_counter() - t0
                    finished_episodes += 1

        if profile:
            t0 = time.perf_counter()
        last_critic_obs = build_critic_observation(self.defender_obs, self.attacker_obs)
        obs_t = torch.tensor(self.defender_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        critic_obs_t = torch.tensor(last_critic_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            last_value = self.local_network.critic_value(critic_obs_t).cpu().numpy().item()
        if profile:
            timings['bootstrap_value'] += time.perf_counter() - t0

        if profile:
            t0 = time.perf_counter()
        mb_obs = np.array(mb_obs, dtype=np.float32)
        mb_critic_obs = np.array(mb_critic_obs, dtype=np.float32)
        mb_actions = np.array(mb_actions, dtype=np.float32)
        mb_log_probs = np.array(mb_log_probs, dtype=np.float32)
        mb_values = np.array(mb_values, dtype=np.float32)
        mb_rewards = np.array(mb_rewards, dtype=np.float32)
        mb_dones = np.array(mb_dones, dtype=np.float32)
        mb_expert_actions = np.array(mb_expert_actions, dtype=np.float32)
        if profile:
            timings['pack_numpy'] += time.perf_counter() - t0

        mb_advs = np.zeros_like(mb_rewards)
        mb_returns = np.zeros_like(mb_rewards)
        lastgaelam = 0.0

        if profile:
            t0 = time.perf_counter()
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_value = last_value
            else:
                next_value = mb_values[t + 1]

            done_t = mb_dones[t]
            delta = mb_rewards[t] + TrainingParameters.GAMMA * next_value * (1.0 - done_t) - mb_values[t]
            lastgaelam = delta + TrainingParameters.GAMMA * TrainingParameters.LAM * (1.0 - done_t) * lastgaelam
            mb_advs[t] = lastgaelam
        if profile:
            timings['gae_compute'] += time.perf_counter() - t0

        mb_returns = mb_advs + mb_values

        ret = {
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
        if profile:
            profiled_total = sum(timings[k] for k in profiled_keys)
            timings['profiled_total'] = profiled_total
            timings['rollout_total'] = time.perf_counter() - rollout_start
            timings['untracked'] = max(0.0, timings['rollout_total'] - profiled_total)
            timings['num_steps'] = float(num_steps)
            timings['finished_episodes'] = float(finished_episodes)
            ret['timings'] = timings
        return ret

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
        trajectory_data = None  # trajectory data for the first episode (for static plot)
        
        for ep_idx in range(num_episodes):
            self._reset(for_eval=True, episode_idx=ep_idx)
            ep_reward = 0.0
            ep_len = 0
            ep_frames = []
            
            # Record initial positions for first episode's trajectory plot
            record_traj = (ep_idx == 0)
            if record_traj:
                priv = self.env.get_privileged_state()
                target_pos = (priv['target']['center_x'], priv['target']['center_y'])
                ep_def_traj = [(priv['defender']['center_x'], priv['defender']['center_y'])]
                ep_atk_traj = [(priv['attacker']['center_x'], priv['attacker']['center_y'])]
                ep_def_theta = [priv['defender'].get('theta', 0.0)]
                ep_atk_theta = [priv['attacker'].get('theta', 0.0)]
            
            while not self.done:
                if record_gif and ep_idx == 0:
                    frame = self.env.render(mode='rgb_array', style='matplotlib')
                    if frame is not None:
                        ep_frames.append(frame)
                
                critic_obs = build_critic_observation(self.defender_obs, self.attacker_obs)
                obs_t = torch.tensor(self.defender_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                critic_obs_t = torch.tensor(critic_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                with torch.no_grad():
                    if greedy:
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
                
                # Record trajectory
                if record_traj:
                    priv = self.env.get_privileged_state()
                    ep_def_traj.append((priv['defender']['center_x'], priv['defender']['center_y']))
                    ep_atk_traj.append((priv['attacker']['center_x'], priv['attacker']['center_y']))
                    ep_def_theta.append(priv['defender'].get('theta', 0.0))
                    ep_atk_theta.append(priv['attacker'].get('theta', 0.0))
                
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
            
            # Save trajectory data from first episode
            if record_traj:
                import map_config as _mc
                trajectory_data = {
                    'defender_traj': ep_def_traj,
                    'attacker_traj': ep_atk_traj,
                    'defender_theta': ep_def_theta,
                    'attacker_theta': ep_atk_theta,
                    'target_pos': target_pos,
                    'obstacles': list(getattr(_mc, 'obstacles', [])),
                    'width': getattr(_mc, 'width', 640),
                    'height': getattr(_mc, 'height', 640),
                    'win': one_ep['win'],
                    'skill_mode': SetupParameters.SKILL_MODE,
                    'episode_len': ep_len,
                    'episode_reward': ep_reward,
                    'capture_radius': getattr(_mc, 'capture_radius', 20),
                    'capture_sector_angle_deg': getattr(_mc, 'capture_sector_angle_deg', 30),
                }
        
        return {
            'perf': perf,
            'frames': frames,
            'trajectory_data': trajectory_data,
        }
