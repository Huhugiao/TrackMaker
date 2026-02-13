import ray
import numpy as np
import torch
import time
from typing import Dict

from ppo.alg_parameters import SetupParameters
from ppo.nets import create_network
from ppo.util import build_critic_observation, update_perf, get_device
from hrl.hrl_env import HRLEnv
from map_config import EnvParameters


class RewardNormalizer:
    """Running return normalization for rollout rewards."""

    def __init__(self, gamma=0.99, epsilon=1e-8, warmup_steps=100, clip_range=10.0):
        self.gamma = gamma
        self.epsilon = epsilon
        self.warmup_steps = warmup_steps
        self.clip_range = clip_range
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.ret = 0.0

    def update(self, reward, done):
        self.ret = self.ret * self.gamma * (1.0 - float(done)) + reward
        self.count += 1

        delta = self.ret - self.mean
        self.mean += delta / self.count
        delta2 = self.ret - self.mean
        if self.count > 1:
            self.var += (delta * delta2 - self.var) / self.count
        else:
            self.var = 0.0
        self.var = max(self.var, 0.0)

        if self.count < self.warmup_steps:
            return reward

        std = max(self.var ** 0.5, self.epsilon)
        norm_reward = reward / std
        return float(np.clip(norm_reward, -self.clip_range, self.clip_range))

    def reset_ret(self):
        self.ret = 0.0


@ray.remote(num_cpus=1, num_gpus=0)
class HRLRunner:
    def __init__(self, meta_agent_id: int, env_configs: Dict = None):
        self.meta_agent_id = meta_agent_id
        self.env_configs = env_configs or {}
        # Default keeps rollout/inference on CPU. Speed tests can opt-in GPU inference.
        runner_use_gpu = bool(self.env_configs.get('runner_use_gpu', False))
        runner_gpu_id = int(self.env_configs.get('runner_gpu_id', SetupParameters.GPU_ID))
        self.device = get_device(prefer_gpu=runner_use_gpu, gpu_id=runner_gpu_id)

        self.gamma = float(self.env_configs.get('gamma', 0.95))
        self.lam = float(self.env_configs.get('lam', 0.95))

        self.local_network = create_network('hrl_top').to(self.device)
        self.local_network.eval()

        self.reward_normalizer = RewardNormalizer(
            gamma=self.gamma
        ) if bool(self.env_configs.get('reward_normalization', True)) else None

        self._init_env()
        self._reset()

    def _init_env(self):
        if 'episode_len' in self.env_configs and self.env_configs.get('episode_len') is not None:
            EnvParameters.EPISODE_LEN = int(self.env_configs.get('episode_len'))
        self.env = HRLEnv(
            protect_model_path=self.env_configs.get('protect_model_path'),
            chase_model_path=self.env_configs.get('chase_model_path'),
            attacker_strategy=self.env_configs.get('attacker_strategy', 'default'),
            device=str(self.device),
            predictor_hidden_dim=int(self.env_configs.get('predictor_hidden_dim', 64)),
            predictor_lr=float(self.env_configs.get('predictor_lr', 1e-3)),
            predictor_train=bool(self.env_configs.get('predictor_train', True)),
            hold_min=int(self.env_configs.get('hold_min', 3)),
            hold_max=int(self.env_configs.get('hold_max', 15)),
            macro_gamma=self.gamma,
            disable_hold_control=bool(self.env_configs.get('disable_hold_control', False)),
        )

    def _reset(self, for_eval: bool = False, episode_idx: int = 0):
        if for_eval:
            reset_seed = SetupParameters.EVAL_FIXED_SEED + episode_idx if not SetupParameters.EVAL_USE_RANDOM_SEED else None
            self.env.set_predictor_train(False)
        else:
            reset_seed = SetupParameters.SEED + self.meta_agent_id * 100 if not SetupParameters.TRAIN_USE_RANDOM_SEED else None
            self.env.set_predictor_train(True)

        obs, _ = self.env.reset(seed=reset_seed)
        self.defender_obs, self.attacker_obs = obs

        self.done = False
        self.episode_reward = 0.0
        self.episode_len = 0
        if self.reward_normalizer is not None:
            self.reward_normalizer.reset_ret()

    def set_weights(self, weights):
        state_dict = {}
        for k, v in weights.items():
            if isinstance(v, torch.Tensor):
                state_dict[k] = v.clone().detach().to(self.device)
            else:
                state_dict[k] = torch.as_tensor(v, device=self.device)
        self.local_network.load_state_dict(state_dict)
        self.local_network.eval()

    def run(self, num_steps: int, profile: bool = False) -> Dict[str, np.ndarray]:
        mb_obs, mb_critic_obs, mb_actions = [], [], []
        mb_log_probs, mb_values, mb_rewards, mb_dones = [], [], [], []
        mb_discounts, mb_gae_discounts = [], []
        predictor_losses = []
        macro_lengths = []
        total_macro_steps = 0

        perf = {'per_r': [], 'per_episode_len': [], 'win': []}
        finished_episodes = 0

        if profile:
            timings = {
                'critic_obs': 0.0,
                'tensorize': 0.0,
                'policy_inference': 0.0,
                'buffer_ops': 0.0,
                'env_macro_step': 0.0,
                'reward_norm': 0.0,
                'episode_reset': 0.0,
                'bootstrap_value': 0.0,
                'pack_numpy': 0.0,
                'gae_compute': 0.0,
            }
            profiled_keys = list(timings.keys())
            rollout_start = time.perf_counter()

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
            top_action = actions.cpu().numpy().flatten()
            pre_tanh_action = pre_tanh.cpu().numpy().flatten()
            log_prob = float(log_probs.cpu().numpy().item())
            value = float(values.cpu().numpy().item())

            mb_obs.append(self.defender_obs.copy())
            mb_critic_obs.append(critic_obs.copy())
            mb_actions.append(pre_tanh_action)
            mb_log_probs.append(log_prob)
            mb_values.append(value)
            if profile:
                timings['buffer_ops'] += time.perf_counter() - t0

            if profile:
                t0 = time.perf_counter()
            obs, macro_reward, terminated, truncated, info = self.env.macro_step(top_action)
            if profile:
                timings['env_macro_step'] += time.perf_counter() - t0
            done = terminated or truncated

            macro_steps = int(info.get('macro_steps', 1))
            discount = float(self.gamma ** macro_steps)
            gae_discount = float((self.gamma * self.lam) ** macro_steps)
            raw_reward_sum = float(info.get('raw_reward_sum', macro_reward))
            total_macro_steps += macro_steps

            self.defender_obs, self.attacker_obs = obs
            self.done = done
            self.episode_reward += raw_reward_sum
            self.episode_len += macro_steps

            if profile:
                t0 = time.perf_counter()
            if self.reward_normalizer is not None:
                train_reward = self.reward_normalizer.update(macro_reward, done)
            else:
                train_reward = macro_reward
            if profile:
                timings['reward_norm'] += time.perf_counter() - t0

            mb_rewards.append(float(train_reward))
            mb_dones.append(done)
            mb_discounts.append(discount)
            mb_gae_discounts.append(gae_discount)

            predictor_losses.append(float(info.get('predictor_loss', 0.0)))
            macro_lengths.append(macro_steps)

            if done:
                one_ep = {
                    'episode_reward': self.episode_reward,
                    'num_step': self.episode_len,
                    'win': info.get('win', False),
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
        mb_discounts = np.array(mb_discounts, dtype=np.float32)
        mb_gae_discounts = np.array(mb_gae_discounts, dtype=np.float32)
        if profile:
            timings['pack_numpy'] += time.perf_counter() - t0

        mb_advs = np.zeros_like(mb_rewards)

        lastgaelam = 0.0
        if profile:
            t0 = time.perf_counter()
        for t in reversed(range(num_steps)):
            next_value = last_value if t == num_steps - 1 else mb_values[t + 1]
            done_t = mb_dones[t]
            delta = mb_rewards[t] + mb_discounts[t] * next_value * (1.0 - done_t) - mb_values[t]
            lastgaelam = delta + mb_gae_discounts[t] * (1.0 - done_t) * lastgaelam
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
            'expert_actions': None,
            'predictor_loss': float(np.mean(predictor_losses)) if predictor_losses else 0.0,
            'macro_len_mean': float(np.mean(macro_lengths)) if macro_lengths else 0.0,
            'perf': perf,
        }
        if profile:
            profiled_total = sum(timings[k] for k in profiled_keys)
            timings['profiled_total'] = profiled_total
            timings['rollout_total'] = time.perf_counter() - rollout_start
            timings['untracked'] = max(0.0, timings['rollout_total'] - profiled_total)
            timings['num_steps'] = float(num_steps)
            timings['num_primitive_steps'] = float(total_macro_steps)
            timings['finished_episodes'] = float(finished_episodes)
            ret['timings'] = timings
        return ret

    def evaluate(self, num_episodes: int = 5, greedy: bool = True, record_gif: bool = False) -> Dict:
        perf = {'per_r': [], 'per_episode_len': [], 'win': []}
        frames = []
        trajectory_data = None
        predictor_losses = []
        macro_lengths = []

        for ep_idx in range(num_episodes):
            self._reset(for_eval=True, episode_idx=ep_idx)
            ep_reward = 0.0
            ep_len = 0
            ep_frames = []
            info = {'win': False}

            # Record the first episode for static trajectory plot.
            record_traj = (ep_idx == 0)
            if record_traj:
                priv = self.env.env.get_privileged_state()
                target_pos = (priv['target']['center_x'], priv['target']['center_y'])
                ep_def_traj = [(priv['defender']['center_x'], priv['defender']['center_y'])]
                ep_atk_traj = [(priv['attacker']['center_x'], priv['attacker']['center_y'])]
                ep_def_theta = [priv['defender'].get('theta', 0.0)]
                ep_atk_theta = [priv['attacker'].get('theta', 0.0)]

            while not self.done and ep_len < EnvParameters.EPISODE_LEN:
                if record_gif and ep_idx == 0:
                    frame = self.env.env.render(mode='rgb_array', style='matplotlib')
                    if frame is not None:
                        ep_frames.append(frame)

                critic_obs = build_critic_observation(self.defender_obs, self.attacker_obs)
                obs_t = torch.tensor(self.defender_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                critic_obs_t = torch.tensor(critic_obs, dtype=torch.float32, device=self.device).unsqueeze(0)

                with torch.no_grad():
                    mean, _, log_std = self.local_network(obs_t, critic_obs_t)
                    if greedy:
                        action = torch.tanh(mean).cpu().numpy().flatten()
                    else:
                        std = torch.exp(log_std)
                        pre_tanh = mean + std * torch.randn_like(mean)
                        action = torch.tanh(pre_tanh).cpu().numpy().flatten()

                obs, _macro_reward, terminated, truncated, info = self.env.macro_step(action)
                done = terminated or truncated

                macro_steps = int(info.get('macro_steps', 1))
                ep_reward += float(info.get('raw_reward_sum', _macro_reward))
                ep_len += macro_steps
                predictor_losses.append(float(info.get('predictor_loss', 0.0)))
                macro_lengths.append(macro_steps)

                self.defender_obs, self.attacker_obs = obs
                self.done = done

                if record_traj:
                    priv = self.env.env.get_privileged_state()
                    ep_def_traj.append((priv['defender']['center_x'], priv['defender']['center_y']))
                    ep_atk_traj.append((priv['attacker']['center_x'], priv['attacker']['center_y']))
                    ep_def_theta.append(priv['defender'].get('theta', 0.0))
                    ep_atk_theta.append(priv['attacker'].get('theta', 0.0))

            one_ep = {
                'episode_reward': ep_reward,
                'num_step': ep_len,
                'win': info.get('win', False),
            }
            update_perf(one_ep, perf)
            perf['win'].append(one_ep['win'])

            if record_gif and ep_idx == 0 and ep_frames:
                frames = ep_frames

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
            'predictor_loss': float(np.mean(predictor_losses)) if predictor_losses else 0.0,
            'macro_len_mean': float(np.mean(macro_lengths)) if macro_lengths else 0.0,
        }
