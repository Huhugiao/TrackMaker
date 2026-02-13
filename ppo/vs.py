"""
D vs A Evaluation Script with Suite & Interactive Modes
Revived and Enhanced.
"""

import os
import sys
import argparse
import glob
import json
import time
import numpy as np
import torch
from datetime import datetime
from typing import Dict, Optional, Tuple, List, Union

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import map_config
from map_config import EnvParameters, set_obstacle_density
import env_lib
from ppo.model import Model
from ppo.util import build_critic_observation, get_device, print_device_info
from ppo.alg_parameters import SetupParameters, NetParameters

# Import Rule Policies
from rule_policies import (
    AttackerGlobalPolicy,
    DefenderGlobalPolicy
)

# Import Environments
# Note: We use specific environments for different strategies to ensure correct observation/action spaces
from hrl.hrl_env import HRLEnv
from hrl.baseline_env import BaselineEnv
from env import TADEnv, TrackingEnv  # Fallback/Standard

# --- Default Paths Configuration ---
DEFAULT_MODEL_PATHS = {
    # HRL High-Level Manager (MLP)
    'hrl': './models/hrl_02-12-14-30/best_model.pth',
    
    # Baseline End-to-End PPO (MLP)
    'baseline': './models/baseline_01-30-22-19/best_model.pth',
    
    # NMN技能模型
    'protect2': './models/defender_protect2_dense_02-11-17-34/best_model.pth',
    'chase': './models/defender_chase_dense_02-11-20-33/best_model.pth',
}

ALL_ATTACKER_STRATEGIES = ['default', 'evasive', 'orbit']

import ray
from ppo.util import get_adjusted_n_envs, get_ray_temp_dir


def _find_latest_checkpoint(model_prefixes: List[str]) -> Optional[str]:
    candidates = []
    for prefix in model_prefixes:
        pattern = os.path.join('models', f'{prefix}_*', 'best_model.pth')
        candidates.extend(glob.glob(pattern))
    candidates = [p for p in candidates if os.path.isfile(p)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _default_model_path(strategy: str) -> Optional[str]:
    path = DEFAULT_MODEL_PATHS.get(strategy)
    if path and os.path.exists(path):
        return path

    prefix_map = {
        'hrl': ['hrl'],
        'baseline': ['baseline'],
        'protect': ['defender_protect_dense', 'defender_protect1_dense', 'defender_protect2_dense'],
        'protect2': ['defender_protect2_dense', 'defender_protect1_dense', 'defender_protect_dense'],
        'chase': ['defender_chase_dense'],
        'protect1_nmn': ['defender_protect1_dense'],
    }
    prefixes = prefix_map.get(strategy)
    if not prefixes:
        return None
    return _find_latest_checkpoint(prefixes)


def _resolve_hrl_skill_paths(
    protect_path: Optional[str] = None,
    chase_path: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    protect_path = protect_path or _default_model_path('protect2')
    chase_path = chase_path or _default_model_path('chase')
    return protect_path, chase_path



def _init_ray():
    """初始化Ray集群（与训练使用相同配置）。"""
    if ray.is_initialized():
        return
    ray_tmp = get_ray_temp_dir()
    ray_num_cpus = os.cpu_count() or 4
    ray_num_gpus = 0  # 评估不使用GPU
    print(f"[Ray] Init with {ray_num_cpus} CPUs for evaluation")
    kwargs = dict(
        num_cpus=ray_num_cpus,
        num_gpus=ray_num_gpus,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False,
    )
    if ray_tmp:
        kwargs['_temp_dir'] = ray_tmp
    ray.init(**kwargs)


@ray.remote
class EvalWorker:
    """Ray远程评估Worker - 每个worker独立运行多个episode。"""

    def __init__(self, worker_id: int):
        self.worker_id = worker_id

    def run_episodes(
        self,
        defender_strategy: str,
        attacker_strategy: str,
        num_episodes: int,
        defender_checkpoint: str,
        network_type: str,
        seed_offset: int,
    ) -> dict:
        """在worker中独立运行 num_episodes 个episode并返回统计数据。"""
        import numpy as np
        import map_config
        from map_config import set_obstacle_density
        import env_lib
        from env import TADEnv, TrackingEnv
        from hrl.hrl_env import HRLEnv
        from hrl.baseline_env import BaselineEnv

        # 环境初始化
        is_gym_wrapper = False
        rl_strategies = ['rl', 'baseline', 'protect', 'protect2', 'chase', 'protect1_nmn']

        if defender_strategy == 'hrl':
            protect_path, chase_path = _resolve_hrl_skill_paths()
            if not protect_path:
                raise FileNotFoundError('HRL评估缺少protect skill checkpoint')
            env = HRLEnv(
                protect_model_path=protect_path,
                chase_model_path=chase_path,
                attacker_strategy=attacker_strategy,
                device='cpu',
            )
            is_gym_wrapper = True
        elif defender_strategy in rl_strategies:
            env = BaselineEnv(attacker_strategy=attacker_strategy)
            is_gym_wrapper = True
        elif defender_strategy in ['astar_to_attacker', 'astar_to_target']:
            env = TrackingEnv()
        else:
            env = TrackingEnv()

        # 障碍物
        set_obstacle_density(map_config.DEFAULT_OBSTACLE_DENSITY)
        map_config.regenerate_obstacles(density_level=map_config.current_obstacle_density)
        env_lib.build_occupancy(
            width=map_config.width, height=map_config.height,
            cell=map_config.pixel_size,
            obstacles=getattr(map_config, 'obstacles', []),
        )

        # Evaluators
        defender_eval = Defenderevaluator(defender_strategy, defender_checkpoint, 'cpu', network_type=network_type)
        attacker_eval = Attackerevaluator(attacker_strategy)

        stats = {
            'rewards': [], 'defender_wins': [], 'attacker_wins': [], 'draws': [],
            'reasons': [], 'episode_lengths': [],
            'defender_captures': [], 'attacker_captures': [],
            'defender_collisions': [], 'attacker_collisions': [],
            'hrl_skill_protect_selected': [], 'hrl_skill_chase_selected': [],
        }

        for ep in range(num_episodes):
            if SetupParameters.EVAL_USE_RANDOM_SEED:
                current_seed = None
            else:
                current_seed = SetupParameters.EVAL_FIXED_SEED + ep + seed_offset

            obs, info = env.reset(seed=current_seed)
            if isinstance(obs, tuple) and len(obs) == 2:
                def_obs, att_obs = obs
            else:
                def_obs = obs
                att_obs = obs

            defender_eval.reset(env)
            attacker_eval.reset()
            done = False
            ep_reward = 0

            while not done:
                def_action = defender_eval.get_action(def_obs, env, att_obs)
                if is_gym_wrapper:
                    output = env.step(def_action)
                else:
                    att_action = attacker_eval.get_action(att_obs)
                    output = env.step(action=def_action, attacker_action=att_action)

                if len(output) == 5:
                    next_obs, reward, term, trunc, info = output
                elif len(output) == 4:
                    next_obs, reward, done_bool, info = output
                    term = done_bool; trunc = False

                if defender_strategy == 'hrl':
                    selected_skill = info.get('selected_skill')
                    if selected_skill == 'protect':
                        stats['hrl_skill_protect_selected'].append(1)
                        stats['hrl_skill_chase_selected'].append(0)
                    elif selected_skill == 'chase':
                        stats['hrl_skill_protect_selected'].append(0)
                        stats['hrl_skill_chase_selected'].append(1)
                    elif info.get('top_skill_idx') == 0:
                        stats['hrl_skill_protect_selected'].append(1)
                        stats['hrl_skill_chase_selected'].append(0)
                    elif info.get('top_skill_idx') == 1:
                        stats['hrl_skill_protect_selected'].append(0)
                        stats['hrl_skill_chase_selected'].append(1)

                if isinstance(next_obs, tuple) and len(next_obs) == 2:
                    def_obs, att_obs = next_obs
                else:
                    def_obs = next_obs

                done = term or trunc
                ep_reward += reward

            # 记录
            stats['rewards'].append(float(ep_reward))
            reason = info.get('reason', 'unknown')
            stats['reasons'].append(reason)
            ep_len = info.get('episode_length', info.get('step', 0))
            if ep_len == 0 and hasattr(env, 'env'):
                ep_len = getattr(env.env, 'step_count', 0)
            stats['episode_lengths'].append(int(ep_len))

            d_cap = 'defender_caught_attacker' in reason
            a_cap = 'attacker_caught_target' in reason or 'attacker_win' in reason
            d_col = 'defender_collision' in reason or 'defender_out' in reason
            a_col = 'attacker_collision' in reason or 'attacker_out' in reason
            timeout = 'timeout' in reason or 'time_limit' in reason or 'max_steps' in reason or 'truncated' in reason

            stats['defender_captures'].append(1 if d_cap else 0)
            stats['attacker_captures'].append(1 if a_cap else 0)
            stats['defender_collisions'].append(1 if d_col else 0)
            stats['attacker_collisions'].append(1 if a_col else 0)

            if d_cap or timeout:
                stats['defender_wins'].append(1); stats['attacker_wins'].append(0); stats['draws'].append(0)
            elif a_cap:
                stats['defender_wins'].append(0); stats['attacker_wins'].append(1); stats['draws'].append(0)
            else:
                stats['defender_wins'].append(0); stats['attacker_wins'].append(0); stats['draws'].append(1)

        return stats


def _merge_stats(all_stats: list) -> dict:
    """合并多个worker返回的stats字典。"""
    merged = {}
    for key in all_stats[0]:
        merged[key] = []
        for s in all_stats:
            merged[key].extend(s[key])
    return merged


def _load_checkpoint(checkpoint_path: str):
    """加载checkpoint并返回 (state_dict, network_type, arch_info)。"""
    try:
        # numpy版本兼容: 旧numpy(< 2.0)没有_core子包
        import numpy as _np
        if not hasattr(_np, '_core'):
            import sys as _sys
            _sys.modules['numpy._core'] = _np.core
            _sys.modules['numpy._core.multiarray'] = _np.core.multiarray

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        keys = set(state_dict.keys())
        has_tracking = any('tracking_branch' in k for k in keys)
        has_actor_backbone = any('actor_backbone' in k for k in keys)

        if has_tracking:
            net_type = 'nmn'
        elif has_actor_backbone:
            # 区分普通MLP(2D action)与HRL顶层MLP(3D action)。
            action_dim = None
            if 'log_std' in state_dict and hasattr(state_dict['log_std'], 'shape'):
                action_dim = int(state_dict['log_std'].shape[0])
            elif 'policy_mean.weight' in state_dict and hasattr(state_dict['policy_mean.weight'], 'shape'):
                action_dim = int(state_dict['policy_mean.weight'].shape[0])
            net_type = 'hrl_top' if action_dim == 3 else 'mlp'
        else:
            net_type = 'nmn'

        arch_info = {
            'hidden_dim': None,
            'action_dim': None,
        }
        if 'actor_backbone.0.weight' in state_dict and hasattr(state_dict['actor_backbone.0.weight'], 'shape'):
            arch_info['hidden_dim'] = int(state_dict['actor_backbone.0.weight'].shape[0])
        if 'log_std' in state_dict and hasattr(state_dict['log_std'], 'shape'):
            arch_info['action_dim'] = int(state_dict['log_std'].shape[0])
        elif 'policy_mean.weight' in state_dict and hasattr(state_dict['policy_mean.weight'], 'shape'):
            arch_info['action_dim'] = int(state_dict['policy_mean.weight'].shape[0])

        return state_dict, net_type, arch_info
    except Exception as e:
        print(f"[警告] 无法加载checkpoint: {e}")
        return None, 'nmn', {'hidden_dim': None, 'action_dim': None}


class Defenderevaluator:
    """Defender Strategy Evaluator Wrapper"""

    def __init__(
        self,
        strategy: str,
        checkpoint_path: Optional[str] = None,
        device: str = 'cpu',
        network_type: Optional[str] = None
    ):
        self.strategy = strategy
        # 使用安全的GPU检测
        self.device = get_device(prefer_gpu=(device == 'cuda'))
        
        # RL策略列表
        rl_strategies = ['rl', 'hrl', 'baseline', 'protect', 'protect2', 'chase', 'protect1_nmn']
        
        # 自动解析checkpoint
        if strategy in rl_strategies and checkpoint_path is None:
            auto_path = _default_model_path(strategy)
            if auto_path is not None:
                checkpoint_path = auto_path
                print(f"[Defender] 自动加载模型: {checkpoint_path}")
            else:
                print(f"[Defender] 警告: 未找到 {strategy} 的默认模型")

        self.model = None
        
        if strategy in rl_strategies:
            if checkpoint_path and os.path.exists(checkpoint_path):
                # 单次加载: 同时检测网络类型和获取权重
                state_dict, detected_type, arch_info = _load_checkpoint(checkpoint_path)
                if network_type is None:
                    network_type = detected_type
                self.network_type = network_type

                # 若checkpoint与当前全局网络超参不一致，临时按checkpoint规格构建网络。
                old_hidden = NetParameters.HIDDEN_DIM
                old_action_dim = NetParameters.ACTION_DIM
                if arch_info.get('hidden_dim') is not None:
                    NetParameters.HIDDEN_DIM = int(arch_info['hidden_dim'])
                if network_type == 'hrl_top':
                    # create_network('hrl_top') 内部会强制 action_dim=3
                    NetParameters.ACTION_DIM = 2
                elif arch_info.get('action_dim') is not None:
                    NetParameters.ACTION_DIM = int(arch_info['action_dim'])

                self.model = Model(self.device, global_model=False, network_type=network_type)
                try:
                    if state_dict is not None:
                        self.model.set_weights(state_dict)
                    self.model.network.eval()
                    print(f"[Defender] 已加载RL模型 (网络={network_type}): {checkpoint_path}")
                except Exception as e:
                    print(f"加载模型错误: {e}")
                    raise e
                finally:
                    NetParameters.HIDDEN_DIM = old_hidden
                    NetParameters.ACTION_DIM = old_action_dim
            else:
                raise ValueError(f"策略 {strategy} 需要有效的checkpoint")

        elif strategy == 'astar_to_attacker':
            # A*导航到攻击者
            self.model = DefenderGlobalPolicy(skill_mode='chase')
            print(f"[Defender] 使用A*导航策略(追击攻击者)")
            
        elif strategy == 'astar_to_target':
            # A*导航到目标
            self.model = DefenderGlobalPolicy(skill_mode='protect')
            print(f"[Defender] 使用A*导航策略(守护目标)")
            
        else:
            raise ValueError(f"未知的defender策略: {strategy}")

    def reset(self, env: Optional[object] = None):
        """Reset evaluator state"""
        if hasattr(self.model, 'reset'):
            self.model.reset()
        if hasattr(self.model, 'reset_gru_sequence'):
            self.model.reset_gru_sequence()

    def get_action(self, obs: np.ndarray, env: object, attacker_obs: np.ndarray = None) -> np.ndarray:
        """Get action from policy"""
        # RL策略（包括各种技能模型）
        rl_strategies = ['rl', 'hrl', 'baseline', 'protect', 'protect2', 'chase', 'protect1_nmn']
        
        if self.strategy in rl_strategies:
            # PPO Model Evaluation
            # 兼容不同网络结构（MLP/NMN）
            if hasattr(self.model, 'update_gru_sequence'):
                try:
                    if hasattr(env, 'env') and hasattr(env.env, 'get_normalized_attacker_info'):
                        rel_x_norm, rel_y_norm, is_visible = env.env.get_normalized_attacker_info()
                        self.model.update_gru_sequence(rel_x_norm, rel_y_norm, is_visible)
                    elif hasattr(env, 'get_normalized_attacker_info'):
                        rel_x_norm, rel_y_norm, is_visible = env.get_normalized_attacker_info()
                        self.model.update_gru_sequence(rel_x_norm, rel_y_norm, is_visible)
                except (AttributeError, TypeError):
                    pass

            # 构建critic观测 - 兼容新的build_critic_observation签名
            try:
                if attacker_obs is not None:
                    # 使用完整的defender和attacker观测构建critic_obs
                    critic_obs = build_critic_observation(obs, attacker_obs)
                else:
                    # 回退到旧方式
                    critic_obs = build_critic_observation(obs)
            except Exception:
                # 如果构建失败，直接使用obs
                critic_obs = obs
            
            with torch.no_grad():
                action, _, _, _ = self.model.evaluate(obs, critic_obs, greedy=True)
            return action

        elif self.strategy in ['astar_to_attacker', 'astar_to_target']:
            # A*导航策略，需要获取privileged state
            if hasattr(env, 'get_privileged_state'):
                p_state = env.get_privileged_state()
            elif hasattr(env, 'env') and hasattr(env.env, 'get_privileged_state'):
                p_state = env.env.get_privileged_state()
            else:
                p_state = None
            
            if p_state:
                return self.model.get_action(obs, p_state)
            else:
                return np.zeros(2)

        return np.zeros(2)


class Attackerevaluator:
    """Attacker Strategy Evaluator"""

    # 支持的策略列表
    VALID_STRATEGIES = ['default', 'evasive', 'orbit', 'switch_random',
                        'attacker_apf', 'attacker_global', 'static', 'random']

    def __init__(
        self,
        strategy: str,
        env_width: float = None,
        env_height: float = None,
        attacker_speed: float = None,
        attacker_max_turn: float = None,
    ):
        self.strategy = strategy
        
        # 使用传入参数或默认值
        env_width = env_width if env_width is not None else map_config.width
        env_height = env_height if env_height is not None else map_config.height
        attacker_speed = attacker_speed if attacker_speed is not None else map_config.attacker_speed
        attacker_max_turn = attacker_max_turn if attacker_max_turn is not None else getattr(map_config, 'attacker_max_angular_speed', 12.0)
        
        if strategy == 'attacker_global':
            self.model = AttackerGlobalPolicy(
                env_width=env_width,
                env_height=env_height,
                attacker_speed=attacker_speed,
                attacker_max_turn=attacker_max_turn,
            )
        elif strategy in ['default', 'evasive', 'orbit', 'switch_random']:
            # 核心策略 + 周期切换策略
            self.model = AttackerGlobalPolicy(
                env_width=env_width,
                env_height=env_height,
                attacker_speed=attacker_speed,
                attacker_max_turn=attacker_max_turn,
                strategy=strategy
            )
        elif strategy == 'static':
            self.model = None 
        elif strategy == 'random':
            self.model = None
        else:
            raise ValueError(f"Unknown attacker strategy: {strategy}. Valid strategies: {self.VALID_STRATEGIES}")

    def reset(self):
        if hasattr(self.model, 'reset'):
            self.model.reset()

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if self.strategy in ['static', 'random']:
            return np.zeros(2) # Random usually handled by env, valid 0 for static
        return self.model.get_action(obs)


# --- Core Evaluation Function ---
def run_evaluation(
    defender_strategy: str,
    attacker_strategy: str,
    num_episodes: int = 100,
    defender_checkpoint: Optional[str] = None,
    device: str = 'cpu',
    save_gif: bool = False,
    gif_path: Optional[str] = None,
    gif_episodes: int = 1,
    save_stats: bool = False,
    stats_path: Optional[str] = None,
    seed_offset: int = 0,
    network_type: Optional[str] = None,
    force_serial: bool = False
) -> Tuple[Dict, str]:

    print(f"[{datetime.now().strftime('%H:%M:%S')}] EVAL START: D={defender_strategy} vs A={attacker_strategy}")

    # 自动检测网络类型（单次加载）
    if network_type is None and defender_checkpoint:
        _, network_type, _ = _load_checkpoint(defender_checkpoint)

    # ---- Ray 并行评估 ----
    n_workers = get_adjusted_n_envs(4)  # 基数4，高内存时自动扩展
    # GIF模式需要串行（需要render），其他情况使用Ray并行
    use_parallel = (not force_serial) and (not save_gif) and num_episodes > 1

    if use_parallel:
        _init_ray()
        # 将episodes均匀分配给workers
        n_workers = min(n_workers, num_episodes)
        episodes_per_worker = [num_episodes // n_workers] * n_workers
        for i in range(num_episodes % n_workers):
            episodes_per_worker[i] += 1

        workers = [EvalWorker.remote(i) for i in range(n_workers)]
        print(f"[Ray] 启动 {n_workers} 个评估worker, 共 {num_episodes} episodes")

        futures = []
        ep_offset = seed_offset
        for w, n_ep in zip(workers, episodes_per_worker):
            if n_ep > 0:
                futures.append(w.run_episodes.remote(
                    defender_strategy, attacker_strategy, n_ep,
                    defender_checkpoint, network_type, ep_offset,
                ))
            ep_offset += n_ep

        all_stats = ray.get(futures)
        stats = _merge_stats(all_stats)

        # 清理workers
        del workers

    else:
        # 串行模式（GIF或单episode）
        stats = _run_serial_evaluation(
            defender_strategy, attacker_strategy, num_episodes,
            defender_checkpoint, device, network_type,
            save_gif, gif_episodes, seed_offset,
        )

    # Final Compilation
    hrl_protect_count = int(sum(stats.get('hrl_skill_protect_selected', [])))
    hrl_chase_count = int(sum(stats.get('hrl_skill_chase_selected', [])))
    hrl_total_count = hrl_protect_count + hrl_chase_count
    if hrl_total_count > 0:
        hrl_protect_selection_rate = hrl_protect_count / hrl_total_count
        hrl_chase_selection_rate = hrl_chase_count / hrl_total_count
    else:
        hrl_protect_selection_rate = None
        hrl_chase_selection_rate = None

    final_results = {
        'defender_strategy': defender_strategy,
        'attacker_strategy': attacker_strategy,
        'episodes': num_episodes,
        'success_rate': np.mean(stats['defender_wins']),
        'defender_win_rate': np.mean(stats['defender_wins']),
        'attacker_win_rate': np.mean(stats['attacker_wins']),
        'defender_capture_rate': np.mean(stats['defender_captures']),
        'attacker_capture_rate': np.mean(stats['attacker_captures']),
        'defender_collision_rate': np.mean(stats['defender_collisions']),
        'attacker_collision_rate': np.mean(stats['attacker_collisions']),
        'mean_episode_length': np.mean(stats['episode_lengths']) if stats['episode_lengths'] else 0,
        'mean_reward': np.mean(stats['rewards']),
        'std_reward': np.std(stats['rewards']),
        'hrl_protect_selection_rate': hrl_protect_selection_rate,
        'hrl_chase_selection_rate': hrl_chase_selection_rate,
    }

    # GIF handling (only in serial mode)
    gif_out = None
    episode_frames = stats.get('_frames', [])
    if save_gif and episode_frames:
        from util import make_gif, make_trajectory_plot
        if not gif_path:
            gif_path = f"./output/{defender_strategy}_vs_{attacker_strategy}.gif"
        os.makedirs(os.path.dirname(gif_path) if os.path.dirname(gif_path) else '.', exist_ok=True)
        saved_count = 0
        for idx, reason, frames in episode_frames:
            p = gif_path.replace('.gif', f'_ep{idx}_{reason}.gif')
            make_gif(frames, p, fps=20, quality='high')
            saved_count += 1
        print(f"  [GIF] Saved {saved_count} gifs")
        gif_out = gif_path

    if save_stats and stats_path:
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        stats_to_save = {k: v for k, v in stats.items() if k != '_frames'}
        with open(stats_path, 'w') as f:
            json.dump({**final_results, 'raw': stats_to_save}, f, indent=2)
        print(f"  [Stats] Saved to {stats_path}")

    d_wr = final_results['defender_win_rate'] * 100
    a_wr = final_results['attacker_win_rate'] * 100
    mode_str = f"parallel({n_workers}w)" if use_parallel else "serial"
    print(f"  RESULT [{mode_str}]: D-win={d_wr:.1f}% A-win={a_wr:.1f}% "
          f"mean_len={final_results['mean_episode_length']:.0f} "
          f"mean_rw={final_results['mean_reward']:.2f}")
    if defender_strategy == 'hrl' and hrl_total_count > 0:
        print(
            f"  HRL Skill选择占比: protect={hrl_protect_selection_rate*100:.1f}% "
            f"chase={hrl_chase_selection_rate*100:.1f}%"
        )

    return final_results, gif_out


def _run_serial_evaluation(
    defender_strategy, attacker_strategy, num_episodes,
    defender_checkpoint, device, network_type,
    save_gif, gif_episodes, seed_offset,
) -> dict:
    """串行运行episodes（用于GIF保存或单episode模式）。"""
    # Environment Selection
    is_gym_wrapper = False
    rl_strategies = ['rl', 'baseline', 'protect', 'protect2', 'chase', 'protect1_nmn']

    if defender_strategy == 'hrl':
        protect_path, chase_path = _resolve_hrl_skill_paths()
        if not protect_path:
            raise FileNotFoundError('HRL评估缺少protect skill checkpoint')
        env = HRLEnv(
            protect_model_path=protect_path,
            chase_model_path=chase_path,
            attacker_strategy=attacker_strategy,
            device=device,
        )
        is_gym_wrapper = True
    elif defender_strategy in rl_strategies:
        env = BaselineEnv(attacker_strategy=attacker_strategy)
        is_gym_wrapper = True
    elif defender_strategy in ['astar_to_attacker', 'astar_to_target']:
        env = TrackingEnv()
    else:
        env = TrackingEnv()

    set_obstacle_density(map_config.DEFAULT_OBSTACLE_DENSITY)
    map_config.regenerate_obstacles(density_level=map_config.current_obstacle_density)
    env_lib.build_occupancy(
        width=map_config.width, height=map_config.height,
        cell=map_config.pixel_size,
        obstacles=getattr(map_config, 'obstacles', []),
    )

    defender_eval = Defenderevaluator(defender_strategy, defender_checkpoint, device, network_type=network_type)
    attacker_eval = Attackerevaluator(attacker_strategy)

    stats = {
        'rewards': [], 'defender_wins': [], 'attacker_wins': [], 'draws': [],
        'reasons': [], 'episode_lengths': [],
        'defender_captures': [], 'attacker_captures': [],
        'defender_collisions': [], 'attacker_collisions': [],
        'hrl_skill_protect_selected': [], 'hrl_skill_chase_selected': [],
    }
    episode_frames = []

    for episode in range(num_episodes):
        if SetupParameters.EVAL_USE_RANDOM_SEED:
            current_seed = None
        else:
            current_seed = SetupParameters.EVAL_FIXED_SEED + episode + seed_offset

        obs, info = env.reset(seed=current_seed)
        if isinstance(obs, tuple) and len(obs) == 2:
            def_obs, att_obs = obs
        else:
            def_obs = obs; att_obs = obs

        defender_eval.reset(env)
        attacker_eval.reset()
        done = False
        ep_reward = 0
        frames = []

        while not done:
            def_action = defender_eval.get_action(def_obs, env, att_obs)
            if is_gym_wrapper:
                output = env.step(def_action)
            else:
                att_action = attacker_eval.get_action(att_obs)
                output = env.step(action=def_action, attacker_action=att_action)

            if len(output) == 5:
                next_obs, reward, term, trunc, info = output
            elif len(output) == 4:
                next_obs, reward, done_bool, info = output
                term = done_bool; trunc = False

            if defender_strategy == 'hrl':
                selected_skill = info.get('selected_skill')
                if selected_skill == 'protect':
                    stats['hrl_skill_protect_selected'].append(1)
                    stats['hrl_skill_chase_selected'].append(0)
                elif selected_skill == 'chase':
                    stats['hrl_skill_protect_selected'].append(0)
                    stats['hrl_skill_chase_selected'].append(1)
                elif info.get('top_skill_idx') == 0:
                    stats['hrl_skill_protect_selected'].append(1)
                    stats['hrl_skill_chase_selected'].append(0)
                elif info.get('top_skill_idx') == 1:
                    stats['hrl_skill_protect_selected'].append(0)
                    stats['hrl_skill_chase_selected'].append(1)

            if isinstance(next_obs, tuple) and len(next_obs) == 2:
                def_obs, att_obs = next_obs
            else:
                def_obs = next_obs

            done = term or trunc
            ep_reward += reward

            if save_gif and episode < gif_episodes and len(frames) < 2000:
                try:
                    if hasattr(env, 'env') and hasattr(env.env, 'render'):
                        f = env.env.render(mode='rgb_array', style='matplotlib')
                    elif hasattr(env, 'render'):
                        f = env.render(mode='rgb_array', style='matplotlib')
                    else:
                        f = None
                    if f is not None:
                        frames.append(f)
                except (NotImplementedError, TypeError):
                    pass

        # Record Stats
        stats['rewards'].append(float(ep_reward))
        reason = info.get('reason', 'unknown')
        stats['reasons'].append(reason)
        ep_len = info.get('episode_length', info.get('step', 0))
        if ep_len == 0 and hasattr(env, 'env'):
            ep_len = getattr(env.env, 'step_count', 0)
        stats['episode_lengths'].append(int(ep_len))

        d_cap = 'defender_caught_attacker' in reason
        a_cap = 'attacker_caught_target' in reason or 'attacker_win' in reason
        d_col = 'defender_collision' in reason or 'defender_out' in reason
        a_col = 'attacker_collision' in reason or 'attacker_out' in reason
        timeout = 'timeout' in reason or 'time_limit' in reason or 'max_steps' in reason or 'truncated' in reason

        stats['defender_captures'].append(1 if d_cap else 0)
        stats['attacker_captures'].append(1 if a_cap else 0)
        stats['defender_collisions'].append(1 if d_col else 0)
        stats['attacker_collisions'].append(1 if a_col else 0)

        if d_cap or timeout:
            stats['defender_wins'].append(1); stats['attacker_wins'].append(0); stats['draws'].append(0)
        elif a_cap:
            stats['defender_wins'].append(0); stats['attacker_wins'].append(1); stats['draws'].append(0)
        else:
            stats['defender_wins'].append(0); stats['attacker_wins'].append(0); stats['draws'].append(1)

        if save_gif and episode < gif_episodes and len(frames) > 0:
            episode_frames.append((episode, reason, frames))

        if (episode + 1) % 10 == 0:
            print(f"  Ep {episode+1}/{num_episodes} | AvgRw: {np.mean(stats['rewards'][-10:]):.2f} | D-Win: {np.mean(stats['defender_wins'][-10:])*100:.0f}%")

    if episode_frames:
        stats['_frames'] = episode_frames
    return stats

# --- Suite Mode ---
def _expand_all_attacker_configs(config_list: List[Dict], global_episodes: int) -> List[Dict]:
    """将attacker=all扩展为动态攻击策略（不含static），每种沿用该配置episodes。"""
    expanded = []
    for config in config_list:
        attacker = config.get('attacker')
        # 兼容历史配置: random 视为 all
        if attacker in ['all', 'random']:
            per_attacker_episodes = int(config.get('episodes', global_episodes))
            for attacker_name in ALL_ATTACKER_STRATEGIES:
                item = dict(config)
                item['attacker'] = attacker_name
                item['episodes'] = per_attacker_episodes
                expanded.append(item)
        else:
            expanded.append(dict(config))
    return expanded


def run_suite(config_list: List[Dict], global_episodes=30, gif_episodes=0):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    suite_dir = f'./output/suite_{timestamp}'
    os.makedirs(suite_dir, exist_ok=True)

    expanded_configs = _expand_all_attacker_configs(config_list, global_episodes)
    if len(expanded_configs) != len(config_list):
        print(
            f"[Suite] 检测到 attacker=all，已展开为{len(ALL_ATTACKER_STRATEGIES)}种动态策略（不含static），"
            f"每种使用该配置的episodes"
        )

    deduped_configs = []
    seen = set()
    for cfg in expanded_configs:
        key = (cfg['defender'], cfg['attacker'], int(cfg.get('episodes', global_episodes)))
        if key in seen:
            continue
        seen.add(key)
        deduped_configs.append(cfg)
    if len(deduped_configs) != len(expanded_configs):
        print("[Suite] 检测到重复对阵，已自动去重")
    expanded_configs = deduped_configs

    summary_results = []
    matchup_results = []

    print(f"\n开始批量评估: 共{len(expanded_configs)}个配置")

    for i, config in enumerate(expanded_configs):
        d_strat = config['defender']
        a_strat = config['attacker']
        n_episodes = int(config.get('episodes', global_episodes))
        checkpoint_path = config.get('checkpoint')
        if checkpoint_path is None:
            checkpoint_path = _default_model_path(d_strat)
        print(
            f"\n[{i+1}/{len(expanded_configs)}] 配置: 防御者={d_strat} "
            f"vs 攻击者={a_strat} | episodes={n_episodes}"
        )
        print(f"  checkpoint={checkpoint_path}")

        metrics, _ = run_evaluation(
            defender_strategy=d_strat,
            attacker_strategy=a_strat,
            num_episodes=n_episodes,
            defender_checkpoint=checkpoint_path,
            save_stats=True,
            stats_path=os.path.join(suite_dir, f'res_{d_strat}_vs_{a_strat}.json'),
            save_gif=gif_episodes > 0,
            gif_episodes=gif_episodes,
            gif_path=os.path.join(suite_dir, f'{d_strat}_vs_{a_strat}.gif')
        )

        row = {
            'defender': d_strat,
            'attacker': a_strat,
            'episodes': metrics['episodes'],
            'success_rate': metrics['success_rate'],
            'defender_capture_rate': metrics['defender_capture_rate'],
            'attacker_capture_rate': metrics['attacker_capture_rate'],
            'defender_collision_rate': metrics['defender_collision_rate'],
            'attacker_collision_rate': metrics['attacker_collision_rate'],
            'mean_episode_length': metrics['mean_episode_length'],
            'mean_reward': metrics['mean_reward'],
            'std_reward': metrics['std_reward'],
            'hrl_protect_selection_rate': metrics['hrl_protect_selection_rate'],
            'hrl_chase_selection_rate': metrics['hrl_chase_selection_rate'],
        }
        summary_results.append(row)
        matchup_results.append(row.copy())

    # Save Summary CSV
    import csv
    csv_path = os.path.join(suite_dir, 'suite_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_results[0].keys())
        writer.writeheader()
        writer.writerows(summary_results)

    # Save Matchup CSV (同一个CSV内按attacker分表)
    matchup_csv_path = os.path.join(suite_dir, 'suite_matchups.csv')
    fieldnames = list(summary_results[0].keys())

    attacker_order = []
    for cfg in expanded_configs:
        attacker = cfg['attacker']
        if attacker not in attacker_order:
            attacker_order.append(attacker)

    with open(matchup_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for idx, attacker in enumerate(attacker_order):
            writer.writerow([f'attacker={attacker}'])
            writer.writerow(fieldnames)

            rows = [r for r in matchup_results if r['attacker'] == attacker]
            rows = sorted(rows, key=lambda x: x['defender'])
            for row in rows:
                writer.writerow([row[k] for k in fieldnames])

            if idx != len(attacker_order) - 1:
                writer.writerow([])

    # 评估完成后关闭Ray
    if ray.is_initialized():
        ray.shutdown()

    print(f"\n评估完成! 汇总保存至 {csv_path}")
    print(f"对阵明细保存至 {matchup_csv_path} (按attacker分表)")
    print("\n========== 结果汇总 ==========")
    print(
        f"{'Defender':<20} {'Attacker':<15} {'胜率':>8} {'D抓获':>8} {'A抓获':>8} "
        f"{'D碰撞':>8} {'平均步数':>10} {'平均奖励':>10} {'Prot%':>8} {'Chase%':>8}"
    )
    print("-" * 120)
    for res in summary_results:
        protect_rate = res['hrl_protect_selection_rate']
        chase_rate = res['hrl_chase_selection_rate']
        protect_str = f"{protect_rate * 100:>7.1f}%" if protect_rate is not None else f"{'-':>8}"
        chase_str = f"{chase_rate * 100:>7.1f}%" if chase_rate is not None else f"{'-':>8}"
        print(
            f"{res['defender']:<20} {res['attacker']:<15} {res['success_rate']*100:>7.1f}% "
            f"{res['defender_capture_rate']*100:>7.1f}% {res['attacker_capture_rate']*100:>7.1f}% "
            f"{res['defender_collision_rate']*100:>7.1f}% {res['mean_episode_length']:>10.1f} "
            f"{res['mean_reward']:>10.2f} {protect_str} {chase_str}"
        )

# --- 交互式模式 ---
def interactive_suite_mode():
    print("\n=== 防御者 vs 攻击者 评估系统 ===")
    
    # 仅保留当前使用的NMN相关评估策略
    defenders = ['hrl', 'protect2', 'chase']
    # 包含所有训练时使用的attacker策略（3种核心策略）+ 周期切换策略 + all + static
    attackers = [
        'all',          # 展开为3种动态策略（不含static），每种回合数=用户输入
        'default',      # 默认策略：A*寻路 + 适度避让
        'evasive',      # 规避策略：最大化距离并避开Defender视野
        'orbit',        # 轨道等待：绕Target运动寻找时机
        'switch_random',# 随机周期在3种核心策略中切换
        'static'        # 静止不动
    ]
    
    defender_names = {
        'hrl': 'HRL分层策略(高层调度)',
        'protect2': 'Protect技能(NMN)',
        'chase': 'Chase技能(NMN)'
    }
    attacker_names = {
        'all': '全量动态策略(3种, 不含static; 每种回合数=你输入的episodes)',
        'default': '默认策略(A*+适度避让)',
        'evasive': '规避策略(避视野)',
        'orbit': '轨道等待',
        'switch_random': '周期切换(随机周期, 3选1)',
        'static': '静止不动'
    }
    
    def multi_select(options, names, prompt):
        """多选函数，返回选中的选项列表"""
        print(f"\n{prompt} (输入序号，多选用逗号分隔，如1,2,3 或输入 a 全选):")
        for i, opt in enumerate(options):
            print(f"  {i+1}. {opt} ({names[opt]})")
        
        choice = input("请输入: ").strip().lower()
        if choice == 'a':
            return options.copy()
        
        selected = []
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(',')]
            for idx in indices:
                if 0 <= idx < len(options):
                    selected.append(options[idx])
        except ValueError:
            pass
        return selected
    
    # 选择防御者（可多选）
    selected_defenders = multi_select(defenders, defender_names, "选择防御者策略")
    if not selected_defenders:
        print("未选择任何防御者，退出。")
        return
    print(f"已选防御者: {[defender_names[d] for d in selected_defenders]}")
    
    # 选择攻击者（可多选）
    selected_attackers = multi_select(attackers, attacker_names, "选择攻击者策略")
    if not selected_attackers:
        print("未选择任何攻击者，退出。")
        return
    print(f"已选攻击者: {[attacker_names[a] for a in selected_attackers]}")
    
    # 设置评估回合数
    ep_input = input("\n评估回合数 (默认30): ").strip()
    episodes = int(ep_input) if ep_input.isdigit() else 30
    
    # 设置GIF数量
    gif_input = input("保存GIF数量 (默认0，不保存): ").strip()
    gif_count = int(gif_input) if gif_input.isdigit() else 0
    
    # 生成所有组合
    configs = []
    for d in selected_defenders:
        for a in selected_attackers:
            configs.append({'defender': d, 'attacker': a})
    
    print(f"\n========================================")
    print(f"评估配置汇总:")
    print(f"  防御者: {len(selected_defenders)}个")
    print(f"  攻击者: {len(selected_attackers)}个")
    print(f"  总组合: {len(configs)}个")
    print(f"  每组回合数: {episodes}")
    print(f"  GIF数量: {gif_count}")
    print(f"========================================")
    
    for i, c in enumerate(configs):
        print(f"  {i+1}. {defender_names[c['defender']]} vs {attacker_names[c['attacker']]}")
    
    confirm = input("\n确认开始评估? (y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消。")
        return
    
    print(f"\n开始评估...")
    run_suite(configs, global_episodes=episodes, gif_episodes=gif_count)

def main():
    parser = argparse.ArgumentParser(description="评估脚本 - 防御者 vs 攻击者")
    parser.add_argument('--suite', action='store_true', help="运行预设套件模式")
    parser.add_argument('--no-interactive', action='store_true', help="跳过交互模式，使用命令行参数")
    
    # 命令行参数（用于非交互模式）
    parser.add_argument('--defender', '-d', default='hrl', help="防御者策略")
    parser.add_argument('--attacker', '-a', default='attacker_global', help="攻击者策略")
    parser.add_argument('--episodes', '-n', type=int, default=30, help="评估回合数")
    parser.add_argument('--gif', action='store_true', help="保存GIF")
    parser.add_argument('--checkpoint', type=str, default=None, help="模型检查点路径")
    parser.add_argument('--network-type', type=str, default=None, choices=['nmn', 'hrl_top'],
                        help="网络类型: nmn/hrl_top，不指定则自动检测")
    parser.add_argument('--save-stats', action='store_true', help="保存评估统计JSON")
    parser.add_argument('--serial', action='store_true', help="禁用Ray并行，使用串行评估（调试用）")
    parser.add_argument('--stats-path', type=str, default=None, help="统计JSON输出路径")
    
    args = parser.parse_args()
    
    # 默认进入交互式界面，除非指定 --no-interactive 或 --suite
    if args.suite:
        # Default fixed suite
        default_suite = [
            {'defender': 'chase', 'attacker': 'all'},
            {'defender': 'protect2', 'attacker': 'all'},
            {'defender': 'hrl', 'attacker': 'all'},
        ]
        run_suite(default_suite, global_episodes=args.episodes)
    elif args.no_interactive:
        if args.attacker == 'all':
            if not args.checkpoint:
                raise ValueError("attacker=all 时必须通过 --checkpoint 指定模型路径")
            print("[CLI] attacker=all 将展开为3种动态策略（不含static），每种使用 --episodes")
            run_suite(
                [{
                    'defender': args.defender,
                    'attacker': 'all',
                    'episodes': args.episodes,
                    'checkpoint': args.checkpoint,
                }],
                global_episodes=args.episodes,
                gif_episodes=10 if args.gif else 0,
            )
            return

        # 单次运行模式（命令行参数）
        ckpt = args.checkpoint
        if not ckpt:
            ckpt = _default_model_path(args.defender)
            
        if args.defender == 'hrl' and args.network_type is None:
            args.network_type = 'hrl_top'

        metrics, _ = run_evaluation(
            defender_strategy=args.defender,
            attacker_strategy=args.attacker,
            num_episodes=args.episodes,
            defender_checkpoint=ckpt,
            save_gif=True if args.gif else False,
            gif_episodes=10 if args.gif else 0,
            save_stats=args.save_stats,
            stats_path=args.stats_path,
            network_type=args.network_type,
            force_serial=args.serial
        )
        print(
            f"[EVAL RESULT] success={metrics['success_rate']:.3f}, "
            f"d_cap={metrics['defender_capture_rate']:.3f}, "
            f"a_cap={metrics['attacker_capture_rate']:.3f}, "
            f"mean_len={metrics['mean_episode_length']:.1f}, "
            f"mean_reward={metrics['mean_reward']:.3f}"
        )
        if args.defender == 'hrl' and metrics['hrl_protect_selection_rate'] is not None:
            print(
                f"[EVAL HRL SKILL] protect={metrics['hrl_protect_selection_rate']:.3f}, "
                f"chase={metrics['hrl_chase_selection_rate']:.3f}"
            )
        if ray.is_initialized():
            ray.shutdown()
    else:
        # 默认进入交互式界面
        interactive_suite_mode()

if __name__ == "__main__":
    main()
