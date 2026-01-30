"""
D vs A 评估脚本

用于评估Defender和Attacker不同策略组合的性能。

Defender策略:
- rl: 使用RL训练的策略（需要指定checkpoint路径）
- defender_apf: 使用APF规则策略

Attacker策略:
- attacker_apf: 使用APF规则策略
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from datetime import datetime
from typing import Dict, Optional, Tuple

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import map_config
from map_config import EnvParameters, set_obstacle_density
from env import TrackingEnv
import env_lib
from ppo.model import Model
from ppo.alg_parameters import SetupParameters
from rule_policies import (
    DefenderAPFPolicy,
    AttackerAPFPolicy,
    AttackerGlobalPolicy,
)


class Defenderevaluator:
    """Defender策略评估器"""

    def __init__(
        self,
        strategy: str,
        checkpoint_path: Optional[str] = None,
        device: str = 'cuda'
    ):
        self.strategy = strategy
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        if strategy == 'rl':
            if checkpoint_path is None or not os.path.exists(checkpoint_path):
                raise ValueError(f"RL strategy requires checkpoint_path: {checkpoint_path}")
            self.model = Model(self.device, global_model=False)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.model.set_weights(checkpoint['model'])
            self.model.network.eval()
            print(f"[Defender] Loaded RL model from {checkpoint_path}")
        elif strategy == 'defender_apf':
            self.model = DefenderAPFPolicy(
                env_width=map_config.width,
                env_height=map_config.height,
                defender_speed=map_config.defender_speed,
                defender_max_turn=getattr(map_config, 'defender_max_angular_speed', 6.0),
            )
            print(f"[Defender] Using APF policy")
        else:
            raise ValueError(f"Unknown defender strategy: {strategy}")

        self.privileged_state = None

    def reset(self, env: Optional[TrackingEnv] = None):
        """重置评估器状态"""
        if hasattr(self.model, 'reset'):
            self.model.reset()
        if hasattr(self.model, 'reset_gru_sequence'):
            self.model.reset_gru_sequence()
        if env is not None:
            self.privileged_state = env.get_privileged_state()

    def get_action(self, defender_obs: np.ndarray, env: TrackingEnv) -> np.ndarray:
        """获取Defender动作"""
        if self.strategy == 'rl':
            if hasattr(self.model, 'update_gru_sequence'):
                rel_x_norm, rel_y_norm, is_visible = env.get_normalized_attacker_info()
                self.model.update_gru_sequence(rel_x_norm, rel_y_norm, is_visible)
            critic_obs = defender_obs[:70]
            action, _, _, _ = self.model.evaluate(defender_obs, critic_obs, greedy=True)
            return action

        elif self.strategy == 'defender_apf':
            if self.privileged_state:
                defender_pos = np.array([
                    self.privileged_state['defender']['center_x'],
                    self.privileged_state['defender']['center_y']
                ])
                defender_heading = self.privileged_state['defender']['theta']
            else:
                defender_pos = None
                defender_heading = None
            action = self.model.get_action(defender_obs, defender_pos, defender_heading)
            return action

        raise ValueError(f"Unknown defender strategy: {self.strategy}")


class Attackerevaluator:
    """Attacker策略评估器"""

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

        if strategy == 'attacker_apf':
            self.model = AttackerAPFPolicy(
                env_width=env_width,
                env_height=env_height,
                attacker_speed=attacker_speed,
                attacker_max_turn=attacker_max_turn,
            )
            print(f"[Attacker] Using APF policy")
        elif strategy == 'attacker_global':
            self.model = AttackerGlobalPolicy(
                env_width=env_width,
                env_height=env_height,
                attacker_speed=attacker_speed,
                attacker_max_turn=attacker_max_turn,
            )
            print(f"[Attacker] Using Global pathfinding policy (A* navigation)")
        else:
            raise ValueError(f"Unknown attacker strategy: {strategy}")

    def reset(self):
        """重置评估器状态"""
        if hasattr(self.model, 'reset'):
            self.model.reset()

    def get_action(self, attacker_obs: np.ndarray) -> np.ndarray:
        """获取Attacker动作"""
        return self.model.get_action(attacker_obs)


def save_stats_to_file(stats: Dict, output_path: str):
    """保存统计数据到JSON文件"""
    save_data = {
        'metadata': {
            'defender_strategy': stats['defender_strategy'],
            'attacker_strategy': stats['attacker_strategy'],
            'num_episodes': stats['num_episodes'],
            'timestamp': datetime.now().isoformat(),
        },
        'summary': {
            'mean_reward': stats['mean_reward'],
            'std_reward': stats['std_reward'],
            'mean_episode_length': stats['mean_episode_length'],
            'defender_win_rate': stats['defender_win_rate'],
            'attacker_win_rate': stats['attacker_win_rate'],
            'draw_rate': stats['draw_rate'],
            'defender_collision_rate': stats['defender_collision_rate'],
            'attacker_collision_rate': stats['attacker_collision_rate'],
            'mean_closest_attacker_dist': stats['mean_closest_attacker_dist'],
            'mean_closest_target_dist': stats['mean_closest_target_dist'],
        },
        'episodes': []
    }

    raw = stats['raw_stats']
    for i in range(stats['num_episodes']):
        episode_data = {
            'episode_id': i + 1,
            'reward': float(raw['rewards'][i]),
            'episode_length': int(raw['episode_lengths'][i]),
            'defender_win': bool(raw['defender_wins'][i]),
            'attacker_win': bool(raw['attacker_wins'][i]),
            'draw': bool(raw['draws'][i]),
            'defender_collision': bool(raw['defender_collisions'][i]),
            'attacker_collision': bool(raw['attacker_collisions'][i]),
            'closest_attacker_dist': float(raw['closest_attacker_distances'][i]),
            'closest_target_dist': float(raw['closest_target_distances'][i]),
            'reason': raw['reasons'][i],
        }
        save_data['episodes'].append(episode_data)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\n[STATS] Saved to {output_path}")


def run_evaluation(
    defender_strategy: str,
    attacker_strategy: str,
    num_episodes: int = 100,
    defender_checkpoint: Optional[str] = None,
    device: str = 'cuda',
    save_gif: bool = False,
    gif_path: Optional[str] = None,
    gif_episodes: int = 1,
    save_stats: bool = False,
    stats_path: Optional[str] = None,
) -> Tuple[Dict, str]:
    """
    运行D vs A评估

    Args:
        defender_strategy: Defender策略 ('rl', 'defender_apf')
        attacker_strategy: Attacker策略 ('attacker_apf')
        num_episodes: 评估回合数
        defender_checkpoint: RL策略的checkpoint路径
        device: 设备
        save_gif: 是否保存GIF
        gif_path: GIF保存路径（模板路径）
        gif_episodes: 保存前N个episode的GIF（每个episode单独一个文件）
        save_stats: 是否保存统计数据到JSON
        stats_path: 统计数据保存路径

    Returns:
        stats: 统计信息字典
        gif_final_path: GIF保存摘要信息或None
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: D={defender_strategy} vs A={attacker_strategy}")
    print(f"Episodes: {num_episodes}")
    if save_gif and gif_episodes > 0:
        print(f"GIF: Recording first {gif_episodes} episode(s) to {gif_path}")
    print(f"{'='*60}\n")

    # 创建环境
    env = TrackingEnv()

    # 初始化障碍物
    set_obstacle_density(map_config.DEFAULT_OBSTACLE_DENSITY)
    map_config.regenerate_obstacles(density_level=map_config.current_obstacle_density, target_pos=None)
    env_lib.build_occupancy(
        width=map_config.width,
        height=map_config.height,
        cell=getattr(map_config, 'occ_cell', getattr(map_config, 'pixel_size', map_config.pixel_size)),
        obstacles=getattr(map_config, 'obstacles', [])
    )

    # 创建评估器
    defender_eval = Defenderevaluator(
        strategy=defender_strategy,
        checkpoint_path=defender_checkpoint,
        device=device,
    )

    attacker_eval = Attackerevaluator(
        strategy=attacker_strategy,
    )

    # 统计信息
    stats = {
        'rewards': [],
        'episode_lengths': [],
        'defender_wins': [],
        'attacker_wins': [],
        'draws': [],
        'defender_collisions': [],
        'attacker_collisions': [],
        'closest_attacker_distances': [],
        'closest_target_distances': [],
        'reasons': [],
    }

    # 用于保存GIF的帧列表
    episode_frames = []  # 每个episode的帧列表
    max_frames_per_episode = 2000

    for episode in range(num_episodes):
        # 根据配置选择随机或固定种子
        if SetupParameters.EVAL_USE_RANDOM_SEED:
            reset_seed = None  # 随机种子
        else:
            reset_seed = SetupParameters.EVAL_FIXED_SEED + episode  # 固定种子
        (defender_obs, attacker_obs), _ = env.reset(seed=reset_seed)
        defender_eval.reset(env)
        attacker_eval.reset()

        done = False
        episode_reward = 0.0
        episode_length = 0

        # 当前回合的帧
        frames = []

        while not done and episode_length < EnvParameters.EPISODE_LEN:
            defender_action = defender_eval.get_action(defender_obs, env)
            attacker_action = attacker_eval.get_action(attacker_obs)

            (defender_obs, attacker_obs), reward, terminated, truncated, info = env.step(
                action=defender_action,
                attacker_action=attacker_action
            )

            done = terminated or truncated
            episode_reward += float(reward)
            episode_length += 1

            # 保存帧（在指定的回合范围内）
            if save_gif and episode < gif_episodes and len(frames) < max_frames_per_episode:
                frame = env.render(mode='rgb_array')
                if frame is not None:
                    frames.append(frame)

        # 记录统计信息
        stats['rewards'].append(episode_reward)
        stats['episode_lengths'].append(episode_length)
        stats['closest_attacker_distances'].append(info.get('closest_attacker_record_value', 999))
        stats['closest_target_distances'].append(info.get('closest_target_record_value', 999))

        reason = info.get('reason', 'unknown')
        stats['reasons'].append(reason)

        # 记录碰撞
        stats['defender_collisions'].append(1 if reason == 'defender_collision' else 0)
        stats['attacker_collisions'].append(1 if reason == 'attacker_collision' else 0)

        if reason == 'defender_caught_attacker':
            stats['defender_wins'].append(1)
            stats['attacker_wins'].append(0)
            stats['draws'].append(0)
        elif reason == 'attacker_caught_target':
            stats['defender_wins'].append(0)
            stats['attacker_wins'].append(1)
            stats['draws'].append(0)
        else:
            stats['defender_wins'].append(0)
            stats['attacker_wins'].append(0)
            stats['draws'].append(1)

        # 保存当前episode的帧（单独保存）
        if save_gif and episode < gif_episodes and len(frames) > 1:
            episode_frames.append((episode, reason, frames))

        # 打印进度
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(stats['rewards'][-10:])
            win_rate = np.mean(stats['defender_wins'][-10:]) * 100
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"D Win Rate: {win_rate:.1f}%")

    # 保存每个episode的GIF
    gif_paths = []
    if save_gif and len(episode_frames) > 0:
        from util import make_gif

        # 创建输出目录
        output_dir = os.path.dirname(gif_path) if gif_path else '.'
        os.makedirs(output_dir, exist_ok=True)

        for episode_idx, episode_reason, frames in episode_frames:
            # 确定结果标记
            if episode_reason == 'defender_caught_attacker':
                result_tag = 'DWin'
            elif episode_reason == 'attacker_caught_target':
                result_tag = 'AWin'
            elif episode_reason == 'defender_collision':
                result_tag = 'DCollision'
            elif episode_reason == 'attacker_collision':
                result_tag = 'ACollision'
            else:
                result_tag = 'Draw'

            # 生成文件名
            base_name = gif_path.replace('_temp.gif', '').replace('.gif', '') if gif_path else os.path.join(output_dir, 'eval')
            episode_gif_path = f"{base_name}_ep{episode_idx + 1}_{result_tag}.gif"

            # 保存GIF
            make_gif(frames, episode_gif_path, fps=30)
            gif_paths.append(episode_gif_path)
            print(f"[GIF] Episode {episode_idx + 1}: {os.path.basename(episode_gif_path)} ({len(frames)} frames, {result_tag})")

        gif_final_path = f"{len(gif_paths)} GIF(s) saved to {output_dir}"
    elif save_gif and gif_episodes > 0:
        print(f"\n[GIF] Warning: No frames collected (check if rendering is working)")
        gif_final_path = None
    else:
        gif_final_path = None

    # 计算最终统计
    final_stats = {
        'defender_strategy': defender_strategy,
        'attacker_strategy': attacker_strategy,
        'num_episodes': num_episodes,
        'mean_reward': float(np.mean(stats['rewards'])),
        'std_reward': float(np.std(stats['rewards'])),
        'mean_episode_length': float(np.mean(stats['episode_lengths'])),
        'defender_win_rate': float(np.mean(stats['defender_wins'])),
        'attacker_win_rate': float(np.mean(stats['attacker_wins'])),
        'draw_rate': float(np.mean(stats['draws'])),
        'defender_collision_rate': float(np.mean(stats['defender_collisions'])),
        'attacker_collision_rate': float(np.mean(stats['attacker_collisions'])),
        'mean_closest_attacker_dist': float(np.mean(stats['closest_attacker_distances'])),
        'mean_closest_target_dist': float(np.mean(stats['closest_target_distances'])),
        'raw_stats': stats,
    }

    # 保存统计数据
    if save_stats and stats_path:
        save_stats_to_file(final_stats, stats_path)

    return final_stats, gif_final_path


def print_stats(stats: Dict):
    """打印统计信息"""
    print(f"\n{'='*60}")
    print(f"Evaluation Results")
    print(f"{'='*60}")
    print(f"Defender Strategy: {stats['defender_strategy']}")
    print(f"Attacker Strategy: {stats['attacker_strategy']}")
    print(f"Episodes: {stats['num_episodes']}")
    print(f"\nPerformance:")
    print(f"  Mean Reward:     {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"  Mean Ep Length:  {stats['mean_episode_length']:.1f}")
    print(f"\nWin Rates:")
    print(f"  Defender Win:    {stats['defender_win_rate']*100:.1f}%")
    print(f"  Attacker Win:    {stats['attacker_win_rate']*100:.1f}%")
    print(f"  Draw/Timeout:    {stats['draw_rate']*100:.1f}%")
    print(f"\nCollision Rates:")
    print(f"  Defender:        {stats['defender_collision_rate']*100:.1f}%")
    print(f"  Attacker:        {stats['attacker_collision_rate']*100:.1f}%")
    print(f"\nClosest Distances:")
    print(f"  D->A:            {stats['mean_closest_attacker_dist']:.1f}")
    print(f"  A->Target:       {stats['mean_closest_target_dist']:.1f}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate D vs A strategies',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Defender配置
    d_group = parser.add_argument_group('Defender Configuration')
    d_group.add_argument(
        '--defender', '-d',
        type=str,
        default='defender_apf',
        choices=['rl', 'defender_apf'],
        help='Defender strategy'
    )
    d_group.add_argument(
        '--defender-checkpoint',
        type=str,
        default=None,
        help='Path to RL defender model checkpoint'
    )

    # Attacker配置
    a_group = parser.add_argument_group('Attacker Configuration')
    a_group.add_argument(
        '--attacker', '-a',
        type=str,
        default='attacker_apf',
        choices=['attacker_apf', 'attacker_global'],
        help='Attacker strategy'
    )

    # 评估配置
    e_group = parser.add_argument_group('Evaluation Configuration')
    e_group.add_argument('--episodes', '-n', type=int, default=30,
                         help='Number of evaluation episodes')
    e_group.add_argument('--device', type=str, default='cuda',
                         choices=['cuda', 'cpu'],
                         help='Device for RL model')

    # 输出配置
    o_group = parser.add_argument_group('Output Configuration')
    o_group.add_argument('--gif', type=str, default=None,
                         metavar='PATH',
                         help='Path template for GIF files (auto-generates if omitted. Use with --gif-episodes)')
    o_group.add_argument('--gif-episodes', type=int, default=10,
                         metavar='N',
                         help='Save first N episodes as separate GIF files (0 = disabled, default: 10)')
    o_group.add_argument('--output', '-o', type=str, default=None,
                         metavar='PATH',
                         help='Path to save evaluation statistics JSON file (auto-generates if omitted)')

    args = parser.parse_args()

    # 如果使用RL策略但没有提供checkpoint，尝试查找
    if args.defender == 'rl' and args.defender_checkpoint is None:
        default_path = './models/defender_chase/best_model/checkpoint.pth'
        if os.path.exists(default_path):
            args.defender_checkpoint = default_path
            print(f"[INFO] Using default checkpoint: {default_path}")
        else:
            print("[ERROR] RL strategy requires --defender-checkpoint")
            print(f"[ERROR] Default path not found: {default_path}")
            sys.exit(1)

    # 生成时间戳（用于自动命名）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 创建输出目录：output/{defender}_vs_{attacker}_{timestamp}/
    output_dir = f'./output/{args.defender}_vs_{args.attacker}_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    # 统计数据文件路径
    if args.output is None:
        stats_path = os.path.join(output_dir, 'eval_stats.json')
    else:
        stats_path = args.output

    # 处理GIF路径和保存标志
    if args.gif_episodes > 0:
        # 需要录制GIF
        if args.gif is None:
            # 未指定路径，使用默认位置（稍后会根据结果重命名）
            gif_path = os.path.join(output_dir, 'eval_temp.gif')
        else:
            # 用户指定了路径
            gif_path = args.gif
            if os.path.isdir(gif_path):
                # 如果是目录，添加文件名
                gif_path = os.path.join(gif_path, 'eval_temp.gif')
            elif not gif_path.endswith('.gif'):
                # 如果不是.gif扩展名，添加扩展名
                gif_path = f'{gif_path}.gif'
        save_gif = True
    else:
        # 不录制GIF
        save_gif = False
        gif_path = None

    # 运行评估
    stats, gif_final_path = run_evaluation(
        defender_strategy=args.defender,
        attacker_strategy=args.attacker,
        num_episodes=args.episodes,
        defender_checkpoint=args.defender_checkpoint,
        device=args.device,
        save_gif=save_gif,
        gif_path=gif_path,
        gif_episodes=args.gif_episodes,
        save_stats=True,
        stats_path=stats_path,
    )

    # 打印结果
    print_stats(stats)

    # 打印GIF最终路径
    if gif_final_path:
        print(f"[GIF] {gif_final_path}")


if __name__ == '__main__':
    main()
