"""
Single Episode Visualization Script

实时可视化Defender和Attacker的对战过程。

特性:
- 只运行一个episode
- 使用pygame实时渲染
- 显示实时统计信息
- 支持慢速/快速模式（调整仿真速度）
- 按Ctrl+C可中断

Defender策略:
- rl: 使用RL训练的策略（需要指定checkpoint路径）
- astar_to_target: 使用A*导航策略（守护目标）
- astar_to_attacker: 使用A*导航策略（追击攻击者）

Attacker策略:
- attacker_global: 使用全局路径规划策略
- default/evasive: 两种保留的全局策略
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import pygame
from typing import Optional

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import map_config
from configs.map_config import EnvParameters, set_obstacle_density
from configs.paths import CHECKPOINTS_DIR
from envs.tad_env import TrackingEnv
from envs import env_lib
from attacker.learned_policy import LearnedAttackerPolicy
from configs.skill_config import NetParameters
from skill.model import Model
from skill.util import build_critic_observation
from policies import (
    AttackerGlobalPolicy,
    DefenderGlobalPolicy,
)


def _display_network_type(network_type: str) -> str:
    return 'mlp' if str(network_type) == 'mlp_noctde' else str(network_type)


def _detect_network_type(state_dict) -> str:
    keys = set(state_dict.keys())
    if 'belief_predictor.0.weight' in keys:
        return 'nmn_ctde_task_shared_distill'
    if 'shared_tracking_branch.0.weight' in keys:
        return 'nmn_ctde_task_shared'
    if 'shared_radar_encoder.net.0.weight' in keys:
        return 'nmn_ctde_shared'
    if 'actor_radar_encoder.net.0.weight' in keys and 'critic_radar_encoder.net.0.weight' in keys:
        return 'nmn_no_shared_radar'
    has_tracking = any('tracking_branch' in k for k in keys)
    has_actor_backbone = any('actor_backbone' in k for k in keys)
    has_actor_gru = any('actor_gru' in k for k in keys)
    has_hrl_top_marker = ('hrl_top_marker' in keys) or ('discrete_policy_marker' in keys)

    if 'actor_tracking_gru.weight_ih_l0' in keys and 'actor_obstacle_gru.weight_ih_l0' in keys:
        return 'nmn_dual_gru_raw'
    if has_tracking and has_actor_gru:
        return 'nmn_gru'
    if has_actor_gru:
        action_dim = None
        if 'log_std' in state_dict and hasattr(state_dict['log_std'], 'shape'):
            action_dim = int(state_dict['log_std'].shape[0])
        elif 'policy_mean.weight' in state_dict and hasattr(state_dict['policy_mean.weight'], 'shape'):
            action_dim = int(state_dict['policy_mean.weight'].shape[0])
        if has_hrl_top_marker or (action_dim is not None and action_dim >= 3):
            return 'hrl_top_gru'
        return 'mlp_gru'
    if has_tracking:
        critic_input_dim = None
        if 'critic_backbone.0.weight' in state_dict and hasattr(state_dict['critic_backbone.0.weight'], 'shape'):
            critic_input_dim = int(state_dict['critic_backbone.0.weight'].shape[1])
        if critic_input_dim == NetParameters.CRITIC_VECTOR_LEN:
            return 'nmn_ctde'
        return 'nmn'
    if has_actor_backbone:
        critic_input_dim = None
        if 'critic_backbone.0.weight' in state_dict and hasattr(state_dict['critic_backbone.0.weight'], 'shape'):
            critic_input_dim = int(state_dict['critic_backbone.0.weight'].shape[1])
        if critic_input_dim == NetParameters.ACTOR_VECTOR_LEN:
            return 'mlp_noctde'
        return 'mlp_ctde'
    return 'nmn'


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
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
            explicit_network_type = None
            if isinstance(checkpoint, dict) and checkpoint.get('network_type'):
                explicit_network_type = str(checkpoint['network_type']).strip().lower()
            network_type = explicit_network_type or _detect_network_type(state_dict)
            old_hrl_top_action_dim = int(getattr(NetParameters, 'HRL_TOP_ACTION_DIM', 3))
            old_hrl_num_skills = int(getattr(NetParameters, 'HRL_NUM_SKILLS', 2))
            old_hrl_duration_bins = tuple(getattr(NetParameters, 'HRL_DURATION_BINS', (1,)))
            old_hrl_num_duration_bins = int(getattr(NetParameters, 'HRL_NUM_DURATION_BINS', len(old_hrl_duration_bins)))
            old_hrl_top_discrete_action_dim = int(
                getattr(NetParameters, 'HRL_TOP_DISCRETE_ACTION_DIM', old_hrl_num_skills)
            )
            try:
                if isinstance(checkpoint, dict):
                    if checkpoint.get('hrl_num_skills') is not None:
                        NetParameters.HRL_NUM_SKILLS = max(2, int(checkpoint['hrl_num_skills']))
                    if checkpoint.get('hrl_duration_bins') is not None:
                        NetParameters.HRL_DURATION_BINS = tuple(int(v) for v in checkpoint['hrl_duration_bins'])
                        NetParameters.HRL_NUM_DURATION_BINS = len(NetParameters.HRL_DURATION_BINS)
                    else:
                        NetParameters.HRL_DURATION_BINS = (1,)
                        NetParameters.HRL_NUM_DURATION_BINS = 1
                    if checkpoint.get('hrl_top_discrete_action_dim') is not None:
                        NetParameters.HRL_TOP_DISCRETE_ACTION_DIM = int(checkpoint['hrl_top_discrete_action_dim'])
                    else:
                        NetParameters.HRL_TOP_DISCRETE_ACTION_DIM = int(
                            NetParameters.HRL_NUM_SKILLS * NetParameters.HRL_NUM_DURATION_BINS
                        )
                    if checkpoint.get('hrl_top_action_dim') is not None:
                        NetParameters.HRL_TOP_ACTION_DIM = int(checkpoint['hrl_top_action_dim'])
                self.model = Model(self.device, global_model=False, network_type=network_type)
                self.model.set_weights(state_dict)
                self.model.network.eval()
                print(f"[Defender] Loaded RL model ({_display_network_type(network_type)}) from {checkpoint_path}")
            finally:
                NetParameters.HRL_TOP_ACTION_DIM = old_hrl_top_action_dim
                NetParameters.HRL_NUM_SKILLS = old_hrl_num_skills
                NetParameters.HRL_DURATION_BINS = old_hrl_duration_bins
                NetParameters.HRL_NUM_DURATION_BINS = old_hrl_num_duration_bins
                NetParameters.HRL_TOP_DISCRETE_ACTION_DIM = old_hrl_top_discrete_action_dim
        elif strategy == 'astar_to_target':
            self.model = DefenderGlobalPolicy(skill_mode='protect')
            print(f"[Defender] Using A* navigation strategy (protect target)")
        elif strategy == 'astar_to_attacker':
            self.model = DefenderGlobalPolicy(skill_mode='chase')
            print(f"[Defender] Using A* navigation strategy (chase attacker)")
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
            attacker_obs = env.current_obs[1] if isinstance(env.current_obs, tuple) and len(env.current_obs) == 2 else None
            critic_obs = build_critic_observation(defender_obs, attacker_obs)
            action, _, _, _ = self.model.evaluate(defender_obs, critic_obs, greedy=True)
            return action

        elif self.strategy in ['astar_to_target', 'astar_to_attacker']:
            privileged_state = self.privileged_state or env.get_privileged_state()
            return self.model.get_action(defender_obs, privileged_state)

        raise ValueError(f"Unknown defender strategy: {self.strategy}")


class Attackerevaluator:
    """Attacker策略评估器"""

    VALID_STRATEGIES = ['default', 'evasive', 'attacker_global', 'attacker_rl']

    def __init__(
        self,
        strategy: str,
        env_width: float = None,
        env_height: float = None,
        attacker_speed: float = None,
        attacker_max_turn: float = None,
        checkpoint_path: Optional[str] = None,
        reward_style: Optional[str] = None,
        device: str = 'cpu',
    ):
        self.strategy = strategy
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')

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
            print(f"[Attacker] Using Global pathfinding policy (A* navigation)")
        elif strategy in ['default', 'evasive']:
            self.model = AttackerGlobalPolicy(
                env_width=env_width,
                env_height=env_height,
                attacker_speed=attacker_speed,
                attacker_max_turn=attacker_max_turn,
                strategy=strategy
            )
            strategy_names = {
                'default': '默认策略(A*+适度避让)',
                'evasive': '规避策略(避视野)',
            }
            print(f"[Attacker] Using {strategy_names.get(strategy, strategy)} strategy")
        elif strategy == 'attacker_rl':
            if checkpoint_path is None or not os.path.exists(checkpoint_path):
                raise ValueError(f"attacker_rl requires --attacker-checkpoint, got {checkpoint_path!r}")
            self.model = LearnedAttackerPolicy(
                checkpoint_path,
                device=self.device,
                alias='attacker_rl',
                reward_style=reward_style,
            )
            print(f"[Attacker] Using learned PPO attacker: {checkpoint_path}")
        else:
            raise ValueError(f"Unknown attacker strategy: {strategy}. Valid strategies: {self.VALID_STRATEGIES}")

    def reset(self):
        """重置评估器状态"""
        if hasattr(self.model, 'reset'):
            self.model.reset()
        if hasattr(self.model, 'reset_recurrent_state'):
            self.model.reset_recurrent_state()
        if hasattr(self.model, 'reset_gru_sequence'):
            self.model.reset_gru_sequence()

    def get_action(self, attacker_obs: np.ndarray) -> np.ndarray:
        """获取Attacker动作"""
        if self.strategy == 'attacker_rl':
            return np.asarray(self.model.get_action(attacker_obs), dtype=np.float32)
        return self.model.get_action(attacker_obs)


def run_episode_viz(
    defender_strategy: str,
    attacker_strategy: str,
    defender_checkpoint: Optional[str] = None,
    attacker_checkpoint: Optional[str] = None,
    attacker_reward_style: Optional[str] = None,
    device: str = 'cuda',
    fps: int = 30,
    slow_motion: float = 1.0,
    show_info: bool = True,
):
    """
    运行单个episode并实时可视化

    Args:
        defender_strategy: Defender策略 ('rl', 'astar_to_target', 'astar_to_attacker')
        attacker_strategy: Attacker策略 ('attacker_global', 'default', 'evasive')
        defender_checkpoint: RL策略的checkpoint路径
        device: 设备
        fps: 渲染帧率（默认30）
        slow_motion: 慢动作倍数（1.0=正常速度，0.5=2倍慢，2.0=2倍快）
        show_info: 是否显示统计信息
    """
    print(f"\n{'='*60}")
    print(f"Single Episode Visualization")
    print(f"Defender: {defender_strategy}")
    print(f"Attacker: {attacker_strategy}")
    print(f"{'='*60}\n")

    # 设置渲染质量
    map_config.set_render_quality('high')

    # 创建环境（TrackingEnv不支持render_mode参数）
    env = TrackingEnv()

    # 初始化pygame显示窗口
    if pygame is None:
        raise ImportError("pygame is required for visualization")
    pygame.init()
    screen_width = map_config.width
    screen_height = map_config.height
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption(f"Single Episode: D={defender_strategy} vs A={attacker_strategy}")

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
        checkpoint_path=attacker_checkpoint,
        reward_style=attacker_reward_style,
        device=device,
    )

    # 重置环境
    (defender_obs, attacker_obs), _ = env.reset()
    defender_eval.reset(env)
    attacker_eval.reset()

    # 统计信息
    episode_reward = 0.0
    step = 0

    # 渲染窗口标题更新
    frame_delay = 1.0 / fps / slow_motion if slow_motion > 0 else 0

    print("[Running episode... Close window to exit]\n")

    try:
        while True:
            # 获取动作
            defender_action = defender_eval.get_action(defender_obs, env)
            attacker_action = attacker_eval.get_action(attacker_obs)

            # 执行步骤
            (defender_obs, attacker_obs), reward, terminated, truncated, info = env.step(
                action=defender_action,
                attacker_action=attacker_action
            )

            done = terminated or truncated
            episode_reward += float(reward)
            step += 1

            # 更新特权状态（用于rule-based策略）
            if hasattr(env, 'get_privileged_state'):
                defender_eval.privileged_state = env.get_privileged_state()

            # 渲染
            canvas = env.render(mode='rgb_array')
            if canvas is not None:
                # 将canvas转换为pygame surface并显示
                if isinstance(canvas, np.ndarray):
                    # 如果是numpy数组，转换为pygame surface
                    canvas = pygame.surfarray.make_surface(np.transpose(canvas, (1, 0, 2)))
                screen.blit(canvas, (0, 0))
                pygame.display.flip()

            # 处理pygame事件（检查窗口关闭和ESC键）
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("\n[Window closed by user]")
                    done = True
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        print("\n[ESC pressed - Exiting...]")
                        done = True
                        break

            # 显示信息
            if show_info and step % 10 == 0:
                # 从defender_obs中提取attacker信息
                # obs[0]: 到attacker的归一化距离
                # obs[1]: 到attacker的归一化方位角（度）
                # obs[3]: attacker是否在FOV内
                # obs[4]: attacker是否被遮挡
                map_diagonal = (map_config.width**2 + map_config.height**2)**0.5

                dist_norm = defender_obs[0]
                bearing_deg = defender_obs[1] * 180.0  # 归一化值转度数
                in_fov = defender_obs[3] > 0.5
                occluded = defender_obs[4] > 0.5
                is_visible = in_fov and not occluded

                # 反归一化距离
                distance_px = ((dist_norm + 1.0) / 2.0) * map_diagonal

                # 可见性符号
                vis_symbol = "✓" if is_visible else "✗"
                if not in_fov:
                    vis_status = "out_FOV"
                elif occluded:
                    vis_status = "blocked"
                else:
                    vis_status = "visible"

                print(f"Step {step:4d} | R: {episode_reward:6.2f} | "
                      f"D->A: {distance_px:6.1f}px ∠{bearing_deg:6.1f}° | "
                      f"Vis: [{vis_symbol}] {vis_status}")

            # 控制帧率
            if frame_delay > 0:
                time.sleep(frame_delay)

            # 检查结束条件
            if done:
                print(f"\n{'='*60}")
                print(f"Episode Finished!")
                print(f"{'='*60}")
                print(f"Total Steps:      {step}")
                print(f"Total Reward:     {episode_reward:.2f}")
                print(f"Reason:           {info.get('reason', 'unknown')}")
                print(f"Closest D->A:     {info.get('closest_attacker_record_value', 999):.1f}px")
                print(f"Closest A->T:     {info.get('closest_target_record_value', 999):.1f}px")
                print(f"{'='*60}\n")
                print("[Episode complete. Press 'R' to restart or any other key to exit.]")

                # 保持窗口打开，等待用户按键
                waiting = True
                restart_episode = False
                clock = pygame.time.Clock()
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            waiting = False
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_r:
                                print("\n[Restarting episode...]")
                                restart_episode = True
                            waiting = False
                    clock.tick(30)

                if restart_episode:
                    # 重置环境和评估器，开始新的episode
                    (defender_obs, attacker_obs), _ = env.reset()
                    defender_eval.reset(env)
                    attacker_eval.reset()
                    episode_reward = 0.0
                    step = 0
                    print(f"\n{'='*60}\n")
                    continue
                else:
                    break

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] User interrupted (Ctrl+C)")
    finally:
        env.close()
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize single episode D vs A',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Defender配置
    d_group = parser.add_argument_group('Defender Configuration')
    d_group.add_argument(
        '--defender', '-d',
        type=str,
        default='astar_to_target',
        choices=['rl', 'astar_to_target', 'astar_to_attacker'],
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
        default='attacker_global',
        choices=['attacker_global', 'default', 'evasive', 'attacker_rl'],
        help='Attacker strategy'
    )
    a_group.add_argument(
        '--attacker-checkpoint',
        type=str,
        default=None,
        help='Path to learned attacker checkpoint'
    )
    a_group.add_argument(
        '--attacker-reward-style',
        type=str,
        default=None,
        help='Reward-style head for a learned multi-style PPO attacker',
    )

    # 评估配置
    e_group = parser.add_argument_group('Visualization Configuration')
    e_group.add_argument('--device', type=str, default='cuda',
                         choices=['cuda', 'cpu'],
                         help='Device for RL model')
    e_group.add_argument('--fps', type=int, default=30,
                         metavar='FPS',
                         help='Rendering frame rate')
    e_group.add_argument('--slow-motion', type=float, default=1.0,
                         metavar='FACTOR',
                         help='Slow motion factor (1.0=normal, 0.5=2x slower, 2.0=2x faster)')
    e_group.add_argument('--no-info', action='store_true',
                         help='Hide step-by-step info output')

    args = parser.parse_args()

    # 如果使用RL策略但没有提供checkpoint，尝试查找
    if args.defender == 'rl' and args.defender_checkpoint is None:
        default_path = str(CHECKPOINTS_DIR / "defender_chase_nmn_dual_gru_raw_dense_05-05-19-12" / "final_model.pth")
        if os.path.exists(default_path):
            args.defender_checkpoint = default_path
            print(f"[INFO] Using default checkpoint: {default_path}")
        else:
            print("[ERROR] RL strategy requires --defender-checkpoint")
            print(f"[ERROR] Default path not found: {default_path}")
            sys.exit(1)

    if args.attacker == 'attacker_rl' and args.attacker_checkpoint is None:
        print("[ERROR] attacker_rl requires --attacker-checkpoint")
        sys.exit(1)

    # 运行可视化
    run_episode_viz(
        defender_strategy=args.defender,
        attacker_strategy=args.attacker,
        defender_checkpoint=args.defender_checkpoint,
        attacker_checkpoint=args.attacker_checkpoint,
        attacker_reward_style=args.attacker_reward_style,
        device=args.device,
        fps=args.fps,
        slow_motion=args.slow_motion,
        show_info=not args.no_info,
    )


if __name__ == '__main__':
    main()
