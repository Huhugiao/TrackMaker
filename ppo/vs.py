"""
D vs A Evaluation Script with Suite & Interactive Modes
Revived and Enhanced.
"""

import os
import sys
import argparse
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
from ppo.alg_parameters import SetupParameters

# Import Rule Policies
from rule_policies import (
    DefenderAPFPolicy,
    AttackerAPFPolicy,
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
    # HRL High-Level Manager
    'hrl': './models/hrl_01-30-17-06/best_model.pth',
    
    # Baseline End-to-End PPO
    'baseline': './models/baseline_01-30-22-19/best_model.pth',
    
    # Pre-trained Skills
    'protect': './models/defender_protect_dense_01-28-11-28/best_model.pth',
    'protect2': './models/defender_protect2_dense_01-29-10-05/best_model.pth',
    'chase': './models/defender_chase_dense_02-02-11-00/best_model.pth' 
}

class Defenderevaluator:
    """Defender Strategy Evaluator Wrapper"""

    def __init__(
        self,
        strategy: str,
        checkpoint_path: Optional[str] = None,
        device: str = 'cpu'
    ):
        self.strategy = strategy
        # 使用安全的GPU检测
        self.device = get_device(prefer_gpu=(device == 'cuda'))
        
        # RL策略列表
        rl_strategies = ['rl', 'hrl', 'baseline', 'protect', 'protect2', 'chase']
        
        # 自动解析checkpoint
        if strategy in rl_strategies and checkpoint_path is None:
            if strategy in DEFAULT_MODEL_PATHS and os.path.exists(DEFAULT_MODEL_PATHS[strategy]):
                checkpoint_path = DEFAULT_MODEL_PATHS[strategy]
                print(f"[Defender] 自动加载模型: {checkpoint_path}")
            else:
                print(f"[Defender] 警告: 未找到 {strategy} 的默认模型")

        self.model = None
        
        if strategy in rl_strategies:
            if checkpoint_path and os.path.exists(checkpoint_path):
                # Load PPO Model
                self.model = Model(self.device, global_model=False)
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                    if 'model' in checkpoint:
                        self.model.set_weights(checkpoint['model'])
                    else:
                        self.model.network.load_state_dict(checkpoint)
                    self.model.network.eval()
                    print(f"[Defender] 已加载RL模型: {checkpoint_path}")
                except Exception as e:
                    print(f"加载模型错误: {e}")
                    raise e
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
        rl_strategies = ['rl', 'hrl', 'baseline', 'protect', 'protect2', 'chase']
        
        if self.strategy in rl_strategies:
            # PPO Model Evaluation
            if hasattr(self.model, 'update_gru_sequence') and hasattr(env, 'env') and hasattr(env.env, 'get_normalized_attacker_info'):
                 try:
                    rel_x_norm, rel_y_norm, is_visible = env.env.get_normalized_attacker_info()
                    self.model.update_gru_sequence(rel_x_norm, rel_y_norm, is_visible)
                 except AttributeError:
                    pass

            # 构建critic观测
            if attacker_obs is not None:
                # 使用完整的defender和attacker观测构建critic_obs
                critic_obs = build_critic_observation(obs, attacker_obs)
            else:
                # 回退到旧方式
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

    def __init__(
        self,
        strategy: str,
    ):
        self.strategy = strategy
        
        if strategy == 'attacker_apf':
            self.model = AttackerAPFPolicy(
                env_width=map_config.width,
                env_height=map_config.height
            )
        elif strategy == 'attacker_global':
            self.model = AttackerGlobalPolicy() 
        elif strategy == 'static':
            self.model = None 
        elif strategy == 'random':
            self.model = None
        else:
            self.model = AttackerGlobalPolicy(strategy=strategy)

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
    seed_offset: int = 0
) -> Tuple[Dict, str]:

    print(f"[{datetime.now().strftime('%H:%M:%S')}] EVAL START: D={defender_strategy} vs A={attacker_strategy}")

    # 1. Environment Selection
    is_gym_wrapper = False
    
    # RL策略使用BaselineEnv（包括各种技能模型）
    rl_strategies = ['rl', 'baseline', 'protect', 'protect2', 'chase']
    
    if defender_strategy == 'hrl':
        protect_path = DEFAULT_MODEL_PATHS.get('protect2')
        if not protect_path or not os.path.exists(protect_path):
             print(f"[警告] Protect模型未找到: {protect_path}")
        
        print(f"[Env] 初始化HRLEnv")
        env = HRLEnv(protect_model_path=protect_path, attacker_strategy=attacker_strategy, device=device)
        is_gym_wrapper = True
    
    elif defender_strategy in rl_strategies:
        print(f"[Env] 初始化BaselineEnv (策略: {defender_strategy})")
        env = BaselineEnv(attacker_strategy=attacker_strategy)
        is_gym_wrapper = True
        
    elif defender_strategy in ['astar_to_attacker', 'astar_to_target']:
        print(f"[Env] 初始化TrackingEnv (A*策略)")
        env = TrackingEnv() 
        is_gym_wrapper = False
        
    else:
        print(f"[Env] 初始化Standard TrackingEnv")
        env = TrackingEnv() 
        is_gym_wrapper = False

    # 2. Obstacles Setup
    set_obstacle_density(map_config.DEFAULT_OBSTACLE_DENSITY)
    map_config.regenerate_obstacles(density_level=map_config.current_obstacle_density)
    env_lib.build_occupancy(
        width=map_config.width,
        height=map_config.height,
        cell=map_config.pixel_size,
        obstacles=getattr(map_config, 'obstacles', [])
    )

    # 3. Evaluators
    defender_eval = Defenderevaluator(defender_strategy, defender_checkpoint, device)
    # Only useful if NOT using gym wrappers that handle attacker
    attacker_eval = Attackerevaluator(attacker_strategy)

    stats = {
        'rewards': [], 'defender_wins': [], 'attacker_wins': [], 'draws': [], 'reasons': [],
        'episode_lengths': [],
        'defender_captures': [],      # defender成功抓住attacker
        'attacker_captures': [],      # attacker成功抓住target  
        'defender_collisions': [],    # defender撞墙/障碍物
        'attacker_collisions': [],    # attacker撞墙/障碍物
    }

    episode_frames = []

    for episode in range(num_episodes):
        if SetupParameters.EVAL_USE_RANDOM_SEED:
            current_seed = None
        else:
            current_seed = SetupParameters.EVAL_FIXED_SEED + episode + seed_offset

        # Reset
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
        frames = []

        while not done:
            def_action = defender_eval.get_action(def_obs, env, att_obs)

            if is_gym_wrapper:
                # Wrappers (HRLEnv/BaselineEnv) usually handle attacker internally
                # and accept only defender action
                output = env.step(def_action)
            else:
                # TrackingEnv needs both
                att_action = attacker_eval.get_action(att_obs)
                output = env.step(action=def_action, attacker_action=att_action)

            # Unpack
            if len(output) == 5:
                next_obs, reward, term, trunc, info = output
            elif len(output) == 4:
                next_obs, reward, done_bool, info = output
                term = done_bool
                trunc = False
            
            if isinstance(next_obs, tuple) and len(next_obs) == 2:
                def_obs, att_obs = next_obs
            else:
                def_obs = next_obs
            
            done = term or trunc
            ep_reward += reward
            
            if save_gif and episode < gif_episodes and len(frames) < 2000:
                try:
                    # 尝试从底层env获取render
                    if hasattr(env, 'env') and hasattr(env.env, 'render'):
                        f = env.env.render(mode='rgb_array')
                    elif hasattr(env, 'render'):
                        f = env.render(mode='rgb_array')
                    else:
                        f = None
                    if f is not None: 
                        frames.append(f)
                except (NotImplementedError, TypeError):
                    pass

        # Record Stats
        stats['rewards'].append(ep_reward)
        reason = info.get('reason', 'unknown')
        stats['reasons'].append(reason)
        
        # 记录episode长度
        ep_len = info.get('episode_length', info.get('step', 0))
        if ep_len == 0 and hasattr(env, 'env'):
            ep_len = getattr(env.env, 'step_count', 0)
        stats['episode_lengths'].append(ep_len)
        
        # 详细分类结果
        defender_capture = 'defender_caught_attacker' in reason
        attacker_capture = 'attacker_caught_target' in reason or 'attacker_win' in reason
        defender_collision = 'defender_collision' in reason or 'defender_out' in reason
        attacker_collision = 'attacker_collision' in reason or 'attacker_out' in reason
        timeout = 'timeout' in reason or 'time_limit' in reason or 'max_steps' in reason or 'truncated' in reason
        
        stats['defender_captures'].append(1 if defender_capture else 0)
        stats['attacker_captures'].append(1 if attacker_capture else 0)
        stats['defender_collisions'].append(1 if defender_collision else 0)
        stats['attacker_collisions'].append(1 if attacker_collision else 0)
        
        # 胜负判定：defender抓住attacker或超时算defender胜利
        if defender_capture or timeout:
            stats['defender_wins'].append(1); stats['attacker_wins'].append(0); stats['draws'].append(0)
        elif attacker_capture:
            stats['defender_wins'].append(0); stats['attacker_wins'].append(1); stats['draws'].append(0)
        else:
            stats['defender_wins'].append(0); stats['attacker_wins'].append(0); stats['draws'].append(1)

        if save_gif and episode < gif_episodes and len(frames) > 0:
            episode_frames.append((episode, reason, frames))

        if (episode + 1) % 10 == 0:
            print(f"  Ep {episode+1}/{num_episodes} | AvgRw: {np.mean(stats['rewards'][-10:]):.2f} | D-Win: {np.mean(stats['defender_wins'][-10:])*100:.0f}%")

    # Final Compilation
    final_results = {
        'defender_strategy': defender_strategy,
        'attacker_strategy': attacker_strategy,
        'episodes': num_episodes,
        'success_rate': np.mean(stats['defender_wins']),  # defender胜率即为成功率
        'defender_win_rate': np.mean(stats['defender_wins']),
        'attacker_win_rate': np.mean(stats['attacker_wins']),
        'defender_capture_rate': np.mean(stats['defender_captures']),
        'attacker_capture_rate': np.mean(stats['attacker_captures']),
        'defender_collision_rate': np.mean(stats['defender_collisions']),
        'attacker_collision_rate': np.mean(stats['attacker_collisions']),
        'mean_episode_length': np.mean(stats['episode_lengths']) if stats['episode_lengths'] else 0,
        'mean_reward': np.mean(stats['rewards']),
        'std_reward': np.std(stats['rewards']),
    }
    
    # Save GIF
    gif_out = None
    if save_gif and episode_frames:
        from util import make_gif
        if not gif_path: gif_path = f"./output/eval_{defender_strategy}_{datetime.now().strftime('%M%S')}.gif"
        
        # Ensure dir
        os.makedirs(os.path.dirname(gif_path) if os.path.dirname(gif_path) else '.', exist_ok=True)
        
        saved_count = 0
        for idx, reason, frames in episode_frames:
            p = gif_path.replace('.gif', f'_ep{idx}_{reason}.gif')
            make_gif(frames, p)
            saved_count += 1
        print(f"  [GIF] Saved {saved_count} gifs")
        gif_out = gif_path

    if save_stats and stats_path:
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        with open(stats_path, 'w') as f:
            json.dump({**final_results, 'raw': stats}, f, indent=2)
        print(f"  [Stats] Saved to {stats_path}")

    return final_results, gif_out

# --- Suite Mode ---
def run_suite(config_list: List[Dict], global_episodes=30, gif_episodes=0):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    suite_dir = f'./output/suite_{timestamp}'
    os.makedirs(suite_dir, exist_ok=True)
    
    summary_results = []
    
    print(f"\n开始批量评估: 共{len(config_list)}个配置")
    
    for i, config in enumerate(config_list):
        d_strat = config['defender']
        a_strat = config['attacker']
        print(f"\n[{i+1}/{len(config_list)}] 配置: 防御者={d_strat} vs 攻击者={a_strat}")
        
        metrics, _ = run_evaluation(
            defender_strategy=d_strat,
            attacker_strategy=a_strat,
            num_episodes=global_episodes,
            defender_checkpoint=DEFAULT_MODEL_PATHS.get(d_strat) if d_strat in DEFAULT_MODEL_PATHS else None,
            save_stats=True,
            stats_path=os.path.join(suite_dir, f'res_{d_strat}_vs_{a_strat}.json'),
            save_gif=gif_episodes > 0,
            gif_episodes=gif_episodes,
            gif_path=os.path.join(suite_dir, f'gif_{d_strat}_vs_{a_strat}.gif')
        )
        
        summary_results.append({
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
        })
    
    # Save Summary CSV
    import csv
    csv_path = os.path.join(suite_dir, 'suite_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_results[0].keys())
        writer.writeheader()
        writer.writerows(summary_results)
    
    print(f"\n评估完成! 汇总保存至 {csv_path}")
    print("\n========== 结果汇总 ==========")
    print(f"{'Defender':<20} {'Attacker':<15} {'胜率':>8} {'D抓获':>8} {'A抓获':>8} {'D碰撞':>8} {'平均步数':>10} {'平均奖励':>10}")
    print("-" * 100)
    for res in summary_results:
        print(f"{res['defender']:<20} {res['attacker']:<15} {res['success_rate']*100:>7.1f}% {res['defender_capture_rate']*100:>7.1f}% {res['attacker_capture_rate']*100:>7.1f}% {res['defender_collision_rate']*100:>7.1f}% {res['mean_episode_length']:>10.1f} {res['mean_reward']:>10.2f}")

# --- 交互式模式 ---
def interactive_suite_mode():
    print("\n=== 防御者 vs 攻击者 评估系统 ===")
    
    defenders = [
        'hrl', 'baseline', 
        'protect', 'protect2', 'chase',
        'astar_to_attacker', 'astar_to_target'
    ]
    # 包含所有训练时使用的attacker策略
    attackers = [
        'random',       # 随机选择以下策略
        'default',      # 默认策略
        'direct',       # 直线冲刺
        'curve',        # 曲线绕行
        'wait_and_attack',  # 等待进攻
        'conservative', # 保守策略
        'zigzag',       # 之字形
        'flank',        # 侧翼包抄
        'orbit',        # 绕行策略
        'aggressive',   # 激进策略
        'stealth',      # 隐蔽策略
        'static'        # 静止不动
    ]
    
    defender_names = {
        'hrl': 'HRL分层强化学习',
        'baseline': 'Baseline端到端PPO',
        'protect': 'Protect技能(v1)',
        'protect2': 'Protect技能(v2)',
        'chase': 'Chase追击技能',
        'astar_to_attacker': 'A*导航-追攻击者',
        'astar_to_target': 'A*导航-守目标'
    }
    attacker_names = {
        'random': '随机(训练用)',
        'default': '默认策略',
        'direct': '直线冲刺',
        'curve': '曲线绕行',
        'wait_and_attack': '等待进攻',
        'conservative': '保守策略',
        'zigzag': '之字形',
        'flank': '侧翼包抄',
        'orbit': '绕行策略',
        'aggressive': '激进策略',
        'stealth': '隐蔽策略',
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
    
    args = parser.parse_args()
    
    # 默认进入交互式界面，除非指定 --no-interactive 或 --suite
    if args.suite:
        # Default fixed suite
        default_suite = [
            {'defender': 'hrl', 'attacker': 'attacker_global'},
            {'defender': 'baseline', 'attacker': 'attacker_global'},
        ]
        run_suite(default_suite, global_episodes=args.episodes)
    elif args.no_interactive:
        # 单次运行模式（命令行参数）
        ckpt = args.checkpoint
        if not ckpt and args.defender in DEFAULT_MODEL_PATHS:
            ckpt = DEFAULT_MODEL_PATHS[args.defender]
            
        run_evaluation(
            defender_strategy=args.defender,
            attacker_strategy=args.attacker,
            num_episodes=args.episodes,
            defender_checkpoint=ckpt,
            save_gif=True if args.gif else False,
            gif_episodes=10 if args.gif else 0
        )
    else:
        # 默认进入交互式界面
        interactive_suite_mode()

if __name__ == "__main__":
    main()
