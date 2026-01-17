"""
TAD PPO Driver - 主训练循环

支持模式:
- 'il': 纯模仿学习
- 'rl': 纯强化学习
- 'mixed': IL+RL混合训练，IL权重使用余弦退火
"""

import os
import os.path as osp
import time
import math
import numpy as np
import torch
import ray
import sys
from datetime import datetime
from typing import Dict, List, Optional

# Add parent directory to sys.path to allow running as a script
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from ppo.alg_parameters import SetupParameters, TrainingParameters, NetParameters, RecordingParameters
from ppo.model import Model
from ppo.runner import Runner
from ppo.util import set_global_seeds, write_to_tensorboard, make_gif

from map_config import EnvParameters


def cosine_anneal_il_weight(current_step: int) -> float:
    """
    计算当前步数对应的IL权重（余弦退火）
    
    Args:
        current_step: 当前训练步数
        
    Returns:
        IL权重，范围 [IL_FINAL_WEIGHT, IL_INITIAL_WEIGHT]
    """
    initial = TrainingParameters.IL_INITIAL_WEIGHT
    final = TrainingParameters.IL_FINAL_WEIGHT
    anneal_steps = TrainingParameters.IL_ANNEAL_STEPS
    
    if current_step >= anneal_steps:
        return final
    
    # 余弦退火: weight = final + (initial - final) * 0.5 * (1 + cos(pi * step / anneal_steps))
    progress = current_step / anneal_steps
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    weight = final + (initial - final) * cosine_decay
    
    return weight


def main():
    set_global_seeds(SetupParameters.SEED)
    
    timestamp = datetime.now().strftime('%m-%d-%H-%M')
    run_name = f"{SetupParameters.SKILL_MODE}_{TrainingParameters.TRAINING_MODE}_{timestamp}"
    
    # 创建 run_name 目录下的 models 和 gifs 子文件夹
    run_dir = osp.join(RecordingParameters.MODEL_PATH, run_name)
    model_dir = osp.join(run_dir, 'models')
    gif_dir = osp.join(run_dir, 'gifs')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(gif_dir, exist_ok=True)
    
    summary_writer = None
    if RecordingParameters.TENSORBOARD:
        from torch.utils.tensorboard import SummaryWriter
        log_dir = osp.join(RecordingParameters.SUMMARY_PATH, run_name)
        os.makedirs(log_dir, exist_ok=True)
        summary_writer = SummaryWriter(log_dir)
    
    print("=" * 60)
    print(f"TAD PPO Training - {run_name}")
    print(f"Skill Mode: {SetupParameters.SKILL_MODE}")
    print(f"Training Mode: {TrainingParameters.TRAINING_MODE}")
    if TrainingParameters.TRAINING_MODE == 'mixed':
        print(f"IL Anneal: {TrainingParameters.IL_INITIAL_WEIGHT} -> {TrainingParameters.IL_FINAL_WEIGHT} over {TrainingParameters.IL_ANNEAL_STEPS:,} steps")
    print(f"Opponent Weights: {TrainingParameters.RANDOM_OPPONENT_WEIGHTS}")
    print(f"Num Runners: {TrainingParameters.N_ENVS}")
    print(f"Steps per Runner: {TrainingParameters.N_STEPS}")
    print("=" * 60)
    
    ray.init(num_cpus=TrainingParameters.N_ENVS, num_gpus=SetupParameters.NUM_GPU)
    
    device = torch.device('cuda' if torch.cuda.is_available() and SetupParameters.USE_GPU_GLOBAL else 'cpu')
    model = Model(device=device, global_model=True)
    
    runners = [Runner.remote(i) for i in range(TrainingParameters.N_ENVS)]
    
    global_step = 0
    best_reward = -float('inf')
    
    total_steps = int(TrainingParameters.N_MAX_STEPS)
    total_updates = int(TrainingParameters.N_MAX_STEPS // (TrainingParameters.N_ENVS * TrainingParameters.N_STEPS))
    
    print(f"\nStarting training for {total_updates} updates...")
    print(f"Total environment steps: {TrainingParameters.N_MAX_STEPS:,}")
    
    start_time = time.time()
    
    for update in range(1, total_updates + 1):
        t_start = time.time()
        
        weights = model.get_weights()
        weight_id = ray.put(weights)
        ray.get([r.set_weights.remote(weight_id) for r in runners])
        
        if TrainingParameters.TRAINING_MODE == 'il':
            rollout_futures = [r.imitation.remote(TrainingParameters.N_STEPS) for r in runners]
        else:
            rollout_futures = [r.run.remote(TrainingParameters.N_STEPS) for r in runners]
        
        rollouts = ray.get(rollout_futures)
        
        all_perf = {'per_r': [], 'per_episode_len': [], 'win': []}
        for rollout in rollouts:
            perf = rollout['perf']
            all_perf['per_r'].extend(perf['per_r'])
            all_perf['per_episode_len'].extend(perf['per_episode_len'])
            all_perf['win'].extend(perf['win'])
        
        # 计算当前IL权重（余弦退火）
        il_weight = cosine_anneal_il_weight(global_step) if TrainingParameters.TRAINING_MODE == 'mixed' else 0.0
        
        if TrainingParameters.TRAINING_MODE == 'il':
            # 纯模仿学习
            obs_all = np.concatenate([r['obs'] for r in rollouts], axis=0)
            critic_obs_all = np.concatenate([r['critic_obs'] for r in rollouts], axis=0)
            expert_actions_all = np.concatenate([r['expert_actions'] for r in rollouts], axis=0)
            
            il_loss = model.imitation_train(obs_all, critic_obs_all, expert_actions_all)
            
            mb_loss = None
            
        elif TrainingParameters.TRAINING_MODE == 'rl':
            # 纯强化学习
            obs_all = np.concatenate([r['obs'] for r in rollouts], axis=0)
            critic_obs_all = np.concatenate([r['critic_obs'] for r in rollouts], axis=0)
            actions_all = np.concatenate([r['actions'] for r in rollouts], axis=0)
            log_probs_all = np.concatenate([r['log_probs'] for r in rollouts], axis=0)
            returns_all = np.concatenate([r['returns'] for r in rollouts], axis=0)
            values_all = np.concatenate([r['values'] for r in rollouts], axis=0)
            
            mb_loss = model.train(
                actor_obs=obs_all,
                critic_obs=critic_obs_all,
                actions=actions_all,
                old_log_probs=log_probs_all,
                returns=returns_all,
                values=values_all
            )
            
            il_loss = None
            
        else:
            # mixed: IL + RL 混合训练
            obs_all = np.concatenate([r['obs'] for r in rollouts], axis=0)
            critic_obs_all = np.concatenate([r['critic_obs'] for r in rollouts], axis=0)
            actions_all = np.concatenate([r['actions'] for r in rollouts], axis=0)
            log_probs_all = np.concatenate([r['log_probs'] for r in rollouts], axis=0)
            returns_all = np.concatenate([r['returns'] for r in rollouts], axis=0)
            values_all = np.concatenate([r['values'] for r in rollouts], axis=0)
            expert_actions_all = np.concatenate([r['expert_actions'] for r in rollouts], axis=0)
            
            # 传入expert_actions和il_weight进行混合训练
            mb_loss = model.train_mixed(
                actor_obs=obs_all,
                critic_obs=critic_obs_all,
                actions=actions_all,
                old_log_probs=log_probs_all,
                returns=returns_all,
                values=values_all,
                expert_actions=expert_actions_all,
                il_weight=il_weight
            )
            
            il_loss = [mb_loss.get('il_loss', 0.0), 0.0]  # [loss, grad_norm]
        
        steps_this_update = TrainingParameters.N_ENVS * TrainingParameters.N_STEPS
        global_step += steps_this_update
        
        if (global_step // TrainingParameters.LOG_EPOCH_STEPS) > ((global_step - steps_this_update) // TrainingParameters.LOG_EPOCH_STEPS):
            mean_reward = np.mean(all_perf['per_r']) if all_perf['per_r'] else 0.0
            mean_ep_len = np.mean(all_perf['per_episode_len']) if all_perf['per_episode_len'] else 0.0
            win_rate = np.mean(all_perf['win']) if all_perf['win'] else 0.0
            progress = global_step / total_steps * 100
            
            il_info = f" | IL_w: {il_weight:.3f}" if TrainingParameters.TRAINING_MODE == 'mixed' else ""
            print(f"Step {global_step:,} ({progress:.1f}%) | "
                  f"Reward: {mean_reward:.2f} | "
                  f"EpLen: {mean_ep_len:.1f} | "
                  f"Win: {win_rate:.2%}{il_info}")
            
            write_to_tensorboard(
                summary_writer, global_step,
                performance_dict=all_perf,
                mb_loss=[mb_loss['losses']] if mb_loss is not None else None,
                imitation_loss=il_loss,
                evaluate=False
            )
            
            # 记录IL权重
            if summary_writer and TrainingParameters.TRAINING_MODE == 'mixed':
                summary_writer.add_scalar('IL/weight', il_weight, global_step)
                if mb_loss and 'il_loss' in mb_loss and mb_loss['il_loss'] is not None:
                    summary_writer.add_scalar('IL/loss', mb_loss['il_loss'], global_step)
        
        if (global_step // RecordingParameters.EVAL_INTERVAL) > ((global_step - steps_this_update) // RecordingParameters.EVAL_INTERVAL):
            print("--- Evaluation ---")
            eval_result = ray.get(runners[0].evaluate.remote(
                num_episodes=RecordingParameters.EVAL_EPISODES,
                greedy=True,
                record_gif=((global_step // RecordingParameters.GIF_INTERVAL) > ((global_step - steps_this_update) // RecordingParameters.GIF_INTERVAL))
            ))
            
            eval_perf = eval_result['perf']
            eval_reward = np.mean(eval_perf['per_r']) if eval_perf['per_r'] else 0.0
            eval_ep_len = np.mean(eval_perf['per_episode_len']) if eval_perf['per_episode_len'] else 0.0
            eval_win = np.mean(eval_perf['win']) if eval_perf['win'] else 0.0
            
            print(f"Eval Reward: {eval_reward:.2f} | "
                  f"Eval EpLen: {eval_ep_len:.1f} | "
                  f"Eval Win: {eval_win:.2%}")
            
            write_to_tensorboard(
                summary_writer, global_step,
                performance_dict=eval_perf,
                evaluate=True,
                greedy=True
            )
            
            if eval_result.get('frames') and len(eval_result['frames']) > 0:
                gif_path = osp.join(gif_dir, f"eval_{global_step}.gif")
                make_gif(eval_result['frames'], gif_path)
            
            if eval_reward > best_reward:
                best_reward = eval_reward
                best_path = osp.join(model_dir, 'best_model.pth')
                model.save(best_path)
                print(f"New best model saved! Reward: {best_reward:.2f}")
            
            print("------------------")
        
        if (global_step // RecordingParameters.SAVE_INTERVAL) > ((global_step - steps_this_update) // RecordingParameters.SAVE_INTERVAL):
            latest_path = osp.join(model_dir, 'latest_model.pth')
            model.save(latest_path)
            print(f"Latest model saved: {latest_path}")
    
    final_path = osp.join(model_dir, 'final_model.pth')
    model.save(final_path)
    print(f"\nFinal model saved: {final_path}")
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time / 3600:.2f} hours")
    print(f"Best reward: {best_reward:.2f}")
    
    if summary_writer:
        summary_writer.close()
    
    ray.shutdown()


if __name__ == '__main__':
    main()
