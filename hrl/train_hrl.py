"""
HRL Top Level Training

训练高层策略来混合 Protect 和 Chase 两个技能。
"""

import os
import sys
import time
import math
import numpy as np
import torch
import ray
from datetime import datetime

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ppo.alg_parameters import SetupParameters, TrainingParameters, NetParameters, RecordingParameters
from ppo.model import Model
from ppo.util import set_global_seeds, write_to_tensorboard, make_gif, get_device, get_num_gpus, print_device_info, get_adjusted_n_envs, print_ram_info, get_ray_temp_dir, is_gpu_available, is_gpu_available
from hrl.hrl_runner import HRLRunner

# Override parameters for HRL
SetupParameters.SKILL_MODE = 'hrl'
RecordingParameters.MODEL_PATH = f"models/hrl_{datetime.now().strftime('%m-%d-%H-%M')}"
RecordingParameters.SUMMARY_PATH = f"models/hrl_{datetime.now().strftime('%m-%d-%H-%M')}/summary"

def cosine_anneal_il_weight(current_step: int) -> float:
    return 0.0 # No IL for HRL yet

def main():
    set_global_seeds(SetupParameters.SEED)
    
    # 打印设备信息
    print_device_info()
    
    timestamp = datetime.now().strftime('%m-%d-%H-%M')
    run_name = f"HRL_TopLevel_{timestamp}"
    
    model_dir = RecordingParameters.MODEL_PATH
    gif_dir = os.path.join(model_dir, 'gifs')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(gif_dir, exist_ok=True)
    
    summary_writer = None
    if RecordingParameters.TENSORBOARD:
        from torch.utils.tensorboard import SummaryWriter
        log_dir = RecordingParameters.SUMMARY_PATH
        os.makedirs(log_dir, exist_ok=True)
        summary_writer = SummaryWriter(log_dir)
    
    # 使用安全的GPU检测
    print_ram_info()
    
    # 根据RAM调整并行环境数量
    n_envs = get_adjusted_n_envs(TrainingParameters.N_ENVS)
    num_gpus = get_num_gpus()
    device = get_device(prefer_gpu=True)
    
    print("=" * 60)
    print(f"HRL Top Level Training - {run_name}")
    print(f"Device: {device} (可用GPU数量: {num_gpus})")
    print(f"Num Runners: {n_envs}")
    print(f"GIF Interval: {RecordingParameters.GIF_INTERVAL:,} steps")
    print("=" * 60)
    
    # 使用检测到的GPU数量初始化Ray
    ray_tmp = get_ray_temp_dir()
    # 只分配1个GPU给Ray，所有Runner共享这个GPU
    ray_num_gpus = 1 if is_gpu_available() else 0
    if ray_tmp:
        ray.init(num_cpus=n_envs, num_gpus=ray_num_gpus, _temp_dir=ray_tmp)
    else:
        ray.init(num_cpus=n_envs, num_gpus=ray_num_gpus)
    
    model = Model(device=device, global_model=True)
    
    # Initialize HRL Runners with random attacker strategies
    runners = [HRLRunner.remote(i, env_configs={'attacker_strategy': 'random'}) for i in range(n_envs)]
    
    global_step = 0
    best_reward = -float('inf')
    
    total_steps = int(TrainingParameters.N_MAX_STEPS)
    total_updates = int(TrainingParameters.N_MAX_STEPS // (n_envs * TrainingParameters.N_STEPS))
    
    print(f"\nStarting training for {total_updates} updates...")
    
    start_time = time.time()
    
    for update in range(1, total_updates + 1):
        # Sync weights
        weights = model.get_weights()
        weight_id = ray.put(weights)
        ray.get([r.set_weights.remote(weight_id) for r in runners])
        
        # Rollout
        rollout_futures = [r.run.remote(TrainingParameters.N_STEPS) for r in runners]
        rollouts = ray.get(rollout_futures)
        
        all_perf = {'per_r': [], 'per_episode_len': [], 'win': []}
        for rollout in rollouts:
            perf = rollout['perf']
            all_perf['per_r'].extend(perf['per_r'])
            all_perf['per_episode_len'].extend(perf['per_episode_len'])
            all_perf['win'].extend(perf['win'])
        
        # Train
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
        
        steps_this_update = n_envs * TrainingParameters.N_STEPS
        global_step += steps_this_update
        
        # Logging
        if (global_step // TrainingParameters.LOG_EPOCH_STEPS) > ((global_step - steps_this_update) // TrainingParameters.LOG_EPOCH_STEPS):
            mean_reward = np.mean(all_perf['per_r']) if all_perf['per_r'] else 0.0
            win_rate = np.mean(all_perf['win']) if all_perf['win'] else 0.0
            print(f"Step {global_step:,} | Reward: {mean_reward:.2f} | Win: {win_rate:.2%}")
            
            write_to_tensorboard(
                summary_writer, global_step,
                performance_dict=all_perf,
                mb_loss=[mb_loss['losses']],
                imitation_loss=None,
                evaluate=False
            )
        
        # Eval (与skill训练同步的GIF生成逻辑)
        if (global_step // RecordingParameters.EVAL_INTERVAL) > ((global_step - steps_this_update) // RecordingParameters.EVAL_INTERVAL):
            # 判断是否需要生成GIF (使用GIF_INTERVAL)
            should_record_gif = (global_step // RecordingParameters.GIF_INTERVAL) > ((global_step - steps_this_update) // RecordingParameters.GIF_INTERVAL)
            
            print("--- Evaluation ---")
            eval_result = ray.get(runners[0].evaluate.remote(
                num_episodes=RecordingParameters.EVAL_EPISODES,
                greedy=True,
                record_gif=should_record_gif
            ))
            
            eval_perf = eval_result['perf']
            eval_reward = np.mean(eval_perf['per_r']) if eval_perf['per_r'] else 0.0
            eval_win = np.mean(eval_perf['win']) if eval_perf['win'] else 0.0
            print(f"Eval Reward: {eval_reward:.2f} | Win: {eval_win:.2%}")
            
            write_to_tensorboard(
                summary_writer, global_step,
                performance_dict=eval_perf,
                evaluate=True,
                greedy=True
            )
            
            if eval_result.get('frames') and len(eval_result['frames']) > 0:
                gif_path = os.path.join(gif_dir, f"eval_{global_step}.gif")
                make_gif(eval_result['frames'], gif_path)
                print(f"GIF saved: {gif_path}")
            
            if eval_reward > best_reward:
                best_reward = eval_reward
                model.save(os.path.join(model_dir, 'best_model.pth'), step=global_step, reward=best_reward)
                print(f"New best model saved!")
            
            print("------------------")
        
        # Save latest model periodically
        if (global_step // RecordingParameters.SAVE_INTERVAL) > ((global_step - steps_this_update) // RecordingParameters.SAVE_INTERVAL):
            latest_path = os.path.join(model_dir, 'latest_model.pth')
            model.save(latest_path, step=global_step, reward=best_reward)
            print(f"Latest model saved: {latest_path}")
                
    model.save(os.path.join(model_dir, 'final_model.pth'), step=global_step, reward=best_reward)
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time / 3600:.2f} hours")
    print(f"Best reward: {best_reward:.2f}")
    
    if summary_writer:
        summary_writer.close()
    
    ray.shutdown()

if __name__ == '__main__':
    main()
