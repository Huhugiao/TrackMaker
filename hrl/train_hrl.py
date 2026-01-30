
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
from ppo.util import set_global_seeds, write_to_tensorboard, make_gif
from hrl.hrl_runner import HRLRunner

# Override parameters for HRL
SetupParameters.SKILL_MODE = 'hrl'
RecordingParameters.MODEL_PATH = f"models/hrl_{datetime.now().strftime('%m-%d-%H-%M')}"
RecordingParameters.SUMMARY_PATH = f"models/hrl_{datetime.now().strftime('%m-%d-%H-%M')}/summary"

def cosine_anneal_il_weight(current_step: int) -> float:
    return 0.0 # No IL for HRL yet

def main():
    set_global_seeds(SetupParameters.SEED)
    
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
    
    print("=" * 60)
    print(f"HRL Top Level Training - {run_name}")
    print(f"Num Runners: {TrainingParameters.N_ENVS}")
    print("=" * 60)
    
    ray.init(num_cpus=TrainingParameters.N_ENVS, num_gpus=SetupParameters.NUM_GPU)
    
    device = torch.device('cuda' if torch.cuda.is_available() and SetupParameters.USE_GPU_GLOBAL else 'cpu')
    model = Model(device=device, global_model=True)
    
    # Initialize HRL Runners with random attacker strategies
    runners = [HRLRunner.remote(i, env_configs={'attacker_strategy': 'random'}) for i in range(TrainingParameters.N_ENVS)]
    
    global_step = 0
    best_reward = -float('inf')
    
    total_steps = int(TrainingParameters.N_MAX_STEPS)
    total_updates = int(TrainingParameters.N_MAX_STEPS // (TrainingParameters.N_ENVS * TrainingParameters.N_STEPS))
    
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
        
        steps_this_update = TrainingParameters.N_ENVS * TrainingParameters.N_STEPS
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
        
        # Eval
        if (global_step // RecordingParameters.EVAL_INTERVAL) > ((global_step - steps_this_update) // RecordingParameters.EVAL_INTERVAL):
            print("--- Evaluation ---")
            eval_result = ray.get(runners[0].evaluate.remote(
                num_episodes=RecordingParameters.EVAL_EPISODES,
                greedy=True,
                record_gif=True
            ))
            
            eval_perf = eval_result['perf']
            eval_reward = np.mean(eval_perf['per_r']) if eval_perf['per_r'] else 0.0
            print(f"Eval Reward: {eval_reward:.2f}")
            
            if eval_result.get('frames') and len(eval_result['frames']) > 0:
                gif_path = os.path.join(gif_dir, f"eval_{global_step}.gif")
                make_gif(eval_result['frames'], gif_path)
            
            if eval_reward > best_reward:
                best_reward = eval_reward
                model.save(os.path.join(model_dir, 'best_model.pth'), step=global_step, reward=best_reward)
                print(f"New best model saved!")
                
    model.save(os.path.join(model_dir, 'final_model.pth'), step=global_step, reward=best_reward)
    ray.shutdown()

if __name__ == '__main__':
    main()
