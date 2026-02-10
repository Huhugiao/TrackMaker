"""
HRL Top Level Training

训练高层策略来调度 Protect / Chase 两个底层技能。
"""

import glob
import os
import sys
import time
import numpy as np
import ray
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ppo.alg_parameters import SetupParameters, TrainingParameters, RecordingParameters
from ppo.model import Model
from ppo.util import (
    set_global_seeds,
    write_to_tensorboard,
    make_gif,
    make_trajectory_plot,
    get_device,
    get_num_gpus,
    print_device_info,
    get_adjusted_n_envs,
    print_ram_info,
    get_ray_temp_dir,
)
from hrl.hrl_runner import HRLRunner


SetupParameters.SKILL_MODE = 'hrl'
RecordingParameters.MODEL_PATH = f"models/hrl_{datetime.now().strftime('%m-%d-%H-%M')}"
RecordingParameters.SUMMARY_PATH = f"models/hrl_{datetime.now().strftime('%m-%d-%H-%M')}/summary"


def _find_latest_checkpoint(model_prefixes):
    candidates = []
    for prefix in model_prefixes:
        for filename in ('best_model.pth',):
            pattern = os.path.join('models', f'{prefix}_*', filename)
            candidates.extend(glob.glob(pattern))
    candidates = [p for p in candidates if os.path.isfile(p)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _resolve_skill_paths():
    protect_path = _find_latest_checkpoint([
        'defender_protect2_dense',
        'defender_protect1_dense',
        'defender_protect_dense',
    ])
    chase_path = _find_latest_checkpoint(['defender_chase_dense'])

    if protect_path is None:
        raise FileNotFoundError('无法找到 protect skill checkpoint (models/defender_protect*_*/best_model.pth).')
    if chase_path is None:
        raise FileNotFoundError('无法找到 chase skill checkpoint (models/defender_chase_dense_*/best_model.pth).')

    return os.path.abspath(protect_path), os.path.abspath(chase_path)


def main():
    set_global_seeds(SetupParameters.SEED)
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

    print_ram_info()

    n_envs = get_adjusted_n_envs(TrainingParameters.N_ENVS)
    num_gpus = get_num_gpus()
    # Keep global PPO update on the configured training GPU.
    device = get_device(prefer_gpu=True, gpu_id=SetupParameters.GPU_ID)

    protect_model_path, chase_model_path = _resolve_skill_paths()

    print('=' * 72)
    print(f'HRL Top Level Training - {run_name}')
    print(f'Device: {device} (可用GPU数量: {num_gpus})')
    print(f'Num Runners: {n_envs}')
    print(f'Protect Skill: {protect_model_path}')
    print(f'Chase Skill:   {chase_model_path}')
    print(f'GIF Interval: {RecordingParameters.GIF_INTERVAL:,} steps')
    print(f'TRAJ Interval: {RecordingParameters.TRAJ_INTERVAL:,} steps')
    print('=' * 72)

    ray_tmp = get_ray_temp_dir()
    # HRLRunner is CPU-only; reserve no GPUs for Ray workers.
    ray_num_gpus = 0
    if ray_tmp:
        ray.init(num_cpus=n_envs, num_gpus=ray_num_gpus, _temp_dir=ray_tmp)
    else:
        ray.init(num_cpus=n_envs, num_gpus=ray_num_gpus)

    model = Model(device=device, global_model=True, network_type='hrl_top')

    env_cfg = {
        'attacker_strategy': 'random',
        'protect_model_path': protect_model_path,
        'chase_model_path': chase_model_path,
        'predictor_hidden_dim': 64,
        'predictor_lr': 1e-3,
        'predictor_train': True,
        'hold_min': 3,
        'hold_max': 15,
    }
    runners = [HRLRunner.remote(i, env_configs=env_cfg) for i in range(n_envs)]

    global_step = 0
    best_reward = -float('inf')
    total_updates = int(TrainingParameters.N_MAX_STEPS // (n_envs * TrainingParameters.N_STEPS))

    print(f'\nStarting training for {total_updates} updates...')
    start_time = time.time()

    for update in range(1, total_updates + 1):
        weights = model.get_weights()
        weight_id = ray.put(weights)
        ray.get([r.set_weights.remote(weight_id) for r in runners])

        rollouts = ray.get([r.run.remote(TrainingParameters.N_STEPS) for r in runners])

        all_perf = {'per_r': [], 'per_episode_len': [], 'win': []}
        rollout_pred_losses = []
        rollout_macro_lens = []
        for rollout in rollouts:
            perf = rollout['perf']
            all_perf['per_r'].extend(perf['per_r'])
            all_perf['per_episode_len'].extend(perf['per_episode_len'])
            all_perf['win'].extend(perf['win'])
            rollout_pred_losses.append(float(rollout.get('predictor_loss', 0.0)))
            rollout_macro_lens.append(float(rollout.get('macro_len_mean', 0.0)))

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
            values=values_all,
        )

        steps_this_update = n_envs * TrainingParameters.N_STEPS
        global_step += steps_this_update

        should_log = (global_step // TrainingParameters.LOG_EPOCH_STEPS) > (
            (global_step - steps_this_update) // TrainingParameters.LOG_EPOCH_STEPS
        )
        if should_log:
            mean_reward = np.mean(all_perf['per_r']) if all_perf['per_r'] else 0.0
            win_rate = np.mean(all_perf['win']) if all_perf['win'] else 0.0
            pred_loss = float(np.mean(rollout_pred_losses)) if rollout_pred_losses else 0.0
            macro_len = float(np.mean(rollout_macro_lens)) if rollout_macro_lens else 0.0
            print(f'Step {global_step:,} | Reward: {mean_reward:.2f} | Win: {win_rate:.2%} | PredLoss: {pred_loss:.4f} | MacroLen: {macro_len:.2f}')

            write_to_tensorboard(
                summary_writer,
                global_step,
                performance_dict=all_perf,
                mb_loss=[mb_loss['losses']],
                imitation_loss=None,
                evaluate=False,
            )
            if summary_writer is not None:
                summary_writer.add_scalar('Train/Predictor_Loss', pred_loss, global_step)
                summary_writer.add_scalar('Train/Macro_Length', macro_len, global_step)

        should_eval = (global_step // RecordingParameters.EVAL_INTERVAL) > (
            (global_step - steps_this_update) // RecordingParameters.EVAL_INTERVAL
        )
        if should_eval:
            should_record_gif = (global_step // RecordingParameters.GIF_INTERVAL) > (
                (global_step - steps_this_update) // RecordingParameters.GIF_INTERVAL
            )

            print('--- Evaluation ---')
            eval_result = ray.get(runners[0].evaluate.remote(
                num_episodes=RecordingParameters.EVAL_EPISODES,
                greedy=True,
                record_gif=should_record_gif,
            ))

            eval_perf = eval_result['perf']
            eval_reward = np.mean(eval_perf['per_r']) if eval_perf['per_r'] else 0.0
            eval_win = np.mean(eval_perf['win']) if eval_perf['win'] else 0.0
            eval_pred_loss = float(eval_result.get('predictor_loss', 0.0))
            eval_macro_len = float(eval_result.get('macro_len_mean', 0.0))
            print(f'Eval Reward: {eval_reward:.2f} | Win: {eval_win:.2%} | PredLoss: {eval_pred_loss:.4f} | MacroLen: {eval_macro_len:.2f}')

            write_to_tensorboard(
                summary_writer,
                global_step,
                performance_dict=eval_perf,
                evaluate=True,
                greedy=True,
            )
            if summary_writer is not None:
                summary_writer.add_scalar('Eval/Predictor_Loss', eval_pred_loss, global_step)
                summary_writer.add_scalar('Eval/Macro_Length', eval_macro_len, global_step)

            if eval_result.get('frames'):
                gif_path = os.path.join(gif_dir, f'eval_{global_step}.gif')
                make_gif(eval_result['frames'], gif_path)

            traj_data = eval_result.get('trajectory_data')
            should_record_traj = (global_step // RecordingParameters.TRAJ_INTERVAL) > (
                (global_step - steps_this_update) // RecordingParameters.TRAJ_INTERVAL
            )
            if traj_data and should_record_traj:
                traj_png = os.path.join(gif_dir, f'traj_{global_step}.png')
                make_trajectory_plot(traj_data, traj_png, dpi=150)

            if eval_reward > best_reward:
                best_reward = eval_reward
                model.save(os.path.join(model_dir, 'best_model.pth'), step=global_step, reward=best_reward)
                print('New best model saved!')

            print('------------------')

        should_save = (global_step // RecordingParameters.SAVE_INTERVAL) > (
            (global_step - steps_this_update) // RecordingParameters.SAVE_INTERVAL
        )
        if should_save:
            latest_path = os.path.join(model_dir, 'latest_model.pth')
            model.save(latest_path, step=global_step, reward=best_reward)
            print(f'Latest model saved: {latest_path}')

    model.save(os.path.join(model_dir, 'final_model.pth'), step=global_step, reward=best_reward)

    total_time = time.time() - start_time
    print(f'\nTraining completed in {total_time / 3600:.2f} hours')
    print(f'Best reward: {best_reward:.2f}')

    if summary_writer:
        summary_writer.close()

    ray.shutdown()


if __name__ == '__main__':
    main()
