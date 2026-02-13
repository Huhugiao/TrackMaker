"""
HRL Top Level Training

训练高层策略来调度 Protect / Chase 两个底层技能。
"""

import glob
import math
import os
import sys
import time
import numpy as np
import ray
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ppo.alg_parameters import SetupParameters, TrainingParameters, RecordingParameters, NetParameters
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
from hrl.hrl_train_parameters import (
    HRLSetupParameters,
    HRLTrainingParameters,
    HRLRecordingParameters,
    HRLEnvTrainParameters,
)
from map_config import EnvParameters


SetupParameters.SKILL_MODE = HRLSetupParameters.SKILL_MODE
SetupParameters.GPU_ID = int(HRLSetupParameters.GPU_ID)
RecordingParameters.MODEL_PATH = f"models/hrl_{datetime.now().strftime('%m-%d-%H-%M')}"
RecordingParameters.SUMMARY_PATH = f"models/hrl_{datetime.now().strftime('%m-%d-%H-%M')}/summary"


def _apply_hrl_parameter_overrides():
    """将HRL独立超参同步到通用参数类，避免复用底层技能配置。"""
    for key, value in HRLTrainingParameters.__dict__.items():
        if key.startswith('_'):
            continue
        setattr(TrainingParameters, key, value)

    for key, value in HRLRecordingParameters.__dict__.items():
        if key.startswith('_'):
            continue
        setattr(RecordingParameters, key, value)

    NetParameters.CONTEXT_WINDOW = int(HRLTrainingParameters.TBPTT_STEPS)
    NetParameters.CONTEXT_LEN = int(HRLTrainingParameters.TBPTT_STEPS)


def _scheduled_lr(current_step: int, total_steps: int) -> float:
    """根据HRL训练配置计算当前学习率。"""
    lr_init = float(HRLTrainingParameters.lr)
    lr_final = float(HRLTrainingParameters.LR_FINAL)
    schedule = str(HRLTrainingParameters.LR_SCHEDULE).lower()
    if total_steps <= 0:
        return lr_init

    progress = min(max(float(current_step) / float(total_steps), 0.0), 1.0)
    if schedule == 'constant':
        return lr_init
    if schedule == 'linear':
        return lr_init + (lr_final - lr_init) * progress
    if schedule == 'cosine':
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_final + (lr_init - lr_final) * cosine_decay
    raise ValueError(f'Unsupported LR_SCHEDULE: {HRLTrainingParameters.LR_SCHEDULE}')


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
    _apply_hrl_parameter_overrides()
    set_global_seeds(SetupParameters.SEED)
    print_device_info()
    EnvParameters.EPISODE_LEN = int(HRLEnvTrainParameters.EPISODE_LEN)

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
    print(f'LR Schedule: {HRLTrainingParameters.LR_SCHEDULE} ({HRLTrainingParameters.lr} -> {HRLTrainingParameters.LR_FINAL})')
    print(f'Episode Max Steps: {EnvParameters.EPISODE_LEN}')
    print(f'Macro Length Range: [{HRLEnvTrainParameters.HOLD_MIN}, {HRLEnvTrainParameters.HOLD_MAX}]')
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
        'episode_len': HRLEnvTrainParameters.EPISODE_LEN,
        'attacker_strategy': HRLEnvTrainParameters.ATTACKER_STRATEGY,
        'protect_model_path': protect_model_path,
        'chase_model_path': chase_model_path,
        'predictor_hidden_dim': HRLEnvTrainParameters.PREDICTOR_HIDDEN_DIM,
        'predictor_lr': HRLEnvTrainParameters.PREDICTOR_LR,
        'predictor_train': HRLEnvTrainParameters.PREDICTOR_TRAIN,
        'hold_min': HRLEnvTrainParameters.HOLD_MIN,
        'hold_max': HRLEnvTrainParameters.HOLD_MAX,
        'disable_hold_control': HRLEnvTrainParameters.DISABLE_HOLD_CONTROL,
        'gamma': HRLTrainingParameters.GAMMA,
        'lam': HRLTrainingParameters.LAM,
        'reward_normalization': HRLTrainingParameters.REWARD_NORMALIZATION,
    }
    runners = [HRLRunner.remote(i, env_configs=env_cfg) for i in range(n_envs)]

    global_step = 0
    best_reward = -float('inf')
    total_updates = int(TrainingParameters.N_MAX_STEPS // (n_envs * TrainingParameters.N_STEPS))

    print(f'\nStarting training for {total_updates} updates...')
    start_time = time.time()

    for update in range(1, total_updates + 1):
        current_lr = _scheduled_lr(global_step, int(TrainingParameters.N_MAX_STEPS))
        model.update_learning_rate(current_lr)
        model.current_lr = current_lr

        weights = model.get_weights()
        weight_id = ray.put(weights)
        ray.get([r.set_weights.remote(weight_id) for r in runners])

        rollouts = ray.get([r.run.remote(TrainingParameters.N_STEPS) for r in runners])

        all_perf = {'per_r': [], 'per_episode_len': [], 'win': []}
        for rollout in rollouts:
            perf = rollout['perf']
            all_perf['per_r'].extend(perf['per_r'])
            all_perf['per_episode_len'].extend(perf['per_episode_len'])
            all_perf['win'].extend(perf['win'])

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
            mean_ep_len = np.mean(all_perf['per_episode_len']) if all_perf['per_episode_len'] else 0.0
            print(f'Step {global_step:,} | LR: {model.current_lr:.2e} | Reward: {mean_reward:.2f} | Win: {win_rate:.2%} | EpLen: {mean_ep_len:.1f}')

            write_to_tensorboard(
                summary_writer,
                global_step,
                performance_dict=all_perf,
                mb_loss=[mb_loss['losses']],
                imitation_loss=None,
                evaluate=False,
            )

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
            eval_ep_len = np.mean(eval_perf['per_episode_len']) if eval_perf['per_episode_len'] else 0.0
            print(f'Eval Reward: {eval_reward:.2f} | Win: {eval_win:.2%} | EpLen: {eval_ep_len:.1f}')

            write_to_tensorboard(
                summary_writer,
                global_step,
                performance_dict=eval_perf,
                evaluate=True,
                greedy=True,
            )

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
