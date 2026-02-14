"""
Baseline End-to-End PPO Training

网络结构: 与HRL顶层一致的 MLP Actor-Critic (CTDE)
奖励设计: TAD奖励 + 时间惩罚(-0.04) + (chase引导 + protect2引导)
"""

import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import ray

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from map_config import EnvParameters
from ppo.alg_parameters import NetParameters, RecordingParameters, SetupParameters, TrainingParameters
from ppo.model import Model
from ppo.runner import Runner
from ppo.util import (
    get_adjusted_n_envs,
    get_device,
    get_num_gpus,
    get_ray_temp_dir,
    make_gif,
    make_trajectory_plot,
    print_device_info,
    print_ram_info,
    set_global_seeds,
    write_to_tensorboard,
)
from hrl.baseline_train_parameters import (
    BaselineEnvTrainParameters,
    BaselineRecordingParameters,
    BaselineSetupParameters,
    BaselineTrainingParameters,
)


SetupParameters.SKILL_MODE = BaselineSetupParameters.SKILL_MODE
SetupParameters.GPU_ID = int(BaselineSetupParameters.GPU_ID)
RecordingParameters.MODEL_PATH = f"models/baseline_{datetime.now().strftime('%m-%d-%H-%M')}"
RecordingParameters.SUMMARY_PATH = f"{RecordingParameters.MODEL_PATH}/summary"


def _apply_baseline_parameter_overrides():
    """将Baseline独立超参同步到通用参数类。"""
    for key, value in BaselineTrainingParameters.__dict__.items():
        if key.startswith('_'):
            continue
        setattr(TrainingParameters, key, value)

    for key, value in BaselineRecordingParameters.__dict__.items():
        if key.startswith('_'):
            continue
        setattr(RecordingParameters, key, value)

    NetParameters.CONTEXT_WINDOW = int(BaselineTrainingParameters.TBPTT_STEPS)
    NetParameters.CONTEXT_LEN = int(BaselineTrainingParameters.TBPTT_STEPS)


def _scheduled_lr(current_step: int, total_steps: int) -> float:
    lr_init = float(BaselineTrainingParameters.lr)
    lr_final = float(BaselineTrainingParameters.LR_FINAL)
    schedule = str(BaselineTrainingParameters.LR_SCHEDULE).lower()
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
    raise ValueError(f'Unsupported LR_SCHEDULE: {BaselineTrainingParameters.LR_SCHEDULE}')


def main():
    _apply_baseline_parameter_overrides()
    set_global_seeds(SetupParameters.SEED)
    print_device_info()
    EnvParameters.EPISODE_LEN = int(BaselineEnvTrainParameters.EPISODE_LEN)

    timestamp = datetime.now().strftime('%m-%d-%H-%M')
    run_name = f'Baseline_PPO_{timestamp}'

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
    device = get_device(prefer_gpu=True, gpu_id=SetupParameters.GPU_ID)

    print('=' * 72)
    print(f'Baseline PPO Training - {run_name}')
    print(f'Device: {device} (可用GPU数量: {num_gpus})')
    print(f'Num Runners: {n_envs}')
    print(f'Network: MLP (与HRL顶层一致)')
    print(f'Reward Mode: {BaselineEnvTrainParameters.REWARD_MODE}')
    print(
        f'LR Schedule: {BaselineTrainingParameters.LR_SCHEDULE} '
        f'({BaselineTrainingParameters.lr} -> {BaselineTrainingParameters.LR_FINAL})'
    )
    print(f'Episode Max Steps: {EnvParameters.EPISODE_LEN}')
    print(f'Attacker Strategy: {BaselineEnvTrainParameters.ATTACKER_STRATEGY}')
    print('=' * 72)

    ray_tmp = get_ray_temp_dir()
    ray_num_cpus = os.cpu_count() or n_envs
    ray_kwargs = dict(
        num_cpus=ray_num_cpus,
        num_gpus=0,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False,
    )
    if ray_tmp:
        ray_kwargs['_temp_dir'] = ray_tmp
    ray.init(**ray_kwargs)

    model = Model(device=device, global_model=True, network_type='mlp')
    env_cfg = {
        'episode_len': BaselineEnvTrainParameters.EPISODE_LEN,
        'reward_mode': BaselineEnvTrainParameters.REWARD_MODE,
        'attacker_strategy': BaselineEnvTrainParameters.ATTACKER_STRATEGY,
        # RL-only训练下该项不会参与训练，但保留兼容性。
        'expert_skill_mode': 'chase',
    }
    runners = [Runner.remote(i, env_configs=env_cfg, network_type='mlp') for i in range(n_envs)]

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
            elapsed = time.time() - start_time
            print(
                f'Step {global_step:,} | LR: {model.current_lr:.2e} | '
                f'Reward: {mean_reward:.2f} | Win: {win_rate:.2%} | '
                f'EpLen: {mean_ep_len:.1f} | Elapsed: {elapsed/3600:.2f}h'
            )

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
            should_record_traj = (global_step // RecordingParameters.TRAJ_INTERVAL) > (
                (global_step - steps_this_update) // RecordingParameters.TRAJ_INTERVAL
            )

            print('--- Evaluation ---')
            eval_result = ray.get(
                runners[0].evaluate.remote(
                    num_episodes=RecordingParameters.EVAL_EPISODES,
                    greedy=True,
                    record_gif=should_record_gif,
                )
            )
            eval_perf = eval_result['perf']
            eval_reward = np.mean(eval_perf['per_r']) if eval_perf['per_r'] else 0.0
            eval_win = np.mean(eval_perf['win']) if eval_perf['win'] else 0.0
            eval_ep_len = np.mean(eval_perf['per_episode_len']) if eval_perf['per_episode_len'] else 0.0
            print(f'Eval Reward: {eval_reward:.2f} | Win: {eval_win:.2%} | EpLen: {eval_ep_len:.1f}')

            write_to_tensorboard(
                summary_writer,
                global_step,
                performance_dict=eval_perf,
                mb_loss=None,
                imitation_loss=None,
                evaluate=True,
            )

            if should_record_gif and eval_result.get('frames'):
                gif_path = os.path.join(gif_dir, f'eval_step_{global_step}.gif')
                make_gif(eval_result['frames'], gif_path, fps=20, quality='high')
                print(f'Saved eval GIF to: {gif_path}')

            if should_record_traj and eval_result.get('trajectory_data'):
                traj_data = eval_result['trajectory_data']
                traj_path = os.path.join(gif_dir, f'traj_step_{global_step}.png')
                make_trajectory_plot(traj_data, traj_path, dpi=150)
                print(f'Saved trajectory plot to: {traj_path}')

            if eval_reward > best_reward:
                best_reward = eval_reward
                best_path = os.path.join(model_dir, 'best_model.pth')
                model.save(best_path, step=global_step, reward=best_reward)
                print(f'New best model saved! Reward: {best_reward:.2f}')

        # 阶段性checkpoint已禁用，仅保留best/final。

    final_path = os.path.join(model_dir, 'final_model.pth')
    model.save(final_path, step=global_step, reward=best_reward)
    print('\nTraining completed!')
    print(f'Final model saved to: {final_path}')
    print(f'Best reward: {best_reward:.2f}')
    print(f'Total training time: {(time.time() - start_time) / 3600:.2f} hours')

    if summary_writer:
        summary_writer.close()
    if ray.is_initialized():
        ray.shutdown()


if __name__ == '__main__':
    main()
