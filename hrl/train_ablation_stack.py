"""
GRU/Macro/NMN/CTDE 消融基线：一键顺序训练
1) chase 底层技能 (MLP no-CTDE)
2) protect2 底层技能 (MLP no-CTDE)
3) 上层策略 (MLP no-CTDE, 无GRU, 无macro length)
"""

import argparse
import math
import os
import sys
import time
import glob
from datetime import datetime
from typing import Dict, Optional, Tuple, Type

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
from hrl.hrl_runner import HRLRunner
from hrl.ablation_stack_parameters import (
    AblationEnvParameters,
    AblationRecordingParameters,
    AblationSetupParameters,
    AblationSkillTrainingParameters,
    AblationTopTrainingParameters,
)


def _find_latest_ablation_run_dir() -> Optional[str]:
    candidates = [
        d for d in glob.glob(os.path.join(AblationSetupParameters.BASE_OUTPUT_DIR, 'ablation_gru_macro_nmn_ctde_*'))
        if os.path.isdir(d)
    ]
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def _resolve_top_only_skill_ckpts(
    protect_ckpt: Optional[str] = None,
    chase_ckpt: Optional[str] = None,
    source_run_dir: Optional[str] = None,
) -> Tuple[str, str]:
    if protect_ckpt and chase_ckpt:
        if not os.path.isfile(protect_ckpt):
            raise FileNotFoundError(f'protect ckpt不存在: {protect_ckpt}')
        if not os.path.isfile(chase_ckpt):
            raise FileNotFoundError(f'chase ckpt不存在: {chase_ckpt}')
        return protect_ckpt, chase_ckpt

    run_dir = source_run_dir or _find_latest_ablation_run_dir()
    if not run_dir:
        raise FileNotFoundError(
            '未找到ablation历史目录，请先跑完整ablation训练，或通过 '
            '--protect-ckpt/--chase-ckpt 显式指定。'
        )

    auto_protect = os.path.join(run_dir, 'ablation_protect2_mlp_noctde', 'best_model.pth')
    auto_chase = os.path.join(run_dir, 'ablation_chase_mlp_noctde', 'best_model.pth')
    protect_path = protect_ckpt or auto_protect
    chase_path = chase_ckpt or auto_chase
    if not os.path.isfile(protect_path):
        raise FileNotFoundError(f'protect ckpt不存在: {protect_path}')
    if not os.path.isfile(chase_path):
        raise FileNotFoundError(f'chase ckpt不存在: {chase_path}')
    return protect_path, chase_path


def _apply_training_overrides(param_cls: Type):
    for key, value in param_cls.__dict__.items():
        if key.startswith('_'):
            continue
        setattr(TrainingParameters, key, value)


def _apply_recording_overrides():
    for key, value in AblationRecordingParameters.__dict__.items():
        if key.startswith('_'):
            continue
        setattr(RecordingParameters, key, value)


def _apply_common_runtime():
    NetParameters.CONTEXT_WINDOW = int(TrainingParameters.TBPTT_STEPS)
    NetParameters.CONTEXT_LEN = int(TrainingParameters.TBPTT_STEPS)
    SetupParameters.GPU_ID = int(AblationSetupParameters.GPU_ID)
    EnvParameters.EPISODE_LEN = int(AblationEnvParameters.EPISODE_LEN)


def _scheduled_lr(current_step: int, total_steps: int) -> float:
    lr_init = float(TrainingParameters.lr)
    lr_final = float(TrainingParameters.LR_FINAL)
    schedule = str(TrainingParameters.LR_SCHEDULE).lower()
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
    raise ValueError(f'Unsupported LR_SCHEDULE: {TrainingParameters.LR_SCHEDULE}')


def _init_ray(n_envs: int):
    ray_tmp = get_ray_temp_dir()
    ray_kwargs = dict(
        num_cpus=os.cpu_count() or n_envs,
        num_gpus=0,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False,
    )
    if ray_tmp:
        ray_kwargs['_temp_dir'] = ray_tmp
    ray.init(**ray_kwargs)


def _train_skill_stage(
    stage_name: str,
    skill_mode: str,
    output_dir: str,
    attacker_strategy: str,
) -> str:
    SetupParameters.SKILL_MODE = skill_mode
    _apply_training_overrides(AblationSkillTrainingParameters)
    _apply_recording_overrides()
    _apply_common_runtime()
    set_global_seeds(SetupParameters.SEED)

    model_dir = os.path.join(output_dir, stage_name)
    gif_dir = os.path.join(model_dir, 'gifs')
    summary_dir = os.path.join(model_dir, 'summary')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(gif_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    summary_writer = None
    if RecordingParameters.TENSORBOARD:
        from torch.utils.tensorboard import SummaryWriter
        summary_writer = SummaryWriter(summary_dir)

    print_ram_info()
    n_envs = get_adjusted_n_envs(TrainingParameters.N_ENVS)
    device = get_device(prefer_gpu=True, gpu_id=SetupParameters.GPU_ID)
    print(f'\n==== Stage: {stage_name} ====')
    print(f'Skill Mode: {skill_mode}, Network: mlp_noctde, Device: {device}, Runners: {n_envs}')

    _init_ray(n_envs)

    model = Model(device=device, global_model=True, network_type='mlp_noctde')
    env_cfg = {
        'episode_len': AblationEnvParameters.EPISODE_LEN,
        'reward_mode': skill_mode,
        'attacker_strategy': attacker_strategy,
        'expert_skill_mode': skill_mode,
    }
    runners = [Runner.remote(i, env_configs=env_cfg, network_type='mlp_noctde') for i in range(n_envs)]

    global_step = 0
    best_reward = -float('inf')
    total_updates = int(TrainingParameters.N_MAX_STEPS // (n_envs * TrainingParameters.N_STEPS))
    start_time = time.time()

    for _update in range(1, total_updates + 1):
        current_lr = _scheduled_lr(global_step, int(TrainingParameters.N_MAX_STEPS))
        model.update_learning_rate(current_lr)
        model.current_lr = current_lr

        weight_id = ray.put(model.get_weights())
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
            mean_win = np.mean(all_perf['win']) if all_perf['win'] else 0.0
            mean_len = np.mean(all_perf['per_episode_len']) if all_perf['per_episode_len'] else 0.0
            elapsed = (time.time() - start_time) / 3600.0
            print(
                f'[{stage_name}] step={global_step:,} lr={model.current_lr:.2e} '
                f'reward={mean_reward:.2f} win={mean_win:.2%} len={mean_len:.1f} elapsed={elapsed:.2f}h'
            )
            write_to_tensorboard(
                summary_writer, global_step,
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
            print(f'[{stage_name}] eval reward={eval_reward:.2f} win={eval_win:.2%}')

            write_to_tensorboard(
                summary_writer, global_step,
                performance_dict=eval_perf,
                mb_loss=None,
                imitation_loss=None,
                evaluate=True,
            )

            if should_record_gif and eval_result.get('frames'):
                gif_path = os.path.join(gif_dir, f'eval_step_{global_step}.gif')
                make_gif(eval_result['frames'], gif_path, fps=20, quality='high')
            if should_record_traj and eval_result.get('trajectory_data'):
                traj_path = os.path.join(gif_dir, f'traj_step_{global_step}.png')
                make_trajectory_plot(eval_result['trajectory_data'], traj_path, dpi=150)

            if eval_reward > best_reward:
                best_reward = eval_reward
                model.save(os.path.join(model_dir, 'best_model.pth'), step=global_step, reward=best_reward)

        # 阶段性checkpoint已禁用，仅保留best/final。

    model.save(os.path.join(model_dir, 'final_model.pth'), step=global_step, reward=best_reward)
    if best_reward == -float('inf'):
        best_ckpt = os.path.join(model_dir, 'final_model.pth')
    else:
        best_ckpt = os.path.join(model_dir, 'best_model.pth')

    if summary_writer is not None:
        summary_writer.close()
    if ray.is_initialized():
        ray.shutdown()

    print(f'[{stage_name}] done. best={best_ckpt}')
    return best_ckpt


def _train_top_stage(output_dir: str, protect_ckpt: str, chase_ckpt: str) -> str:
    SetupParameters.SKILL_MODE = 'hrl'
    _apply_training_overrides(AblationTopTrainingParameters)
    _apply_recording_overrides()
    _apply_common_runtime()
    set_global_seeds(SetupParameters.SEED)

    stage_name = 'ablation_hrl_top_mlp_noctde'
    model_dir = os.path.join(output_dir, stage_name)
    gif_dir = os.path.join(model_dir, 'gifs')
    summary_dir = os.path.join(model_dir, 'summary')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(gif_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    summary_writer = None
    if RecordingParameters.TENSORBOARD:
        from torch.utils.tensorboard import SummaryWriter
        summary_writer = SummaryWriter(summary_dir)

    print_ram_info()
    n_envs = get_adjusted_n_envs(TrainingParameters.N_ENVS)
    device = get_device(prefer_gpu=True, gpu_id=SetupParameters.GPU_ID)
    print(f'\n==== Stage: {stage_name} ====')
    print(f'Network: hrl_top_noctde, Device: {device}, Runners: {n_envs}')
    print('Ablations: disable_predictor=True, hold_min=1, hold_max=1, disable_hold_control=True')

    _init_ray(n_envs)

    model = Model(device=device, global_model=True, network_type='hrl_top_noctde')
    env_cfg = {
        'network_type': 'hrl_top_noctde',
        'episode_len': AblationEnvParameters.EPISODE_LEN,
        'attacker_strategy': AblationEnvParameters.ATTACKER_STRATEGY,
        'protect_model_path': os.path.abspath(protect_ckpt),
        'chase_model_path': os.path.abspath(chase_ckpt),
        'predictor_hidden_dim': 64,
        'predictor_lr': 1e-3,
        'predictor_train': False,
        'disable_predictor': True,   # 关闭GRU预测器
        'hold_min': 1,               # 关闭macro length
        'hold_max': 1,
        'disable_hold_control': True,
        'gamma': TrainingParameters.GAMMA,
        'lam': TrainingParameters.LAM,
        'reward_normalization': TrainingParameters.REWARD_NORMALIZATION,
    }
    runners = [HRLRunner.remote(i, env_configs=env_cfg) for i in range(n_envs)]

    global_step = 0
    best_reward = -float('inf')
    total_updates = int(TrainingParameters.N_MAX_STEPS // (n_envs * TrainingParameters.N_STEPS))
    start_time = time.time()

    for _update in range(1, total_updates + 1):
        current_lr = _scheduled_lr(global_step, int(TrainingParameters.N_MAX_STEPS))
        model.update_learning_rate(current_lr)
        model.current_lr = current_lr

        weight_id = ray.put(model.get_weights())
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
            mean_win = np.mean(all_perf['win']) if all_perf['win'] else 0.0
            mean_len = np.mean(all_perf['per_episode_len']) if all_perf['per_episode_len'] else 0.0
            elapsed = (time.time() - start_time) / 3600.0
            print(
                f'[{stage_name}] step={global_step:,} lr={model.current_lr:.2e} '
                f'reward={mean_reward:.2f} win={mean_win:.2%} len={mean_len:.1f} elapsed={elapsed:.2f}h'
            )
            write_to_tensorboard(
                summary_writer, global_step,
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
            print(f'[{stage_name}] eval reward={eval_reward:.2f} win={eval_win:.2%}')

            write_to_tensorboard(
                summary_writer, global_step,
                performance_dict=eval_perf,
                mb_loss=None,
                imitation_loss=None,
                evaluate=True,
            )

            if should_record_gif and eval_result.get('frames'):
                gif_path = os.path.join(gif_dir, f'eval_step_{global_step}.gif')
                make_gif(eval_result['frames'], gif_path, fps=20, quality='high')
            if should_record_traj and eval_result.get('trajectory_data'):
                traj_path = os.path.join(gif_dir, f'traj_step_{global_step}.png')
                make_trajectory_plot(eval_result['trajectory_data'], traj_path, dpi=150)

            if eval_reward > best_reward:
                best_reward = eval_reward
                model.save(os.path.join(model_dir, 'best_model.pth'), step=global_step, reward=best_reward)

        # 阶段性checkpoint已禁用，仅保留best/final。

    model.save(os.path.join(model_dir, 'final_model.pth'), step=global_step, reward=best_reward)
    best_ckpt = os.path.join(model_dir, 'best_model.pth') if best_reward > -float('inf') else os.path.join(model_dir, 'final_model.pth')

    if summary_writer is not None:
        summary_writer.close()
    if ray.is_initialized():
        ray.shutdown()

    print(f'[{stage_name}] done. best={best_ckpt}')
    return best_ckpt


def main():
    parser = argparse.ArgumentParser(description='Ablation training pipeline')
    parser.add_argument('--top-only', action='store_true', help='只训练ablation上层')
    parser.add_argument('--source-run-dir', type=str, default=None, help='top-only时复用技能模型的ablation目录')
    parser.add_argument('--protect-ckpt', type=str, default=None, help='top-only时指定protect skill ckpt')
    parser.add_argument('--chase-ckpt', type=str, default=None, help='top-only时指定chase skill ckpt')
    parser.add_argument('--output-dir', type=str, default=None, help='输出目录（默认自动按时间戳创建）')
    args = parser.parse_args()

    print_device_info()
    timestamp = datetime.now().strftime('%m-%d-%H-%M')
    if args.output_dir:
        output_dir = args.output_dir
    elif args.top_only:
        output_dir = os.path.join(
            AblationSetupParameters.BASE_OUTPUT_DIR,
            f'ablation_top_only_{timestamp}',
        )
    else:
        output_dir = os.path.join(
            AblationSetupParameters.BASE_OUTPUT_DIR,
            f'ablation_gru_macro_nmn_ctde_{timestamp}',
        )
    os.makedirs(output_dir, exist_ok=True)

    if args.top_only:
        print('\n========== Ablation Top-Only Training ==========')
        protect_ckpt, chase_ckpt = _resolve_top_only_skill_ckpts(
            protect_ckpt=args.protect_ckpt,
            chase_ckpt=args.chase_ckpt,
            source_run_dir=args.source_run_dir,
        )
        print(f'Protect Skill: {protect_ckpt}')
        print(f'Chase Skill:   {chase_ckpt}')
        top_ckpt = _train_top_stage(output_dir=output_dir, protect_ckpt=protect_ckpt, chase_ckpt=chase_ckpt)
        print('\n========== Done ==========')
        print(f'Output Dir: {output_dir}')
        print(f'Top Policy: {top_ckpt}')
        return

    print('\n========== Ablation Training Pipeline ==========')
    print('Stage 1/3: chase (MLP no-CTDE)')
    chase_ckpt = _train_skill_stage(
        stage_name='ablation_chase_mlp_noctde',
        skill_mode='chase',
        output_dir=output_dir,
        attacker_strategy=AblationEnvParameters.ATTACKER_STRATEGY,
    )

    print('\nStage 2/3: protect2 (MLP no-CTDE)')
    protect_ckpt = _train_skill_stage(
        stage_name='ablation_protect2_mlp_noctde',
        skill_mode='protect2',
        output_dir=output_dir,
        attacker_strategy=AblationEnvParameters.ATTACKER_STRATEGY,
    )

    print('\nStage 3/3: HRL top (MLP no-CTDE, no GRU, no macro-length)')
    top_ckpt = _train_top_stage(output_dir=output_dir, protect_ckpt=protect_ckpt, chase_ckpt=chase_ckpt)

    print('\n========== Done ==========')
    print(f'Output Dir: {output_dir}')
    print(f'Chase Skill:   {chase_ckpt}')
    print(f'Protect Skill: {protect_ckpt}')
    print(f'Top Policy:    {top_ckpt}')


if __name__ == '__main__':
    main()
