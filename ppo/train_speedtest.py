"""
训练速度剖析脚本（支持 PPO 技能训练 / HRL 顶层训练）。

用途：统计训练过程中的平均耗时占比，包括：
- 采样阶段内部：环境计算、网络推理、GAE 等
- 主循环阶段：权重同步、采样等待、数据拼接、网络更新
- 训练设备传输：D2H(权重拉回CPU) 与 H2D(训练批次拷贝到训练设备)

示例：
    # 复现 protect2 训练测速
    python -m ppo.train_speedtest --target protect2 --updates 30 --warmup-updates 3

    # HRL 顶层训练测速（自动寻找最新 protect/chase checkpoint）
    python -m ppo.train_speedtest --target hrl --updates 20 --warmup-updates 2
"""

import argparse
import glob
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import ray
import torch

from ppo.alg_parameters import SetupParameters, TrainingParameters
from ppo.model import Model
from ppo.runner import Runner
from ppo.util import (
    get_adjusted_n_envs,
    get_device,
    get_num_gpus,
    get_ray_temp_dir,
    set_global_seeds,
)
from rule_policies.attacker_global import SUPPORTED_STRATEGIES


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _pct(num: float, den: float) -> float:
    return 100.0 * _safe_div(num, den)


def _sync_cuda(device: torch.device) -> None:
    if isinstance(device, torch.device) and device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _format_table(headers: List[str], rows: List[List[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(row_vals: List[str]) -> str:
        return ' | '.join(val.ljust(widths[i]) for i, val in enumerate(row_vals))

    line = '-+-'.join('-' * w for w in widths)
    out = [fmt_row(headers), line]
    out.extend(fmt_row(r) for r in rows)
    return '\n'.join(out)


def _sum_numeric_dict(dst: Dict[str, float], src: Dict[str, float]) -> None:
    for k, v in src.items():
        if isinstance(v, (int, float)):
            dst[k] += float(v)


def _find_latest_checkpoint(model_prefixes: List[str]) -> Optional[str]:
    candidates = []
    for prefix in model_prefixes:
        pattern = os.path.join('models', f'{prefix}_*', 'best_model.pth')
        candidates.extend(glob.glob(pattern))
    candidates = [p for p in candidates if os.path.isfile(p)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _resolve_hrl_skill_paths(protect_model_path: Optional[str], chase_model_path: Optional[str]) -> Tuple[str, str]:
    if protect_model_path is None:
        protect_model_path = _find_latest_checkpoint([
            'defender_protect2_dense',
            'defender_protect1_dense',
            'defender_protect_dense',
        ])
    if chase_model_path is None:
        chase_model_path = _find_latest_checkpoint(['defender_chase_dense'])

    if protect_model_path is None:
        raise FileNotFoundError('无法找到 protect checkpoint，请传 --protect-model-path')
    if chase_model_path is None:
        raise FileNotFoundError('无法找到 chase checkpoint，请传 --chase-model-path')

    return os.path.abspath(protect_model_path), os.path.abspath(chase_model_path)


def _build_batch(rollouts: List[Dict], target: str, training_mode: str) -> Dict[str, np.ndarray]:
    common = {
        'obs': np.concatenate([r['obs'] for r in rollouts], axis=0),
        'critic_obs': np.concatenate([r['critic_obs'] for r in rollouts], axis=0),
        'actions': np.concatenate([r['actions'] for r in rollouts], axis=0),
        'log_probs': np.concatenate([r['log_probs'] for r in rollouts], axis=0),
        'returns': np.concatenate([r['returns'] for r in rollouts], axis=0),
        'values': np.concatenate([r['values'] for r in rollouts], axis=0),
    }
    if target == 'hrl':
        return common

    if training_mode == 'rl':
        return common
    if training_mode == 'mixed':
        common['expert_actions'] = np.concatenate([r['expert_actions'] for r in rollouts], axis=0)
        return common

    raise ValueError(f'不支持的 training_mode: {training_mode}，目前仅支持 rl/mixed。')


def _batch_to_device(batch: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device=device, dtype=torch.float32)
        else:
            out[k] = torch.as_tensor(v, dtype=torch.float32, device=device)
    return out


def _train_update(model: Model, batch: Dict[str, torch.Tensor], target: str, training_mode: str) -> None:
    if target == 'hrl' or training_mode == 'rl':
        model.train(
            actor_obs=batch['obs'],
            critic_obs=batch['critic_obs'],
            actions=batch['actions'],
            old_log_probs=batch['log_probs'],
            returns=batch['returns'],
            values=batch['values'],
        )
        return

    if training_mode == 'mixed':
        model.train_mixed(
            actor_obs=batch['obs'],
            critic_obs=batch['critic_obs'],
            actions=batch['actions'],
            old_log_probs=batch['log_probs'],
            returns=batch['returns'],
            values=batch['values'],
            expert_actions=batch['expert_actions'],
            il_weight=0.5,
        )
        return

    raise ValueError(f'不支持的 training_mode: {training_mode}，目前仅支持 rl/mixed。')


def _runner_rows(
    runner_totals: Dict[str, float],
    driver_totals: Dict[str, float],
    target: str,
) -> List[List[str]]:
    if target == 'hrl':
        labels: List[Tuple[str, str]] = [
            ('env_macro_step', '环境执行(macro_step)'),
            ('policy_inference', '网络推理(actor act)'),
            ('tensorize', 'obs转Tensor'),
            ('critic_obs', '构建critic obs'),
            ('reward_norm', '奖励归一化'),
            ('buffer_ops', '缓存写入/CPU转换'),
            ('gae_compute', 'GAE计算'),
            ('pack_numpy', 'list->numpy打包'),
            ('bootstrap_value', 'bootstrap value'),
            ('episode_reset', 'episode reset'),
            ('untracked', '采样内其他'),
        ]
    else:
        labels = [
            ('env_step', '环境计算(env.step)'),
            ('policy_inference', '网络推理(actor act)'),
            ('opponent_policy', '对手策略动作'),
            ('expert_policy', '专家策略动作'),
            ('tensorize', 'obs转Tensor'),
            ('critic_obs', '构建critic obs'),
            ('reward_norm', '奖励归一化'),
            ('buffer_ops', '缓存写入/CPU转换'),
            ('gae_compute', 'GAE计算'),
            ('pack_numpy', 'list->numpy打包'),
            ('bootstrap_value', 'bootstrap value'),
            ('episode_reset', 'episode reset'),
            ('untracked', '采样内其他'),
        ]

    rows = []
    rollout_total = runner_totals.get('rollout_total', 0.0)
    rollout_share = _safe_div(driver_totals.get('rollout_collect', 0.0), driver_totals.get('update_total', 0.0))
    total_steps = runner_totals.get('num_steps', 0.0)

    for key, name in labels:
        t = runner_totals.get(key, 0.0)
        pct_rollout = _pct(t, rollout_total)
        pct_total_est = rollout_share * pct_rollout
        ms_per_step = 1000.0 * _safe_div(t, total_steps)
        rows.append([
            name,
            f'{t:.3f}',
            f'{ms_per_step:.4f}',
            f'{pct_rollout:.2f}%',
            f'{pct_total_est:.2f}%',
        ])
    return rows


def _driver_rows(driver_totals: Dict[str, float], total_steps: float) -> List[List[str]]:
    labels = [
        ('weights_d2h', '权重拷贝到CPU(D2H)'),
        ('weights_put', '权重放入Ray对象存储'),
        ('weights_broadcast', '广播权重到Runner'),
        ('rollout_collect', '采样等待(并行总耗时)'),
        ('merge_rollouts', '拼接rollout数据(CPU)'),
        ('batch_to_device', '训练批次拷贝到设备(H2D)'),
        ('train_compute', '训练计算(前向+反向+优化器)'),
        ('driver_other', 'Driver其他开销'),
        ('update_total', '总计'),
    ]
    rows = []
    total = driver_totals.get('update_total', 0.0)
    for key, name in labels:
        t = driver_totals.get(key, 0.0)
        ms_per_step = 1000.0 * _safe_div(t, total_steps)
        rows.append([
            name,
            f'{t:.3f}',
            f'{ms_per_step:.4f}',
            f'{_pct(t, total):.2f}%',
        ])
    return rows


def _create_runners(args, n_envs: int, network_type: str):
    if args.target == 'hrl':
        from hrl.hrl_runner import HRLRunner

        protect_model_path, chase_model_path = _resolve_hrl_skill_paths(
            args.protect_model_path,
            args.chase_model_path,
        )
        env_cfg = {
            'attacker_strategy': args.attacker_strategy,
            'protect_model_path': protect_model_path,
            'chase_model_path': chase_model_path,
            'predictor_hidden_dim': int(args.predictor_hidden_dim),
            'predictor_lr': float(args.predictor_lr),
            'predictor_train': bool(not args.disable_predictor_train),
            'hold_min': int(args.hold_min),
            'hold_max': int(args.hold_max),
            'runner_use_gpu': bool(args.hrl_runner_use_gpu),
        }

        runner_num_gpus = 0.0
        if args.hrl_runner_use_gpu:
            if args.hrl_runner_num_gpus is not None:
                runner_num_gpus = float(args.hrl_runner_num_gpus)
            else:
                # Auto-share one visible GPU across all runners by default.
                runner_num_gpus = 1.0 / float(max(1, n_envs))

        print(f'Protect Skill: {protect_model_path}')
        print(f'Chase Skill:   {chase_model_path}')
        print(f'HRL Runner Inference Device: {"GPU" if args.hrl_runner_use_gpu else "CPU"}')
        print(f'HRL Runner num_gpus per actor: {runner_num_gpus:.6f}')
        return [
            HRLRunner.options(num_gpus=runner_num_gpus).remote(i, env_configs=env_cfg)
            for i in range(n_envs)
        ]

    return [Runner.remote(i, network_type=network_type) for i in range(n_envs)]


def main() -> None:
    parser = argparse.ArgumentParser(description='训练耗时剖析（PPO/HRL）')
    parser.add_argument('--target', type=str, default='protect2', choices=['protect1', 'protect2', 'chase', 'hrl'],
                        help='要测速的训练目标')
    parser.add_argument('--training-mode', type=str, default='rl', choices=['rl', 'mixed'], help='PPO训练模式')
    parser.add_argument('--updates', type=int, default=30, help='总update轮数')
    parser.add_argument('--warmup-updates', type=int, default=3, help='预热update，不计入统计')
    parser.add_argument('--n-envs', type=int, default=None, help='并行Runner数量，默认读取配置')
    parser.add_argument('--n-steps', type=int, default=None, help='每个Runner rollout步数，默认读取配置')
    parser.add_argument('--network-type', type=str, default='nmn', help='PPO网络类型(nmn/mlp)')
    parser.add_argument('--cpu-only', action='store_true', help='强制在CPU上做网络更新')
    parser.add_argument('--disable-auto-adjust-envs', action='store_true', help='关闭按RAM自动扩容并行环境')
    parser.add_argument('--disable-runner-profile', action='store_true', help='关闭runner内部细分profile')

    parser.add_argument('--protect-model-path', type=str, default=None, help='HRL: protect skill checkpoint路径')
    parser.add_argument('--chase-model-path', type=str, default=None, help='HRL: chase skill checkpoint路径')
    parser.add_argument(
        '--attacker-strategy',
        type=str,
        default='random',
        choices=['random'] + SUPPORTED_STRATEGIES,
        help='HRL: attacker策略（random 或具体策略）'
    )
    parser.add_argument('--predictor-hidden-dim', type=int, default=64, help='HRL: GRU hidden dim')
    parser.add_argument('--predictor-lr', type=float, default=1e-3, help='HRL: GRU学习率')
    parser.add_argument('--disable-predictor-train', action='store_true', help='HRL: 关闭GRU在线训练')
    parser.add_argument('--hold-min', type=int, default=3, help='HRL: macro最小保持步数')
    parser.add_argument('--hold-max', type=int, default=15, help='HRL: macro最大保持步数')
    parser.add_argument('--hrl-runner-use-gpu', action='store_true', help='HRL: runner推理改为GPU')
    parser.add_argument('--hrl-runner-num-gpus', type=float, default=None,
                        help='HRL: 每个runner申请的GPU资源，默认自动=1/n_envs')

    args = parser.parse_args()

    if args.warmup_updates >= args.updates:
        raise ValueError('--warmup-updates 必须小于 --updates')

    if args.target == 'hrl' and args.training_mode != 'rl':
        raise ValueError('HRL 顶层测速仅支持 --training-mode rl')

    SetupParameters.SKILL_MODE = args.target
    TrainingParameters.TRAINING_MODE = args.training_mode

    if args.n_steps is not None:
        TrainingParameters.N_STEPS = int(args.n_steps)

    if args.n_envs is not None:
        n_envs = int(args.n_envs)
    else:
        n_envs = int(TrainingParameters.N_ENVS)
        if not args.disable_auto_adjust_envs:
            n_envs = get_adjusted_n_envs(n_envs)

    n_steps = int(TrainingParameters.N_STEPS)

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(getattr(SetupParameters, 'GPU_ID', 0)))

    set_global_seeds(SetupParameters.SEED)
    device = get_device(prefer_gpu=not args.cpu_only, gpu_id=getattr(SetupParameters, 'GPU_ID', 0))

    network_type = 'hrl_top' if args.target == 'hrl' else args.network_type
    profile_runner = not args.disable_runner_profile
    step_unit = 'macro_step' if args.target == 'hrl' else 'env_step'

    print('=' * 72)
    print('训练速度剖析')
    print(f'Target: {args.target}')
    print(f'Training Mode: {TrainingParameters.TRAINING_MODE}')
    print(f'Network Type: {network_type}')
    print(f'Device: {device} (GPU count visible: {get_num_gpus()})')
    print(f'Runners: {n_envs}, Steps/Runner: {n_steps}, Updates: {args.updates}, Warmup: {args.warmup_updates}')
    print(f'Runner Profile: {profile_runner}')
    print('=' * 72)

    ray_kwargs = {
        'num_cpus': os.cpu_count() or n_envs,
        'num_gpus': 0,
        'include_dashboard': False,
        'ignore_reinit_error': True,
        'log_to_driver': False,
    }
    if args.target == 'hrl' and args.hrl_runner_use_gpu:
        ray_kwargs['num_gpus'] = max(1, get_num_gpus())

    ray_tmp = get_ray_temp_dir()
    if ray_tmp:
        ray_kwargs['_temp_dir'] = ray_tmp

    driver_totals = defaultdict(float)
    runner_totals = defaultdict(float)
    counted_updates = 0

    ray.init(**ray_kwargs)
    try:
        model = Model(device=device, global_model=True, network_type=network_type)
        runners = _create_runners(args, n_envs=n_envs, network_type=network_type)

        for update_idx in range(1, args.updates + 1):
            phase = defaultdict(float)
            update_start = time.perf_counter()

            _sync_cuda(model.device)
            t0 = time.perf_counter()
            weights = model.get_weights()
            _sync_cuda(model.device)
            phase['weights_d2h'] = time.perf_counter() - t0

            t0 = time.perf_counter()
            weight_id = ray.put(weights)
            phase['weights_put'] = time.perf_counter() - t0

            t0 = time.perf_counter()
            ray.get([r.set_weights.remote(weight_id) for r in runners])
            phase['weights_broadcast'] = time.perf_counter() - t0

            t0 = time.perf_counter()
            rollout_futures = [r.run.remote(n_steps, profile_runner) for r in runners]
            rollouts = ray.get(rollout_futures)
            phase['rollout_collect'] = time.perf_counter() - t0

            if profile_runner:
                rollout_timing_sum = defaultdict(float)
                for rollout in rollouts:
                    _sum_numeric_dict(rollout_timing_sum, rollout.get('timings', {}))
            else:
                rollout_timing_sum = {}

            t0 = time.perf_counter()
            batch_cpu = _build_batch(rollouts, target=args.target, training_mode=args.training_mode)
            phase['merge_rollouts'] = time.perf_counter() - t0

            _sync_cuda(model.device)
            t0 = time.perf_counter()
            batch_device = _batch_to_device(batch_cpu, model.device)
            _sync_cuda(model.device)
            phase['batch_to_device'] = time.perf_counter() - t0

            _sync_cuda(model.device)
            t0 = time.perf_counter()
            _train_update(model, batch_device, target=args.target, training_mode=args.training_mode)
            _sync_cuda(model.device)
            phase['train_compute'] = time.perf_counter() - t0

            phase['network_update'] = phase['batch_to_device'] + phase['train_compute']
            phase['update_total'] = time.perf_counter() - update_start
            tracked = (
                phase['weights_d2h'] +
                phase['weights_put'] +
                phase['weights_broadcast'] +
                phase['rollout_collect'] +
                phase['merge_rollouts'] +
                phase['batch_to_device'] +
                phase['train_compute']
            )
            phase['driver_other'] = max(0.0, phase['update_total'] - tracked)

            include = update_idx > args.warmup_updates
            tag = 'COUNT' if include else 'WARMUP'
            print(
                f"[{update_idx:03d}/{args.updates}] {tag} "
                f"update={phase['update_total']:.3f}s "
                f"rollout={phase['rollout_collect']:.3f}s "
                f"train_compute={phase['train_compute']:.3f}s "
                f"h2d={phase['batch_to_device']:.3f}s d2h={phase['weights_d2h']:.3f}s"
            )

            if include:
                counted_updates += 1
                _sum_numeric_dict(driver_totals, phase)
                _sum_numeric_dict(runner_totals, rollout_timing_sum)

        if counted_updates <= 0:
            raise RuntimeError('没有可统计的update，请增大 --updates 或减小 --warmup-updates')

        total_steps = counted_updates * n_envs * n_steps
        wall_time = driver_totals['update_total']
        steps_per_sec = _safe_div(total_steps, wall_time)
        ms_per_step = 1000.0 * _safe_div(wall_time, total_steps)

        transfer_roundtrip = driver_totals['weights_d2h'] + driver_totals['batch_to_device']
        transfer_pct = _pct(transfer_roundtrip, wall_time)

        print('\n' + '=' * 72)
        print('1) 主训练循环(按update墙钟时间)')
        print(_format_table(
            headers=['阶段', '累计秒', f'每{step_unit}毫秒', '占总训练比例'],
            rows=_driver_rows(driver_totals, total_steps),
        ))

        print('\n' + '=' * 72)
        print('2) CPU/GPU传输关键指标')
        print(f"D2H(权重get_weights): {driver_totals['weights_d2h']:.3f}s ({_pct(driver_totals['weights_d2h'], wall_time):.2f}%)")
        print(f"H2D(训练批次拷贝):    {driver_totals['batch_to_device']:.3f}s ({_pct(driver_totals['batch_to_device'], wall_time):.2f}%)")
        print(f'来回传输合计(D2H+H2D): {transfer_roundtrip:.3f}s ({transfer_pct:.2f}%)')
        print(f"训练计算(不含H2D):     {driver_totals['train_compute']:.3f}s ({_pct(driver_totals['train_compute'], wall_time):.2f}%)")

        if profile_runner:
            print('\n' + '=' * 72)
            print('3) 采样阶段内部细分(来自Runner profile)')
            print(_format_table(
                headers=['采样子阶段', '累计秒', f'每{step_unit}毫秒', '占采样比例', '估算占总训练比例'],
                rows=_runner_rows(runner_totals, driver_totals, target=args.target),
            ))

        print('\n' + '=' * 72)
        print('4) 总体吞吐')
        print(f'统计update数: {counted_updates}')
        print(f'总{step_unit}s: {total_steps}')
        print(f'总墙钟时间: {wall_time:.3f}s')
        print(f'吞吐: {steps_per_sec:.2f} {step_unit}s/s')
        print(f'平均: {ms_per_step:.4f} ms/{step_unit}')

        if args.target == 'hrl' and runner_totals.get('num_primitive_steps', 0.0) > 0:
            primitive_steps = runner_totals['num_primitive_steps']
            primitive_sps = _safe_div(primitive_steps, wall_time)
            primitive_ms = 1000.0 * _safe_div(wall_time, primitive_steps)
            print(f'估算primitive steps: {int(primitive_steps)}')
            print(f'估算primitive吞吐: {primitive_sps:.2f} steps/s')
            print(f'估算primitive平均: {primitive_ms:.4f} ms/step')

        print('=' * 72)

    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()
