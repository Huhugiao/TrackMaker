"""Train a feed-forward Attacker with competence-gated multi-style PPO."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def build_env_config(defender_checkpoint: Path) -> Dict:
    """Build the no-shield Attacker environment shared by all PPO workers."""
    from configs.attacker_config import RewardParameters, SetupParameters

    checkpoint = Path(defender_checkpoint).expanduser().resolve()
    return {
        "defender_strategy": "skill_protect",
        "defender_strategy_pool": ("skill_protect",),
        "defender_strategy_weights": {"skill_protect": 1.0},
        "defender_strategy_params": {
            "protect_skill_path": str(checkpoint),
            "device": "cpu",
        },
        "reward_mode": str(SetupParameters.ENV_REWARD_MODE),
        "curriculum_enabled": True,
        "curriculum_bins": tuple(dict(item) for item in SetupParameters.CURRICULUM_BINS),
        "curriculum_keep_base_speeds": True,
        "target_progress_metric": "astar",
        "path_grid_size": float(RewardParameters.PATH_GRID_SIZE),
        "path_obstacle_padding": float(RewardParameters.PATH_OBSTACLE_PADDING),
        "reward_parameters": {
            "ATTACKER_COLLISION_MAX_EVENTS": float(
                RewardParameters.ATTACKER_COLLISION_MAX_EVENTS
            ),
        },
        "reward_styles": tuple(dict(item) for item in RewardParameters.MULTISTYLE_REWARDS),
    }


def build_run_contract(args) -> Dict:
    from attacker.multistyle_rewards import reward_weight_matrix, validate_reward_styles
    from configs.attacker_config import (
        MultiStylePPOParameters,
        RewardParameters,
        SetupParameters,
    )

    styles = validate_reward_styles(RewardParameters.MULTISTYLE_REWARDS)
    stages = _stage_definitions(args.curriculum_profile)
    style_assignments, style_env_counts = _resolve_style_assignments(
        tuple(style["name"] for style in styles),
        int(args.num_envs),
        args.style_env_counts,
    )
    return {
        "status": "active",
        "algorithm": "multi_style_ppo",
        "network_type": "attacker_nmn_mlp_multistyle_ppo",
        "recurrent": False,
        "action_safety_layer": False,
        "on_policy": True,
        "reward_relabeling": False,
        "worker_style_assignment": "fixed_explicit",
        "style_env_counts": style_env_counts,
        "worker_style_ids": style_assignments,
        "reward_styles": [dict(style) for style in styles],
        "reward_weight_matrix": reward_weight_matrix(styles).tolist(),
        "num_envs": int(args.num_envs),
        "rollout_steps_per_env": int(args.rollout_steps),
        "max_environment_steps": int(args.max_steps),
        "minibatch_size": int(MultiStylePPOParameters.MINIBATCH_SIZE),
        "epochs": int(MultiStylePPOParameters.N_EPOCHS),
        "gamma": float(MultiStylePPOParameters.GAMMA),
        "gae_lambda": float(MultiStylePPOParameters.GAE_LAMBDA),
        "clip_range": float(MultiStylePPOParameters.CLIP_RANGE),
        "learning_rate": float(args.learning_rate),
        "final_learning_rate": float(args.final_learning_rate),
        "evaluation": {
            "interval": int(args.eval_interval),
            "gate_episodes_per_style": int(args.gate_eval_episodes_per_style),
            "default_episodes_per_style": int(args.default_eval_episodes_per_style),
            "paired_seeds_across_styles": True,
            "promotion_requires_all_styles": True,
        },
        "curriculum_stages": _jsonable(stages),
        "curriculum_profile": str(args.curriculum_profile),
        "initial_checkpoint": (
            None
            if args.init_checkpoint is None
            else str(Path(args.init_checkpoint).expanduser().resolve())
        ),
        "seed": int(args.seed),
        "eval_seed": int(args.eval_seed),
        "defender_checkpoint": str(Path(args.defender_checkpoint).resolve()),
        "env_config": _jsonable(build_env_config(args.defender_checkpoint)),
    }


def _stage_definitions(profile="bootstrap"):
    from attacker.curriculum import build_gate_bins, build_weighted_bins
    from configs.attacker_config import SetupParameters

    profile = str(profile).strip().lower()
    if profile == "bootstrap":
        raw_stages = SetupParameters.CURRICULUM_STAGES
    elif profile == "diversity_continuation":
        raw_stages = SetupParameters.DIVERSITY_CONTINUATION_STAGES
    else:
        raise ValueError(f"unknown Attacker curriculum profile: {profile!r}")
    stages = []
    for raw_stage in raw_stages:
        stage = dict(raw_stage)
        stage["training_bins"] = build_weighted_bins(
            SetupParameters.CURRICULUM_BINS,
            stage["train_weights"],
        )
        stage["gate_bins"] = build_gate_bins(
            SetupParameters.CURRICULUM_BINS,
            stage["gate_bin"],
        )
        stages.append(stage)
    return tuple(stages)


def _resolve_style_assignments(style_names, num_envs, specification=None):
    style_names = tuple(str(name) for name in style_names)
    num_envs = int(num_envs)
    if specification is None or not str(specification).strip():
        if num_envs < len(style_names) or num_envs % len(style_names) != 0:
            raise ValueError(
                f"--num-envs must be a positive multiple of {len(style_names)} "
                "unless --style-env-counts is supplied"
            )
        counts = {name: num_envs // len(style_names) for name in style_names}
    else:
        counts = {name: 0 for name in style_names}
        for item in str(specification).split(","):
            name, separator, raw_count = item.strip().partition("=")
            if not separator or name not in counts:
                raise ValueError(
                    "--style-env-counts must use known name=count pairs; "
                    f"valid={style_names}"
                )
            count = int(raw_count)
            if count <= 0:
                raise ValueError("every active reward style requires at least one environment")
            counts[name] = count
        if any(count <= 0 for count in counts.values()):
            raise ValueError("--style-env-counts must assign every active reward style")
        if sum(counts.values()) != num_envs:
            raise ValueError(
                f"style environment counts sum to {sum(counts.values())}, expected {num_envs}"
            )
    assignments = []
    for style_id, style_name in enumerate(style_names):
        assignments.extend([style_id] * int(counts[style_name]))
    return tuple(assignments), counts


def _merge_rollouts(
    worker_batches,
    expected_policy_version,
    style_names,
    expected_style_steps=None,
):
    import numpy as np

    stale = sorted(
        {
            int(batch.get("policy_version", -1))
            for batch in worker_batches
            if int(batch.get("policy_version", -1)) != int(expected_policy_version)
        }
    )
    if stale:
        raise RuntimeError(
            f"PPO rejected stale rollout versions {stale}; expected {expected_policy_version}"
        )
    array_keys = (
        "obs",
        "pre_tanh_actions",
        "log_probs",
        "values",
        "returns",
        "advantages",
        "dones",
        "style_ids",
    )
    merged = {
        key: np.concatenate([batch[key] for batch in worker_batches], axis=0)
        for key in array_keys
    }
    episodes = []
    curriculum_steps = Counter()
    worker_steps = Counter()
    for batch in worker_batches:
        episodes.extend(batch.get("episodes", ()))
        curriculum_steps.update(batch.get("curriculum_steps", {}))
        worker_steps[str(batch["style_name"])] += int(batch["obs"].shape[0])
    if expected_style_steps is None:
        invalid = any(worker_steps[name] <= 0 for name in style_names)
        expected_description = "at least one sample per style"
    else:
        invalid = any(
            worker_steps[name] != int(expected_style_steps[name])
            for name in style_names
        )
        expected_description = str(dict(expected_style_steps))
    if invalid:
        raise RuntimeError(
            f"PPO style rollout allocation mismatch; got={dict(worker_steps)}, "
            f"expected={expected_description}"
        )
    merged["episodes"] = episodes
    merged["curriculum_steps"] = dict(curriculum_steps)
    merged["style_steps"] = dict(worker_steps)
    merged["policy_version"] = int(expected_policy_version)
    return merged


def _evaluate_parallel(ray, workers, weights, style_names, episodes_per_style, bins):
    weights_ref = ray.put(weights)
    bins = tuple(dict(item) for item in bins)
    pending = []
    for style_id, _style_name in enumerate(style_names):
        worker = workers[style_id % len(workers)]
        pending.append(
            worker.evaluate_style.remote(
                weights_ref,
                style_id=style_id,
                num_episodes=int(episodes_per_style),
                curriculum_bins=bins,
            )
        )
    evaluations = {}
    for result in ray.get(pending):
        evaluations.update(result)
    return evaluations


def _print_evaluation(step: int, label: str, evaluations: Dict) -> None:
    print(f"\n[eval step={step:,} distribution={label}]", flush=True)
    for style, metrics in evaluations.items():
        print(
            f"  {style:12s} success={metrics['target_success_rate']:.2%} "
            f"captured={metrics['defender_capture_rate']:.2%} "
            f"timeout={metrics['timeout_rate']:.2%} "
            f"reward={metrics['mean_reward']:.3f}",
            flush=True,
        )


def train(args) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")

    import numpy as np
    import ray
    import torch

    from attacker.curriculum import CompetenceCurriculum, build_gate_bins
    from attacker.multistyle_rewards import validate_reward_styles
    from attacker.ppo_model import AttackerMultiStylePPO
    from attacker.ppo_runner import MultiStylePPORunner
    from configs.attacker_config import (
        MultiStylePPOParameters,
        RewardParameters,
        SetupParameters,
    )

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = Path(args.defender_checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"learned Defender checkpoint not found: {checkpoint}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    resume_path = (
        Path(args.resume_checkpoint).expanduser().resolve()
        if args.resume_checkpoint is not None
        else None
    )
    init_path = (
        Path(args.init_checkpoint).expanduser().resolve()
        if args.init_checkpoint is not None
        else None
    )
    if resume_path is None and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty Attacker run directory: {output_dir}"
        )
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(f"PPO resume checkpoint not found: {resume_path}")
    if init_path is not None and not init_path.is_file():
        raise FileNotFoundError(f"PPO initialization checkpoint not found: {init_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    styles = validate_reward_styles(RewardParameters.MULTISTYLE_REWARDS)
    style_names = tuple(style["name"] for style in styles)
    style_assignments, style_env_counts = _resolve_style_assignments(
        style_names,
        int(args.num_envs),
        args.style_env_counts,
    )
    stages = _stage_definitions(args.curriculum_profile)
    curriculum = CompetenceCurriculum(
        stages=stages,
        style_names=style_names,
        plateau_patience=int(MultiStylePPOParameters.PLATEAU_EVAL_PATIENCE),
        plateau_min_delta=float(MultiStylePPOParameters.PLATEAU_MIN_DELTA),
    )
    contract = build_run_contract(args)
    (output_dir / "run_config.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    print(
        f"Multi-style PPO | device={device} | envs={args.num_envs} | "
        f"styles={style_names} | shield=False | recurrent=False",
        flush=True,
    )
    print(f"Output: {output_dir}", flush=True)

    if args.dry_run:
        from attacker.ppo_runner import LocalMultiStylePPORunner

        model = AttackerMultiStylePPO(styles, device="cpu", training=True)
        if init_path is not None:
            source_model, _source_checkpoint = AttackerMultiStylePPO.from_checkpoint(
                init_path,
                device="cpu",
                training=False,
            )
            if tuple(source_model.reward_styles) != tuple(styles):
                raise ValueError("initial checkpoint reward styles differ from active styles")
            model.set_weights(source_model.weights())
        runners = [
            LocalMultiStylePPORunner(
                worker_id=style_id,
                env_config=build_env_config(checkpoint),
                reward_styles=styles,
                seed=args.seed,
                eval_seed=args.eval_seed,
            )
            for style_id in range(len(styles))
        ]
        try:
            version = 0
            batches = [
                runner.collect(
                    min(8, int(args.rollout_steps)),
                    model.weights(),
                    stages[0]["training_bins"],
                    version,
                )
                for runner in runners
            ]
            batch = _merge_rollouts(batches, version, style_names)
            metrics = model.update(batch)
            print(
                f"Dry-run OK: batch={batch['obs'].shape}, "
                f"policy_loss={metrics['policy_loss']:.4f}, "
                f"value_loss={metrics['value_loss']:.4f}",
                flush=True,
            )
        finally:
            for runner in runners:
                runner.close()
        return 0

    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        num_cpus=max(1, int(args.num_envs) + len(styles)),
        _temp_dir=str(Path("/tmp") / f"trackmaker_multistyle_ppo_{args.seed}"),
        log_to_driver=False,
    )
    env_config = build_env_config(checkpoint)
    workers = [
        MultiStylePPORunner.remote(
            worker_id=index,
            env_config=env_config,
            reward_styles=styles,
            seed=args.seed,
            eval_seed=args.eval_seed,
            style_id=style_assignments[index],
        )
        for index in range(int(args.num_envs))
    ]
    # Evaluation must not reset or otherwise perturb an in-progress training
    # episode.  One dedicated fixed-style worker per head is sufficient.
    eval_workers = [
        MultiStylePPORunner.remote(
            worker_id=int(args.num_envs) + style_id,
            env_config=env_config,
            reward_styles=styles,
            seed=args.seed,
            eval_seed=args.eval_seed,
            style_id=style_id,
        )
        for style_id in range(len(styles))
    ]

    resume_contract = {
        "reward_styles": [dict(style) for style in styles],
        "curriculum_stages": _jsonable(stages),
        "defender_checkpoint": str(checkpoint),
        "num_envs": int(args.num_envs),
        "rollout_steps": int(args.rollout_steps),
        "style_env_counts": dict(style_env_counts),
        "curriculum_profile": str(args.curriculum_profile),
        "learning_rate": float(args.learning_rate),
        "final_learning_rate": float(args.final_learning_rate),
    }

    worker_sampling_states = []
    if resume_path is None:
        model = AttackerMultiStylePPO(styles, device=device, training=True)
        if init_path is not None:
            source_model, source_checkpoint = AttackerMultiStylePPO.from_checkpoint(
                init_path,
                device=device,
                training=False,
            )
            if tuple(source_model.reward_styles) != tuple(styles):
                raise ValueError("initial checkpoint reward styles differ from active styles")
            model.set_weights(source_model.weights())
            print(
                f"Initialized policy weights from {init_path} "
                f"(source_step={int(source_checkpoint.get('step', 0)):,}); optimizer reset",
                flush=True,
            )
        step = 0
        policy_version = 0
        best_scores = {name: -float("inf") for name in style_names}
    else:
        model, restored = AttackerMultiStylePPO.from_checkpoint(
            resume_path,
            device=device,
            training=True,
        )
        step = int(restored.get("step", 0))
        policy_version = int(restored.get("policy_version", 0))
        best_scores = {
            name: float(restored.get("best_target_success_rates", {}).get(name, -float("inf")))
            for name in style_names
        }
        curriculum.load_state_dict(restored.get("curriculum_state", {}))
        if tuple(model.reward_styles) != tuple(styles):
            raise ValueError(
                "resume checkpoint reward styles differ from the active ordered contract"
            )
        if restored.get("resume_contract") != resume_contract:
            raise ValueError(
                "resume checkpoint training contract differs from reward/course/Defender/"
                "sampler configuration"
            )
        restored_sampling_states = restored.get("worker_sampling_states")
        if not isinstance(restored_sampling_states, (tuple, list)) or len(
            restored_sampling_states
        ) != len(workers):
            raise ValueError("resume checkpoint has incomplete PPO worker sampling states")
        ray.get(
            [
                worker.set_sampling_state.remote(state)
                for worker, state in zip(workers, restored_sampling_states)
            ]
        )
        worker_sampling_states = list(restored_sampling_states)
        print(
            f"Resumed: step={step:,} policy_version={policy_version} "
            f"stage={curriculum.stage['name']}",
            flush=True,
        )

    next_eval = ((step // int(args.eval_interval)) + 1) * int(args.eval_interval)
    next_save = ((step // int(args.save_interval)) + 1) * int(args.save_interval)
    next_log = ((step // int(args.log_interval)) + 1) * int(args.log_interval)
    recent_episodes = deque(maxlen=400)
    latest_metrics = {}
    start_time = time.monotonic()
    start_step = step
    log_path = output_dir / "training_log.jsonl"
    default_bins = build_gate_bins(SetupParameters.CURRICULUM_BINS, "default")

    def checkpoint_metadata():
        return {
            "policy_version": int(policy_version),
            "curriculum_state": curriculum.state_dict(),
            "curriculum_stage": str(curriculum.stage["name"]),
            "best_target_success_rates": dict(best_scores),
            "resume_contract": resume_contract,
            "worker_sampling_states": list(worker_sampling_states),
        }

    try:
        while step < int(args.max_steps):
            active_stage = curriculum.stage
            weights_ref = ray.put(model.weights())
            pending = [
                worker.collect.remote(
                    int(args.rollout_steps),
                    weights_ref,
                    active_stage["training_bins"],
                    int(policy_version),
                )
                for worker in workers
            ]
            worker_batches = ray.get(pending)
            worker_sampling_states = [
                batch["sampling_state"] for batch in worker_batches
            ]
            merged = _merge_rollouts(
                worker_batches,
                expected_policy_version=policy_version,
                style_names=style_names,
                expected_style_steps={
                    name: int(style_env_counts[name]) * int(args.rollout_steps)
                    for name in style_names
                },
            )
            collected = int(merged["obs"].shape[0])
            step += collected
            recent_episodes.extend(merged["episodes"])
            learning_rate = model.update_learning_rate(
                min(1.0, step / float(max(1, int(args.max_steps)))),
                initial=float(args.learning_rate),
                final=float(args.final_learning_rate),
            )
            latest_metrics = model.update(merged)
            policy_version += 1

            if step >= next_log:
                elapsed = max(1e-6, time.monotonic() - start_time)
                trained = max(0, step - start_step)
                rate = trained / elapsed
                remaining = max(0, int(args.max_steps) - step)
                eta = datetime.now() + timedelta(seconds=remaining / max(rate, 1e-6))
                recent_by_style = {}
                for style_name in style_names:
                    relevant = [
                        episode["target_success"]
                        for episode in recent_episodes
                        if episode["behavior_style"] == style_name
                    ]
                    recent_by_style[style_name] = (
                        float(np.mean(relevant)) if relevant else 0.0
                    )
                record = {
                    "step": int(step),
                    "policy_version": int(policy_version),
                    "stage": str(curriculum.stage["name"]),
                    "steps_per_second": float(rate),
                    "recent_target_success_rate_by_style": recent_by_style,
                    "style_steps": merged["style_steps"],
                    "curriculum_steps": merged["curriculum_steps"],
                    "learning_rate": float(learning_rate),
                    "training": latest_metrics,
                    "eta": eta.isoformat(),
                }
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                recent_text = " ".join(
                    f"{name}={recent_by_style[name]:.1%}" for name in style_names
                )
                print(
                    f"step={step:,}/{args.max_steps:,} stage={curriculum.stage['name']} "
                    f"rate={rate:.1f} step/s KL={latest_metrics['approx_kl']:.4f} "
                    f"recent[{recent_text}] ETA={eta:%Y-%m-%d %H:%M}",
                    flush=True,
                )
                while next_log <= step:
                    next_log += int(args.log_interval)

            if step >= next_eval:
                evaluated_stage = dict(curriculum.stage)
                gate_evaluations = _evaluate_parallel(
                    ray,
                    eval_workers,
                    model.weights(),
                    style_names,
                    args.gate_eval_episodes_per_style,
                    evaluated_stage["gate_bins"],
                )
                if str(evaluated_stage["gate_bin"]) == "default":
                    default_evaluations = gate_evaluations
                else:
                    default_evaluations = _evaluate_parallel(
                        ray,
                        eval_workers,
                        model.weights(),
                        style_names,
                        args.default_eval_episodes_per_style,
                        default_bins,
                    )
                _print_evaluation(
                    step,
                    f"frontier:{evaluated_stage['gate_bin']}",
                    gate_evaluations,
                )
                _print_evaluation(step, "default", default_evaluations)
                decision = curriculum.observe(step, gate_evaluations)
                print(
                    f"[gate] stage={decision['stage']} frontier_min="
                    f"{decision['frontier_score']:.2%} threshold={decision['threshold']} "
                    f"passes={decision['consecutive_passes']}/"
                    f"{decision['required_consecutive_passes']} "
                    f"promoted={decision['promoted']} "
                    f"criteria={decision['criteria_checks']}",
                    flush=True,
                )
                evaluation_record = {
                    "step": int(step),
                    "policy_version": int(policy_version),
                    "evaluated_stage": evaluated_stage["name"],
                    "gate_bin": evaluated_stage["gate_bin"],
                    "gate": gate_evaluations,
                    "default": default_evaluations,
                    "decision": decision,
                    "curriculum_state": curriculum.state_dict(),
                }
                (output_dir / f"evaluation_{step:09d}.json").write_text(
                    json.dumps(evaluation_record, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                for style_name, metrics in default_evaluations.items():
                    if style_name == "evasive" and "goal_rush" in default_evaluations:
                        capture_reduction = float(
                            default_evaluations["goal_rush"]["defender_capture_rate"]
                            - metrics["defender_capture_rate"]
                        )
                        score = float(
                            metrics["target_success_rate"]
                            + 0.5 * max(0.0, capture_reduction)
                            - 0.1 * metrics["timeout_rate"]
                        )
                        selection_metric = (
                            "default_success_plus_capture_diversity_minus_timeout"
                        )
                    else:
                        capture_reduction = None
                        score = float(metrics["target_success_rate"])
                        selection_metric = "default_target_success_rate"
                    if score > best_scores[style_name]:
                        best_scores[style_name] = score
                        model.save(
                            output_dir / f"best_{style_name}.pth",
                            step=step,
                            extra_metadata={
                                **checkpoint_metadata(),
                                "selected_style": style_name,
                                "selection_metric": selection_metric,
                                "selection_score": score,
                                "capture_reduction_vs_goal_rush": capture_reduction,
                                "default_evaluation": metrics,
                            },
                        )
                if decision["promoted"]:
                    model.save(
                        output_dir
                        / f"promotion_{decision['stage']}_to_{decision['next_stage']}.pth",
                        step=step,
                        extra_metadata=checkpoint_metadata(),
                    )
                    print(
                        f"[curriculum] promoted {decision['stage']} -> "
                        f"{decision['next_stage']}; all workers switch next rollout",
                        flush=True,
                    )
                if decision["plateau_alert"]:
                    print(
                        f"[ALERT] stage={decision['stage']} frontier has not improved "
                        f"for {MultiStylePPOParameters.PLATEAU_EVAL_PATIENCE} evaluations",
                        flush=True,
                    )
                while next_eval <= step:
                    next_eval += int(args.eval_interval)

            if step >= next_save:
                model.save(
                    output_dir / "latest_model.pth",
                    step=step,
                    extra_metadata=checkpoint_metadata(),
                )
                while next_save <= step:
                    next_save += int(args.save_interval)

        model.save(
            output_dir / "final_model.pth",
            step=step,
            extra_metadata=checkpoint_metadata(),
        )
        print(f"Training complete: {step:,} environment steps", flush=True)
        return 0
    except KeyboardInterrupt:
        model.save(
            output_dir / "interrupted_model.pth",
            step=step,
            extra_metadata=checkpoint_metadata(),
        )
        print(f"Interrupted safely at step={step:,}; checkpoint saved", flush=True)
        return 130
    finally:
        for worker in [*workers, *eval_workers]:
            worker.close.remote()
        ray.shutdown()


def parse_args(argv: Optional[Sequence[str]] = None):
    from configs.attacker_config import MultiStylePPOParameters, SetupParameters

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = (
        PROJECT_ROOT / "models" / f"attacker_nmn_mlp_multistyle_ppo_{timestamp}"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--resume-checkpoint", type=Path, default=None)
    checkpoint_group.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--curriculum-profile",
        choices=("bootstrap", "diversity_continuation"),
        default="bootstrap",
    )
    parser.add_argument(
        "--style-env-counts",
        type=str,
        default=None,
        help="Explicit allocation such as goal_rush=4,evasive=8",
    )
    parser.add_argument(
        "--defender-checkpoint",
        type=Path,
        default=Path(SetupParameters.PROTECT_SKILL_PATH),
    )
    parser.add_argument("--seed", type=int, default=int(SetupParameters.SEED))
    parser.add_argument("--eval-seed", type=int, default=int(SetupParameters.EVAL_FIXED_SEED))
    parser.add_argument("--gpu-id", type=int, default=int(SetupParameters.GPU_ID))
    parser.add_argument("--num-envs", type=int, default=int(MultiStylePPOParameters.N_ENVS))
    parser.add_argument(
        "--rollout-steps", type=int, default=int(MultiStylePPOParameters.ROLLOUT_STEPS)
    )
    parser.add_argument("--max-steps", type=int, default=int(MultiStylePPOParameters.N_MAX_STEPS))
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=float(MultiStylePPOParameters.LEARNING_RATE),
    )
    parser.add_argument(
        "--final-learning-rate",
        type=float,
        default=float(MultiStylePPOParameters.LR_FINAL),
    )
    parser.add_argument(
        "--eval-interval", type=int, default=int(MultiStylePPOParameters.EVAL_INTERVAL)
    )
    parser.add_argument(
        "--gate-eval-episodes-per-style",
        type=int,
        default=int(MultiStylePPOParameters.GATE_EVAL_EPISODES_PER_STYLE),
    )
    parser.add_argument(
        "--default-eval-episodes-per-style",
        type=int,
        default=int(MultiStylePPOParameters.DEFAULT_EVAL_EPISODES_PER_STYLE),
    )
    parser.add_argument(
        "--save-interval", type=int, default=int(MultiStylePPOParameters.SAVE_INTERVAL)
    )
    parser.add_argument(
        "--log-interval", type=int, default=int(MultiStylePPOParameters.LOG_INTERVAL)
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for name in (
        "num_envs",
        "rollout_steps",
        "max_steps",
        "eval_interval",
        "gate_eval_episodes_per_style",
        "default_eval_episodes_per_style",
        "save_interval",
        "log_interval",
    ):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if float(args.learning_rate) <= 0.0 or float(args.final_learning_rate) <= 0.0:
        parser.error("learning rates must be positive")
    if float(args.final_learning_rate) > float(args.learning_rate):
        parser.error("--final-learning-rate cannot exceed --learning-rate")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    return train(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
