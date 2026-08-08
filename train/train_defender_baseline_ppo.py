"""Train a fresh Protect Defender with the proven legacy baseline reward."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attacker.frozen_pool import (  # noqa: E402
    FROZEN_ATTACKER_POOL,
    PROGRAMMATIC_ATTACKER_ALIASES,
    resolved_learned_attacker_specs,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Resume step/LR progress and weights from --restore-checkpoint.",
    )
    parser.add_argument("--restore-checkpoint")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--eval-seed", type=int, default=187000)
    parser.add_argument("--network-type", default="mlp_ctde")
    parser.add_argument("--max-steps", type=int, default=30_000_000)
    parser.add_argument("--num-envs", type=int, default=6)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--minibatch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--tbptt-steps", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--final-learning-rate", type=float, default=3e-4)
    parser.add_argument("--learning-rate-schedule", choices=("constant", "linear", "cosine"), default="cosine")
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument(
        "--reward-normalization",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--eval-interval", type=int, default=250_000)
    parser.add_argument("--save-interval", type=int, default=500_000)
    parser.add_argument("--eval-episodes", type=int, default=48)
    parser.add_argument("--balanced-eval-episodes", type=int, default=32)
    return parser


def _default_output_dir(network_type: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "models" / f"defender_protect_{network_type}_frozen6_{timestamp}"


def _configure_environment(
    args,
    output_dir: Path,
    *,
    skill_mode: str = "baseline",
    policy_name: str = "protect",
) -> None:
    learned_specs = resolved_learned_attacker_specs(PROJECT_ROOT)
    equal_weights = [1.0 / len(FROZEN_ATTACKER_POOL)] * len(FROZEN_ATTACKER_POOL)
    values = {
        "SKILL_GPU_ID": args.gpu_id,
        "SKILL_MODE": str(skill_mode),
        "BOTTOM_NETWORK_TYPE": args.network_type,
        "SKILL_MODEL_PATH": output_dir,
        "SKILL_SEED": args.seed,
        "SKILL_TRAIN_USE_RANDOM_SEED": 1,
        "SKILL_EVAL_USE_RANDOM_SEED": 0,
        "SKILL_EVAL_FIXED_SEED": args.eval_seed,
        "SKILL_N_MAX_STEPS": args.max_steps,
        "SKILL_N_ENVS": args.num_envs,
        "SKILL_N_STEPS": args.rollout_steps,
        "SKILL_MINIBATCH_SIZE": args.minibatch_size,
        "SKILL_N_EPOCHS": args.epochs,
        "SKILL_TBPTT_STEPS": args.tbptt_steps,
        "SKILL_LR": args.learning_rate,
        "SKILL_LR_FINAL": args.final_learning_rate,
        "SKILL_LR_SCHEDULE": args.learning_rate_schedule,
        "SKILL_GAMMA": args.gamma,
        "SKILL_LAM": 0.95,
        "SKILL_CLIP_RANGE": 0.20,
        "SKILL_VALUE_CLIP_RANGE": 0.20,
        "SKILL_ENTROPY_COEF": args.entropy_coef,
        "SKILL_REWARD_NORMALIZATION": int(args.reward_normalization),
        "SKILL_ADV_ACTION_BC_COEF": 0,
        "SKILL_POLICY_ANCHOR_ENABLE": 0,
        "SKILL_RETRAIN": int(args.retrain),
        "SKILL_FRESH_RETRAIN": 0,
        "DEFENDER_POLICY_NAME": policy_name,
        "SKILL_BEST_MODEL_METRIC": "win",
        "SKILL_EVAL_INTERVAL": args.eval_interval,
        "SKILL_SAVE_INTERVAL": args.save_interval,
        "SKILL_EVAL_EPISODES": args.eval_episodes,
        "SKILL_BALANCED_EVAL_ATTACKERS": ",".join(FROZEN_ATTACKER_POOL),
        "SKILL_BALANCED_EVAL_EPISODES": args.balanced_eval_episodes,
        "SKILL_BALANCED_EVAL_METRIC": "min_win",
        "TRAIN_ATTACKER_STRATEGIES": ",".join(FROZEN_ATTACKER_POOL),
        "TRAIN_ATTACKER_STRATEGY_WEIGHTS": ",".join(str(v) for v in equal_weights),
        "TRAIN_LEARNED_ATTACKER_SPECS": json.dumps(learned_specs, sort_keys=True),
    }
    if args.restore_checkpoint:
        values["SKILL_RESTORE_DIR"] = Path(args.restore_checkpoint).expanduser().resolve()
    for name, value in values.items():
        os.environ[name] = str(value)


def _dry_run(seed: int, *, reward_mode: str = "baseline") -> int:
    """Load every frozen opponent and complete one real-environment transition."""
    import numpy as np

    from configs import map_config
    from envs.tad_env import TADEnv
    from attacker.learned_policy import LearnedAttackerPolicy
    from attacker_heuristics.registry import create_policy

    if float(map_config.attacker_speed) != 2.0 or float(map_config.defender_speed) != 2.6:
        raise RuntimeError(
            f"unexpected speeds A={map_config.attacker_speed}, D={map_config.defender_speed}"
        )

    learned_specs = resolved_learned_attacker_specs(PROJECT_ROOT)
    rows = []
    for index, alias in enumerate(FROZEN_ATTACKER_POOL):
        episode_seed = int(seed) + index
        env = TADEnv(reward_mode=str(reward_mode))
        try:
            observations, _ = env.reset(seed=episode_seed)
            _defender_obs, attacker_obs = observations
            if alias in PROGRAMMATIC_ATTACKER_ALIASES:
                policy = create_policy(
                    PROGRAMMATIC_ATTACKER_ALIASES[alias],
                    seed=episode_seed,
                )
                policy.reset(seed=episode_seed)
                source = PROGRAMMATIC_ATTACKER_ALIASES[alias]
            else:
                spec = learned_specs[alias]
                checkpoint = Path(spec["checkpoint"])
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"missing frozen Attacker {alias}: {checkpoint}")
                policy = LearnedAttackerPolicy(
                    checkpoint,
                    device="cpu",
                    alias=alias,
                    reward_style=spec.get("reward_style"),
                )
                policy.reset()
                source = str(checkpoint)
            action = np.asarray(policy.get_action(attacker_obs), dtype=np.float32)
            if not env.action_space.contains(action):
                raise RuntimeError(f"illegal action from {alias}: {action}")
            _obs, reward, terminated, truncated, info = env.step(
                np.zeros(2, dtype=np.float32),
                action,
            )
            rows.append({
                "alias": alias,
                "source": source,
                "action": action.tolist(),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "reason": info.get("reason"),
                "real_environment_step": True,
            })
        finally:
            env.close()
    print(json.dumps({
        "status": "ok",
        "reward_mode": str(reward_mode),
        "attacker_speed": float(map_config.attacker_speed),
        "defender_speed": float(map_config.defender_speed),
        "smoke_tests": rows,
    }, ensure_ascii=False, indent=2))
    return 0


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.dry_run:
        return _dry_run(args.seed)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else _default_output_dir(args.network_type).resolve()
    )
    if args.retrain and not args.restore_checkpoint:
        raise ValueError("--retrain requires --restore-checkpoint")
    if args.restore_checkpoint and not Path(args.restore_checkpoint).expanduser().is_file():
        raise FileNotFoundError(f"restore checkpoint not found: {args.restore_checkpoint}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.retrain:
        raise FileExistsError(f"refusing to overwrite non-empty run: {output_dir}")
    _configure_environment(args, output_dir)
    print(
        f"Protect Defender PPO | reward=legacy_baseline | network={args.network_type} | "
        f"steps={args.max_steps:,} | attackers={len(FROZEN_ATTACKER_POOL)} | "
        f"output={output_dir}",
        flush=True,
    )

    from train.train_skill import main as train_skill_main

    train_skill_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
