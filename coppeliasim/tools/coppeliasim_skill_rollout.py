#!/usr/bin/env python3
"""Evaluate an existing bottom skill in the CoppeliaTrackEnv."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coppelia_env.track_env import CoppeliaTrackEnv  # noqa: E402
from coppelia_env.rollout_diagnostics import summarize_episode_diagnostics  # noqa: E402
from coppelia_env.skill_adapter import (  # noqa: E402
    CoppeliaSkillObservationProvider,
    NormalizedCmdVelAdapter,
    load_skill_model,
)


DEFAULT_MANIFEST = PROJECT_ROOT / "coppeliasim/scenes/trackmaker_turtlebot4_scene.json"
DEFAULT_BASELINE_CHECKPOINT = PROJECT_ROOT / "models/defender_baseline_mlp_ctde_repro_20260526/final_model.pth"
DEFAULT_CHASE_CHECKPOINT = PROJECT_ROOT / "models/defender_chase_nmn_dual_gru_raw_dense_05-05-19-12/final_model.pth"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/coppeliasim/skill_rollout.json"


@dataclass(frozen=True)
class SkillDefaults:
    checkpoint: Path
    network_type: str
    reward_mode: str


def skill_defaults(skill: str) -> SkillDefaults:
    normalized = str(skill).strip().lower()
    if normalized in {"baseline", "protect"}:
        return SkillDefaults(
            checkpoint=DEFAULT_BASELINE_CHECKPOINT,
            network_type="mlp_ctde",
            reward_mode="baseline",
        )
    if normalized == "chase":
        return SkillDefaults(
            checkpoint=DEFAULT_CHASE_CHECKPOINT,
            network_type="nmn_dual_gru_raw",
            reward_mode="chase",
        )
    raise ValueError(f"unsupported skill {skill!r}; expected baseline, protect, or chase")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--skill", default="chase", choices=["baseline", "protect", "chase"])
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--network-type", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--allow-reverse", action="store_true")
    parser.add_argument("--record-steps", action="store_true")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-start-simulation", action="store_true")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; using CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(str(name))


def _jsonable_action(action: np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(action, dtype=np.float32).reshape(-1)]


def run_episode(
    env: CoppeliaTrackEnv,
    model: Any,
    provider: CoppeliaSkillObservationProvider,
    action_adapter: NormalizedCmdVelAdapter,
    seed: int,
    max_steps: int,
    deterministic: bool,
    record_steps: bool,
) -> dict[str, Any]:
    model.reset_recurrent_state()
    transition = env.reset(seed=int(seed))
    total_reward = 0.0
    steps = 0
    done = False
    info: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    diagnostic_records: list[dict[str, Any]] = []

    while not done and steps < max(1, int(max_steps)):
        actor_obs, _privileged_obs, critic_obs = provider.observations(transition)
        if deterministic:
            action, pre_tanh, value, log_prob = model.evaluate(actor_obs, critic_obs, greedy=True)
        else:
            action, pre_tanh, value, log_prob = model.step(actor_obs, critic_obs)
        cmd = action_adapter.defender_command(action)
        next_transition = env.step(cmd)
        reward = float(next_transition["reward"])
        done = bool(next_transition["done"])
        info = dict(next_transition.get("info", {}) or {})
        steps += 1
        total_reward += reward
        state = next_transition.get("state", {})
        diagnostic_record = {"state": state, "info": info}
        diagnostic_records.append(diagnostic_record)

        if record_steps:
            records.append(
                {
                    "step": int(steps),
                    "state": state,
                    "action": _jsonable_action(action),
                    "pre_tanh": _jsonable_action(pre_tanh),
                    "value": float(value),
                    "log_prob": float(log_prob),
                    "defender_cmd": asdict(cmd),
                    "reward": reward,
                    "done": done,
                    "info": info,
                }
            )
        transition = next_transition

    diagnostics = summarize_episode_diagnostics(diagnostic_records, transition.get("state", {}))
    return {
        "seed": int(seed),
        "steps": int(steps),
        "return": float(total_reward),
        "done": bool(done),
        "win": bool(info.get("win", False)),
        "reason": str(info.get("reason") or ("step_limit" if not done else "")),
        "diagnostics": diagnostics,
        "final_state": transition.get("state", {}),
        "records": records,
    }


def _diagnostic_means(episodes: list[dict[str, Any]]) -> dict[str, float | None]:
    keys = [
        "min_defender_attacker_dist_px",
        "final_defender_attacker_dist_px",
        "min_attacker_target_dist_px",
        "final_attacker_target_dist_px",
        "defender_collision_steps",
        "attacker_collision_steps",
    ]
    out: dict[str, float | None] = {}
    for key in keys:
        values = [
            float(value)
            for ep in episodes
            for value in [dict(ep.get("diagnostics", {}) or {}).get(key)]
            if value is not None
        ]
        out[f"{key}_mean"] = float(np.mean(values)) if values else None
    return out


def summarize_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        return {
            "return_mean": 0.0,
            "len_mean": 0.0,
            "win_rate": 0.0,
            "done_rate": 0.0,
            **_diagnostic_means([]),
        }
    return {
        "return_mean": float(np.mean([float(ep["return"]) for ep in episodes])),
        "len_mean": float(np.mean([float(ep["steps"]) for ep in episodes])),
        "win_rate": float(np.mean([1.0 if ep.get("win") else 0.0 for ep in episodes])),
        "done_rate": float(np.mean([1.0 if ep.get("done") else 0.0 for ep in episodes])),
        **_diagnostic_means(episodes),
    }


def main() -> int:
    args = parse_args()
    defaults = skill_defaults(args.skill)
    checkpoint = args.checkpoint.resolve() if args.checkpoint is not None else defaults.checkpoint
    network_type = str(args.network_type or defaults.network_type)
    device = resolve_device(str(args.device))
    started = time.time()

    model = load_skill_model(
        checkpoint=checkpoint,
        network_type=network_type,
        device=device,
        global_model=False,
    )
    model.network.eval()

    env = CoppeliaTrackEnv(
        manifest=args.manifest,
        host=args.host,
        port=args.port,
        reward_mode=defaults.reward_mode,
        start_simulation=not args.no_start_simulation,
    )
    try:
        provider = CoppeliaSkillObservationProvider(env)
        action_adapter = NormalizedCmdVelAdapter(env, allow_reverse=bool(args.allow_reverse))
        episodes = [
            run_episode(
                env=env,
                model=model,
                provider=provider,
                action_adapter=action_adapter,
                seed=int(args.seed) + idx,
                max_steps=max(1, int(args.steps)),
                deterministic=bool(args.deterministic),
                record_steps=bool(args.record_steps),
            )
            for idx in range(max(1, int(args.episodes)))
        ]
    finally:
        env.close()

    metrics = summarize_episodes(episodes)
    summary = {
        "manifest": str(args.manifest.resolve()),
        "skill": str(args.skill),
        "checkpoint": str(checkpoint),
        "network_type": network_type,
        "reward_mode": defaults.reward_mode,
        "deterministic": bool(args.deterministic),
        "allow_reverse": bool(args.allow_reverse),
        "episodes": episodes,
        **metrics,
        "elapsed_sec": float(time.time() - started),
    }
    out = args.output_json.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
