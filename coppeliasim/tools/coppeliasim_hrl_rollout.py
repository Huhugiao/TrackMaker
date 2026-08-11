#!/usr/bin/env python3
"""Run a 2-skill HRL top policy in CoppeliaTrackEnv."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
from configs.skill_config import NetParameters  # noqa: E402
from coppeliasim.tools.coppeliasim_skill_rollout import (  # noqa: E402
    DEFAULT_PROTECT_CHECKPOINT,
    DEFAULT_CHASE_CHECKPOINT,
    DEFAULT_MANIFEST,
    resolve_device,
)


DEFAULT_TOP_CHECKPOINT = PROJECT_ROOT / "models/hrl_ch2_m1_astar_cached_top_20260606_170036/best_model.pth"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/coppeliasim/hrl_rollout.json"
DEFAULT_TOP_NETWORK_TYPE = "hrl_top_dual_gru_raw"
SKILL_NAMES = ("protect", "chase")


def hrl_skill_paths(protect_checkpoint: Path | None, chase_checkpoint: Path | None) -> dict[str, Path]:
    return {
        "protect": Path(protect_checkpoint) if protect_checkpoint is not None else DEFAULT_PROTECT_CHECKPOINT,
        "chase": Path(chase_checkpoint) if chase_checkpoint is not None else DEFAULT_CHASE_CHECKPOINT,
    }


def decode_skill_index(action: np.ndarray | list[float] | tuple[float, ...], skill_names: tuple[str, ...] = SKILL_NAMES) -> str:
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    raw = int(round(float(arr[0]))) if arr.size else 0
    idx = max(0, min(raw, len(skill_names) - 1))
    return str(skill_names[idx])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--top-checkpoint", type=Path, default=DEFAULT_TOP_CHECKPOINT)
    parser.add_argument("--top-network-type", default=DEFAULT_TOP_NETWORK_TYPE)
    parser.add_argument("--protect-checkpoint", type=Path, default=None)
    parser.add_argument("--chase-checkpoint", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--allow-reverse", action="store_true")
    parser.add_argument("--record-steps", action="store_true")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-start-simulation", action="store_true")
    return parser.parse_args()


def _set_top_net_parameters() -> tuple[int, tuple[int, ...], int, int]:
    old_num_skills = int(getattr(NetParameters, "HRL_NUM_SKILLS", 2))
    old_duration_bins = tuple(getattr(NetParameters, "HRL_DURATION_BINS", (1,)))
    old_num_duration_bins = int(getattr(NetParameters, "HRL_NUM_DURATION_BINS", len(old_duration_bins)))
    old_discrete_dim = int(getattr(NetParameters, "HRL_TOP_DISCRETE_ACTION_DIM", old_num_skills))
    NetParameters.HRL_NUM_SKILLS = 2
    NetParameters.HRL_DURATION_BINS = (1,)
    NetParameters.HRL_NUM_DURATION_BINS = 1
    NetParameters.HRL_TOP_DISCRETE_ACTION_DIM = 2
    return old_num_skills, old_duration_bins, old_num_duration_bins, old_discrete_dim


def _restore_top_net_parameters(old: tuple[int, tuple[int, ...], int, int]) -> None:
    (
        NetParameters.HRL_NUM_SKILLS,
        NetParameters.HRL_DURATION_BINS,
        NetParameters.HRL_NUM_DURATION_BINS,
        NetParameters.HRL_TOP_DISCRETE_ACTION_DIM,
    ) = old


def load_hrl_top_model(checkpoint: Path, network_type: str, device: torch.device):
    old = _set_top_net_parameters()
    try:
        return load_skill_model(
            checkpoint=checkpoint,
            network_type=network_type,
            device=device,
            global_model=False,
        )
    finally:
        _restore_top_net_parameters(old)


def _jsonable_action(action: np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(action, dtype=np.float32).reshape(-1)]


def run_episode(
    env: CoppeliaTrackEnv,
    top_model: Any,
    bottom_models: dict[str, Any],
    provider: CoppeliaSkillObservationProvider,
    adapter: NormalizedCmdVelAdapter,
    seed: int,
    max_steps: int,
    deterministic: bool,
    record_steps: bool,
) -> dict[str, Any]:
    top_model.reset_recurrent_state()
    for model in bottom_models.values():
        model.reset_recurrent_state()
    transition = env.reset(seed=int(seed))
    total_reward = 0.0
    steps = 0
    done = False
    info: dict[str, Any] = {}
    skill_counts = {name: 0 for name in SKILL_NAMES}
    records: list[dict[str, Any]] = []
    diagnostic_records: list[dict[str, Any]] = []

    while not done and steps < max(1, int(max_steps)):
        actor_obs, _privileged_obs, critic_obs = provider.observations(transition)
        top_action, _top_pre_tanh, top_value, top_log_prob = top_model.evaluate(
            actor_obs,
            critic_obs,
            greedy=bool(deterministic),
        )
        skill_name = decode_skill_index(top_action)
        skill_counts[skill_name] = int(skill_counts.get(skill_name, 0)) + 1
        bottom_model = bottom_models[skill_name]
        if deterministic:
            bottom_action, bottom_pre_tanh, bottom_value, bottom_log_prob = bottom_model.evaluate(
                actor_obs,
                critic_obs,
                greedy=True,
            )
        else:
            bottom_action, bottom_pre_tanh, bottom_value, bottom_log_prob = bottom_model.step(actor_obs, critic_obs)
        cmd = adapter.defender_command(bottom_action)
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
                    "top_action": _jsonable_action(top_action),
                    "selected_skill": skill_name,
                    "top_value": float(top_value),
                    "top_log_prob": float(top_log_prob),
                    "bottom_action": _jsonable_action(bottom_action),
                    "bottom_pre_tanh": _jsonable_action(bottom_pre_tanh),
                    "bottom_value": float(bottom_value),
                    "bottom_log_prob": float(bottom_log_prob),
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
        "skill_counts": skill_counts,
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
            "skill_counts": {name: 0 for name in SKILL_NAMES},
            **_diagnostic_means([]),
        }
    counts = {name: 0 for name in SKILL_NAMES}
    for ep in episodes:
        for name, value in dict(ep.get("skill_counts", {}) or {}).items():
            counts[str(name)] = int(counts.get(str(name), 0)) + int(value)
    return {
        "return_mean": float(np.mean([float(ep["return"]) for ep in episodes])),
        "len_mean": float(np.mean([float(ep["steps"]) for ep in episodes])),
        "win_rate": float(np.mean([1.0 if ep.get("win") else 0.0 for ep in episodes])),
        "done_rate": float(np.mean([1.0 if ep.get("done") else 0.0 for ep in episodes])),
        "skill_counts": counts,
        **_diagnostic_means(episodes),
    }


def main() -> int:
    args = parse_args()
    device = resolve_device(str(args.device))
    started = time.time()
    top_model = load_hrl_top_model(
        checkpoint=args.top_checkpoint.resolve(),
        network_type=str(args.top_network_type),
        device=device,
    )
    top_model.network.eval()
    paths = hrl_skill_paths(args.protect_checkpoint, args.chase_checkpoint)
    bottom_models = {
        "protect": load_skill_model(paths["protect"], "mlp_ctde", device=device, global_model=False),
        "chase": load_skill_model(paths["chase"], "nmn_dual_gru_raw", device=device, global_model=False),
    }
    for model in bottom_models.values():
        model.network.eval()

    env = CoppeliaTrackEnv(
        manifest=args.manifest,
        host=args.host,
        port=args.port,
        reward_mode="hrl",
        start_simulation=not args.no_start_simulation,
    )
    try:
        provider = CoppeliaSkillObservationProvider(env)
        adapter = NormalizedCmdVelAdapter(env, allow_reverse=bool(args.allow_reverse))
        episodes = [
            run_episode(
                env=env,
                top_model=top_model,
                bottom_models=bottom_models,
                provider=provider,
                adapter=adapter,
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
        "top_checkpoint": str(args.top_checkpoint.resolve()),
        "top_network_type": str(args.top_network_type),
        "skill_names": list(SKILL_NAMES),
        "protect_checkpoint": str(paths["protect"]),
        "chase_checkpoint": str(paths["chase"]),
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
