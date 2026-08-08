#!/usr/bin/env python3
"""Run a rollout in the Coppelia-native TrackMaker environment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coppelia_env import CoppeliaTrackEnv, VelocityCommand


DEFAULT_MANIFEST = PROJECT_ROOT / "coppeliasim/scenes/trackmaker_turtlebot4_scene.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--reward-mode", default="hrl", choices=["hrl", "baseline", "chase"])
    parser.add_argument("--policy", default="seek", choices=["zero", "seek", "circle"])
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "outputs/coppeliasim/native_rollout.json")
    parser.add_argument("--no-start-simulation", action="store_true")
    return parser.parse_args()


def _state_center(agent: dict[str, float], pixel_size: float) -> tuple[float, float]:
    return float(agent["x"]) + pixel_size * 0.5, float(agent["y"]) + pixel_size * 0.5


def seek_command(
    env: CoppeliaTrackEnv,
    role: str,
    target_role: str,
    speed_scale: float = 1.0,
) -> VelocityCommand:
    state = env.get_state()
    ego = state[role]
    target = state[target_role]
    ex, ey = _state_center(ego, env.pixel_size)
    tx, ty = _state_center(target, env.pixel_size)
    desired = math.degrees(math.atan2(ty - ey, tx - ex))
    err = ((desired - float(ego["theta"]) + 180.0) % 360.0) - 180.0
    max_v = env.defender_max_v if role == "defender" else env.attacker_max_v
    max_w = env.defender_max_w if role == "defender" else env.attacker_max_w
    w = max(-max_w, min(max_w, math.radians(err) / max(env.sim_step_dt, 1e-6)))
    turn_scale = max(0.2, 1.0 - min(abs(err), 120.0) / 150.0)
    return VelocityCommand(linear_mps=max_v * float(speed_scale) * turn_scale, angular_radps=w)


def policy_commands(env: CoppeliaTrackEnv, policy: str, step_idx: int) -> tuple[VelocityCommand, VelocityCommand]:
    if policy == "zero":
        return VelocityCommand(), VelocityCommand()
    if policy == "circle":
        return (
            VelocityCommand(env.defender_max_v * 0.7, env.defender_max_w * 0.35),
            VelocityCommand(env.attacker_max_v * 0.8, -env.attacker_max_w * 0.25),
        )
    return seek_command(env, "defender", "attacker"), seek_command(env, "attacker", "target", speed_scale=0.9)


def main() -> int:
    args = parse_args()
    env = CoppeliaTrackEnv(
        manifest=args.manifest,
        host=args.host,
        port=args.port,
        reward_mode=args.reward_mode,
        start_simulation=not args.no_start_simulation,
    )
    records: list[dict[str, Any]] = []
    try:
        transition = env.reset(seed=args.seed)
        records.append({"step": 0, **transition})
        for step_idx in range(1, max(1, int(args.steps)) + 1):
            defender_cmd, attacker_cmd = policy_commands(env, args.policy, step_idx)
            transition = env.step(defender_cmd, attacker_cmd)
            records.append(
                {
                    "step": step_idx,
                    "defender_cmd": defender_cmd.__dict__,
                    "attacker_cmd": attacker_cmd.__dict__,
                    **transition,
                }
            )
            if transition["done"]:
                break
    finally:
        env.close()

    final = records[-1]
    summary = {
        "manifest": str(args.manifest),
        "seed": int(args.seed),
        "policy": str(args.policy),
        "steps": int(final["step"]),
        "done": bool(final["done"]),
        "reward_total": float(sum(float(r.get("reward", 0.0)) for r in records[1:])),
        "reason": str(final.get("info", {}).get("reason", "")),
        "win": bool(final.get("info", {}).get("win", False)),
        "final_state": final.get("state", {}),
        "records": records,
    }
    out = args.output_json.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
