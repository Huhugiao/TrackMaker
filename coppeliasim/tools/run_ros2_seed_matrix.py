#!/usr/bin/env python3
"""Run and aggregate the fixed paired-seed ROS 2 CoppeliaSim acceptance matrix."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = PROJECT_ROOT / "coppeliasim/tools/run_ros2_demo.sh"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=20260326)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=449)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--with-media", action="store_true", help="also record MP4 and rosbag for every seed")
    args = parser.parse_args()
    output = args.output_dir or (
        PROJECT_ROOT
        / "outputs/coppeliasim/ros2_matrix"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    episodes = []
    for index in range(max(1, int(args.count))):
        seed = int(args.seed_start) + index
        episode_dir = output / f"seed_{seed}"
        episode_json = episode_dir / "episode.json"
        command = [
            str(LAUNCHER),
            "--seed",
            str(seed),
            "--max-steps",
            str(max(1, int(args.max_steps))),
            "--output-dir",
            str(episode_dir),
        ]
        if not args.with_media:
            command.append("--no-media")
        print(f"[{index + 1}/{args.count}] seed={seed}", flush=True)
        valid_reasons = {
            "defender_caught_attacker",
            "attacker_caught_target",
            "timeout_defender_wins",
            "defender_collision",
        }
        reusable = False
        if episode_json.is_file():
            try:
                reusable = json.loads(episode_json.read_text(encoding="utf-8")).get("reason") in valid_reasons
            except (json.JSONDecodeError, OSError):
                reusable = False
        result_code = 0
        if reusable:
            print("  -> reusing completed episode", flush=True)
        else:
            result_code = subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode
        if result_code != 0 or not episode_json.is_file():
            print(f"seed {seed} failed with code {result_code}")
            return result_code or 3
        payload = json.loads(episode_json.read_text(encoding="utf-8"))
        records = payload.get("records", [])
        mean_speeds = {}
        for role in ("defender", "attacker"):
            samples = [float(row[f"{role}_cmd"]["linear_mps"]) for row in records]
            mean_speeds[role] = sum(samples) / len(samples) if samples else 0.0
        episodes.append(
            {
                "seed": seed,
                "reason": payload["reason"],
                "outcome": payload["outcome"],
                "steps": payload["steps"],
                "elapsed_s": payload["elapsed_s"],
                "skill_counts": payload["skill_counts"],
                "mean_linear_mps": mean_speeds,
                "attacker_collision_steps": sum(bool(row.get("attacker_collision")) for row in records),
                "controller_env_obstacle_mask": payload["controller_env_obstacle_mask"],
                "action_shield": payload["action_shield"],
                "episode_json": str(episode_json),
            }
        )
        print(f"  -> {payload['reason']} at step {payload['steps']}", flush=True)

    reasons = Counter(row["reason"] for row in episodes)
    summary = {
        "paired_seeds": [row["seed"] for row in episodes],
        "episode_count": len(episodes),
        "reason_counts": dict(reasons),
        "capture_rate": reasons["defender_caught_attacker"] / len(episodes),
        "target_success_rate": reasons["attacker_caught_target"] / len(episodes),
        "timeout_rate": reasons["timeout_defender_wins"] / len(episodes),
        "defender_collision_draw_rate": reasons["defender_collision"] / len(episodes),
        "attacker_collision_episode_rate": sum(
            row["attacker_collision_steps"] > 0 for row in episodes
        ) / len(episodes),
        "controller_env_obstacle_mask": False,
        "action_shield": False,
        "episodes": episodes,
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
