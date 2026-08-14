#!/usr/bin/env python3
"""Fail unless ROS projection and action scaling match the authoritative Gym contract."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coppelia_env.ros2_deployment import (  # noqa: E402
    DefenderObservationMemory,
    DeploymentConfig,
    LinearSlewLimiter,
    Pose2D,
    build_policy_observations,
    normalized_action_to_command,
)
from configs import map_config  # noqa: E402
from envs import env_lib  # noqa: E402
from envs.tad_env import TADEnv  # noqa: E402


def _pose(state: dict[str, float], config: DeploymentConfig) -> Pose2D:
    center_x = float(state["x"]) + 0.5 * config.pixel_size_px
    center_y = float(state["y"]) + 0.5 * config.pixel_size_px
    return Pose2D(
        (center_x - 0.5 * config.width_px) * config.scale_m_per_px,
        (0.5 * config.height_px - center_y) * config.scale_m_per_px,
        -math.radians(float(state["theta"])),
    )


def run(samples: int, seed: int, manifest_path: Path) -> dict[str, float | int | bool]:
    config = DeploymentConfig()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    obstacles = list(manifest["obstacles"])
    old_obstacles = list(map_config.obstacles)
    rng = random.Random(seed)
    max_actor_error = max_privileged_error = max_critic_error = 0.0
    max_action_error = max_acceleration_error = 0.0

    try:
        map_config.obstacles = obstacles
        env_lib.build_occupancy(width=640, height=640, cell=4.0, obstacles=obstacles)
        env = TADEnv()

        def sample_agent() -> dict[str, float]:
            for _ in range(10000):
                state = {
                    "x": rng.uniform(20.0, 616.0),
                    "y": rng.uniform(20.0, 616.0),
                    "theta": rng.uniform(0.0, 360.0),
                }
                if not env_lib.is_point_blocked(state["x"] + 2.0, state["y"] + 2.0, padding=8.0):
                    return state
            raise RuntimeError("unable to sample a clear parity state")

        for _ in range(samples):
            env.defender, env.attacker, env.target = sample_agent(), sample_agent(), sample_agent()
            env.last_observed_attacker_pos = None
            env.steps_since_observed = 0
            defender_radar = np.asarray([rng.uniform(-1.0, 1.0) for _ in range(64)], dtype=np.float32)
            attacker_radar = np.asarray([rng.uniform(-1.0, 1.0) for _ in range(64)], dtype=np.float32)
            env._sense_agent_radar = (  # type: ignore[method-assign]
                lambda agent, num_rays=64, full_circle=True: defender_radar
                if agent is env.defender
                else attacker_radar
            )

            expected_actor = env._get_defender_observation()
            expected_privileged = env._get_attacker_observation()
            poses = {
                "defender": _pose(env.defender, config),
                "attacker": _pose(env.attacker, config),
                "target": _pose(env.target, config),
            }
            actual_actor, actual_privileged, actual_critic = build_policy_observations(
                poses,
                defender_radar,
                attacker_radar,
                config,
                defender_memory=DefenderObservationMemory(),
                obstacles=obstacles,
            )
            expected_critic = np.concatenate((expected_actor, expected_privileged)).astype(np.float32)
            max_actor_error = max(max_actor_error, float(np.max(np.abs(actual_actor - expected_actor))))
            max_privileged_error = max(
                max_privileged_error,
                float(np.max(np.abs(actual_privileged - expected_privileged))),
            )
            max_critic_error = max(max_critic_error, float(np.max(np.abs(actual_critic - expected_critic))))

        for role, max_linear, max_angular in (
            ("defender", config.defender_max_linear_mps, config.defender_max_angular_radps),
            ("attacker", config.attacker_max_linear_mps, config.attacker_max_angular_radps),
        ):
            for _ in range(samples):
                action = np.asarray([rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)], dtype=np.float32)
                expected_turn, expected_speed = env._control_to_physical(action, role)
                command = normalized_action_to_command(
                    action,
                    max_linear_mps=max_linear,
                    max_angular_radps=max_angular,
                )
                actual_turn = -math.degrees(command.angular_radps / config.decision_hz)
                actual_speed = command.linear_mps / (config.scale_m_per_px * config.decision_hz)
                max_action_error = max(
                    max_action_error,
                    abs(actual_turn - expected_turn),
                    abs(actual_speed - expected_speed),
                )

        limiter = LinearSlewLimiter(config.attacker_max_accel_mps2)
        gym_speed = 0.0
        for requested_px_per_step in (2.0, 2.0, 0.0, 1.7, 0.2):
            gym_speed += float(np.clip(requested_px_per_step - gym_speed, -0.6, 0.6))
            requested_mps = requested_px_per_step * config.scale_m_per_px * config.decision_hz
            ros_speed = limiter.apply(requested_mps, 1.0 / config.decision_hz)
            ros_px_per_step = ros_speed / (config.scale_m_per_px * config.decision_hz)
            max_acceleration_error = max(max_acceleration_error, abs(ros_px_per_step - gym_speed))
    finally:
        map_config.obstacles = old_obstacles
        env_lib.build_occupancy(width=640, height=640, cell=4.0, obstacles=old_obstacles)

    tolerance = 2e-5
    result = {
        "samples": int(samples),
        "seed": int(seed),
        "max_actor_abs_error": max_actor_error,
        "max_privileged_abs_error": max_privileged_error,
        "max_critic_abs_error": max_critic_error,
        "max_action_contract_error": max_action_error,
        "max_attacker_acceleration_error": max_acceleration_error,
    }
    result["passed"] = bool(max(result[key] for key in result if key.startswith("max_")) <= tolerance)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "coppeliasim/scenes/trackmaker_turtlebot4_scene.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(max(1, args.samples), args.seed, args.manifest)
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
