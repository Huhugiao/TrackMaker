#!/usr/bin/env python3
"""Load all deployment checkpoints and verify deterministic recurrent reset."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attacker.learned_policy import LearnedAttackerPolicy  # noqa: E402
from coppelia_env.ros2_deployment import CheckpointSpec, validate_checkpoint  # noqa: E402
from coppeliasim.tools.trackmaker_ros2_policy import (  # noqa: E402
    CHECKSUMS,
    DEFAULT_ATTACKER,
    DEFAULT_CHASE,
    DEFAULT_PROTECT,
    DEFAULT_TOP,
    load_hrl_top_model,
)
from coppelia_env.skill_adapter import load_skill_model  # noqa: E402


def repeated_action(model, actor: np.ndarray, critic: np.ndarray) -> tuple[list[float], bool]:
    model.reset_recurrent_state()
    first = np.asarray(model.evaluate(actor, critic, greedy=True)[0], dtype=np.float32).reshape(-1)
    model.reset_recurrent_state()
    second = np.asarray(model.evaluate(actor, critic, greedy=True)[0], dtype=np.float32).reshape(-1)
    return first.tolist(), bool(np.array_equal(first, second))


def main() -> int:
    paths = {
        "top": (DEFAULT_TOP, "hrl_top_dual_gru_raw"),
        "protect": (DEFAULT_PROTECT, "mlp_ctde"),
        "chase": (DEFAULT_CHASE, "nmn_dual_gru_raw"),
        "attacker": (DEFAULT_ATTACKER, "attacker_nmn_mlp_multistyle_ppo"),
    }
    metadata = {
        name: validate_checkpoint(CheckpointSpec(path, network, CHECKSUMS[name]))
        for name, (path, network) in paths.items()
    }
    device = torch.device("cpu")
    top = load_hrl_top_model(DEFAULT_TOP, paths["top"][1], device)
    protect = load_skill_model(DEFAULT_PROTECT, paths["protect"][1], device=device, global_model=False)
    chase = load_skill_model(DEFAULT_CHASE, paths["chase"][1], device=device, global_model=False)
    attacker = LearnedAttackerPolicy(
        str(DEFAULT_ATTACKER),
        device="cpu",
        alias="rl_ppo_goal_rush",
        reward_style="goal_rush",
    )
    actor = np.zeros(71, dtype=np.float32)
    privileged = np.zeros(72, dtype=np.float32)
    critic = np.zeros(143, dtype=np.float32)
    top_action, top_repeat = repeated_action(top, actor, critic)
    protect_action, protect_repeat = repeated_action(protect, actor, critic)
    chase_action, chase_repeat = repeated_action(chase, actor, critic)
    attacker.reset()
    attacker_first = np.asarray(attacker.get_action(privileged), dtype=np.float32).reshape(-1)
    attacker.reset()
    attacker_second = np.asarray(attacker.get_action(privileged), dtype=np.float32).reshape(-1)
    repeatable = {
        "top": top_repeat,
        "protect": protect_repeat,
        "chase": chase_repeat,
        "attacker": bool(np.array_equal(attacker_first, attacker_second)),
    }
    report = {
        "checkpoints": metadata,
        "actions": {
            "top": top_action,
            "protect": protect_action,
            "chase": chase_action,
            "attacker": attacker_first.tolist(),
        },
        "deterministic_after_reset": repeatable,
    }
    print(json.dumps(report, indent=2))
    return 0 if all(repeatable.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
