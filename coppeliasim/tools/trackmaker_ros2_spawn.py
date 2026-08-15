#!/usr/bin/env python3
"""Print the deterministic CoppeliaSim spawn string for one paired seed."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coppelia_env.ros2_deployment import sample_fixed_spawn  # noqa: E402


DEFAULT_MANIFEST = PROJECT_ROOT / "coppeliasim/scenes/trackmaker_turtlebot4_v2_1_scene.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--json", action="store_true", help="print structured JSON instead of the bridge string")
    args = parser.parse_args()
    spawn = sample_fixed_spawn(args.manifest, args.seed)
    if args.json:
        print(json.dumps({name: asdict(pose) for name, pose in spawn.items()}, sort_keys=True))
    else:
        values = [
            spawn["defender"].x,
            spawn["defender"].y,
            spawn["defender"].yaw,
            spawn["attacker"].x,
            spawn["attacker"].y,
            spawn["attacker"].yaw,
            spawn["target"].x,
            spawn["target"].y,
        ]
        print(",".join(f"{value:.9f}" for value in values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
