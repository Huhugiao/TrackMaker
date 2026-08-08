"""Canonical six-member Attacker pool used for Defender training/evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict


FROZEN_ATTACKER_POOL = (
    "heuristic_default",
    "heuristic_evasive",
    "heuristic_geometry_feint_v3",
    "heuristic_occlusion_dash_v2",
    "rl_ppo_goal_rush",
    "rl_ppo_evasive",
)

PROGRAMMATIC_ATTACKER_ALIASES = {
    "heuristic_default": "default",
    "heuristic_evasive": "evasive",
    "heuristic_geometry_feint_v3": "geometry_feint_v3",
    "heuristic_occlusion_dash_v2": "occlusion_dash_v2",
}

LEARNED_ATTACKER_SPECS = {
    "rl_ppo_goal_rush": {
        "checkpoint": "models/attacker_nmn_mlp_diversity_continuation_20260715_120331/best_goal_rush.pth",
        "reward_style": "goal_rush",
    },
    "rl_ppo_evasive": {
        "checkpoint": "models/attacker_nmn_mlp_diversity_continuation_20260715_120331/best_evasive.pth",
        "reward_style": "evasive",
    },
}


def resolved_learned_attacker_specs(project_root: Path) -> Dict[str, Dict[str, str]]:
    """Return checkpoint specs with absolute paths for Ray workers."""
    root = Path(project_root).expanduser().resolve()
    return {
        alias: {
            **spec,
            "checkpoint": str((root / spec["checkpoint"]).resolve()),
        }
        for alias, spec in LEARNED_ATTACKER_SPECS.items()
    }


__all__ = [
    "FROZEN_ATTACKER_POOL",
    "PROGRAMMATIC_ATTACKER_ALIASES",
    "LEARNED_ATTACKER_SPECS",
    "resolved_learned_attacker_specs",
]
