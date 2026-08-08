"""Registry for the four programmatic Attackers in the frozen pool."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from attacker_heuristics.active_policies import (
    GeometryGatedFeintV3Policy,
    OcclusionDashV2Policy,
)
from attacker_heuristics.base import DIAGNOSTIC_KEYS, decode_world, interception_risk
from policies.attacker_global import AttackerGlobalPolicy


@dataclass(frozen=True)
class PolicySpec:
    name: str
    family: str
    hypothesis: str
    factory: Callable[..., object]
    parent: Optional[str] = None
    generation: int = 1
    baseline: bool = False


class GlobalPolicyAdapter:
    """Expose retained global policies through the diagnostic interface."""

    mechanism = "A* baseline"

    def __init__(self, strategy: str, *, seed: int = 0):
        self.strategy = str(strategy)
        self.base_seed = int(seed)
        self.policy = AttackerGlobalPolicy(strategy=self.strategy)
        self._step = 0
        self._previous = None
        self._diagnostics = {key: None for key in DIAGNOSTIC_KEYS}
        self.reset()

    def reset(self, seed: Optional[int] = None):
        random.seed(self.base_seed if seed is None else int(seed))
        self.policy.reset()
        self._step = 0
        self._previous = None
        self._diagnostics = {
            "mode": self.strategy,
            "state": "reset",
            "subgoal": [0.0, 0.0],
            "risk": 0.0,
            "switch_reason": "reset",
            "replan_count": 0,
            "mechanism": self.mechanism,
            "details": {},
        }

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        action, policy_info = self.policy.get_action_with_info(obs)
        world = decode_world(obs, self._previous, self._step)
        path = getattr(self.policy, "path", []) or []
        if path:
            index = min(int(getattr(self.policy, "current_path_index", 0)), len(path) - 1)
            subgoal = np.asarray(path[index], dtype=np.float64)
        else:
            subgoal = world.target
        self._diagnostics = {
            "mode": self.strategy,
            "state": str(policy_info.get("active_strategy", self.strategy)),
            "subgoal": [float(subgoal[0]), float(subgoal[1])],
            "risk": interception_risk(world),
            "switch_reason": "policy_internal",
            "replan_count": int(
                getattr(self.policy, "step_count", 0)
                // max(1, getattr(self.policy, "replan_interval", 20))
            ),
            "mechanism": self.mechanism,
            "details": {
                "active_strategy": str(policy_info.get("active_strategy", self.strategy)),
                "path_length": int(policy_info.get("path_length", 0)),
            },
        }
        self._previous = world
        self._step += 1
        return np.asarray(action, dtype=np.float32)

    def get_action_with_info(self, obs: np.ndarray):
        action = self.get_action(obs)
        return action, dict(self._diagnostics)

    @property
    def diagnostics(self):
        return dict(self._diagnostics)


def _global_factory(strategy: str):
    return lambda *, seed=0: GlobalPolicyAdapter(strategy, seed=seed)


POLICY_REGISTRY: Dict[str, PolicySpec] = {
    "default": PolicySpec(
        "default",
        "direct_astar",
        "Direct A* attack with local Defender clearance.",
        _global_factory("default"),
        baseline=True,
    ),
    "evasive": PolicySpec(
        "evasive",
        "visibility_evasion",
        "Long-range Defender-view evasion anchor.",
        _global_factory("evasive"),
        baseline=True,
    ),
    "geometry_feint_v3": PolicySpec(
        "geometry_feint_v3",
        "geometry_gated_deception",
        "Feint only when initial A-D-T geometry shows Defender controls the direct corridor.",
        GeometryGatedFeintV3Policy,
        generation=3,
    ),
    "occlusion_dash_v2": PolicySpec(
        "occlusion_dash_v2",
        "bounded_map_occlusion",
        "A single bounded shadow stage avoids repeated occlusion replanning and timeout.",
        OcclusionDashV2Policy,
        generation=2,
    ),
}


def policy_names(*, include_baselines: bool = True) -> Tuple[str, ...]:
    return tuple(
        name
        for name, spec in POLICY_REGISTRY.items()
        if include_baselines or not spec.baseline
    )


def create_policy(name: str, *, seed: int = 0):
    key = str(name).strip().lower()
    if key not in POLICY_REGISTRY:
        raise KeyError(f"Unknown heuristic policy {name!r}; valid={tuple(POLICY_REGISTRY)}")
    return POLICY_REGISTRY[key].factory(seed=int(seed))


__all__ = ["POLICY_REGISTRY", "PolicySpec", "create_policy", "policy_names"]
