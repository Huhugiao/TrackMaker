"""Transition-local reward relabeling for multi-style Attacker training.

The environment emits reward-independent features once per transition.  A
matrix multiply then evaluates every configured reward style without stepping
another environment.  Keeping the reward as a linear function of transition
features also makes the training contract auditable and checkpointable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Dict, Tuple

import numpy as np


REWARD_FEATURE_NAMES: Tuple[str, ...] = (
    "step",
    "target_progress",
    "evade_progress",
    "attacker_collision",
    "target_success",
    "defender_capture",
    "timeout",
    "defender_collision",
)

REWARD_WEIGHT_KEYS: Tuple[str, ...] = (
    "STEP_PENALTY",
    "TARGET_PROGRESS_SCALE",
    "EVADE_PROGRESS_SCALE",
    "ATTACKER_COLLISION_PENALTY",
    "SUCCESS_REWARD",
    "CAPTURE_PENALTY",
    "TIMEOUT_PENALTY",
    "DEFENDER_COLLISION_REWARD",
)


def validate_reward_styles(styles: Sequence[Mapping]) -> Tuple[Dict, ...]:
    """Normalize and validate an ordered collection of reward specifications."""
    if not styles:
        raise ValueError("at least one Attacker reward style is required")

    normalized = []
    names = set()
    allowed = {"name", "description", *REWARD_WEIGHT_KEYS}
    for index, raw_style in enumerate(styles):
        if not isinstance(raw_style, Mapping):
            raise TypeError(f"reward style {index} must be a mapping")
        unknown = set(raw_style) - allowed
        if unknown:
            raise ValueError(f"reward style {index} has unknown keys: {sorted(unknown)}")
        name = str(raw_style.get("name", "")).strip().lower()
        if not name:
            raise ValueError(f"reward style {index} requires a non-empty name")
        if name in names:
            raise ValueError(f"duplicate reward style name: {name!r}")
        names.add(name)
        style = {
            "name": name,
            "description": str(raw_style.get("description", "")).strip(),
        }
        for key in REWARD_WEIGHT_KEYS:
            value = float(raw_style.get(key, 0.0))
            if not np.isfinite(value):
                raise ValueError(f"reward style {name!r} has non-finite {key}={value}")
            style[key] = value
        normalized.append(style)
    return tuple(normalized)


def reward_weight_matrix(styles: Sequence[Mapping]) -> np.ndarray:
    """Return ``[num_styles, num_features]`` reward weights."""
    normalized = validate_reward_styles(styles)
    return np.asarray(
        [[style[key] for key in REWARD_WEIGHT_KEYS] for style in normalized],
        dtype=np.float32,
    )


def reward_feature_vector(features: Mapping[str, float]) -> np.ndarray:
    """Return an ordered, finite transition feature vector."""
    vector = np.asarray(
        [float(features.get(name, 0.0)) for name in REWARD_FEATURE_NAMES],
        dtype=np.float32,
    )
    if not bool(np.isfinite(vector).all()):
        raise FloatingPointError(f"non-finite Attacker reward features: {dict(features)}")
    return vector


def compute_reward_vector(
    features: Mapping[str, float],
    styles: Sequence[Mapping],
) -> np.ndarray:
    """Relabel one transition under every configured reward style."""
    vector = reward_weight_matrix(styles) @ reward_feature_vector(features)
    if not bool(np.isfinite(vector).all()):
        raise FloatingPointError("non-finite multi-style Attacker rewards")
    return vector.astype(np.float32, copy=False)


__all__ = [
    "REWARD_FEATURE_NAMES",
    "REWARD_WEIGHT_KEYS",
    "compute_reward_vector",
    "reward_feature_vector",
    "reward_weight_matrix",
    "validate_reward_styles",
]
