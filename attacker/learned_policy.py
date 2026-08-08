"""Deployment wrapper for frozen learned attacker checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from attacker.ppo_model import NETWORK_TYPE as MULTISTYLE_PPO_NETWORK_TYPE
from attacker.ppo_model import AttackerMultiStylePPO


DEFAULT_LEARNED_ATTACKER_ALIAS = "attacker_rl"


def normalize_learned_attacker_alias(alias: str = DEFAULT_LEARNED_ATTACKER_ALIAS) -> str:
    """Return the normalized, non-empty name used in attacker pools."""
    normalized = str(alias).strip().lower()
    if not normalized:
        raise ValueError("learned attacker alias must be non-empty")
    return normalized


def is_learned_attacker_alias(value: str, alias: str = DEFAULT_LEARNED_ATTACKER_ALIAS) -> bool:
    """Check whether a pool entry selects the configured learned attacker."""
    return str(value).strip().lower() == normalize_learned_attacker_alias(alias)


def normalize_learned_attacker_specs(
    policy_specs: Optional[Mapping[str, Mapping]] = None,
) -> Dict[str, Dict[str, Optional[str]]]:
    """Validate checkpoint-backed Attacker aliases used by Defender pools."""
    if policy_specs is None:
        return {}
    if not isinstance(policy_specs, Mapping):
        raise ValueError("learned_attacker_specs must be a JSON object keyed by alias")

    normalized: Dict[str, Dict[str, Optional[str]]] = {}
    for raw_alias, raw_spec in policy_specs.items():
        alias = normalize_learned_attacker_alias(raw_alias)
        if alias in normalized:
            raise ValueError(
                f"duplicate learned attacker alias after normalization: {alias!r}"
            )
        if not isinstance(raw_spec, Mapping):
            raise ValueError(
                f"learned_attacker_specs[{alias!r}] must be a JSON object"
            )
        unknown_fields = set(raw_spec) - {"checkpoint", "reward_style"}
        if unknown_fields:
            raise ValueError(
                f"learned_attacker_specs[{alias!r}] has unknown fields: "
                f"{sorted(unknown_fields)}"
            )
        checkpoint = str(raw_spec.get("checkpoint", "")).strip()
        if not checkpoint:
            raise ValueError(
                f"learned_attacker_specs[{alias!r}].checkpoint must be non-empty"
            )
        raw_style = raw_spec.get("reward_style")
        reward_style = None if raw_style is None else str(raw_style).strip()
        if raw_style is not None and not reward_style:
            raise ValueError(
                f"learned_attacker_specs[{alias!r}].reward_style must be non-empty"
            )
        normalized[alias] = {
            "checkpoint": checkpoint,
            "reward_style": reward_style,
        }
    return normalized


class LearnedAttackerPolicy:
    """Frozen active multi-style PPO checkpoint with deterministic inference."""

    strategy = DEFAULT_LEARNED_ATTACKER_ALIAS

    def __init__(
        self,
        checkpoint_path: str,
        device: Any = "cpu",
        alias: str = DEFAULT_LEARNED_ATTACKER_ALIAS,
        reward_style: Optional[str] = None,
    ):
        path = Path(str(checkpoint_path)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"learned attacker checkpoint not found: {path}")

        self.checkpoint_path = str(path.resolve())
        self.device = torch.device(device)
        self.strategy = normalize_learned_attacker_alias(alias)
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        checkpoint_type = (
            str(checkpoint.get("network_type", ""))
            if isinstance(checkpoint, dict)
            else ""
        )
        if checkpoint_type != MULTISTYLE_PPO_NETWORK_TYPE:
            raise ValueError(
                "unsupported learned attacker checkpoint type "
                f"{checkpoint_type!r}; expected {MULTISTYLE_PPO_NETWORK_TYPE!r}"
            )
        self.model, checkpoint = AttackerMultiStylePPO.from_checkpoint(
            self.checkpoint_path,
            device=self.device,
            training=False,
        )
        requested_style = reward_style or checkpoint.get("selected_style")
        if requested_style is None:
            requested_style = self.model.style_names[0]
        self.reward_style = self.model.style_names[
            self.model.resolve_style_id(requested_style)
        ]
        self.model.network.eval()

    def __deepcopy__(self, memo):
        clone = object.__new__(type(self))
        memo[id(self)] = clone
        clone.checkpoint_path = self.checkpoint_path
        clone.device = self.device
        clone.strategy = self.strategy
        clone.model = self.model
        clone.reward_style = self.reward_style
        return clone

    def reset(self) -> None:
        return None

    def get_action_with_info(self, obs: np.ndarray) -> Tuple[np.ndarray, Dict[str, str]]:
        action_array = self.model.act(
            obs,
            style=self.reward_style,
            deterministic=True,
        )[0]
        return action_array, {
            "strategy": self.strategy,
            "reward_style": str(self.reward_style),
        }

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        action, _info = self.get_action_with_info(obs)
        return action


__all__ = [
    "DEFAULT_LEARNED_ATTACKER_ALIAS",
    "LearnedAttackerPolicy",
    "is_learned_attacker_alias",
    "normalize_learned_attacker_alias",
    "normalize_learned_attacker_specs",
]
