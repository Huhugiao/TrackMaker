"""Calibration helpers for two-skill HRL top policies."""

from __future__ import annotations

from typing import Sequence

import torch


def _skill_index(skill_names: Sequence[str], name: str) -> int | None:
    target = str(name).strip().lower()
    for index, skill in enumerate(skill_names):
        if str(skill).strip().lower() == target:
            return index
    return None


def apply_chase_logit_bias(
    logits: torch.Tensor,
    *,
    skill_names: Sequence[str],
    chase_logit_bias: float = 0.0,
) -> torch.Tensor:
    """Return logits with an additive bias applied to the chase skill."""
    bias = float(chase_logit_bias)
    if bias == 0.0:
        return logits
    chase_idx = _skill_index(skill_names, "chase")
    if chase_idx is None or chase_idx >= int(logits.shape[-1]):
        return logits
    adjusted = logits.clone()
    adjusted[..., chase_idx] = adjusted[..., chase_idx] + bias
    return adjusted


def build_two_skill_class_weights(
    skill_names: Sequence[str],
    *,
    baseline_weight: float = 1.0,
    chase_weight: float = 1.0,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Build class weights ordered like skill_names for baseline/chase training."""
    weights = []
    for skill in skill_names:
        key = str(skill).strip().lower()
        if key == "chase":
            weights.append(float(chase_weight))
        elif key == "baseline":
            weights.append(float(baseline_weight))
        else:
            weights.append(1.0)
    return torch.as_tensor(weights, dtype=dtype, device=device)
