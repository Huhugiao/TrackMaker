"""Deterministic curriculum construction and promotion gates for Attacker PPO."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy

import numpy as np


def build_weighted_bins(
    bin_catalog: Sequence[Mapping],
    weights: Mapping[str, float],
):
    """Build an AttackerEnv distribution from named curriculum weights."""
    catalog = {str(item["name"]): dict(item) for item in bin_catalog}
    output = []
    for name, raw_weight in weights.items():
        weight = float(raw_weight)
        if weight < 0.0:
            raise ValueError(f"negative curriculum weight for {name!r}: {weight}")
        if weight == 0.0:
            continue
        if str(name) == "default":
            output.append(
                {
                    "name": "default",
                    "default_distribution": True,
                    "weight": weight,
                }
            )
            continue
        if str(name) not in catalog:
            raise ValueError(f"unknown curriculum bin {name!r}; valid={tuple(catalog)}")
        entry = deepcopy(catalog[str(name)])
        entry["weight"] = weight
        output.append(entry)
    if not output or sum(float(item["weight"]) for item in output) <= 0.0:
        raise ValueError("curriculum distribution requires a positive total weight")
    return tuple(output)


def build_gate_bins(bin_catalog: Sequence[Mapping], gate_bin: str):
    return build_weighted_bins(bin_catalog, {str(gate_bin): 1.0})


class CompetenceCurriculum:
    """Require every style to pass repeated frontier evaluations before promotion."""

    def __init__(
        self,
        stages: Sequence[Mapping],
        style_names: Sequence[str],
        plateau_patience: int,
        plateau_min_delta: float,
    ):
        if not stages:
            raise ValueError("at least one curriculum stage is required")
        self.stages = tuple(dict(stage) for stage in stages)
        self.style_names = tuple(str(name) for name in style_names)
        if not self.style_names:
            raise ValueError("at least one reward style is required")
        self.plateau_patience = int(plateau_patience)
        self.plateau_min_delta = float(plateau_min_delta)
        self.stage_index = 0
        self.stage_start_step = 0
        self.consecutive_passes = 0
        self.best_frontier_score = -float("inf")
        self.evals_without_improvement = 0

    @property
    def stage(self):
        return self.stages[self.stage_index]

    @property
    def is_final(self) -> bool:
        return self.stage_index >= len(self.stages) - 1

    def state_dict(self):
        return {
            "stage_index": int(self.stage_index),
            "stage_start_step": int(self.stage_start_step),
            "consecutive_passes": int(self.consecutive_passes),
            "best_frontier_score": float(self.best_frontier_score),
            "evals_without_improvement": int(self.evals_without_improvement),
        }

    def load_state_dict(self, state: Mapping) -> None:
        stage_index = int(state.get("stage_index", 0))
        if stage_index < 0 or stage_index >= len(self.stages):
            raise ValueError(f"invalid curriculum stage index: {stage_index}")
        self.stage_index = stage_index
        self.stage_start_step = int(state.get("stage_start_step", 0))
        self.consecutive_passes = int(state.get("consecutive_passes", 0))
        self.best_frontier_score = float(
            state.get("best_frontier_score", -float("inf"))
        )
        self.evals_without_improvement = int(
            state.get("evals_without_improvement", 0)
        )

    def observe(self, global_step: int, evaluations: Mapping[str, Mapping]):
        rates = {
            style_name: float(evaluations[style_name]["target_success_rate"])
            for style_name in self.style_names
        }
        if not all(np.isfinite(value) for value in rates.values()):
            raise FloatingPointError(f"non-finite curriculum evaluation: {rates}")
        frontier_score = min(rates.values())
        if frontier_score >= self.best_frontier_score + self.plateau_min_delta:
            self.best_frontier_score = frontier_score
            self.evals_without_improvement = 0
        else:
            self.evals_without_improvement += 1

        threshold = self.stage.get("gate_success_rate")
        criteria = self.stage.get("gate_criteria")
        criteria_checks = {}
        if criteria:
            for style_name, style_criteria in dict(criteria.get("styles", {})).items():
                if style_name not in evaluations:
                    raise ValueError(
                        f"curriculum gate references unknown style {style_name!r}"
                    )
                metrics = evaluations[style_name]
                if "min_success_rate" in style_criteria:
                    key = f"{style_name}.min_success_rate"
                    criteria_checks[key] = bool(
                        float(metrics["target_success_rate"])
                        >= float(style_criteria["min_success_rate"])
                    )
                if "max_capture_rate" in style_criteria:
                    key = f"{style_name}.max_capture_rate"
                    criteria_checks[key] = bool(
                        float(metrics["defender_capture_rate"])
                        <= float(style_criteria["max_capture_rate"])
                    )
                if "max_timeout_rate" in style_criteria:
                    key = f"{style_name}.max_timeout_rate"
                    criteria_checks[key] = bool(
                        float(metrics["timeout_rate"])
                        <= float(style_criteria["max_timeout_rate"])
                    )
            reduction_key = "min_evasive_capture_reduction_vs_goal_rush"
            if reduction_key in criteria:
                if not {"goal_rush", "evasive"}.issubset(evaluations):
                    raise ValueError(
                        "capture-reduction gate requires goal_rush and evasive"
                    )
                capture_reduction = float(
                    evaluations["goal_rush"]["defender_capture_rate"]
                    - evaluations["evasive"]["defender_capture_rate"]
                )
                criteria_checks[reduction_key] = bool(
                    capture_reduction >= float(criteria[reduction_key])
                )
            else:
                capture_reduction = None
        else:
            capture_reduction = None
        elapsed = int(global_step) - int(self.stage_start_step)
        minimum = int(self.stage.get("min_environment_steps", 0))
        required = int(self.stage.get("required_consecutive_passes", 0))
        gate_configured = threshold is not None or bool(criteria_checks)
        eligible = gate_configured and elapsed >= minimum and not self.is_final
        success_passed = bool(
            threshold is None
            or all(value >= float(threshold) for value in rates.values())
        )
        passed_now = bool(
            eligible
            and success_passed
            and all(criteria_checks.values())
        )
        self.consecutive_passes = self.consecutive_passes + 1 if passed_now else 0
        promoted = bool(required > 0 and self.consecutive_passes >= required)
        previous_stage = str(self.stage["name"])
        if promoted:
            self.stage_index += 1
            self.stage_start_step = int(global_step)
            self.consecutive_passes = 0
            self.best_frontier_score = -float("inf")
            self.evals_without_improvement = 0

        return {
            "stage": previous_stage,
            "rates": rates,
            "frontier_score": float(frontier_score),
            "threshold": None if threshold is None else float(threshold),
            "criteria_checks": criteria_checks,
            "capture_reduction": capture_reduction,
            "stage_environment_steps": int(elapsed),
            "minimum_environment_steps": int(minimum),
            "eligible": bool(eligible),
            "passed_now": bool(passed_now),
            "consecutive_passes": int(
                required if promoted else self.consecutive_passes
            ),
            "required_consecutive_passes": int(required),
            "promoted": bool(promoted),
            "next_stage": str(self.stage["name"]) if promoted else None,
            "plateau_alert": bool(
                self.plateau_patience > 0
                and self.evals_without_improvement >= self.plateau_patience
            ),
        }


__all__ = [
    "CompetenceCurriculum",
    "build_gate_bins",
    "build_weighted_bins",
]
