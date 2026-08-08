"""Outcome, confidence interval, behavioral signature, and diversity metrics."""

from __future__ import annotations

from collections import Counter
import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


OUTCOMES = ("target_success", "defender_capture", "timeout", "defender_collision", "other")


def classify_outcome(reason: str, *, terminated: bool = False, truncated: bool = False) -> str:
    """Classify an episode without treating Defender collision as A success."""
    del terminated
    normalized = str(reason or "").strip().lower()
    if normalized == "attacker_caught_target":
        return "target_success"
    if normalized == "defender_caught_attacker":
        return "defender_capture"
    if normalized == "defender_collision" or "defender_out" in normalized:
        return "defender_collision"
    if truncated or any(token in normalized for token in ("timeout", "time_limit", "max_steps", "truncated")):
        return "timeout"
    return "other"


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = float(successes) / float(total)
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / total + z2 / (4.0 * total * total)) / denominator
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


def _mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    return float(arr.mean()) if arr.size else 0.0


def _polyline_length(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def behavior_signature(trace: Sequence[Mapping[str, object]], initial_target_distance: float) -> Dict[str, float]:
    """Compute one episode's policy-agnostic behavioral signature."""
    if not trace:
        return {key: 0.0 for key in signature_feature_names()}
    attacker = np.asarray([row["attacker"] for row in trace], dtype=np.float64)
    defender = np.asarray([row["defender"] for row in trace], dtype=np.float64)
    target = np.asarray([row["target"] for row in trace], dtype=np.float64)
    actions = np.asarray([row["action"] for row in trace], dtype=np.float64)
    subgoals = np.asarray([row["diagnostics"]["subgoal"] for row in trace], dtype=np.float64)
    steps = max(1, len(trace))

    deltas = np.diff(attacker, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1) if deltas.size else np.zeros(0)
    path_length = _polyline_length(attacker)
    headings = np.arctan2(deltas[:, 1], deltas[:, 0]) if deltas.shape[0] else np.zeros(0)
    heading_changes = np.diff(np.unwrap(headings)) if headings.size > 1 else np.zeros(0)
    defender_distances = np.linalg.norm(attacker - defender, axis=1)
    target_distances = np.linalg.norm(attacker - target, axis=1)
    initial_distance = max(float(initial_target_distance), 1e-6)

    initial_axis = target[0] - attacker[0]
    axis_norm = max(float(np.linalg.norm(initial_axis)), 1e-9)
    relative = attacker - attacker[0]
    cross = initial_axis[0] * relative[:, 1] - initial_axis[1] * relative[:, 0]
    left_fraction = float(np.mean(cross > 0.05 * axis_norm))
    right_fraction = float(np.mean(cross < -0.05 * axis_norm))

    movement_to_target = np.zeros(max(0, deltas.shape[0]), dtype=np.float64)
    if deltas.shape[0]:
        target_dirs = target[:-1] - attacker[:-1]
        target_norms = np.maximum(np.linalg.norm(target_dirs, axis=1), 1e-9)
        movement_to_target = np.sum(deltas * target_dirs, axis=1) / target_norms
    reverse_fraction = float(np.mean(movement_to_target < -0.05)) if movement_to_target.size else 0.0
    wait_fraction = float(np.mean(segment_lengths < 0.25)) if segment_lengths.size else 1.0

    if subgoals.shape[0] > 1:
        subgoal_switches = int(np.sum(np.linalg.norm(np.diff(subgoals, axis=0), axis=1) > 18.0))
    else:
        subgoal_switches = 0
    replans = [int(row["diagnostics"].get("replan_count", 0)) for row in trace]
    replan_count = max(replans, default=0)
    first_close_indices = np.flatnonzero(defender_distances < 120.0)
    first_close_turn = float(actions[first_close_indices[0], 0]) if first_close_indices.size else 0.0
    first_close_speed = float(actions[first_close_indices[0], 1]) if first_close_indices.size else 0.0

    signature = {
        "path_length": path_length,
        "detour_ratio": path_length / initial_distance,
        "mean_abs_curvature": _mean(np.abs(heading_changes)),
        "total_abs_curvature": float(np.abs(heading_changes).sum()),
        "min_defender_distance": float(defender_distances.min()),
        "mean_defender_distance": float(defender_distances.mean()),
        "target_progress_per_step": float((target_distances[0] - target_distances[-1]) / steps),
        "left_fraction": left_fraction,
        "right_fraction": right_fraction,
        "wait_fraction": wait_fraction,
        "reverse_fraction": reverse_fraction,
        "mean_speed_action": float(actions[:, 1].mean()),
        "std_speed_action": float(actions[:, 1].std()),
        "mean_abs_turn_action": float(np.abs(actions[:, 0]).mean()),
        "std_turn_action": float(actions[:, 0].std()),
        "subgoal_switch_rate": float(subgoal_switches / steps),
        "replan_rate": float(replan_count / steps),
        "first_close_turn": first_close_turn,
        "first_close_speed": first_close_speed,
    }
    # Coarse occupancy captures route topology without making dimensionality
    # dominate the scalar mechanism features.
    bins = 4
    hist, _, _ = np.histogram2d(
        attacker[:, 0], attacker[:, 1], bins=bins, range=((0.0, 640.0), (0.0, 640.0))
    )
    hist /= max(float(hist.sum()), 1.0)
    for index, value in enumerate(hist.reshape(-1)):
        signature[f"occupancy_{index:02d}"] = float(value)
    return signature


def signature_feature_names() -> Tuple[str, ...]:
    scalar = (
        "path_length", "detour_ratio", "mean_abs_curvature", "total_abs_curvature",
        "min_defender_distance", "mean_defender_distance", "target_progress_per_step",
        "left_fraction", "right_fraction", "wait_fraction", "reverse_fraction",
        "mean_speed_action", "std_speed_action", "mean_abs_turn_action", "std_turn_action",
        "subgoal_switch_rate", "replan_rate", "first_close_turn", "first_close_speed",
    )
    return scalar + tuple(f"occupancy_{index:02d}" for index in range(16))


def summarize_records(records: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    total = len(records)
    counts = Counter(str(row["outcome"]) for row in records)
    summary: Dict[str, object] = {"episodes": total}
    for outcome in OUTCOMES:
        count = int(counts.get(outcome, 0))
        rate = count / total if total else 0.0
        low, high = wilson_interval(count, total)
        summary[f"{outcome}_count"] = count
        summary[f"{outcome}_rate"] = rate
        summary[f"{outcome}_ci_low"] = low
        summary[f"{outcome}_ci_high"] = high
    summary["episode_length_mean"] = _mean(float(row["episode_length"]) for row in records)
    summary["episode_length_std"] = float(np.std([float(row["episode_length"]) for row in records])) if records else 0.0
    summary["attacker_collision_episode_rate"] = _mean(
        float(bool(row.get("attacker_collision_events", 0))) for row in records
    )
    summary["attacker_collision_events_mean"] = _mean(
        float(row.get("attacker_collision_events", 0)) for row in records
    )
    for feature in signature_feature_names():
        summary[f"behavior_{feature}"] = _mean(
            float(row.get("behavior", {}).get(feature, 0.0)) for row in records
        )
    return summary


def policy_behavior_vectors(records: Sequence[Mapping[str, object]]) -> Tuple[List[str], np.ndarray, List[str]]:
    names = sorted({str(row["policy"]) for row in records})
    features = list(signature_feature_names())
    matrix = np.zeros((len(names), len(features)), dtype=np.float64)
    for row_index, name in enumerate(names):
        selected = [row for row in records if str(row["policy"]) == name]
        for col_index, feature in enumerate(features):
            matrix[row_index, col_index] = _mean(
                float(row.get("behavior", {}).get(feature, 0.0)) for row in selected
            )
    return names, matrix, features


def standardized_behavior_distances(
    names: Sequence[str], matrix: np.ndarray, *, epsilon: float = 1e-9
) -> Dict[Tuple[str, str], float]:
    """Pairwise RMS z-distance; identical behavior remains exactly zero."""
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != len(names):
        raise ValueError("behavior matrix shape does not match names")
    scale = arr.std(axis=0)
    usable = scale > epsilon
    standardized = np.zeros_like(arr)
    if np.any(usable):
        standardized[:, usable] = (arr[:, usable] - arr[:, usable].mean(axis=0)) / scale[usable]
    distances = {}
    for i, left in enumerate(names):
        for j in range(i + 1, len(names)):
            right = names[j]
            diff = standardized[i] - standardized[j]
            distances[(str(left), str(right))] = float(math.sqrt(float(np.mean(diff * diff))))
    return distances


def cluster_by_distance(
    names: Sequence[str], distances: Mapping[Tuple[str, str], float], threshold: float = 0.70
) -> Dict[str, int]:
    parent = {str(name): str(name) for name in names}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for (left, right), distance in distances.items():
        if float(distance) < float(threshold):
            union(str(left), str(right))
    roots = {}
    assignments = {}
    for name in names:
        root = find(str(name))
        if root not in roots:
            roots[root] = len(roots)
        assignments[str(name)] = int(roots[root])
    return assignments

