"""Episode-level diagnostics for Coppelia rollout JSON files."""

from __future__ import annotations

import math
from typing import Any


def _agent_center(state: dict[str, Any], role: str) -> tuple[float, float] | None:
    agent = state.get(role)
    if not isinstance(agent, dict):
        return None
    try:
        return float(agent["x"]), float(agent["y"])
    except (KeyError, TypeError, ValueError):
        return None


def _distance(state: dict[str, Any], a: str, b: str) -> float | None:
    pa = _agent_center(state, a)
    pb = _agent_center(state, b)
    if pa is None or pb is None:
        return None
    return float(math.hypot(pb[0] - pa[0], pb[1] - pa[1]))


def _normalize_angle_deg(angle: float) -> float:
    angle = float(angle) % 360.0
    if angle > 180.0:
        angle -= 360.0
    return float(angle)


def _capture_heading_error(state: dict[str, Any]) -> float | None:
    defender = state.get("defender")
    attacker = state.get("attacker")
    if not isinstance(defender, dict) or not isinstance(attacker, dict):
        return None
    try:
        dx = float(attacker["x"]) - float(defender["x"])
        dy = float(attacker["y"]) - float(defender["y"])
        heading = float(defender["theta"])
    except (KeyError, TypeError, ValueError):
        return None
    bearing = math.degrees(math.atan2(dy, dx))
    return float(abs(_normalize_angle_deg(bearing - heading)))


def _min_or_none(values: list[float]) -> float | None:
    return float(min(values)) if values else None


def summarize_episode_diagnostics(
    records: list[dict[str, Any]],
    final_state: dict[str, Any],
    capture_radius_px: float = 20.0,
    capture_half_angle_deg: float = 15.0,
) -> dict[str, Any]:
    states = [dict(record.get("state", {}) or {}) for record in records if isinstance(record, dict)]
    if final_state and (not states or states[-1] != final_state):
        states.append(dict(final_state))

    def_att = [value for value in (_distance(state, "defender", "attacker") for state in states) if value is not None]
    att_tgt = [value for value in (_distance(state, "attacker", "target") for state in states) if value is not None]
    capture_heading = [value for value in (_capture_heading_error(state) for state in states) if value is not None]
    inside_capture_heading = [
        heading
        for state in states
        for dist, heading in [(_distance(state, "defender", "attacker"), _capture_heading_error(state))]
        if dist is not None and heading is not None and float(dist) <= float(capture_radius_px)
    ]

    final_def_att = _distance(dict(final_state or {}), "defender", "attacker")
    final_att_tgt = _distance(dict(final_state or {}), "attacker", "target")
    final_capture_heading = _capture_heading_error(dict(final_state or {}))
    defender_collisions = 0
    attacker_collisions = 0
    capture_progress_values: list[int] = []
    capture_required_steps = 0
    for record in records:
        info = dict(record.get("info", {}) or {})
        defender_collisions += 1 if info.get("defender_collision") else 0
        attacker_collisions += 1 if info.get("attacker_collision") else 0
        try:
            capture_progress_values.append(int(info.get("capture_progress_defender", 0) or 0))
        except (TypeError, ValueError):
            capture_progress_values.append(0)
        try:
            capture_required_steps = max(capture_required_steps, int(info.get("capture_required_steps", 0) or 0))
        except (TypeError, ValueError):
            pass

    inside_capture_steps = int(len(inside_capture_heading))
    inside_aligned_steps = int(sum(1 for value in inside_capture_heading if float(value) <= float(capture_half_angle_deg)))
    inside_unaligned_steps = int(inside_capture_steps - inside_aligned_steps)
    inside_heading_mean = (
        float(sum(float(value) for value in inside_capture_heading) / len(inside_capture_heading))
        if inside_capture_heading
        else None
    )

    return {
        "min_defender_attacker_dist_px": _min_or_none(def_att),
        "final_defender_attacker_dist_px": float(final_def_att) if final_def_att is not None else None,
        "min_attacker_target_dist_px": _min_or_none(att_tgt),
        "final_attacker_target_dist_px": float(final_att_tgt) if final_att_tgt is not None else None,
        "min_capture_heading_error_deg": _min_or_none(capture_heading),
        "final_capture_heading_error_deg": float(final_capture_heading) if final_capture_heading is not None else None,
        "max_capture_progress_defender": int(max(capture_progress_values) if capture_progress_values else 0),
        "capture_required_steps": int(capture_required_steps),
        "inside_capture_radius_steps": int(inside_capture_steps),
        "inside_aligned_capture_steps": int(inside_aligned_steps),
        "inside_unaligned_capture_steps": int(inside_unaligned_steps),
        "inside_capture_heading_error_deg_mean": inside_heading_mean,
        "defender_collision_steps": int(defender_collisions),
        "attacker_collision_steps": int(attacker_collisions),
    }


def classify_episode_outcome(
    episode: dict[str, Any],
    entry_radius_px: float,
    capture_radius_px: float,
) -> str:
    """Classify one rollout episode into a coarse chase failure mode."""
    if bool(episode.get("win", False)):
        return "win"
    diagnostics = dict(episode.get("diagnostics", {}) or {})
    if int(diagnostics.get("defender_collision_steps") or 0) > 0:
        return "defender_collision"
    if int(diagnostics.get("attacker_collision_steps") or 0) > 0:
        return "attacker_collision"
    min_dist = diagnostics.get("min_defender_attacker_dist_px")
    if min_dist is None:
        return "missing_distance_diagnostics"
    min_dist = float(min_dist)
    if min_dist > float(entry_radius_px):
        return "never_entered_entry_radius"
    if min_dist > float(capture_radius_px):
        return "entered_entry_not_capture"
    return "entered_capture_not_win"
