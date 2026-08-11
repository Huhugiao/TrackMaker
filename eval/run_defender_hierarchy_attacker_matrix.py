#!/usr/bin/env python3
"""Paired-seed matrix for frozen Defender candidates and the Attacker pool."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.attacker_env import AttackerEnv  # noqa: E402
from attacker.learned_policy import LearnedAttackerPolicy  # noqa: E402
from attacker.frozen_pool import (  # noqa: E402
    FROZEN_ATTACKER_POOL,
    LEARNED_ATTACKER_SPECS,
    PROGRAMMATIC_ATTACKER_ALIASES,
)
from attacker_heuristics.metrics import classify_outcome, wilson_interval  # noqa: E402
from attacker_heuristics.registry import (  # noqa: E402
    POLICY_REGISTRY,
    create_policy,
)


LEGACY_ATTACKERS = (
    "default",
    "evasive",
)
NEW_ATTACKERS = (
    "geometry_feint_v3",
    "occlusion_dash_v2",
)
DEFAULT_ATTACKERS = LEGACY_ATTACKERS + NEW_ATTACKERS
DEFAULT_DEFENDERS = ("hrl", "protect_skill", "chase_skill")

# User-confirmed active Defender pool.  Historical variants remain registered
# below for reproducibility, but are not selected by the default matrix.
POOL_DEFENDERS = (
    "protect_skill",
    "chase_skill",
    "hrl",
)


@dataclass(frozen=True)
class DefenderVariant:
    name: str
    env_strategy: str
    role: str
    top_checkpoint: Optional[str]
    protect_checkpoint: str
    chase_checkpoint: str


DEFENDER_VARIANTS: Dict[str, DefenderVariant] = {
    "hrl": DefenderVariant(
        name="hrl",
        env_strategy="hrl",
        role="learned top over frozen Protect/Chase skills",
        top_checkpoint="models/hrl_ch2_m1_astar_cached_top_20260606_170036/best_model.pth",
        protect_checkpoint="models/defender_protect_mlp_ctde_repro_20260526/final_model.pth",
        chase_checkpoint="models/defender_chase_nmn_dual_gru_raw_dense_05-05-19-12/final_model.pth",
    ),
    "protect_skill": DefenderVariant(
        name="protect_skill",
        env_strategy="skill_protect",
        role="active 20260721 Protect best-balanced; trained equally against frozen6 Attackers",
        top_checkpoint=None,
        protect_checkpoint="models/defender_protect_mlp_ctde_frozen6_20260721_105148/best_balanced_model.pth",
        chase_checkpoint="models/defender_chase_nmn_dual_gru_raw_dense_05-05-19-12/final_model.pth",
    ),
    "chase_skill": DefenderVariant(
        name="chase_skill",
        env_strategy="skill_chase",
        role="frozen recurrent chase bottom skill only",
        top_checkpoint=None,
        protect_checkpoint="models/defender_protect_mlp_ctde_repro_20260526/final_model.pth",
        chase_checkpoint="models/defender_chase_nmn_dual_gru_raw_dense_05-05-19-12/final_model.pth",
    ),
}


def _split_csv(value: str) -> Tuple[str, ...]:
    return tuple(token.strip().lower() for token in str(value).split(",") if token.strip())


class LearnedAttackerEvalAdapter:
    """Give frozen PPO attackers the same reset/diagnostic API as heuristics."""

    def __init__(self, name: str, spec: Mapping[str, str]):
        self.name = str(name)
        self.policy = LearnedAttackerPolicy(
            checkpoint_path=str(_checkpoint_path(spec["checkpoint"])),
            device="cpu",
            alias=self.name,
            reward_style=spec.get("reward_style"),
        )
        self._diagnostics: Dict[str, object] = {}

    def reset(self, seed: Optional[int] = None):
        del seed
        self.policy.reset()
        self._diagnostics = {
            "mode": self.name,
            "state": "reset",
            "mechanism": "frozen PPO attacker checkpoint",
            "details": {"reward_style": self.policy.reward_style},
        }

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        action, info = self.policy.get_action_with_info(obs)
        self._diagnostics = {
            "mode": self.name,
            "state": str(info.get("reward_style", self.name)),
            "mechanism": "frozen PPO attacker checkpoint",
            "details": dict(info),
        }
        return np.asarray(action, dtype=np.float32)

    def get_action_with_info(self, obs: np.ndarray):
        action = self.get_action(obs)
        return action, dict(self._diagnostics)

    @property
    def diagnostics(self):
        return dict(self._diagnostics)


def create_attacker(name: str, seed: int):
    key = str(name).strip().lower()
    if key in PROGRAMMATIC_ATTACKER_ALIASES:
        return create_policy(PROGRAMMATIC_ATTACKER_ALIASES[key], seed=int(seed))
    if key in LEARNED_ATTACKER_SPECS:
        return LearnedAttackerEvalAdapter(key, LEARNED_ATTACKER_SPECS[key])
    if key in POLICY_REGISTRY:
        return create_policy(key, seed=int(seed))
    valid = sorted(
        set(POLICY_REGISTRY)
        | set(PROGRAMMATIC_ATTACKER_ALIASES)
        | set(LEARNED_ATTACKER_SPECS)
    )
    raise KeyError(f"Unknown attacker {name!r}; valid={valid}")


def _checkpoint_path(relative: Optional[str]) -> Optional[Path]:
    return None if relative is None else (PROJECT_ROOT / relative).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def defender_audit(names: Sequence[str]) -> List[Dict[str, object]]:
    rows = []
    for name in names:
        spec = DEFENDER_VARIANTS[name]
        checkpoints = {}
        for field, relative in (
            ("top", spec.top_checkpoint),
            ("protect", spec.protect_checkpoint),
            ("chase", spec.chase_checkpoint),
        ):
            path = _checkpoint_path(relative)
            if path is None:
                checkpoints[field] = None
            else:
                checkpoints[field] = {
                    "path": str(path),
                    "exists": path.is_file(),
                    "size_bytes": path.stat().st_size if path.is_file() else None,
                    "sha256": _sha256(path) if path.is_file() else None,
                }
        rows.append({**asdict(spec), "checkpoints": checkpoints})
    return rows


def _controller_config(spec: DefenderVariant) -> Dict[str, object]:
    return {
        "device": "cpu",
        "hrl_num_skills": 2,
        "top_policy_path": None if spec.top_checkpoint is None else str(_checkpoint_path(spec.top_checkpoint)),
        "primary_skill_path": str(_checkpoint_path(spec.protect_checkpoint)),
        "chase_skill_path": str(_checkpoint_path(spec.chase_checkpoint)),
        "protect_skill_path": str(_checkpoint_path(spec.protect_checkpoint)),
    }


def _positions(env: AttackerEnv) -> Dict[str, np.ndarray]:
    state = env.get_privileged_state()
    return {
        name: np.asarray([state[name]["center_x"], state[name]["center_y"]], dtype=np.float64)
        for name in ("attacker", "defender", "target")
    }


def run_episode(
    env: AttackerEnv,
    defender_name: str,
    attacker_name: str,
    seed: int,
    policy=None,
) -> Dict[str, object]:
    obs, _reset_info = env.reset(seed=int(seed))
    policy = policy or create_attacker(attacker_name, int(seed))
    policy.reset(seed=int(seed))
    initial = _positions(env)
    previous = initial
    attacker_path = 0.0
    defender_path = 0.0
    distances = [float(np.linalg.norm(initial["attacker"] - initial["defender"]))]
    skill_counts = Counter()
    skill_switches = 0
    previous_skill = None
    first_skill = None
    last_skill = None
    collision_events = 0
    safety_steps = 0
    safety_interventions = 0
    safety_unavoidable = 0
    safety_action_delta = 0.0
    done = False
    terminated = truncated = False
    info: Mapping[str, object] = {}

    while not done:
        action = policy.get_action(obs)
        obs, _reward, terminated, truncated, info = env.step(action)
        current = _positions(env)
        attacker_path += float(np.linalg.norm(current["attacker"] - previous["attacker"]))
        defender_path += float(np.linalg.norm(current["defender"] - previous["defender"]))
        distances.append(float(np.linalg.norm(current["attacker"] - current["defender"])))
        previous = current
        collision_events = max(collision_events, int(info.get("attacker_collision_event_count", 0)))
        if bool(info.get("defender_radar_safety_enabled", False)):
            safety_steps += 1
            safety_interventions += int(bool(info.get("defender_radar_safety_intervened", False)))
            safety_unavoidable += int(info.get("defender_radar_safety_mode") == "unavoidable")
            safety_action_delta += float(info.get("defender_radar_safety_action_delta", 0.0))

        controller = env.defender_policy
        skill_name = getattr(controller, "last_skill_name", None)
        if skill_name in ("protect", "chase"):
            skill_name = str(skill_name)
            skill_counts[skill_name] += 1
            if first_skill is None:
                first_skill = skill_name
            if previous_skill is not None and skill_name != previous_skill:
                skill_switches += 1
            previous_skill = skill_name
            last_skill = skill_name
        done = bool(terminated or truncated)

    reason = str(info.get("reason", "unknown"))
    outcome = classify_outcome(reason, terminated=terminated, truncated=truncated)
    total_skill_steps = int(sum(skill_counts.values()))
    final_target_distance = float(np.linalg.norm(previous["attacker"] - previous["target"]))
    initial_target_distance = float(np.linalg.norm(initial["attacker"] - initial["target"]))
    return {
        "defender": defender_name,
        "attacker": attacker_name,
        "seed": int(seed),
        "reason": reason,
        "outcome": outcome,
        "target_success": outcome == "target_success",
        "defender_capture": outcome == "defender_capture",
        "timeout": outcome == "timeout",
        "defender_collision": outcome == "defender_collision",
        "defender_success": outcome in ("defender_capture", "timeout"),
        "env_obstacle_mask": False,
        "controller_obstacle_mask": False,
        "radar_steering_filter": bool(getattr(env.env, "defender_radar_safety_enabled", False)),
        "radar_steering_intervention_rate": (
            float(safety_interventions / safety_steps) if safety_steps else 0.0
        ),
        "radar_steering_unavoidable_rate": (
            float(safety_unavoidable / safety_steps) if safety_steps else 0.0
        ),
        "radar_steering_action_delta_mean": (
            float(safety_action_delta / safety_steps) if safety_steps else 0.0
        ),
        "episode_length": int(env.step_count),
        "attacker_collision_events": int(collision_events),
        "attacker_path_length": attacker_path,
        "defender_path_length": defender_path,
        "mean_attacker_defender_distance": float(np.mean(distances)),
        "initial_target_distance": initial_target_distance,
        "final_target_distance": final_target_distance,
        "target_progress": initial_target_distance - final_target_distance,
        "protect_skill_rate": float(skill_counts["protect"] / total_skill_steps) if total_skill_steps else 0.0,
        "chase_skill_rate": float(skill_counts["chase"] / total_skill_steps) if total_skill_steps else 0.0,
        "skill_switches": int(skill_switches),
        "first_skill": first_skill,
        "last_skill": last_skill,
        "final_attacker_diagnostics": policy.diagnostics,
    }


def _mean(rows: Sequence[Mapping[str, object]], key: str) -> float:
    return float(np.mean([float(row.get(key, 0.0)) for row in rows])) if rows else 0.0


def summarize(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    total = len(rows)
    summary: Dict[str, object] = {"episodes": total}
    for key in ("target_success", "defender_capture", "timeout", "defender_collision", "defender_success"):
        count = sum(bool(row.get(key, False)) for row in rows)
        low, high = wilson_interval(count, total)
        summary[f"{key}_count"] = int(count)
        summary[f"{key}_rate"] = count / total if total else 0.0
        summary[f"{key}_ci_low"] = low
        summary[f"{key}_ci_high"] = high
    for key in (
        "episode_length",
        "attacker_path_length",
        "defender_path_length",
        "mean_attacker_defender_distance",
        "target_progress",
        "protect_skill_rate",
        "chase_skill_rate",
        "skill_switches",
        "radar_steering_intervention_rate",
        "radar_steering_unavoidable_rate",
        "radar_steering_action_delta_mean",
    ):
        summary[f"{key}_mean"] = _mean(rows, key)
    summary["attacker_collision_episode_rate"] = (
        sum(bool(row.get("attacker_collision_events", 0)) for row in rows) / total if total else 0.0
    )
    return summary


def _exact_discordant_p(left: int, right: int) -> float:
    n = int(left + right)
    if n <= 0:
        return 1.0
    tail = min(int(left), int(right))
    probability = 2.0 * sum(math.comb(n, k) for k in range(tail + 1)) / float(2**n)
    return float(min(1.0, probability))


def paired_comparison(
    records: Sequence[Mapping[str, object]],
    attacker_names: Sequence[str],
    comparators: Sequence[str] = ("protect_skill", "chase_skill"),
) -> List[Dict[str, object]]:
    keyed = {
        (str(row["defender"]), str(row["attacker"]), int(row["seed"])): row
        for row in records
    }
    rows = []
    for comparator in comparators:
        for attacker in tuple(attacker_names) + ("__all__",):
            selected_attackers = tuple(attacker_names) if attacker == "__all__" else (attacker,)
            hrl_blocks = comparator_blocks = target_ties = 0
            hrl_defense_wins = comparator_defense_wins = defense_ties = 0
            hrl_target_count = comparator_target_count = 0
            pairs = 0
            for selected in selected_attackers:
                seeds = sorted(
                    seed for defender, name, seed in keyed
                    if defender == "hrl" and name == selected
                )
                for seed in seeds:
                    hrl = keyed[("hrl", selected, seed)]
                    other = keyed[(comparator, selected, seed)]
                    hrl_target = bool(hrl["target_success"])
                    other_target = bool(other["target_success"])
                    hrl_defense = bool(hrl["defender_success"])
                    other_defense = bool(other["defender_success"])
                    hrl_target_count += int(hrl_target)
                    comparator_target_count += int(other_target)
                    hrl_blocks += int(other_target and not hrl_target)
                    comparator_blocks += int(hrl_target and not other_target)
                    target_ties += int(hrl_target == other_target)
                    hrl_defense_wins += int(hrl_defense and not other_defense)
                    comparator_defense_wins += int(other_defense and not hrl_defense)
                    defense_ties += int(hrl_defense == other_defense)
                    pairs += 1
            rows.append({
                "attacker": "all" if attacker == "__all__" else attacker,
                "comparator": comparator,
                "pairs": pairs,
                "hrl_target_success_count": hrl_target_count,
                "comparator_target_success_count": comparator_target_count,
                "hrl_target_success_delta": (hrl_target_count - comparator_target_count) / pairs if pairs else 0.0,
                "hrl_unique_blocks": hrl_blocks,
                "comparator_unique_blocks": comparator_blocks,
                "target_ties": target_ties,
                "target_discordant_exact_p": _exact_discordant_p(hrl_blocks, comparator_blocks),
                "hrl_unique_defense_successes": hrl_defense_wins,
                "comparator_unique_defense_successes": comparator_defense_wins,
                "defense_ties": defense_ties,
                "defense_discordant_exact_p": _exact_discordant_p(hrl_defense_wins, comparator_defense_wins),
            })
    return rows


def hrl_skill_outcome_summary(
    records: Sequence[Mapping[str, object]],
    attacker_names: Sequence[str],
) -> List[Dict[str, object]]:
    """Describe top-policy skill use conditional on the terminal outcome.

    Both the episode-mean and step-weighted selection rates are reported.  The
    latter prevents a collection of short episodes from dominating the skill
    allocation estimate.
    """
    hrl_records = [row for row in records if str(row.get("defender")) == "hrl"]
    rows: List[Dict[str, object]] = []
    scopes = [("all", hrl_records)] + [
        (name, [row for row in hrl_records if str(row.get("attacker")) == name])
        for name in attacker_names
    ]
    for scope, scope_records in scopes:
        for outcome in ("target_success", "defender_capture", "timeout", "defender_collision"):
            selected = [row for row in scope_records if bool(row.get(outcome, False))]
            if not selected:
                continue
            steps = sum(int(row.get("episode_length", 0)) for row in selected)
            protect_steps = sum(
                float(row.get("protect_skill_rate", 0.0)) * int(row.get("episode_length", 0))
                for row in selected
            )
            rows.append({
                "attacker": scope,
                "outcome": outcome,
                "episodes": len(selected),
                "protect_skill_rate_episode_mean": _mean(selected, "protect_skill_rate"),
                "chase_skill_rate_episode_mean": _mean(selected, "chase_skill_rate"),
                "protect_skill_rate_step_weighted": protect_steps / steps if steps else 0.0,
                "chase_skill_rate_step_weighted": 1.0 - protect_steps / steps if steps else 0.0,
                "first_skill_protect_rate": (
                    sum(row.get("first_skill") == "protect" for row in selected) / len(selected)
                ),
                "last_skill_protect_rate": (
                    sum(row.get("last_skill") == "protect" for row in selected) / len(selected)
                ),
                "skill_switches_mean": _mean(selected, "skill_switches"),
                "skill_switches_median": float(np.median([int(row.get("skill_switches", 0)) for row in selected])),
                "episode_length_mean": _mean(selected, "episode_length"),
            })
    return rows


def hrl_switch_bin_summary(records: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Report outcome rates by top-policy switch-count bin (descriptive, not causal)."""
    hrl_records = [row for row in records if str(row.get("defender")) == "hrl"]
    bins = ((0, 0, "0"), (1, 5, "1-5"), (6, 15, "6-15"), (16, 30, "16-30"), (31, None, "31+"))
    rows = []
    for low, high, label in bins:
        selected = [
            row for row in hrl_records
            if int(row.get("skill_switches", 0)) >= low
            and (high is None or int(row.get("skill_switches", 0)) <= high)
        ]
        if selected:
            rows.append({
                "switch_bin": label,
                "episodes": len(selected),
                "target_success_rate": sum(bool(row.get("target_success")) for row in selected) / len(selected),
                "defender_capture_rate": sum(bool(row.get("defender_capture")) for row in selected) / len(selected),
                "timeout_rate": sum(bool(row.get("timeout")) for row in selected) / len(selected),
                "protect_skill_rate_mean": _mean(selected, "protect_skill_rate"),
                "episode_length_mean": _mean(selected, "episode_length"),
            })
    return rows


def paired_discordant_episodes(records: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Return every paired seed where HRL and a frozen skill differ on breach."""
    keyed = {
        (str(row["defender"]), str(row["attacker"]), int(row["seed"])): row
        for row in records
    }
    rows = []
    for (defender, attacker, seed), hrl in sorted(keyed.items()):
        if defender != "hrl":
            continue
        for comparator in ("protect_skill", "chase_skill"):
            other = keyed.get((comparator, attacker, seed))
            if other is None or bool(hrl["target_success"]) == bool(other["target_success"]):
                continue
            rows.append({
                "attacker": attacker,
                "seed": seed,
                "comparator": comparator,
                "direction": (
                    "comparator_blocks_hrl_breach"
                    if bool(hrl["target_success"])
                    else "hrl_blocks_comparator_breach"
                ),
                "hrl_outcome": hrl["outcome"],
                "comparator_outcome": other["outcome"],
                "hrl_protect_skill_rate": hrl.get("protect_skill_rate", 0.0),
                "hrl_chase_skill_rate": hrl.get("chase_skill_rate", 0.0),
                "hrl_skill_switches": hrl.get("skill_switches", 0),
                "hrl_first_skill": hrl.get("first_skill"),
                "hrl_last_skill": hrl.get("last_skill"),
                "hrl_episode_length": hrl.get("episode_length", 0),
            })
    return rows


def new_vs_legacy_comparison(
    records: Sequence[Mapping[str, object]],
    defenders: Sequence[str],
    new_attackers: Sequence[str] = NEW_ATTACKERS,
    legacy_attackers: Sequence[str] = LEGACY_ATTACKERS,
) -> List[Dict[str, object]]:
    """Pair every new Attacker with every legacy Attacker on identical seeds."""
    keyed = {
        (str(row["defender"]), str(row["attacker"]), int(row["seed"])): row
        for row in records
    }
    rows: List[Dict[str, object]] = []
    binary_metrics = (
        "target_success",
        "defender_capture",
        "timeout",
        "defender_collision",
        "attacker_collision_episode",
    )
    continuous_metrics = (
        "episode_length",
        "attacker_path_length",
        "mean_attacker_defender_distance",
        "target_progress",
    )

    def metric_value(row: Mapping[str, object], metric: str):
        if metric == "attacker_collision_episode":
            return bool(row.get("attacker_collision_events", 0))
        return row.get(metric, False if metric in binary_metrics else 0.0)

    for defender in defenders:
        for new_attacker in new_attackers:
            new_seeds = {
                seed for d, attacker, seed in keyed
                if d == defender and attacker == new_attacker
            }
            for legacy_attacker in legacy_attackers:
                legacy_seeds = {
                    seed for d, attacker, seed in keyed
                    if d == defender and attacker == legacy_attacker
                }
                seeds = sorted(new_seeds & legacy_seeds)
                pair_rows = [
                    (keyed[(defender, new_attacker, seed)], keyed[(defender, legacy_attacker, seed)])
                    for seed in seeds
                ]
                result: Dict[str, object] = {
                    "defender": defender,
                    "new_attacker": new_attacker,
                    "legacy_attacker": legacy_attacker,
                    "pairs": len(pair_rows),
                }
                for metric in binary_metrics:
                    new_values = [bool(metric_value(new, metric)) for new, _old in pair_rows]
                    old_values = [bool(metric_value(old, metric)) for _new, old in pair_rows]
                    new_count = sum(new_values)
                    old_count = sum(old_values)
                    new_unique = sum(new and not old for new, old in zip(new_values, old_values))
                    old_unique = sum(old and not new for new, old in zip(new_values, old_values))
                    total = len(pair_rows)
                    result[f"new_{metric}_rate"] = new_count / total if total else 0.0
                    result[f"legacy_{metric}_rate"] = old_count / total if total else 0.0
                    result[f"{metric}_delta"] = (new_count - old_count) / total if total else 0.0
                    result[f"new_unique_{metric}"] = new_unique
                    result[f"legacy_unique_{metric}"] = old_unique
                    result[f"{metric}_discordant_exact_p"] = _exact_discordant_p(new_unique, old_unique)
                for metric in continuous_metrics:
                    new_values = [float(metric_value(new, metric)) for new, _old in pair_rows]
                    old_values = [float(metric_value(old, metric)) for _new, old in pair_rows]
                    deltas = [new - old for new, old in zip(new_values, old_values)]
                    result[f"new_{metric}_mean"] = float(np.mean(new_values)) if new_values else 0.0
                    result[f"legacy_{metric}_mean"] = float(np.mean(old_values)) if old_values else 0.0
                    result[f"{metric}_delta_mean"] = float(np.mean(deltas)) if deltas else 0.0
                rows.append(result)
    return rows


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(type(value).__name__)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_heatmap(
    path: Path,
    summary_rows: Sequence[Mapping[str, object]],
    attackers: Sequence[str],
    defenders: Sequence[str],
    metric: str,
    title: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = {(str(row["defender"]), str(row["attacker"])): float(row[metric]) for row in summary_rows}
    matrix = np.asarray([[lookup[(d, a)] for a in attackers] for d in defenders], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(max(9.0, 1.2 * len(attackers)), 4.2), dpi=150)
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="RdYlGn_r" if metric == "target_success_rate" else "RdYlGn")
    ax.set_xticks(range(len(attackers)), labels=attackers, rotation=35, ha="right")
    ax.set_yticks(range(len(defenders)), labels=defenders)
    ax.set_title(title)
    for i in range(len(defenders)):
        for j in range(len(attackers)):
            ax.text(j, i, f"{100.0 * matrix[i, j]:.1f}%", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.03)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _plot_hrl_skill_mix(
    path: Path,
    summary_rows: Sequence[Mapping[str, object]],
    attackers: Sequence[str],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = {
        str(row["attacker"]): row
        for row in summary_rows
        if str(row["defender"]) == "hrl"
    }
    protect = np.asarray([float(lookup[name]["protect_skill_rate_mean"]) for name in attackers])
    chase = np.asarray([float(lookup[name]["chase_skill_rate_mean"]) for name in attackers])
    fig, ax = plt.subplots(figsize=(max(9.0, 1.2 * len(attackers)), 4.5), dpi=150)
    x = np.arange(len(attackers))
    ax.bar(x, protect, label="protect skill", color="#4C78A8")
    ax.bar(x, chase, bottom=protect, label="chase skill", color="#F58518")
    ax.set_xticks(x, labels=attackers, rotation=35, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("HRL selection fraction")
    ax.set_title("Learned HRL top-policy skill allocation")
    ax.legend(loc="upper right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _plot_new_vs_legacy_delta(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    defenders: Sequence[str],
    new_attackers: Sequence[str],
    legacy_attackers: Sequence[str],
    metric: str,
    title: str,
    percentage: bool = True,
) -> None:
    """Plot one 2x6 new-minus-legacy matrix for every Defender."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = {
        (str(row["defender"]), str(row["new_attacker"]), str(row["legacy_attacker"])): float(row[metric])
        for row in rows
    }
    matrices = [
        np.asarray(
            [[lookup[(defender, new, old)] for old in legacy_attackers] for new in new_attackers],
            dtype=np.float64,
        )
        for defender in defenders
    ]
    max_abs = max(1e-9, max(float(np.max(np.abs(matrix))) for matrix in matrices))
    fig, axes = plt.subplots(
        len(defenders), 1,
        figsize=(max(10.0, 1.55 * len(legacy_attackers)), 3.0 * len(defenders)),
        dpi=160,
        squeeze=False,
    )
    image = None
    for index, (defender, matrix) in enumerate(zip(defenders, matrices)):
        ax = axes[index, 0]
        image = ax.imshow(matrix, vmin=-max_abs, vmax=max_abs, cmap="coolwarm", aspect="auto")
        ax.set_xticks(range(len(legacy_attackers)), labels=legacy_attackers, rotation=25, ha="right")
        ax.set_yticks(range(len(new_attackers)), labels=new_attackers)
        ax.set_title(defender)
        for i in range(len(new_attackers)):
            for j in range(len(legacy_attackers)):
                label = f"{100.0 * matrix[i, j]:+.1f} pp" if percentage else f"{matrix[i, j]:+.1f}"
                ax.text(j, i, label, ha="center", va="center", fontsize=8)
    fig.suptitle(f"{title}\ncell = new Attacker minus legacy Attacker on paired seeds", fontsize=12)
    if image is not None:
        fig.colorbar(image, ax=axes[:, 0].tolist(), fraction=0.025, pad=0.02)
    fig.subplots_adjust(left=0.22, right=0.91, top=0.90, bottom=0.10, hspace=0.70)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def run_matrix(
    output_dir: Path,
    attackers: Sequence[str],
    defenders: Sequence[str],
    seeds: Sequence[int],
    safety_mode: str = "raw",
    radar_safety_params: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    safety_mode = str(safety_mode).strip().lower()
    if safety_mode not in {"raw", "radar_steer"}:
        raise ValueError(f"Unknown Defender safety mode: {safety_mode!r}")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    audit = defender_audit(defenders)
    missing = [
        f"{row['name']}:{key}"
        for row in audit
        for key, value in row["checkpoints"].items()
        if value is not None and not value["exists"]
    ]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")
    config = {
        "git_head": _git_head(),
        "attackers": list(attackers),
        "defenders": list(defenders),
        "seeds": list(seeds),
        "speed_contract": {"attacker": 2.0, "defender": 2.6},
        "outcome_contract": {
            "target_success": "reason == attacker_caught_target only",
            "defender_success": "defender_capture or timeout; defender_collision excluded",
        },
        "safety_contract": {
            "mode": str(safety_mode),
            "env_obstacle_mask": False,
            "controller_obstacle_mask": False,
            "radar_steering_filter": safety_mode == "radar_steer",
            "radar_safety_params": dict(radar_safety_params or {}),
            "speed_modified": False,
        },
        "defender_audit": audit,
        "learned_attacker_audit": {
            name: {
                **spec,
                "path": str(_checkpoint_path(spec["checkpoint"])),
                "exists": bool(_checkpoint_path(spec["checkpoint"]).is_file()),
                "sha256": (
                    _sha256(_checkpoint_path(spec["checkpoint"]))
                    if _checkpoint_path(spec["checkpoint"]).is_file()
                    else None
                ),
            }
            for name, spec in LEARNED_ATTACKER_SPECS.items()
            if name in attackers
        },
    }
    _write_json(output_dir / "run_config.json", config)

    attacker_policies = {
        attacker_name: create_attacker(attacker_name, int(seeds[0]))
        for attacker_name in attackers
    }
    records = []
    for defender_name in defenders:
        spec = DEFENDER_VARIANTS[defender_name]
        env = AttackerEnv(
            defender_strategy=spec.env_strategy,
            defender_strategy_params=_controller_config(spec),
            env_kwargs={
                "defender_radar_safety": safety_mode == "radar_steer",
                "defender_radar_safety_params": dict(radar_safety_params or {}),
            },
        )
        try:
            for attacker_name in attackers:
                group = []
                for seed in seeds:
                    record = run_episode(
                        env,
                        defender_name,
                        attacker_name,
                        int(seed),
                        policy=attacker_policies[attacker_name],
                    )
                    records.append(record)
                    group.append(record)
                group_summary = summarize(group)
                print(
                    f"D={defender_name:14s} A={attacker_name:26s} "
                    f"target={group_summary['target_success_rate']:.1%} "
                    f"capture={group_summary['defender_capture_rate']:.1%} "
                    f"timeout={group_summary['timeout_rate']:.1%}",
                    flush=True,
                )
        finally:
            env.close()

    _write_jsonl(output_dir / "episodes.jsonl", records)
    summary_rows = []
    for defender_name in defenders:
        for attacker_name in attackers:
            selected = [
                row for row in records
                if row["defender"] == defender_name and row["attacker"] == attacker_name
            ]
            summary_rows.append({
                "defender": defender_name,
                "defender_role": DEFENDER_VARIANTS[defender_name].role,
                "attacker": attacker_name,
                **summarize(selected),
            })
    _write_csv(output_dir / "summary.csv", summary_rows)

    overall_rows = []
    for defender_name in defenders:
        selected = [row for row in records if row["defender"] == defender_name]
        overall_rows.append({
            "defender": defender_name,
            "scope": "all_attackers",
            **summarize(selected),
        })
        for scope, scope_attackers in (
            ("legacy_attackers", [name for name in attackers if name in LEGACY_ATTACKERS]),
            ("new_attackers", [name for name in attackers if name in NEW_ATTACKERS]),
            (
                "programmatic_pool_attackers",
                [name for name in attackers if name in PROGRAMMATIC_ATTACKER_ALIASES],
            ),
            (
                "rl_pool_attackers",
                [name for name in attackers if name in LEARNED_ATTACKER_SPECS],
            ),
        ):
            scoped = [row for row in selected if row["attacker"] in scope_attackers]
            if scoped:
                overall_rows.append({"defender": defender_name, "scope": scope, **summarize(scoped)})
    _write_csv(output_dir / "defender_overall.csv", overall_rows)

    hrl_comparators = tuple(
        name for name in ("protect_skill", "chase_skill")
        if name in defenders
    )
    paired_rows = (
        paired_comparison(records, attackers, comparators=hrl_comparators)
        if "hrl" in defenders
        else []
    )
    _write_csv(output_dir / "paired_hrl_vs_skills.csv", paired_rows)
    _write_csv(
        output_dir / "hrl_skill_outcomes.csv",
        hrl_skill_outcome_summary(records, attackers),
    )
    _write_csv(output_dir / "hrl_switch_bins.csv", hrl_switch_bin_summary(records))
    _write_csv(output_dir / "paired_discordant_episodes.csv", paired_discordant_episodes(records))
    if set(NEW_ATTACKERS).issubset(attackers) and set(LEGACY_ATTACKERS).issubset(attackers):
        new_vs_legacy_rows = new_vs_legacy_comparison(records, defenders)
        _write_csv(output_dir / "new_vs_legacy_pairwise.csv", new_vs_legacy_rows)
        for metric, title, percentage in (
            ("target_success_delta", "Target breach rate delta", True),
            ("defender_capture_delta", "Defender capture rate delta", True),
            ("timeout_delta", "Timeout rate delta", True),
            ("attacker_collision_episode_delta", "Attacker collision-episode rate delta", True),
            ("episode_length_delta_mean", "Episode length delta", False),
        ):
            _plot_new_vs_legacy_delta(
                output_dir / "plots" / f"new_vs_legacy_{metric}.png",
                new_vs_legacy_rows,
                defenders,
                NEW_ATTACKERS,
                LEGACY_ATTACKERS,
                metric,
                title,
                percentage=percentage,
            )
    _plot_heatmap(
        output_dir / "plots" / "target_success_heatmap.png",
        summary_rows, attackers, defenders, "target_success_rate",
        "Attacker Target success rate (lower is better for Defender)",
    )
    _plot_heatmap(
        output_dir / "plots" / "defender_capture_heatmap.png",
        summary_rows, attackers, defenders, "defender_capture_rate",
        "Defender capture rate",
    )
    if "hrl" in defenders:
        _plot_hrl_skill_mix(output_dir / "plots" / "hrl_skill_mix.png", summary_rows, attackers)
    result = {
        "output_dir": str(output_dir),
        "episodes": len(records),
        "summary_rows": len(summary_rows),
        "paired_rows": len(paired_rows),
    }
    _write_json(output_dir / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="active_defender_pool_attacker_matrix")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "defender_hierarchy_eval")
    parser.add_argument("--attackers", default=",".join(FROZEN_ATTACKER_POOL))
    parser.add_argument("--defenders", default=",".join(POOL_DEFENDERS))
    parser.add_argument("--seed-start", type=int, default=184000)
    parser.add_argument("--num-seeds", type=int, default=48)
    parser.add_argument(
        "--defender-safety-mode",
        choices=("raw", "radar_steer"),
        default="raw",
    )
    parser.add_argument("--radar-safety-margin", type=float, default=4.0)
    parser.add_argument("--radar-safety-immediate-margin", type=float, default=4.0)
    parser.add_argument("--radar-safety-horizon", type=int, default=6)
    parser.add_argument("--radar-safety-turn-samples", type=int, default=33)
    parser.add_argument(
        "--radar-safety-selection",
        choices=("closest", "max_clearance"),
        default="closest",
    )
    args = parser.parse_args()

    attackers = _split_csv(args.attackers)
    defenders = _split_csv(args.defenders)
    unknown_defenders = sorted(set(defenders) - set(DEFENDER_VARIANTS))
    if unknown_defenders:
        raise KeyError(f"Unknown defender variants: {unknown_defenders}")
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive")
    seeds = tuple(range(int(args.seed_start), int(args.seed_start) + int(args.num_seeds)))
    result = run_matrix(
        args.output_root / args.run_id,
        attackers,
        defenders,
        seeds,
        safety_mode=args.defender_safety_mode,
        radar_safety_params={
            "safety_margin": args.radar_safety_margin,
            "immediate_margin": args.radar_safety_immediate_margin,
            "horizon_steps": args.radar_safety_horizon,
            "turn_samples": args.radar_safety_turn_samples,
            "selection_mode": args.radar_safety_selection,
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
