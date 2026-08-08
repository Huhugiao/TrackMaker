"""Real-environment paired-seed experiment runner and artifact writer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from envs.attacker_env import AttackerEnv
from attacker_heuristics.metrics import (
    behavior_signature,
    classify_outcome,
    cluster_by_distance,
    policy_behavior_vectors,
    standardized_behavior_distances,
    summarize_records,
)
from attacker_heuristics.registry import POLICY_REGISTRY, create_policy
from configs import map_config


SEED_SETS = {
    "discovery": tuple(range(181000, 181016)),
    "development": tuple(range(182000, 182064)),
    "holdout": tuple(range(183000, 183064)),
}


@dataclass(frozen=True)
class DefenderSpec:
    name: str
    env_strategy: str
    kind: str
    mechanism: str
    checkpoint: Optional[str] = None


DEFENDER_SPECS: Dict[str, DefenderSpec] = {
    "rule_chase": DefenderSpec("rule_chase", "chase", "rule", "direct A* pursuit"),
    "rule_protect": DefenderSpec("rule_protect", "protect", "rule", "direct A* Target protection"),
    "learned_baseline": DefenderSpec(
        "learned_baseline", "hl_learned_baseline", "learned", "formal MLP CTDE baseline",
        "models/defender_baseline_mlp_ctde_repro_20260526/final_model.pth",
    ),
    "learned_chase_skill": DefenderSpec(
        "learned_chase_skill", "skill_chase", "learned", "formal recurrent chase skill",
        "models/defender_chase_nmn_dual_gru_raw_dense_05-05-19-12/final_model.pth",
    ),
    "learned_protect_skill": DefenderSpec(
        "learned_protect_skill", "skill_protect", "learned", "formal protect2 skill",
        "models/defender_protect2_dense_02-11-17-34/best_model.pth",
    ),
    "learned_hrl": DefenderSpec(
        "learned_hrl", "hrl", "learned", "Chapter 2 learned A* path-risk top with frozen skills",
        "models/hrl_ch2_m1_astar_cached_top_20260606_170036/best_model.pth",
    ),
}


def _policy_spec_aliases(root: Path) -> Dict[str, Dict[str, str]]:
    aliases = {}
    for spec in DEFENDER_SPECS.values():
        if not spec.env_strategy.startswith("hl_"):
            continue
        checkpoint = (root / str(spec.checkpoint)).resolve()
        aliases[spec.env_strategy] = {
            "strategy": "skill_baseline",
            "checkpoint": str(checkpoint),
        }
    return aliases


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON encode {type(value).__name__}")


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]], *, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], *, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    exists = path.exists() and path.stat().st_size > 0
    mode = "a" if append else "w"
    with path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not append or not exists:
            writer.writeheader()
        writer.writerows(rows)


def _rebuild_run_summary(run_root: Path) -> None:
    rows = []
    for path in sorted(run_root.glob("*/summary.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    if rows:
        _write_csv(run_root / "summary.csv", rows)


def source_version(root: Path) -> Dict[str, str]:
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        head = "unknown"
    digest = hashlib.sha256()
    source_paths = sorted((root / "attacker_heuristics").glob("*.py"))
    cli_path = root / "eval" / "run_attacker_heuristic_learning.py"
    if cli_path.exists():
        source_paths.append(cli_path)
    for path in source_paths:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    try:
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip())
    except Exception:
        dirty = True
    return {"git_head": head, "source_sha256": digest.hexdigest(), "worktree_dirty": str(dirty).lower()}


def checkpoint_audit(root: Path, defender_names: Sequence[str]) -> List[Dict[str, object]]:
    rows = []
    for name in defender_names:
        spec = DEFENDER_SPECS[name]
        path = None if spec.checkpoint is None else (root / spec.checkpoint).resolve()
        rows.append({
            **asdict(spec),
            "checkpoint": None if path is None else str(path),
            "checkpoint_exists": True if path is None else path.is_file(),
            "load_status": "pending_real_episode",
        })
    return rows


def _privileged_positions(env: AttackerEnv):
    state = env.get_privileged_state()
    return {
        role: [float(state[role]["center_x"]), float(state[role]["center_y"])]
        for role in ("attacker", "defender", "target")
    }


def _clean_record(row: Mapping[str, object]) -> Dict[str, object]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def _failure_reason(summary: Mapping[str, object]) -> str:
    if float(summary.get("target_success_rate", 0.0)) <= 0.0:
        if float(summary.get("timeout_rate", 0.0)) >= 0.5:
            return "zero_target_success_and_timeout_dominant"
        if float(summary.get("defender_capture_rate", 0.0)) >= 0.5:
            return "zero_target_success_and_capture_dominant"
        return "zero_target_success"
    if float(summary.get("timeout_rate", 0.0)) >= 0.5:
        return "survivability_probe_timeout_dominant"
    if float(summary.get("attacker_collision_episode_rate", 0.0)) >= 0.5:
        return "attacker_collision_dominant"
    return ""


def _trial_conclusion(summary: Mapping[str, object], baseline: Optional[Mapping[str, object]]):
    success = float(summary.get("target_success_rate", 0.0))
    timeout = float(summary.get("timeout_rate", 0.0))
    if baseline is None:
        return "measured", None
    delta = success - float(baseline.get("target_success_rate", 0.0))
    if timeout >= 0.5 and success < 0.1:
        conclusion = "survivability_probe"
    elif delta > 0.0:
        conclusion = "promising_on_current_budget"
    elif delta == 0.0 and success > 0.0:
        conclusion = "tie_requires_more_evidence"
    else:
        conclusion = "no_pareto_improvement"
    return conclusion, delta


class ExperimentRunner:
    def __init__(
        self,
        *,
        root: Path,
        output_dir: Path,
        run_id: str,
        stage: str,
        phase: str,
        policies: Sequence[str],
        defenders: Sequence[str],
        seeds: Sequence[int],
        evidence: str = "",
        mechanism_change: str = "",
        expected_metric: str = "",
        regression_risk: str = "",
        validation_cases: str = "",
        trace_limit_per_policy: int = 3,
    ):
        self.root = root.resolve()
        self.output_dir = output_dir.resolve()
        self.run_id = str(run_id)
        self.stage = str(stage)
        self.phase = str(phase)
        self.policies = tuple(policies)
        self.defenders = tuple(defenders)
        self.seeds = tuple(int(seed) for seed in seeds)
        self.evidence = str(evidence)
        self.mechanism_change = str(mechanism_change)
        self.expected_metric = str(expected_metric)
        self.regression_risk = str(regression_risk)
        self.validation_cases = str(validation_cases)
        self.trace_limit_per_policy = max(0, int(trace_limit_per_policy))
        self.version = source_version(self.root)

        unknown_policies = sorted(set(self.policies) - set(POLICY_REGISTRY))
        unknown_defenders = sorted(set(self.defenders) - set(DEFENDER_SPECS))
        if unknown_policies:
            raise KeyError(f"Unknown policies: {unknown_policies}")
        if unknown_defenders:
            raise KeyError(f"Unknown defenders: {unknown_defenders}")
        if not self.seeds:
            raise ValueError("At least one fixed seed is required")
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(f"Stage output already exists and is non-empty: {self.output_dir}")

    def _make_env(self, defender_name: str) -> AttackerEnv:
        spec = DEFENDER_SPECS[defender_name]
        return AttackerEnv(
            defender_strategy=spec.env_strategy,
            defender_policy_specs=_policy_spec_aliases(self.root),
            env_kwargs={"hard_action_mask": False},
        )

    def _episode(self, env: AttackerEnv, policy_name: str, defender_name: str, seed: int):
        obs, reset_info = env.reset(seed=int(seed))
        policy = create_policy(policy_name, seed=int(seed))
        policy.reset(seed=int(seed))
        initial = _privileged_positions(env)
        initial_target_distance = float(np.linalg.norm(
            np.asarray(initial["target"]) - np.asarray(initial["attacker"])
        ))
        trace = []
        done = False
        info = {}
        terminated = False
        truncated = False
        collision_events = 0
        while not done:
            action, diagnostics = policy.get_action_with_info(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            collision_events = max(
                collision_events,
                int(info.get("attacker_collision_event_count", 0)),
            )
            positions = _privileged_positions(env)
            trace.append({
                "step": int(env.step_count),
                **positions,
                "action": [float(action[0]), float(action[1])],
                "reward": float(reward),
                "diagnostics": diagnostics,
                "attacker_collision": bool(info.get("attacker_collision", False)),
            })
            done = bool(terminated or truncated)

        reason = str(info.get("reason", "unknown"))
        outcome = classify_outcome(reason, terminated=terminated, truncated=truncated)
        behavior = behavior_signature(trace, initial_target_distance)
        record = {
            "run_id": self.run_id,
            "stage": self.stage,
            "phase": self.phase,
            "trial_id": f"{self.stage}:{policy_name}:{defender_name}",
            "policy": policy_name,
            "defender": defender_name,
            "defender_kind": DEFENDER_SPECS[defender_name].kind,
            "seed": int(seed),
            "reason": reason,
            "outcome": outcome,
            "target_success": outcome == "target_success",
            "defender_capture": outcome == "defender_capture",
            "timeout": outcome == "timeout",
            "defender_collision": outcome == "defender_collision",
            "episode_length": int(env.step_count),
            "attacker_collision_events": int(collision_events),
            "obstacle_density": str(reset_info.get("obstacle_density", map_config.current_obstacle_density)),
            "initial_attacker": initial["attacker"],
            "initial_defender": initial["defender"],
            "initial_target": initial["target"],
            "initial_target_distance": initial_target_distance,
            "obstacles": [dict(item) for item in getattr(map_config, "obstacles", [])],
            "behavior": behavior,
            "final_diagnostics": trace[-1]["diagnostics"] if trace else policy.diagnostics,
            "_trace": trace,
        }
        return record

    def run(self) -> Dict[str, object]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        run_root = self.output_dir.parent
        _write_json(run_root / "seed_sets.json", {key: list(values) for key, values in SEED_SETS.items()})
        audit = checkpoint_audit(self.root, self.defenders)
        config = {
            "run_id": self.run_id,
            "stage": self.stage,
            "phase": self.phase,
            "policies": list(self.policies),
            "defenders": list(self.defenders),
            "seeds": list(self.seeds),
            "speed_contract": {"attacker": 2.0, "defender": 2.6},
            "episode_limit": 449,
            "outcome_contract": {
                "target_success": "reason == attacker_caught_target only",
                "defender_collision": "draw; never target success",
            },
            "change_record": {
                "evidence": self.evidence,
                "mechanism_change": self.mechanism_change,
                "expected_metric": self.expected_metric,
                "regression_risk": self.regression_risk,
                "validation_cases": self.validation_cases,
            },
            "code_version": self.version,
            "defender_audit": audit,
        }
        _write_json(self.output_dir / "run_config.json", config)

        records = []
        for defender_name in self.defenders:
            env = self._make_env(defender_name)
            try:
                for policy_name in self.policies:
                    for seed in self.seeds:
                        record = self._episode(env, policy_name, defender_name, seed)
                        records.append(record)
                        print(
                            f"[{self.stage}] A={policy_name} D={defender_name} seed={seed} "
                            f"outcome={record['outcome']} len={record['episode_length']}",
                            flush=True,
                        )
            finally:
                env.close()

        clean_records = [_clean_record(row) for row in records]
        for item in audit:
            completed = sum(1 for row in clean_records if row["defender"] == item["name"])
            item["load_status"] = "real_episode_completed" if completed else "not_evaluated"
            item["completed_episodes"] = int(completed)
        config["defender_audit"] = audit
        _write_json(self.output_dir / "run_config.json", config)
        _write_jsonl(self.output_dir / "episodes.jsonl", clean_records)
        paired_rows = [
            {
                "stage": self.stage,
                "seed": row["seed"],
                "policy": row["policy"],
                "defender": row["defender"],
                "outcome": row["outcome"],
                "episode_length": row["episode_length"],
            }
            for row in clean_records
        ]
        _write_csv(self.output_dir / "paired_outcomes.csv", paired_rows)

        summary_rows = []
        for policy_name in self.policies:
            for defender_name in self.defenders:
                selected = [
                    row for row in clean_records
                    if row["policy"] == policy_name and row["defender"] == defender_name
                ]
                summary = summarize_records(selected)
                summary_rows.append({
                    "run_id": self.run_id,
                    "stage": self.stage,
                    "phase": self.phase,
                    "policy": policy_name,
                    "family": POLICY_REGISTRY[policy_name].family,
                    "defender": defender_name,
                    "defender_kind": DEFENDER_SPECS[defender_name].kind,
                    **summary,
                })
        _write_csv(self.output_dir / "summary.csv", summary_rows)
        _rebuild_run_summary(run_root)

        baseline_by_defender = {
            row["defender"]: row for row in summary_rows if row["policy"] == "default"
        }
        trial_rows = []
        for summary in summary_rows:
            spec = POLICY_REGISTRY[str(summary["policy"])]
            conclusion, baseline_delta = _trial_conclusion(
                summary, baseline_by_defender.get(str(summary["defender"]))
            )
            trial_rows.append({
                "trial_id": f"{self.stage}:{summary['policy']}:{summary['defender']}",
                "run_id": self.run_id,
                "stage": self.stage,
                "phase": self.phase,
                "parent_policy": spec.parent,
                "policy": spec.name,
                "family": spec.family,
                "generation": spec.generation,
                "mechanism_hypothesis": spec.hypothesis,
                "code_version": self.version,
                "config": {"seeds": list(self.seeds), "defender": summary["defender"]},
                "seed": list(self.seeds),
                "defender": summary["defender"],
                "metrics": {
                    key: value for key, value in summary.items()
                    if key.endswith("_rate") or key.endswith("_mean") or "ci_" in key
                },
                "baseline_target_success_delta": baseline_delta,
                "conclusion": conclusion,
                "failure_reason": _failure_reason(summary),
                "change_record": config["change_record"],
            })
        _write_jsonl(self.output_dir / "trials.jsonl", trial_rows)
        _write_jsonl(run_root / "trials.jsonl", trial_rows, append=True)

        names, matrix, features = policy_behavior_vectors(clean_records)
        distances = standardized_behavior_distances(names, matrix)
        clusters = cluster_by_distance(names, distances)
        behavior_rows = []
        for row_index, name in enumerate(names):
            behavior_rows.append({
                "policy": name,
                "behavior_cluster": clusters[name],
                **{feature: float(matrix[row_index, col]) for col, feature in enumerate(features)},
            })
        distance_rows = [
            {"policy_a": left, "policy_b": right, "standardized_distance": distance}
            for (left, right), distance in sorted(distances.items())
        ]
        _write_csv(self.output_dir / "behavior_summary.csv", behavior_rows)
        _write_csv(self.output_dir / "behavior_distances.csv", distance_rows)
        _write_json(self.output_dir / "behavior_clusters.json", {
            "threshold": 0.70,
            "assignments": clusters,
            "feature_names": features,
        })

        selected_traces = self._save_representative_traces(records)
        regression = self._regression_seeds(clean_records)
        _write_json(self.output_dir / "regression_seeds.json", regression)

        # A compact latest-stage index is intentionally overwritten within the
        # isolated run directory; formal repository results remain untouched.
        _write_json(run_root / "latest_stage.json", {
            "stage": self.stage,
            "phase": self.phase,
            "path": str(self.output_dir),
            "summary": str(self.output_dir / "summary.csv"),
            "trials": str(self.output_dir / "trials.jsonl"),
        })
        return {
            "records": len(clean_records),
            "summary_rows": len(summary_rows),
            "traces": selected_traces,
            "output_dir": str(self.output_dir),
            "clusters": clusters,
        }

    def _save_representative_traces(self, records: Sequence[Mapping[str, object]]) -> int:
        if self.trace_limit_per_policy <= 0:
            return 0
        selected = []
        for policy_name in self.policies:
            policy_rows = [row for row in records if row["policy"] == policy_name]
            candidates = []
            for outcome in ("target_success", "defender_capture", "timeout", "defender_collision"):
                matching = [row for row in policy_rows if row["outcome"] == outcome]
                if matching:
                    candidates.append(matching[0])
            if policy_rows:
                divergent = max(policy_rows, key=lambda row: float(row["behavior"].get("detour_ratio", 0.0)))
                candidates.append(divergent)
            seen = set()
            for row in candidates:
                key = (row["defender"], row["seed"])
                if key in seen:
                    continue
                seen.add(key)
                selected.append(row)
                if len(seen) >= self.trace_limit_per_policy:
                    break

        for row in selected:
            stem = f"{row['policy']}__{row['defender']}__seed{row['seed']}__{row['outcome']}"
            trace_path = self.output_dir / "traces" / f"{stem}.json"
            _write_json(trace_path, {
                "episode": _clean_record(row),
                "trace": row["_trace"],
            })
            self._plot_trace(row, self.output_dir / "plots" / f"{stem}.png")
        return len(selected)

    def _plot_trace(self, row: Mapping[str, object], path: Path) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        trace = row["_trace"]
        attacker = np.asarray([item["attacker"] for item in trace], dtype=np.float64)
        defender = np.asarray([item["defender"] for item in trace], dtype=np.float64)
        target = np.asarray(trace[-1]["target"], dtype=np.float64)
        fig, ax = plt.subplots(figsize=(5.4, 5.4), dpi=130)
        from matplotlib.patches import Circle, Rectangle
        for obstacle in row.get("obstacles", []):
            kind = obstacle.get("type")
            if kind == "rect":
                ax.add_patch(Rectangle(
                    (obstacle["x"], obstacle["y"]), obstacle["w"], obstacle["h"],
                    facecolor="#5b5b66", edgecolor="none", alpha=0.42,
                ))
            elif kind == "circle":
                ax.add_patch(Circle(
                    (obstacle["cx"], obstacle["cy"]), obstacle["r"],
                    facecolor="#5b5b66", edgecolor="none", alpha=0.42,
                ))
            elif kind == "segment":
                ax.plot(
                    [obstacle["x1"], obstacle["x2"]], [obstacle["y1"], obstacle["y2"]],
                    color="#5b5b66", lw=max(1.0, float(obstacle.get("thick", 4.0))),
                    alpha=0.42, solid_capstyle="round",
                )
        ax.plot(attacker[:, 0], attacker[:, 1], color="#d94841", lw=1.8, label="Attacker")
        ax.plot(defender[:, 0], defender[:, 1], color="#3274b5", lw=1.4, label="Defender")
        ax.scatter(*target, marker="*", s=110, color="#e5a100", label="Target", zorder=5)
        ax.scatter(*attacker[0], s=30, color="#7f1d1d")
        ax.scatter(*defender[0], s=30, color="#173c68")
        ax.set(xlim=(0, map_config.width), ylim=(0, map_config.height), aspect="equal")
        ax.set_title(f"{row['policy']} vs {row['defender']} | seed={row['seed']} | {row['outcome']}")
        ax.grid(alpha=0.18)
        ax.legend(loc="upper right", fontsize=8)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def _regression_seeds(self, records: Sequence[Mapping[str, object]]):
        output = {"stage": self.stage, "cases": []}
        for policy_name in self.policies:
            selected = [row for row in records if row["policy"] == policy_name]
            for label in ("target_success", "defender_capture", "timeout", "defender_collision"):
                candidates = [row for row in selected if row["outcome"] == label]
                if not candidates:
                    continue
                row = candidates[0]
                output["cases"].append({
                    "policy": policy_name,
                    "defender": row["defender"],
                    "seed": row["seed"],
                    "expected_outcome": label,
                    "purpose": f"preserve representative {label} behavior",
                })
        return output
