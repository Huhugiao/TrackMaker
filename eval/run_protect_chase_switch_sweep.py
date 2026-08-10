#!/usr/bin/env python3
"""Evaluate same-snapshot complement or direct Protect-to-Chase switching."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
import sys
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attacker.defender_pool import (  # noqa: E402
    BaseDefenderController,
    _load_policy_model,
    _predict_skill_action,
)
from attacker_heuristics.metrics import classify_outcome  # noqa: E402
from configs import map_config  # noqa: E402
from envs.tad_env import TADEnv  # noqa: E402
from eval.run_defender_hierarchy_attacker_matrix import create_attacker  # noqa: E402


DEFAULT_PROTECT = (
    PROJECT_ROOT
    / "models/defender_protect_mlp_ctde_frozen6_20260721_105148/best_balanced_model.pth"
)
DEFAULT_CHASE = (
    PROJECT_ROOT
    / "models/defender_chase_nmn_dual_gru_raw_dense_05-05-19-12/final_model.pth"
)


def _split_csv(value: str) -> Tuple[str, ...]:
    return tuple(token.strip() for token in str(value).split(",") if token.strip())


def _parse_int_csv(value: str) -> Tuple[int, ...]:
    return tuple(int(token) for token in _split_csv(value))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _jsonl_dump(path: Path, rows: Iterable[Mapping]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _clone_value(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    return copy.deepcopy(value)


class SkillBank:
    """Two frozen skills with explicit recurrent-state snapshots."""

    _STATE_FIELDS = ("_actor_hidden", "_critic_hidden", "_prev_actor_obs_for_context")

    def __init__(self, protect_path: Path, chase_path: Path):
        self.models = {
            "protect": _load_policy_model(str(protect_path), torch.device("cpu"), "protect")[0],
            "chase": _load_policy_model(str(chase_path), torch.device("cpu"), "chase")[0],
        }

    def reset(self) -> None:
        for model in self.models.values():
            model.reset_gru_sequence()

    def snapshot(self) -> Dict[str, Dict[str, object]]:
        return {
            name: {field: _clone_value(getattr(model, field, None)) for field in self._STATE_FIELDS}
            for name, model in self.models.items()
        }

    def restore(self, state: Mapping[str, Mapping[str, object]]) -> None:
        for name, model in self.models.items():
            for field in self._STATE_FIELDS:
                setattr(model, field, _clone_value(state[name].get(field)))

    def reset_chase_hidden(self) -> None:
        self.models["chase"].reset_gru_sequence()

    def raw_actions(self, defender_obs: np.ndarray, attacker_obs: np.ndarray):
        return {
            name: _predict_skill_action(model, defender_obs, attacker_obs)
            for name, model in self.models.items()
        }

    @staticmethod
    def masked_action(action: np.ndarray, env: TADEnv) -> np.ndarray:
        return BaseDefenderController._apply_defender_hard_obstacle_mask(action, env)


def _centers(env: TADEnv) -> Dict[str, np.ndarray]:
    state = env.get_privileged_state()
    return {
        name: np.asarray([state[name]["center_x"], state[name]["center_y"]], dtype=np.float64)
        for name in ("attacker", "defender", "target")
    }


def _state_features(
    env: TADEnv,
    bank: SkillBank,
    target_distance_history: Sequence[float],
) -> Dict[str, object]:
    points = _centers(env)
    attacker = points["attacker"]
    defender = points["defender"]
    target = points["target"]
    d_at = float(np.linalg.norm(attacker - target))
    d_dt = float(np.linalg.norm(defender - target))
    d_da = float(np.linalg.norm(defender - attacker))
    reach_radius = float(getattr(map_config, "target_radius", 16.0) + getattr(map_config, "agent_radius", 8.0))
    capture_radius = float(getattr(map_config, "capture_radius", 20.0))
    attacker_time = max(0.0, d_at - reach_radius) / float(map_config.attacker_speed)
    defender_guard_time = max(0.0, d_dt - reach_radius) / float(map_config.defender_speed)
    defender_capture_time = max(0.0, d_da - capture_radius) / float(map_config.defender_speed)

    axis = target - attacker
    axis_sq = max(float(np.dot(axis, axis)), 1e-9)
    projection = float(np.dot(defender - attacker, axis) / axis_sq)
    nearest = attacker + np.clip(projection, 0.0, 1.0) * axis
    corridor_distance = float(np.linalg.norm(defender - nearest))
    history = list(target_distance_history)
    progress_16 = float(history[-17] - history[-1]) if len(history) >= 17 else 0.0

    model_state = bank.snapshot()
    defender_obs, attacker_obs = env.current_obs
    raw = bank.raw_actions(defender_obs, attacker_obs)
    masked = {name: bank.masked_action(action, env) for name, action in raw.items()}
    bank.restore(model_state)
    return {
        "step": int(env.step_count),
        "steps_remaining": int(map_config.EnvParameters.EPISODE_LEN - env.step_count),
        "attacker_target_distance": d_at,
        "defender_target_distance": d_dt,
        "defender_attacker_distance": d_da,
        "target_time_margin": float(attacker_time - defender_guard_time),
        "defender_capture_time": defender_capture_time,
        "corridor_projection": projection,
        "corridor_distance": corridor_distance,
        "attacker_target_progress_16": progress_16,
        "attacker_speed": float(env.attacker.get("v", 0.0)),
        "defender_speed": float(env.defender.get("v", 0.0)),
        "raw_action_disagreement": float(np.linalg.norm(raw["protect"] - raw["chase"])),
        "masked_action_disagreement": float(np.linalg.norm(masked["protect"] - masked["chase"])),
    }


def _snapshot(env: TADEnv, bank: SkillBank, attacker) -> Dict[str, object]:
    return {
        "env": env.snapshot_state(),
        "bank": bank.snapshot(),
        "attacker": copy.deepcopy(attacker),
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_random": torch.get_rng_state().clone(),
    }


def _restore(env: TADEnv, bank: SkillBank, state: Mapping[str, object]):
    env.restore_state(state["env"])
    bank.restore(state["bank"])
    random.setstate(state["python_random"])
    np.random.set_state(state["numpy_random"])
    torch.set_rng_state(state["torch_random"])
    return copy.deepcopy(state["attacker"])


def _step(env: TADEnv, bank: SkillBank, attacker, selected_skill: str):
    defender_obs, attacker_obs = env.current_obs
    actions = bank.raw_actions(defender_obs, attacker_obs)
    defender_action = bank.masked_action(actions[selected_skill], env)
    attacker_action = attacker.get_action(attacker_obs)
    return env.step(action=defender_action, attacker_action=attacker_action)


def _rollout_branch(
    env: TADEnv,
    bank: SkillBank,
    attacker,
    chase_steps: Optional[int],
    hidden_mode: str,
) -> Dict[str, object]:
    if hidden_mode == "reset":
        bank.reset_chase_hidden()
    start_step = int(env.step_count)
    terminated = truncated = False
    info: Mapping[str, object] = {}
    chase_used = 0
    while not (terminated or truncated):
        use_chase = chase_steps is None or chase_used < chase_steps
        selected = "chase" if use_chase else "protect"
        _obs, _reward, terminated, truncated, info = _step(env, bank, attacker, selected)
        chase_used += int(use_chase)
        if int(env.step_count) > int(map_config.EnvParameters.EPISODE_LEN) + 2:
            raise RuntimeError("rollout exceeded episode limit without terminal state")
    reason = str(info.get("reason", "unknown"))
    return {
        "reason": reason,
        "outcome": classify_outcome(reason, terminated=terminated, truncated=truncated),
        "branch_steps": int(env.step_count - start_step),
        "chase_steps_used": int(chase_used),
    }


def _run_case(
    env: TADEnv,
    bank: SkillBank,
    attacker_name: str,
    seed: int,
    snapshot_interval: int,
    snapshot_start: int,
    burst_steps: Sequence[int],
    hidden_modes: Sequence[str],
):
    env.reset(seed=int(seed))
    attacker = create_attacker(attacker_name, int(seed))
    attacker.reset(seed=int(seed))
    bank.reset()
    target_history = [float(np.linalg.norm(_centers(env)["attacker"] - _centers(env)["target"]))]
    rows = []
    terminated = truncated = False
    final_info: Mapping[str, object] = {}
    snapshot_count = 0

    while not (terminated or truncated):
        step = int(env.step_count)
        should_snapshot = step >= snapshot_start and (step - snapshot_start) % snapshot_interval == 0
        if should_snapshot:
            snapshot_count += 1
            state = _snapshot(env, bank, attacker)
            features = _state_features(
                env,
                bank,
                target_history,
            )
            for hidden_mode in hidden_modes:
                for burst in burst_steps:
                    branch_attacker = _restore(env, bank, state)
                    result = _rollout_branch(
                        env,
                        bank,
                        branch_attacker,
                        chase_steps=None if burst == 0 else int(burst),
                        hidden_mode=hidden_mode,
                    )
                    rows.append(
                        {
                            "attacker": attacker_name,
                            "seed": int(seed),
                            "hidden_mode": hidden_mode,
                            "branch": "chase_forever" if burst == 0 else f"chase_{burst}_then_protect",
                            "requested_chase_steps": None if burst == 0 else int(burst),
                            **features,
                            **result,
                        }
                    )
            attacker = _restore(env, bank, state)

        _obs, _reward, terminated, truncated, final_info = _step(env, bank, attacker, "protect")
        points = _centers(env)
        target_history.append(float(np.linalg.norm(points["attacker"] - points["target"])))

    reason = str(final_info.get("reason", "unknown"))
    episode = {
        "attacker": attacker_name,
        "seed": int(seed),
        "reason": reason,
        "outcome": classify_outcome(reason, terminated=terminated, truncated=truncated),
        "episode_length": int(env.step_count),
        "snapshots": int(snapshot_count),
    }
    for row in rows:
        row["protect_reason"] = episode["reason"]
        row["protect_outcome"] = episode["outcome"]
    return episode, rows


def _gate_accepts(
    features: Mapping[str, object],
    min_target_time_margin: float,
    min_defender_target_distance: float,
    max_attacker_target_progress_16: Optional[float],
) -> bool:
    return bool(
        float(features["target_time_margin"]) >= float(min_target_time_margin)
        and float(features["defender_target_distance"]) >= float(min_defender_target_distance)
        and (
            max_attacker_target_progress_16 is None
            or float(features["attacker_target_progress_16"])
            <= float(max_attacker_target_progress_16)
        )
    )


def _reset_chase_protect_action_disagreement(env: TADEnv, bank: SkillBank) -> float:
    state = bank.snapshot()
    defender_obs, attacker_obs = env.current_obs
    raw = bank.raw_actions(defender_obs, attacker_obs)
    protect_action = bank.masked_action(raw["protect"], env)
    bank.restore(state)
    bank.reset_chase_hidden()
    raw = bank.raw_actions(defender_obs, attacker_obs)
    chase_action = bank.masked_action(raw["chase"], env)
    bank.restore(state)
    return float(np.linalg.norm(chase_action - protect_action))


def _run_direct_episode(
    env: TADEnv,
    bank: SkillBank,
    attacker_name: str,
    seed: int,
    gate: Optional[Mapping[str, float]] = None,
):
    env.reset(seed=int(seed))
    attacker = create_attacker(attacker_name, int(seed))
    attacker.reset(seed=int(seed))
    bank.reset()
    target_history = [float(np.linalg.norm(_centers(env)["attacker"] - _centers(env)["target"]))]
    switched = False
    switch_features = None
    chase_gate_trigger_count = 0
    chase_veto_count = 0
    chase_veto_step = None
    chase_authorization_blocked = False
    chase_protect_action_disagreement_max = None
    terminated = truncated = False
    info: Mapping[str, object] = {}

    while not (terminated or truncated):
        step = int(env.step_count)
        if gate is not None and not switched and not chase_authorization_blocked:
            start = int(gate["start"])
            end = int(gate["end"])
            interval = int(gate["interval"])
            if start <= step <= end and (step - start) % interval == 0:
                features = _state_features(
                    env,
                    bank,
                    target_history,
                )
                chase_gate_accepted = _gate_accepts(
                    features,
                    min_target_time_margin=float(gate["min_target_time_margin"]),
                    min_defender_target_distance=float(gate["min_defender_target_distance"]),
                    max_attacker_target_progress_16=gate.get("max_attacker_target_progress_16"),
                )
                if chase_gate_accepted:
                    chase_gate_trigger_count += 1
                    max_disagreement = gate.get("max_protect_action_disagreement")
                    disagreement = None
                    if max_disagreement is not None:
                        disagreement = _reset_chase_protect_action_disagreement(env, bank)
                        chase_protect_action_disagreement_max = max(
                            disagreement,
                            float(chase_protect_action_disagreement_max or 0.0),
                        )
                    if max_disagreement is None or disagreement <= float(max_disagreement):
                        bank.reset_chase_hidden()
                        switched = True
                        switch_features = features
                    else:
                        chase_veto_count += 1
                        chase_veto_step = int(step)
                        chase_authorization_blocked = True

        selected_skill = "chase" if switched else "protect"
        _obs, _reward, terminated, truncated, info = _step(
            env,
            bank,
            attacker,
            selected_skill,
        )
        points = _centers(env)
        target_history.append(float(np.linalg.norm(points["attacker"] - points["target"])))

    reason = str(info.get("reason", "unknown"))
    return {
        "attacker": attacker_name,
        "seed": int(seed),
        "reason": reason,
        "outcome": classify_outcome(reason, terminated=terminated, truncated=truncated),
        "episode_length": int(env.step_count),
        "switched": switched,
        "switch_step": None if switch_features is None else int(switch_features["step"]),
        "switch_features": switch_features,
        "chase_gate_trigger_count": int(chase_gate_trigger_count),
        "chase_veto_count": int(chase_veto_count),
        "chase_veto_step": chase_veto_step,
        "chase_authorization_blocked": chase_authorization_blocked,
        "chase_protect_action_disagreement_max": chase_protect_action_disagreement_max,
    }


def _direct_record(protect: Mapping, candidate: Mapping) -> Dict[str, object]:
    features = candidate.get("switch_features") or {}
    return {
        "attacker": str(protect["attacker"]),
        "seed": int(protect["seed"]),
        "protect_reason": str(protect["reason"]),
        "protect_outcome": str(protect["outcome"]),
        "candidate_reason": str(candidate["reason"]),
        "candidate_outcome": str(candidate["outcome"]),
        "switched": bool(candidate["switched"]),
        "switch_step": candidate.get("switch_step"),
        "target_time_margin": features.get("target_time_margin"),
        "defender_target_distance": features.get("defender_target_distance"),
        "attacker_target_progress_16": features.get("attacker_target_progress_16"),
        "chase_gate_trigger_count": int(candidate.get("chase_gate_trigger_count", 0)),
        "chase_veto_count": int(candidate.get("chase_veto_count", 0)),
        "chase_veto_step": candidate.get("chase_veto_step"),
        "chase_authorization_blocked": bool(
            candidate.get("chase_authorization_blocked", False)
        ),
        "chase_protect_action_disagreement_max": candidate.get(
            "chase_protect_action_disagreement_max"
        ),
    }


def _summary(episodes: Sequence[Mapping], branches: Sequence[Mapping]) -> Dict[str, object]:
    episode_counts = Counter(str(row["outcome"]) for row in episodes)
    grouped: Dict[Tuple[str, str], Counter] = {}
    for row in branches:
        key = (str(row["hidden_mode"]), str(row["branch"]))
        grouped.setdefault(key, Counter())[str(row["outcome"])] += 1
    return {
        "protect_episodes": {"total": len(episodes), "outcomes": dict(episode_counts)},
        "branch_states": len(branches),
        "branch_outcomes": {
            f"{hidden}/{branch}": {"total": sum(counts.values()), "outcomes": dict(counts)}
            for (hidden, branch), counts in sorted(grouped.items())
        },
    }


def _gated_records(
    episodes: Sequence[Mapping],
    branches: Sequence[Mapping],
    switch_step: int,
    min_target_time_margin: float,
    min_defender_target_distance: float,
    max_attacker_target_progress_16: Optional[float] = None,
):
    branch_by_case = {
        (str(row["attacker"]), int(row["seed"])): row
        for row in branches
        if int(row["step"]) == int(switch_step)
        and row["hidden_mode"] == "reset"
        and row["branch"] == "chase_forever"
    }
    records = []
    for episode in episodes:
        key = (str(episode["attacker"]), int(episode["seed"]))
        branch = branch_by_case.get(key)
        switched = bool(
            branch is not None
            and float(branch["target_time_margin"]) >= float(min_target_time_margin)
            and float(branch["defender_target_distance"]) >= float(min_defender_target_distance)
            and (
                max_attacker_target_progress_16 is None
                or float(branch["attacker_target_progress_16"])
                <= float(max_attacker_target_progress_16)
            )
        )
        records.append(
            {
                "attacker": key[0],
                "seed": key[1],
                "protect_reason": str(episode["reason"]),
                "protect_outcome": str(episode["outcome"]),
                "candidate_reason": str(branch["reason"] if switched else episode["reason"]),
                "candidate_outcome": str(branch["outcome"] if switched else episode["outcome"]),
                "switched": switched,
                "switch_step": int(switch_step) if switched else None,
                "target_time_margin": None if branch is None else float(branch["target_time_margin"]),
                "defender_target_distance": (
                    None if branch is None else float(branch["defender_target_distance"])
                ),
                "attacker_target_progress_16": (
                    None if branch is None else float(branch["attacker_target_progress_16"])
                ),
            }
        )
    return records


def _gated_summary(records: Sequence[Mapping]) -> Dict[str, object]:
    protect_counts = Counter(str(row["protect_outcome"]) for row in records)
    candidate_counts = Counter(str(row["candidate_outcome"]) for row in records)
    transitions = Counter(
        (str(row["protect_outcome"]), str(row["candidate_outcome"])) for row in records
    )
    per_attacker = {}
    for attacker in sorted({str(row["attacker"]) for row in records}):
        selected = [row for row in records if row["attacker"] == attacker]
        per_attacker[attacker] = {
            "episodes": len(selected),
            "switched": sum(bool(row["switched"]) for row in selected),
            "chase_vetoed": sum(
                int(row.get("chase_veto_count", 0)) > 0 for row in selected
            ),
            "protect_outcomes": dict(Counter(str(row["protect_outcome"]) for row in selected)),
            "candidate_outcomes": dict(Counter(str(row["candidate_outcome"]) for row in selected)),
        }
    return {
        "episodes": len(records),
        "switched": sum(bool(row["switched"]) for row in records),
        "chase_vetoed": sum(
            int(row.get("chase_veto_count", 0)) > 0 for row in records
        ),
        "protect_outcomes": dict(protect_counts),
        "candidate_outcomes": dict(candidate_counts),
        "capture_gain_count": int(
            candidate_counts["defender_capture"] - protect_counts["defender_capture"]
        ),
        "capture_gain_rate": float(
            (candidate_counts["defender_capture"] - protect_counts["defender_capture"])
            / max(1, len(records))
        ),
        "breach_delta_count": int(candidate_counts["target_success"] - protect_counts["target_success"]),
        "new_breach_count": int(
            sum(
                row["protect_outcome"] != "target_success"
                and row["candidate_outcome"] == "target_success"
                for row in records
            )
        ),
        "breach_rescued_count": int(
            sum(
                row["protect_outcome"] == "target_success"
                and row["candidate_outcome"] != "target_success"
                for row in records
            )
        ),
        "timeout_to_capture_count": int(transitions[("timeout", "defender_capture")]),
        "transitions": {
            f"{source}->{target}": int(count)
            for (source, target), count in sorted(transitions.items())
        },
        "per_attacker": per_attacker,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attackers", required=True, help="Comma-separated attacker names")
    parser.add_argument("--seeds", required=True, help="Comma-separated fixed seeds")
    parser.add_argument("--snapshot-interval", type=int, default=16)
    parser.add_argument("--snapshot-start", type=int, default=0)
    parser.add_argument(
        "--burst-steps",
        default="0,32,64",
        help="0 means permanent Chase; other values switch back to Protect",
    )
    parser.add_argument("--hidden-modes", default="shadow", choices=("shadow", "reset", "shadow,reset"))
    parser.add_argument("--gate-target-time-margin", type=float)
    parser.add_argument("--gate-min-defender-target-distance", type=float)
    parser.add_argument("--gate-max-attacker-target-progress-16", type=float)
    parser.add_argument("--chase-max-protect-action-disagreement", type=float)
    parser.add_argument("--direct-gate", action="store_true")
    parser.add_argument("--gate-decision-end", type=int, default=64)
    parser.add_argument("--protect-checkpoint", type=Path, default=DEFAULT_PROTECT)
    parser.add_argument("--chase-checkpoint", type=Path, default=DEFAULT_CHASE)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.snapshot_interval <= 0:
        parser.error("--snapshot-interval must be positive")
    attackers = _split_csv(args.attackers)
    seeds = _parse_int_csv(args.seeds)
    bursts = _parse_int_csv(args.burst_steps)
    hidden_modes = _split_csv(args.hidden_modes)
    if not attackers or not seeds or not bursts:
        parser.error("attackers, seeds, and burst-steps must be non-empty")
    if any(value < 0 for value in bursts):
        parser.error("burst-steps cannot be negative")
    if (
        args.chase_max_protect_action_disagreement is not None
        and args.chase_max_protect_action_disagreement < 0.0
    ):
        parser.error("--chase-max-protect-action-disagreement cannot be negative")
    gate_values = (args.gate_target_time_margin, args.gate_min_defender_target_distance)
    if args.gate_max_attacker_target_progress_16 is not None and all(
        value is None for value in gate_values
    ):
        parser.error("progress gate requires both base gate thresholds")
    if any(value is not None for value in gate_values):
        if any(value is None for value in gate_values):
            parser.error("both gate thresholds must be provided")
        if hidden_modes != ("reset",) or bursts != (0,):
            parser.error("gated evaluation requires --hidden-modes reset --burst-steps 0")
    if args.direct_gate and args.gate_target_time_margin is None:
        parser.error("--direct-gate requires gate thresholds")
    if args.chase_max_protect_action_disagreement is not None and not args.direct_gate:
        parser.error("Chase/Protect action compatibility is supported only by direct gate")
    if args.direct_gate and args.gate_decision_end < args.snapshot_start:
        parser.error("--gate-decision-end cannot precede --snapshot-start")
    if args.direct_gate and (
        args.gate_decision_end - args.snapshot_start
    ) % args.snapshot_interval:
        parser.error("direct gate decision window must align with --snapshot-interval")
    for path in (args.protect_checkpoint, args.chase_checkpoint):
        if not path.is_file():
            parser.error(f"checkpoint not found: {path}")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    config = {
        "attackers": list(attackers),
        "seeds": list(seeds),
        "snapshot_interval": int(args.snapshot_interval),
        "snapshot_start": int(args.snapshot_start),
        "burst_steps": list(bursts),
        "hidden_modes": list(hidden_modes),
        "gate": (
            None
            if args.gate_target_time_margin is None
            else {
                "decision_start": int(args.snapshot_start),
                "decision_end": int(args.gate_decision_end),
                "decision_interval": int(args.snapshot_interval),
                "decision_steps": list(
                    range(
                        int(args.snapshot_start),
                        int(args.gate_decision_end) + 1,
                        int(args.snapshot_interval),
                    )
                ),
                "min_target_time_margin": float(args.gate_target_time_margin),
                "min_defender_target_distance": float(args.gate_min_defender_target_distance),
                "max_attacker_target_progress_16": args.gate_max_attacker_target_progress_16,
                "max_protect_action_disagreement": (
                    args.chase_max_protect_action_disagreement
                ),
                "decision_count": int(
                    (args.gate_decision_end - args.snapshot_start) // args.snapshot_interval + 1
                ),
                "on_switch": "reset Chase recurrent state and use Chase until terminal",
                "on_action_conflict": "latch Protect until terminal and cancel later decisions",
                "on_reject": "continue Protect until terminal",
            }
        ),
        "contract": {
            "evaluation_mode": (
                "direct_online_gate" if args.direct_gate else "snapshot_counterfactual"
            ),
            "greedy": True,
            "paired_fixed_seeds": True,
            "snapshot_branching": not args.direct_gate,
            "controller_obstacle_mask": True,
            "defender_collision_is_draw": True,
            "attacker_speed": float(map_config.attacker_speed),
            "defender_speed": float(map_config.defender_speed),
        },
        "checkpoints": {
            "protect": {"path": str(args.protect_checkpoint.resolve()), "sha256": _sha256(args.protect_checkpoint)},
            "chase": {"path": str(args.chase_checkpoint.resolve()), "sha256": _sha256(args.chase_checkpoint)},
        },
    }
    _json_dump(output_dir / "run_config.json", config)

    env = TADEnv(reward_mode="standard")
    bank = SkillBank(args.protect_checkpoint, args.chase_checkpoint)
    episodes = []
    branches = []
    direct_records = []
    try:
        for attacker_name in attackers:
            for seed in seeds:
                if args.direct_gate:
                    protect = _run_direct_episode(env, bank, attacker_name, seed)
                    candidate = _run_direct_episode(
                        env,
                        bank,
                        attacker_name,
                        seed,
                        gate={
                            "start": int(args.snapshot_start),
                            "end": int(args.gate_decision_end),
                            "interval": int(args.snapshot_interval),
                            "min_target_time_margin": float(args.gate_target_time_margin),
                            "min_defender_target_distance": float(
                                args.gate_min_defender_target_distance
                            ),
                            "max_attacker_target_progress_16": (
                                args.gate_max_attacker_target_progress_16
                            ),
                            "max_protect_action_disagreement": (
                                args.chase_max_protect_action_disagreement
                            ),
                        },
                    )
                    episodes.append(protect)
                    direct_records.append(_direct_record(protect, candidate))
                    print(
                        f"A={attacker_name} seed={seed} Protect={protect['outcome']} "
                        f"Gate={candidate['outcome']} switch={candidate['switch_step']}",
                        flush=True,
                    )
                    continue
                episode, rows = _run_case(
                    env,
                    bank,
                    attacker_name,
                    seed,
                    snapshot_interval=int(args.snapshot_interval),
                    snapshot_start=int(args.snapshot_start),
                    burst_steps=bursts,
                    hidden_modes=hidden_modes,
                )
                episodes.append(episode)
                branches.extend(rows)
                print(
                    f"A={attacker_name} seed={seed} Protect={episode['outcome']} "
                    f"snapshots={episode['snapshots']} branches={len(rows)}",
                    flush=True,
                )
    finally:
        env.close()

    _jsonl_dump(output_dir / "episodes.jsonl", episodes)
    if args.direct_gate:
        _jsonl_dump(output_dir / "candidate_episodes.jsonl", direct_records)
        _json_dump(output_dir / "candidate_summary.json", _gated_summary(direct_records))
        print(json.dumps(_gated_summary(direct_records), ensure_ascii=False, indent=2))
        return
    _jsonl_dump(output_dir / "branches.jsonl", branches)
    _json_dump(output_dir / "summary.json", _summary(episodes, branches))
    if args.gate_target_time_margin is not None:
        gated_records = _gated_records(
            episodes,
            branches,
            switch_step=int(args.snapshot_start),
            min_target_time_margin=float(args.gate_target_time_margin),
            min_defender_target_distance=float(args.gate_min_defender_target_distance),
            max_attacker_target_progress_16=args.gate_max_attacker_target_progress_16,
        )
        _jsonl_dump(output_dir / "candidate_episodes.jsonl", gated_records)
        _json_dump(output_dir / "candidate_summary.json", _gated_summary(gated_records))
    print(json.dumps(_summary(episodes, branches), ensure_ascii=False, indent=2))
    if args.gate_target_time_margin is not None:
        print(json.dumps(_gated_summary(gated_records), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
