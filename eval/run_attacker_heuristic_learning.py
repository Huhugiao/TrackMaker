#!/usr/bin/env python3
"""Run paired-seed Attacker heuristic experiments in the real TAD environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the environment first to respect the repository's legacy env/policy
# initialization order.
from envs.attacker_env import AttackerEnv as _AttackerEnvImportGuard  # noqa: E402,F401
from attacker_heuristics.experiment import (  # noqa: E402
    DEFENDER_SPECS,
    ExperimentRunner,
    SEED_SETS,
    source_version,
)
from attacker_heuristics.registry import POLICY_REGISTRY  # noqa: E402
from configs import map_config  # noqa: E402


def _split_csv(value: str):
    return tuple(token.strip().lower() for token in str(value).split(",") if token.strip())


def _resolve_names(value: str, registry, *, candidates_only: bool = False):
    names = _split_csv(value)
    if names == ("all",):
        if candidates_only:
            return tuple(name for name, spec in registry.items() if not spec.baseline)
        return tuple(registry)
    return names


def _seed_selection(args):
    if args.seeds:
        seeds = tuple(int(token) for token in _split_csv(args.seeds))
        if args.phase == "holdout":
            raise ValueError("holdout must use the registered holdout seed set, not --seeds")
        return seeds
    available = SEED_SETS[args.seed_set]
    start = max(0, int(args.seed_offset))
    stop = len(available) if args.num_seeds <= 0 else start + int(args.num_seeds)
    selected = available[start:stop]
    if not selected:
        raise ValueError("seed selection is empty")
    return selected


def _freeze(run_root: Path, policies):
    payload = {
        "policies": list(policies),
        "code_version": source_version(PROJECT_ROOT),
        "holdout_seed_set": list(SEED_SETS["holdout"]),
        "contract": "No tuning or code changes after this manifest before holdout.",
    }
    run_root.mkdir(parents=True, exist_ok=True)
    path = run_root / "freeze_manifest.json"
    if path.exists():
        raise FileExistsError(f"freeze manifest already exists: {path}")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"frozen": str(path), **payload}, ensure_ascii=False, indent=2))


def _validate_holdout(run_root: Path, policies):
    path = run_root / "freeze_manifest.json"
    if not path.is_file():
        raise RuntimeError("holdout requires a prior --freeze-only manifest")
    frozen = json.loads(path.read_text())
    if tuple(frozen["policies"]) != tuple(policies):
        raise RuntimeError(
            f"holdout policy list differs from frozen list: frozen={frozen['policies']} active={list(policies)}"
        )
    current = source_version(PROJECT_ROOT)
    if current["source_sha256"] != frozen["code_version"]["source_sha256"]:
        raise RuntimeError("heuristic source changed after freeze; establish a new holdout set")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="attacker_heuristic_learning")
    parser.add_argument("--stage", default="round_01_smoke")
    parser.add_argument("--phase", choices=tuple(SEED_SETS), default="discovery")
    parser.add_argument("--seed-set", choices=tuple(SEED_SETS), default=None)
    parser.add_argument("--num-seeds", type=int, default=8)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--seeds", default=None, help="Explicit development/discovery seeds")
    parser.add_argument(
        "--policies",
        default="default,evasive,geometry_feint_v3,occlusion_dash_v2",
    )
    parser.add_argument(
        "--defenders",
        default="learned_protect,learned_chase_skill",
    )
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "heuristic_learning")
    parser.add_argument("--trace-limit-per-policy", type=int, default=3)
    parser.add_argument("--evidence", default="")
    parser.add_argument("--mechanism-change", default="")
    parser.add_argument("--expected-metric", default="")
    parser.add_argument("--regression-risk", default="")
    parser.add_argument("--validation-cases", default="")
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("Policies:")
        for name, spec in POLICY_REGISTRY.items():
            print(f"  {name:22s} family={spec.family:28s} baseline={spec.baseline}")
        print("Defenders:")
        for name, spec in DEFENDER_SPECS.items():
            print(f"  {name:22s} kind={spec.kind:8s} mechanism={spec.mechanism}")
        return

    args.seed_set = args.seed_set or args.phase
    if args.phase == "holdout" and args.seed_set != "holdout":
        raise ValueError("holdout phase must use --seed-set holdout")
    if float(map_config.attacker_speed) != 2.0 or float(map_config.defender_speed) != 2.6:
        raise RuntimeError(
            f"speed contract violated: A={map_config.attacker_speed}, D={map_config.defender_speed}"
        )

    policies = _resolve_names(args.policies, POLICY_REGISTRY)
    defenders = _resolve_names(args.defenders, DEFENDER_SPECS)
    run_root = args.output_root.resolve() / args.run_id
    if args.freeze_only:
        _freeze(run_root, policies)
        return
    if args.phase == "holdout":
        _validate_holdout(run_root, policies)
    seeds = _seed_selection(args)
    output_dir = run_root / args.stage

    runner = ExperimentRunner(
        root=PROJECT_ROOT,
        output_dir=output_dir,
        run_id=args.run_id,
        stage=args.stage,
        phase=args.phase,
        policies=policies,
        defenders=defenders,
        seeds=seeds,
        evidence=args.evidence,
        mechanism_change=args.mechanism_change,
        expected_metric=args.expected_metric,
        regression_risk=args.regression_risk,
        validation_cases=args.validation_cases,
        trace_limit_per_policy=args.trace_limit_per_policy,
    )
    result = runner.run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
