#!/usr/bin/env python3
"""Fit a portable ROS 2 calibration JSON into a separate measured V2.1 profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coppelia_env.digital_twin import load_profile  # noqa: E402
from coppelia_env.digital_twin_calibration import (  # noqa: E402
    fit_calibration_dataset,
    measured_profile_from_fit,
    validate_calibration_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--prior-profile",
        type=Path,
        default=PROJECT_ROOT / "coppeliasim/profiles/trackmaker_turtlebot4_v2_1_prior.json",
    )
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--profile-id")
    parser.add_argument("--split-seed", type=int, default=90210)
    args = parser.parse_args()
    for output in (args.output_profile, args.output_report):
        if output.exists():
            raise SystemExit(f"refusing to overwrite existing fit output: {output}")
    dataset = validate_calibration_dataset(json.loads(args.dataset.read_text(encoding="utf-8")))
    prior = load_profile(args.prior_profile)
    fit = fit_calibration_dataset(dataset, prior, split_seed=args.split_seed)
    profile_id = args.profile_id or f"trackmaker_turtlebot4_v2_1_measured_{fit['dataset_checksum'][-12:]}"
    measured = measured_profile_from_fit(
        prior,
        fit,
        profile_id=profile_id,
        calibration_seed=int(dataset["seed"]),
    )
    report = {
        **fit,
        "dataset": str(args.dataset.resolve()),
        "prior_profile": str(args.prior_profile.resolve()),
        "measured_profile": str(args.output_profile.resolve()),
        "measured_profile_id": measured["profile_id"],
        "measured_profile_checksum": measured["checksum"],
        "input_provenance": dataset["source_profile"]["provenance"],
        "input_calibration_state": dataset["source_profile"]["calibration_state"],
    }
    args.output_profile.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_profile.write_text(json.dumps(measured, indent=2) + "\n", encoding="utf-8")
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("dataset_checksum", "measured_profile_id", "measured_profile_checksum", "split")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
