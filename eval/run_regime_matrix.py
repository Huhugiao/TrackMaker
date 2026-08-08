import argparse
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)


DEFAULT_BASELINE = "models/defender_baseline_mlp_ctde_repro_20260526/final_model.pth"
DEFAULT_CHASE = "models/defender_chase_nmn_dual_gru_raw_dense_05-05-19-12/final_model.pth"


def _parse_list(raw: str) -> List[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _parse_regimes(raw: str) -> List[Tuple[str, str]]:
    if str(raw).strip().lower() == "all":
        names = ("advantage", "neutral", "disadvantage")
        return [(speed, margin) for speed in names for margin in names]
    pairs = []
    for item in _parse_list(raw):
        if ":" not in item:
            raise ValueError(f"Regime pair must be speed:margin, got {item!r}")
        speed, margin = item.split(":", 1)
        pairs.append((speed.strip(), margin.strip()))
    return pairs


def _rate(values: Iterable[int]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return 100.0 * sum(float(x) for x in values) / len(values)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(mean(float(x) for x in values))


def _bool_env(value: bool) -> str:
    return "1" if bool(value) else "0"


def _run_one(
    *,
    env: Dict[str, str],
    defender: str,
    attacker: str,
    checkpoint: str,
    episodes: int,
    stats_path: Path,
    hrl_baseline_path: str,
    hrl_chase_path: str,
    hrl_network_type: str,
    hrl_chase_logit_bias: float = 0.0,
    eval_deterministic_seeds: bool = False,
    eval_fixed_seed: int = 42,
    eval_seed_offset: int = 0,
    hrl_step_trace: bool = False,
    hrl_step_trace_max_steps: int = 0,
) -> Dict:
    cmd = [
        str(PYTHON),
        "eval/vs.py",
        "--no-interactive",
        "--defender",
        defender,
        "--attacker",
        attacker,
        "--episodes",
        str(episodes),
        "--checkpoint",
        checkpoint,
        "--save-stats",
        "--stats-path",
        str(stats_path),
    ]
    if eval_deterministic_seeds:
        cmd.extend(
            [
                "--eval-deterministic-seeds",
                "--eval-fixed-seed",
                str(int(eval_fixed_seed)),
                "--eval-seed-offset",
                str(int(eval_seed_offset)),
            ]
        )
    if defender == "hrl":
        cmd.extend(
            [
                "--hrl-num-skills",
                "2",
                "--hrl-baseline-skill-path",
                hrl_baseline_path,
                "--hrl-chase-skill-path",
                hrl_chase_path,
                "--network-type",
                hrl_network_type,
            ]
        )
    elif defender == "chase":
        if hrl_network_type:
            cmd.extend(["--network-type", hrl_network_type])

    completed = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env={
            **env,
            "HRL_TOP_CHASE_LOGIT_BIAS": str(float(hrl_chase_logit_bias)),
            "VS_HRL_STEP_TRACE_ENABLE": _bool_env(bool(hrl_step_trace)),
            "VS_HRL_STEP_TRACE_MAX_STEPS": str(max(0, int(hrl_step_trace_max_steps))),
            "TAD_ATTACKER_STRATEGY": str(attacker),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Eval failed defender={defender} attacker={attacker} rc={completed.returncode}\n"
            f"{completed.stdout[-4000:]}"
        )
    with stats_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _summarize(raw: Dict) -> Dict[str, float]:
    stats = raw.get("raw", {})
    return {
        "success": _rate(stats.get("defender_wins", [])),
        "capture": _rate(stats.get("defender_captures", [])),
        "attacker_win": _rate(stats.get("attacker_wins", [])),
        "collision": _rate(stats.get("defender_collisions", [])),
        "mean_len": _mean(stats.get("episode_lengths", [])),
        "chase_rate": 100.0 * _mean(stats.get("episode_hrl_skill_chase_selection_rate", [])),
        "baseline_rate": 100.0 * _mean(stats.get("episode_hrl_skill_baseline_selection_rate", [])),
    }


def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_token(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "eval"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate defenders on decoupled speed/margin regimes.")
    parser.add_argument("--name", default="regime_matrix")
    parser.add_argument("--checkpoint", required=True, help="HRL top checkpoint.")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--attackers", default="default,evasive")
    parser.add_argument("--defenders", default="hrl,baseline,chase")
    parser.add_argument(
        "--regimes",
        default="advantage:advantage,neutral:neutral,disadvantage:disadvantage,advantage:disadvantage,disadvantage:advantage",
        help="Comma-separated speed:margin pairs, or all.",
    )
    parser.add_argument("--baseline-path", default=DEFAULT_BASELINE)
    parser.add_argument("--chase-path", default=DEFAULT_CHASE)
    parser.add_argument("--hrl-network-type", default="hrl_top_dual_gru_raw")
    parser.add_argument(
        "--hrl-chase-logit-bias",
        type=float,
        default=0.0,
        help="Additive bias applied to HRL top-policy chase logits during evaluation.",
    )
    parser.add_argument(
        "--chase-network-type",
        default="",
        help="Optional network type override for defender=chase; empty means auto-detect from checkpoint.",
    )
    parser.add_argument(
        "--old-regime",
        action="store_true",
        help="Evaluate the original fixed environment distribution instead of randomized regimes.",
    )
    parser.add_argument(
        "--eval-deterministic-seeds",
        action="store_true",
        help="Use fixed per-episode seeds in eval/vs.py for paired comparisons.",
    )
    parser.add_argument("--eval-fixed-seed", type=int, default=42)
    parser.add_argument("--eval-seed-offset", type=int, default=0)
    parser.add_argument(
        "--hrl-step-trace",
        action="store_true",
        help="Record HRL per-step top-policy traces into eval JSON raw output.",
    )
    parser.add_argument("--hrl-step-trace-max-steps", type=int, default=0)
    parser.add_argument("--out-dir", default="log/regime_matrix")
    args = parser.parse_args()

    attackers = _parse_list(args.attackers)
    defenders = _parse_list(args.defenders)
    regimes = [("old", "old")] if args.old_regime else _parse_regimes(args.regimes)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = REPO_ROOT / args.out_dir / f"{timestamp}_{_safe_token(args.name)}"
    root.mkdir(parents=True, exist_ok=True)

    base_env = os.environ.copy()
    base_env.update(
        {
            "TAD_REGIME_RANDOMIZATION": "0" if args.old_regime else "1",
            "TAD_REGIME_CONTINUOUS": "0",
            "TAD_REGIME_DECOUPLED": "0" if args.old_regime else "1",
            "TAD_UED_SELF_PACED_ENABLE": "0",
            "TAD_REGIME_TERMINAL_REWARD_ENABLE": "0",
            "VS_HRL_SAFE_MASK_ENABLE": "0",
        }
    )

    rows: List[Dict] = []
    with (root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)

    for speed, margin in regimes:
        for attacker in attackers:
            for defender in defenders:
                checkpoint = args.checkpoint
                if defender == "baseline":
                    checkpoint = args.baseline_path
                elif defender == "chase":
                    checkpoint = args.chase_path

                env = base_env.copy()
                if not args.old_regime:
                    env["TAD_SPEED_REGIME"] = speed
                    env["TAD_MARGIN_REGIME"] = margin
                stats_path = root / f"{_safe_token(speed)}__{_safe_token(margin)}__{_safe_token(attacker)}__{_safe_token(defender)}.json"
                print(
                    f"[RegimeMatrix] speed={speed} margin={margin} attacker={attacker} "
                    f"defender={defender} episodes={args.episodes}",
                    flush=True,
                )
                raw = _run_one(
                    env=env,
                    defender=defender,
                    attacker=attacker,
                    checkpoint=checkpoint,
                    episodes=args.episodes,
                    stats_path=stats_path,
                    hrl_baseline_path=args.baseline_path,
                    hrl_chase_path=args.chase_path,
                    hrl_network_type=args.chase_network_type if defender == "chase" else args.hrl_network_type,
                    hrl_chase_logit_bias=float(args.hrl_chase_logit_bias) if defender == "hrl" else 0.0,
                    eval_deterministic_seeds=bool(args.eval_deterministic_seeds),
                    eval_fixed_seed=int(args.eval_fixed_seed),
                    eval_seed_offset=int(args.eval_seed_offset),
                    hrl_step_trace=bool(args.hrl_step_trace) if defender == "hrl" else False,
                    hrl_step_trace_max_steps=int(args.hrl_step_trace_max_steps),
                )
                row = {
                    "speed_regime": speed,
                    "margin_regime": margin,
                    "attacker": attacker,
                    "defender": defender,
                    "episodes": args.episodes,
                    **_summarize(raw),
                    "stats_path": str(stats_path),
                }
                rows.append(row)
                _write_csv(root / "summary.csv", rows)
                print(
                    f"  success={row['success']:.1f}% capture={row['capture']:.1f}% "
                    f"collision={row['collision']:.1f}% chase={row['chase_rate']:.1f}% "
                    f"len={row['mean_len']:.1f}",
                    flush=True,
                )

    _write_csv(root / "summary.csv", rows)
    print(f"[RegimeMatrix] done: {root}", flush=True)


if __name__ == "__main__":
    main()
