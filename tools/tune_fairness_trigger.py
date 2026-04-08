import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.aggregate_matrix import aggregate


def _parse_int_list(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _parse_float_list(value: str) -> List[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def _parse_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _safe_float(row: Dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in {"", "NA", None}:
        return 0.0
    return float(value)


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _build_run_args(
    csv_path: Path,
    seed: int,
    run_dir: Path,
    max_rows: int,
    label_delay_batches: int,
    fairness_threshold: float,
    fairness_threshold_low: float,
    consecutive_k: int,
    cooldown: int,
    fairness_min_support: int,
    fairness_min_group_support: int,
) -> List[str]:
    return [
        sys.executable,
        "-m",
        "src.runner.offline_run",
        "--csv",
        str(csv_path),
        "--task",
        "classification",
        "--label-column",
        "income",
        "--target-column",
        "income",
        "--protected-attr",
        "sex",
        "--dataset-id",
        "adult_openml",
        "--window-size",
        "200",
        "--max-rows",
        str(max_rows),
        "--label-delay-batches",
        str(label_delay_batches),
        "--seed",
        str(seed),
        "--policy",
        "fairness_trigger",
        "--threshold",
        "0.01",
        "--consecutive-k",
        str(consecutive_k),
        "--cooldown",
        str(cooldown),
        "--fairness-threshold",
        str(fairness_threshold),
        "--fairness-threshold-low",
        str(fairness_threshold_low),
        "--fairness-min-support",
        str(fairness_min_support),
        "--fairness-min-group-support",
        str(fairness_min_group_support),
        "--replay-output",
        str(run_dir / "replay.jsonl"),
        "--state-output",
        str(run_dir / "state.jsonl"),
        "--actions-output",
        str(run_dir / "actions.jsonl"),
        "--trainer-output",
        str(run_dir / "trainer_output.jsonl"),
        "--canonical-output",
        str(run_dir / "events.jsonl"),
        "--processed-index-file",
        str(run_dir / "processed_actions.jsonl"),
        "--dlq",
        str(run_dir / "dlq.jsonl"),
        "--clean-artifacts",
    ]


def _summarize_combos(summary_rows: List[Dict[str, str]], target_action_rate: float) -> List[Dict[str, float]]:
    grouped: Dict[str, Dict[str, List[float]]] = {}
    for row in summary_rows:
        run_name = row["run_name"]
        combo = "_".join(run_name.split("_")[:-1]) if "_seed" in run_name else run_name
        slot = grouped.setdefault(
            combo,
            {
                "action_rate": [],
                "fine_tune_count": [],
                "full_retrain_count": [],
                "update_count": [],
                "total_cost_usd": [],
                "balanced_accuracy_mean": [],
                "f1_mean": [],
                "dpd_mean": [],
                "eod_mean": [],
            },
        )
        for key in slot:
            slot[key].append(_safe_float(row, key))

    results: List[Dict[str, float]] = []
    for combo, vals in grouped.items():
        row = {"combo": combo}
        for key, series in vals.items():
            row[key] = _mean(series)
        row["action_gap"] = abs(row["action_rate"] - target_action_rate)
        results.append(row)
    results.sort(
        key=lambda r: (
            r["action_gap"],
            r["dpd_mean"],
            r["total_cost_usd"],
            -r["f1_mean"],
        )
    )
    return results


def _write_rows(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune fairness_trigger by sweeping threshold, persistence K, and cooldown."
    )
    parser.add_argument("--csv", default="data/adult.csv")
    parser.add_argument("--pilot-seeds", default="111")
    parser.add_argument("--confirm-seeds", default="111,222,333")
    parser.add_argument("--fairness-thresholds", default="0.05,0.06,0.08,0.10,0.12")
    parser.add_argument("--fairness-threshold-lows", default="0.03,0.04,0.05,0.06,0.08")
    parser.add_argument("--ks", default="2,3,4")
    parser.add_argument("--cooldowns", default="0,2,4,6")
    parser.add_argument("--max-rows", type=int, default=40000)
    parser.add_argument("--label-delay-batches", type=int, default=2)
    parser.add_argument("--fairness-min-support", type=int, default=200)
    parser.add_argument("--fairness-min-group-support", type=int, default=50)
    parser.add_argument("--target-action-rate", type=float, default=0.10)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--pilot-outdir", default="runs/fairness_tuning_pilot")
    parser.add_argument("--confirm-outdir", default="runs/fairness_tuning_confirm")
    return parser


def _run_grid(
    outdir: Path,
    csv_path: Path,
    seeds: List[int],
    fairness_thresholds: List[float],
    fairness_threshold_lows: List[float],
    ks: List[int],
    cooldowns: List[int],
    max_rows: int,
    label_delay_batches: int,
    fairness_min_support: int,
    fairness_min_group_support: int,
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    for fairness_threshold in fairness_thresholds:
        for fairness_threshold_low in fairness_threshold_lows:
            if fairness_threshold_low > fairness_threshold:
                continue
            for consecutive_k in ks:
                for cooldown in cooldowns:
                    combo_name = (
                        f"fairT={fairness_threshold:.2f}_low={fairness_threshold_low:.2f}_K={consecutive_k}_cd={cooldown}"
                    )
                    for seed in seeds:
                        run_dir = outdir / f"{combo_name}_seed{seed}"
                        run_dir.mkdir(parents=True, exist_ok=True)
                        subprocess.run(
                            _build_run_args(
                                csv_path=csv_path,
                                seed=seed,
                                run_dir=run_dir,
                                max_rows=max_rows,
                            label_delay_batches=label_delay_batches,
                            fairness_threshold=fairness_threshold,
                            fairness_threshold_low=fairness_threshold_low,
                            consecutive_k=consecutive_k,
                            cooldown=cooldown,
                            fairness_min_support=fairness_min_support,
                            fairness_min_group_support=fairness_min_group_support,
                        ),
                        cwd=REPO_ROOT,
                        check=True,
                        )
    summary_path = outdir / "summary.csv"
    aggregate(outdir, summary_path)
    return summary_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    csv_path = (REPO_ROOT / args.csv).resolve()
    pilot_outdir = (REPO_ROOT / args.pilot_outdir).resolve()
    confirm_outdir = (REPO_ROOT / args.confirm_outdir).resolve()

    pilot_seeds = _parse_int_list(args.pilot_seeds)
    confirm_seeds = _parse_int_list(args.confirm_seeds)
    fairness_thresholds = _parse_float_list(args.fairness_thresholds)
    fairness_threshold_lows = _parse_float_list(args.fairness_threshold_lows)
    ks = _parse_int_list(args.ks)
    cooldowns = _parse_int_list(args.cooldowns)

    pilot_summary = _run_grid(
        outdir=pilot_outdir,
        csv_path=csv_path,
        seeds=pilot_seeds,
        fairness_thresholds=fairness_thresholds,
        fairness_threshold_lows=fairness_threshold_lows,
        ks=ks,
        cooldowns=cooldowns,
        max_rows=args.max_rows,
        label_delay_batches=args.label_delay_batches,
        fairness_min_support=args.fairness_min_support,
        fairness_min_group_support=args.fairness_min_group_support,
    )
    pilot_rows = _summarize_combos(_parse_csv(pilot_summary), args.target_action_rate)
    _write_rows(pilot_outdir / "pilot_ranking.csv", pilot_rows)

    top_rows = pilot_rows[: args.top_k]
    confirm_thresholds: List[float] = []
    confirm_ks: List[int] = []
    confirm_cooldowns: List[int] = []
    combos: List[tuple[float, int, int]] = []
    for row in top_rows:
        combo = row["combo"]
        parts = combo.split("_")
        fair_t = float(parts[0].split("=")[1])
        fair_low = float(parts[1].split("=")[1])
        k = int(parts[2].split("=")[1])
        cooldown = int(parts[3].split("=")[1])
        combos.append((fair_t, fair_low, k, cooldown))

    # Run only the shortlisted combinations on all confirmation seeds.
    confirm_outdir.mkdir(parents=True, exist_ok=True)
    for fair_t, fair_low, k, cooldown in combos:
        combo_name = f"fairT={fair_t:.2f}_low={fair_low:.2f}_K={k}_cd={cooldown}"
        for seed in confirm_seeds:
            run_dir = confirm_outdir / f"{combo_name}_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                _build_run_args(
                    csv_path=csv_path,
                    seed=seed,
                    run_dir=run_dir,
                    max_rows=args.max_rows,
                    label_delay_batches=args.label_delay_batches,
                    fairness_threshold=fair_t,
                    fairness_threshold_low=fair_low,
                    consecutive_k=k,
                    cooldown=cooldown,
                    fairness_min_support=args.fairness_min_support,
                    fairness_min_group_support=args.fairness_min_group_support,
                ),
                cwd=REPO_ROOT,
                check=True,
            )

    confirm_summary = confirm_outdir / "summary.csv"
    aggregate(confirm_outdir, confirm_summary)
    confirm_rows = _summarize_combos(_parse_csv(confirm_summary), args.target_action_rate)
    _write_rows(confirm_outdir / "confirm_ranking.csv", confirm_rows)


if __name__ == "__main__":
    main()
