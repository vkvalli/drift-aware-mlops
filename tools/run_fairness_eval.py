import argparse
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.aggregate_matrix import aggregate

try:
    if "MPLCONFIGDIR" not in os.environ:
        mpl_dir = Path(tempfile.gettempdir()) / "matplotlib-cache"
        mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_dir)
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


POLICY_CONFIGS = {
    "no_retrain": [],
    "fixed_interval": ["--fixed-interval", "10"],
    "threshold_persistence": ["--threshold", "0.01", "--consecutive-k", "3"],
    "fairness_trigger": [
        "--threshold",
        "0.01",
        "--consecutive-k",
        "3",
        "--fairness-threshold",
        "0.05",
    ],
}

POLICY_LABELS = {
    "no_retrain": "no_retrain",
    "fixed_interval": "fixed_interval (N=10)",
    "threshold_persistence": "threshold_persistence (T=0.01, K=3)",
    "fairness_trigger": "fairness_trigger (F=0.05, T=0.01)",
}


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


def _policy_means(rows: List[Dict[str, str]]) -> List[Dict[str, float]]:
    grouped: Dict[str, Dict[str, List[float]]] = {}
    for row in rows:
        policy = row.get("policy", "unknown")
        slot = grouped.setdefault(
            policy,
            {
                "accuracy_mean": [],
                "balanced_accuracy_mean": [],
                "f1_mean": [],
                "total_cost_usd": [],
                "dpd_mean": [],
                "eod_mean": [],
                "fine_tune_count": [],
                "action_rate": [],
            },
        )
        for key in slot:
            slot[key].append(_safe_float(row, key))

    policy_rows: List[Dict[str, float]] = []
    for policy, vals in grouped.items():
        policy_rows.append(
            {
                "policy": policy,
                "policy_label": POLICY_LABELS.get(policy, policy),
                "accuracy_mean": _mean(vals["accuracy_mean"]),
                "balanced_accuracy_mean": _mean(vals["balanced_accuracy_mean"]),
                "f1_mean": _mean(vals["f1_mean"]),
                "total_cost_usd": _mean(vals["total_cost_usd"]),
                "dpd_mean": _mean(vals["dpd_mean"]),
                "eod_mean": _mean(vals["eod_mean"]),
                "fine_tune_count": _mean(vals["fine_tune_count"]),
                "action_rate": _mean(vals["action_rate"]),
            }
        )
    return sorted(policy_rows, key=lambda row: row["policy_label"])


def _write_policy_means(path: Path, rows: List[Dict[str, float]]) -> None:
    fieldnames = [
        "policy",
        "policy_label",
        "accuracy_mean",
        "balanced_accuracy_mean",
        "f1_mean",
        "total_cost_usd",
        "dpd_mean",
        "eod_mean",
        "fine_tune_count",
        "action_rate",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_metric(
    rows: List[Dict[str, float]],
    x_key: str,
    x_label: str,
    y_key: str,
    y_label: str,
    title: str,
    out_path: Path,
) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    for idx, row in enumerate(rows):
        x = float(row[x_key])
        y = float(row[y_key])
        label = row["policy_label"]
        color = colors[idx % len(colors)]
        ax.scatter(x, y, color=color, s=140, zorder=3)
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(8, 4),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def _build_report(
    rows: List[Dict[str, float]],
    report_path: Path,
    seeds: List[int],
    label_delay_batches: int,
    max_rows: int,
) -> None:
    lines: List[str] = []
    lines.append("# Fairness Evaluation Results - Adult OpenML")
    lines.append("")
    lines.append(
        f"**Experiment:** Adult OpenML dataset, seeds ({', '.join(str(s) for s in seeds)}), "
        f"`window_size=200`, `max_rows={max_rows}`, `label_delay={label_delay_batches}`"
    )
    lines.append("")
    lines.append("## Results (Mean Across Seeds)")
    lines.append("")
    lines.append(
        "| Policy | Accuracy | Bal. Acc | F1 | Cost ($) | DPD | EOD | Updates | Action Rate |"
    )
    lines.append(
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    for row in rows:
        lines.append(
            f"| {row['policy_label']} | "
            f"{row['accuracy_mean']:.4f} | "
            f"{row['balanced_accuracy_mean']:.4f} | "
            f"{row['f1_mean']:.4f} | "
            f"{row['total_cost_usd']:.2f} | "
            f"{row['dpd_mean']:.4f} | "
            f"{row['eod_mean']:.4f} | "
            f"{row['fine_tune_count']:.2f} | "
            f"{row['action_rate']:.4f} |"
        )
    lines.append("")
    lines.append(
        "> **Note on no_retrain:** DPD and EOD are reported from the logged evaluation stream. "
        "When the system never updates, those values should be read as a degenerate baseline rather than as evidence of fairness."
    )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    positive_cost_rows = [r for r in rows if r["total_cost_usd"] > 0]
    if positive_cost_rows:
        best_f1 = max(positive_cost_rows, key=lambda r: r["f1_mean"])
        fairest = min(positive_cost_rows, key=lambda r: r["dpd_mean"])
        cheapest = min(positive_cost_rows, key=lambda r: r["total_cost_usd"])
        lines.append(
            f"`{best_f1['policy_label']}` achieves the highest mean F1 "
            f"({best_f1['f1_mean']:.4f}) among the retraining policies in this run."
        )
        lines.append(
            f"`{fairest['policy_label']}` has the lowest mean DPD "
            f"({fairest['dpd_mean']:.4f}) among policies that actually retrain."
        )
        lines.append(
            f"`{cheapest['policy_label']}` is the lowest-cost nontrivial policy "
            f"at ${cheapest['total_cost_usd']:.2f} mean cost."
        )
    else:
        lines.append("No retraining policies produced nonzero cost in this run.")
    lines.append("")
    lines.append("## Methodological Notes")
    lines.append("")
    lines.append(
        "- `fairness_trigger` uses a lagged fairness proxy derived only from batches whose labels would already have arrived under the configured delay. It does not read current-batch labels at decision time."
    )
    lines.append(
        "- DPD and EOD in the table are post-hoc evaluation metrics computed on model predictions; they are distinct from the lagged proxy used by the trigger policy."
    )
    lines.append(
        "- All tables and plots in this file are generated from the aggregated seed runs, not from hardcoded values."
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_run_args(
    csv_path: Path,
    policy: str,
    seed: int,
    run_dir: Path,
    max_rows: int,
    label_delay_batches: int,
) -> List[str]:
    args = [
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
        policy,
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
    args.extend(POLICY_CONFIGS.get(policy, []))
    return args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run reproducible fairness benchmark evaluation and generate report artifacts."
    )
    parser.add_argument("--csv", default="data/adult.csv")
    parser.add_argument(
        "--seeds", default="111,222,333", help="Comma-separated seed list"
    )
    parser.add_argument(
        "--policies",
        default="no_retrain,fixed_interval,threshold_persistence,fairness_trigger",
        help="Comma-separated policy list",
    )
    parser.add_argument("--max-rows", type=int, default=40000)
    parser.add_argument("--label-delay-batches", type=int, default=2)
    parser.add_argument(
        "--outdir", default="runs/fairness_eval_matrix", help="Directory for per-seed runs"
    )
    parser.add_argument(
        "--summary-out", default="runs/fairness_eval_matrix/summary.csv"
    )
    parser.add_argument(
        "--policy-means-out", default="runs/fairness_eval_matrix/policy_means.csv"
    )
    parser.add_argument(
        "--report-out", default="docs/fairness_results.md"
    )
    parser.add_argument(
        "--plot-dir", default="docs/figures"
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    repo_root = REPO_ROOT
    csv_path = (repo_root / args.csv).resolve()
    outdir = (repo_root / args.outdir).resolve()
    summary_out = (repo_root / args.summary_out).resolve()
    policy_means_out = (repo_root / args.policy_means_out).resolve()
    report_out = (repo_root / args.report_out).resolve()
    plot_dir = (repo_root / args.plot_dir).resolve()

    seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    policies = [part.strip() for part in args.policies.split(",") if part.strip()]

    outdir.mkdir(parents=True, exist_ok=True)

    for policy in policies:
        if policy not in POLICY_CONFIGS:
            raise ValueError(f"Unknown policy '{policy}'")
        for seed in seeds:
            run_name = f"policy={policy}_seed{seed}"
            run_dir = outdir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                _build_run_args(
                    csv_path=csv_path,
                    policy=policy,
                    seed=seed,
                    run_dir=run_dir,
                    max_rows=args.max_rows,
                    label_delay_batches=args.label_delay_batches,
                ),
                cwd=repo_root,
                check=True,
            )

    aggregate(outdir, summary_out)
    summary_rows = _parse_csv(summary_out)
    policy_rows = _policy_means(summary_rows)
    _write_policy_means(policy_means_out, policy_rows)
    _build_report(
        rows=policy_rows,
        report_path=report_out,
        seeds=seeds,
        label_delay_batches=args.label_delay_batches,
        max_rows=args.max_rows,
    )
    _plot_metric(
        rows=policy_rows,
        x_key="balanced_accuracy_mean",
        x_label="Mean Balanced Accuracy",
        y_key="dpd_mean",
        y_label="Mean DPD (lower = fairer)",
        title="Fairness vs Performance (Adult OpenML)",
        out_path=plot_dir / "fairness_vs_performance.svg",
    )
    _plot_metric(
        rows=policy_rows,
        x_key="total_cost_usd",
        x_label="Mean Cost (USD)",
        y_key="dpd_mean",
        y_label="Mean DPD (lower = fairer)",
        title="Fairness vs Cost (Adult OpenML)",
        out_path=plot_dir / "fairness_vs_cost.svg",
    )


if __name__ == "__main__":
    main()
