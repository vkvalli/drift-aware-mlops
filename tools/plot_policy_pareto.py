import argparse
import csv
from pathlib import Path
from statistics import mean


def _load_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _agg(rows):
    return {
        "mean_cost": mean(float(r["total_cost_usd"]) for r in rows),
        "mean_accuracy": mean(float(r["accuracy_mean"]) for r in rows if r["accuracy_mean"] != "NA"),
        "mean_updates": mean(float(r["fine_tune_count"]) for r in rows),
    }


def build_pareto(
    baseline_summary: Path,
    linucb_summary: Path,
    out_path: Path,
) -> None:
    baseline_rows = _load_rows(baseline_summary)
    linucb_rows = _load_rows(linucb_summary)

    fixed_rows = [r for r in baseline_rows if r["policy"] == "fixed_interval" and r["fixed_interval"] == "10"]
    threshold_rows = [
        r for r in baseline_rows if r["policy"] == "threshold_persistence" and r["consecutive_k"] == "5"
    ]
    linucb_rows = [r for r in linucb_rows if r["policy"] == "linucb"]

    fixed = _agg(fixed_rows)
    threshold = _agg(threshold_rows)
    linucb = _agg(linucb_rows)

    points = [
        ("fixed_interval (N=10)", fixed),
        ("threshold_persistence (K=5)", threshold),
        ("linucb (epsilon2=0.04)", linucb),
    ]

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping pareto plot.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    for label, stats in points:
        x = stats["mean_cost"]
        y = stats["mean_accuracy"]
        plt.scatter(x, y, s=60)
        plt.text(x, y, label, fontsize=8, ha="left", va="bottom")
    plt.xlabel("Mean Cost (USD)")
    plt.ylabel("Mean Accuracy")
    plt.title("Accuracy vs Cost (Policy Pareto)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot Accuracy vs Cost for policy comparison")
    parser.add_argument("--baseline-summary", required=True, help="Baseline summary.csv (fixed/threshold)")
    parser.add_argument("--linucb-summary", required=True, help="LinUCB summary.csv")
    parser.add_argument("--out", default="runs/pareto/accuracy_cost.svg")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    build_pareto(Path(args.baseline_summary), Path(args.linucb_summary), Path(args.out))


if __name__ == "__main__":
    main()
