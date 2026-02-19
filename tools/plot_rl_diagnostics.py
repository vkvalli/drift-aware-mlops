import argparse
import csv
from pathlib import Path


def _load_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _rolling_mean(values, window):
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def plot_diagnostics(dataset_path: Path, out_dir: Path, window: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping RL diagnostics plot.")
        return

    rows = _load_rows(dataset_path)
    if not rows:
        print("No rows found; skipping RL diagnostics plot.")
        return

    t_vals = list(range(len(rows)))
    reward_vals = [float(r.get("reward", 0.0) or 0.0) for r in rows]
    action_vals = [1.0 if r.get("action_type") == "fine_tune" else 0.0 for r in rows]

    reward_roll = _rolling_mean(reward_vals, window)
    action_roll = _rolling_mean(action_vals, window)

    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 3))
    plt.plot(t_vals, reward_vals, alpha=0.3, label="reward")
    plt.plot(t_vals, reward_roll, label=f"reward (rolling {window})")
    plt.xlabel("t")
    plt.ylabel("reward")
    plt.title("Reward Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "reward_over_time.svg")
    plt.close()

    plt.figure(figsize=(8, 3))
    plt.plot(t_vals, action_roll, label=f"fine_tune rate (rolling {window})")
    plt.xlabel("t")
    plt.ylabel("rate")
    plt.title("Action Probability (fine_tune)")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "action_prob_fine_tune_over_time.svg")
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot RL diagnostics from RL dataset CSV")
    parser.add_argument("--dataset", required=True, help="Path to RL dataset CSV")
    parser.add_argument("--outdir", required=True, help="Output directory for plots")
    parser.add_argument("--window", type=int, default=10, help="Rolling window size")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    plot_diagnostics(Path(args.dataset), Path(args.outdir), args.window)


if __name__ == "__main__":
    main()
