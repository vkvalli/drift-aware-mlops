"""
Fairness evaluation plots.

Functions:
  plot_fairness_over_time(events_path, outdir)  → fairness_over_time.svg
  plot_fairness_tradeoff(summary_csv, outdir)   → fairness_vs_performance.svg
  plot_fairness_cost(summary_csv, outdir)       → fairness_vs_cost.svg
"""
import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    if "MPLCONFIGDIR" not in os.environ:
        _mpl_dir = Path(tempfile.gettempdir()) / "matplotlib-cache"
        _mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(_mpl_dir)
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    _MPL = True
except Exception:
    plt = None
    _MPL = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _batch_index(batch_id: str) -> int:
    try:
        return int(batch_id.split("_")[-1])
    except Exception:
        return 0


def _load_events(path: Path) -> List[Dict]:
    records: List[Dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _load_summary(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _safe_float(value, default: float = 0.0) -> Optional[float]:
    if value is None or str(value).strip() in {"", "NA", "nan", "None"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Plot 1: fairness_core per batch over time
# ---------------------------------------------------------------------------

def plot_fairness_over_time(events_path: Path, outdir: Path) -> Path:
    """Line plot of fairness_core per batch index, saved as fairness_over_time.svg."""
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "fairness_over_time.svg"

    events = _load_events(events_path)
    states = [e for e in events if e.get("event_type") in {"state", "StateEvent"}]
    states_sorted = sorted(states, key=lambda s: _batch_index(s.get("batch_id", "b_0")))

    xs = [_batch_index(s.get("batch_id", "b_0")) for s in states_sorted]
    ys = [float(s.get("fairness_core", 0.0)) for s in states_sorted]

    if not xs:
        _write_empty_svg(out_path, "No state events found")
        return out_path

    if _MPL and plt is not None:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(xs, ys, color="#e15759", linewidth=1.2, label="fairness_core (DPD proxy)")
        ax.axhline(y=0.1, color="#888", linestyle="--", linewidth=0.8, label="threshold=0.1")
        ax.set_title("Fairness Core (DPD Proxy) Over Time")
        ax.set_xlabel("Batch Index")
        ax.set_ylabel("DPD Proxy")
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        fig.savefig(out_path, format="svg")
        plt.close(fig)
    else:
        _write_svg_line(out_path, xs, ys, "Fairness Core (DPD Proxy) Over Time",
                        "Batch Index", "DPD Proxy", "#e15759")

    return out_path


# ---------------------------------------------------------------------------
# Plot 2: accuracy vs mean_fairness_core scatter (one point per policy)
# ---------------------------------------------------------------------------

def plot_fairness_tradeoff(summary_csv: Path, outdir: Path) -> Path:
    """Scatter: x=accuracy_mean, y=mean_fairness_core, colour per policy."""
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "fairness_vs_performance.svg"

    rows = _load_summary(summary_csv)
    policy_data: Dict[str, Tuple[List[float], List[float]]] = {}
    for row in rows:
        policy = str(row.get("policy") or "unknown")
        acc = _safe_float(row.get("accuracy_mean"))
        fc = _safe_float(row.get("mean_fairness_core"))
        if acc is None or fc is None:
            continue
        if policy not in policy_data:
            policy_data[policy] = ([], [])
        policy_data[policy][0].append(acc)
        policy_data[policy][1].append(fc)

    if not policy_data:
        _write_empty_svg(out_path, "No data for fairness tradeoff plot")
        return out_path

    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
              "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]

    if _MPL and plt is not None:
        fig, ax = plt.subplots(figsize=(7, 5))
        for i, (policy, (accs, fcs)) in enumerate(sorted(policy_data.items())):
            color = colors[i % len(colors)]
            # Plot individual seeds as small dots
            ax.scatter(accs, fcs, color=color, alpha=0.5, s=30)
            # Plot mean as larger marker
            mean_acc = sum(accs) / len(accs)
            mean_fc = sum(fcs) / len(fcs)
            ax.scatter([mean_acc], [mean_fc], color=color, s=120, zorder=5,
                       label=f"{policy} (n={len(accs)})", edgecolors="black", linewidths=0.5)
        ax.set_title("Fairness vs Performance")
        ax.set_xlabel("Mean Accuracy")
        ax.set_ylabel("Mean Fairness Core (DPD Proxy)")
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(out_path, format="svg")
        plt.close(fig)
    else:
        _write_empty_svg(out_path, "matplotlib unavailable — install matplotlib")

    return out_path


# ---------------------------------------------------------------------------
# Plot 3: cost vs mean_fairness_core scatter
# ---------------------------------------------------------------------------

def plot_fairness_cost(summary_csv: Path, outdir: Path) -> Path:
    """Scatter: x=total_cost_usd, y=mean_fairness_core, colour per policy."""
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "fairness_vs_cost.svg"

    rows = _load_summary(summary_csv)
    policy_data: Dict[str, Tuple[List[float], List[float]]] = {}
    for row in rows:
        policy = str(row.get("policy") or "unknown")
        cost = _safe_float(row.get("total_cost_usd"))
        fc = _safe_float(row.get("mean_fairness_core"))
        if cost is None or fc is None:
            continue
        if policy not in policy_data:
            policy_data[policy] = ([], [])
        policy_data[policy][0].append(cost)
        policy_data[policy][1].append(fc)

    if not policy_data:
        _write_empty_svg(out_path, "No data for fairness cost plot")
        return out_path

    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
              "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]

    if _MPL and plt is not None:
        fig, ax = plt.subplots(figsize=(7, 5))
        for i, (policy, (costs, fcs)) in enumerate(sorted(policy_data.items())):
            color = colors[i % len(colors)]
            ax.scatter(costs, fcs, color=color, alpha=0.5, s=30)
            mean_cost = sum(costs) / len(costs)
            mean_fc = sum(fcs) / len(fcs)
            ax.scatter([mean_cost], [mean_fc], color=color, s=120, zorder=5,
                       label=f"{policy} (n={len(costs)})", edgecolors="black", linewidths=0.5)
        ax.set_title("Fairness vs Cost")
        ax.set_xlabel("Total Cost (USD)")
        ax.set_ylabel("Mean Fairness Core (DPD Proxy)")
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(out_path, format="svg")
        plt.close(fig)
    else:
        _write_empty_svg(out_path, "matplotlib unavailable — install matplotlib")

    return out_path


# ---------------------------------------------------------------------------
# SVG fallback helpers
# ---------------------------------------------------------------------------

def _write_empty_svg(path: Path, message: str) -> None:
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360">'
        f'<rect width="640" height="360" fill="white"/>'
        f'<text x="320" y="180" text-anchor="middle" font-size="14" font-family="Arial">{message}</text>'
        f"</svg>",
        encoding="utf-8",
    )


def _write_svg_line(
    path: Path,
    xs: List[float],
    ys: List[float],
    title: str,
    xlabel: str,
    ylabel: str,
    color: str,
) -> None:
    width, height, margin = 640, 360, 50
    chart_w = width - 2 * margin
    chart_h = height - 2 * margin
    max_x = max(xs) if xs else 1
    max_y = max(ys) if ys else 1.0
    if max_y == 0.0:
        max_y = 1.0

    lines = [
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width/2}" y="24" text-anchor="middle" font-size="15" font-family="Arial">{title}</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#333"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#333"/>',
        f'<text x="{width/2}" y="{height-8}" text-anchor="middle" font-size="11" font-family="Arial">{xlabel}</text>',
    ]
    pts = []
    for xv, yv in zip(xs, ys):
        px = margin + (xv / max(max_x, 1)) * chart_w
        py = height - margin - (yv / max_y) * chart_h
        pts.append((px, py))
        lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2" fill="{color}"/>')
    for i in range(1, len(pts)):
        x1, y1 = pts[i - 1]
        x2, y2 = pts[i]
        lines.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}"/>')

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    svg.extend(lines)
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate fairness evaluation plots")
    parser.add_argument("--summary", type=str, default=None,
                        help="Path to summary.csv from aggregate_matrix.py")
    parser.add_argument("--events", type=str, default=None,
                        help="Path to a single run events.jsonl for time-series plot")
    parser.add_argument("--outdir", type=str, default="runs/fairness_eval/plots",
                        help="Output directory for SVG plots")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    outdir = Path(args.outdir)

    if args.events:
        p = plot_fairness_over_time(Path(args.events), outdir)
        print(f"Saved: {p}")

    if args.summary:
        p2 = plot_fairness_tradeoff(Path(args.summary), outdir)
        print(f"Saved: {p2}")
        p3 = plot_fairness_cost(Path(args.summary), outdir)
        print(f"Saved: {p3}")


if __name__ == "__main__":
    main()
