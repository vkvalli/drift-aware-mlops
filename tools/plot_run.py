import argparse
import json
from pathlib import Path
from typing import Dict, List

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def _load_events(path: Path) -> List[Dict]:
    records: List[Dict] = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record.get("event_type") == "run_meta":
            continue
        records.append(record)
    return records


def _batch_index_from_id(batch_id: str) -> int:
    try:
        return int(batch_id.split("_")[-1])
    except Exception:
        return 0


def plot_run(events_path: Path, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    events = _load_events(events_path)

    states = [e for e in events if e.get("event_type") == "state"]
    actions = [e for e in events if e.get("event_type") == "action"]
    costs = [e for e in events if e.get("event_type") == "cost"]
    updates = [e for e in events if e.get("event_type") == "model_update"]

    state_by_id = {s.get("state_id"): s for s in states}

    states_sorted = sorted(states, key=lambda e: _batch_index_from_id(e.get("batch_id", "b_000000")))
    actions_sorted = sorted(
        actions,
        key=lambda e: _batch_index_from_id(
            state_by_id.get(e.get("state_id"), {}).get("batch_id", "b_000000")
        ),
    )
    costs_sorted = sorted(costs, key=lambda e: _batch_index_from_id(e.get("action_id", "a_000000")))

    # Plot 1: Drift over time
    if plt is not None and states_sorted:
        x = [_batch_index_from_id(s["batch_id"]) for s in states_sorted]
        y = [s.get("drift_core", 0.0) for s in states_sorted]
        y_max = [s.get("drift_max", 0.0) for s in states_sorted]
        ready = [s.get("drift_ready", False) for s in states_sorted]

        plt.figure(figsize=(8, 4))
        plt.plot(x, y, label="drift_core")
        plt.plot(x, y_max, label="drift_max", linestyle="--")
        for xi, r in zip(x, ready):
            if not r:
                plt.axvspan(xi - 0.5, xi + 0.5, color="gray", alpha=0.1)
        plt.title("Drift Over Time")
        plt.xlabel("Batch Index")
        plt.ylabel("Drift")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "drift_timeseries.png")
        plt.close()

    # Plot 2: Action timeline
    if plt is not None and actions_sorted:
        x = [
            _batch_index_from_id(
                state_by_id.get(a.get("state_id"), {}).get("batch_id", "b_000000")
            )
            for a in actions_sorted
        ]
        colors = ["red" if a.get("action_type") == "fine_tune" else "blue" for a in actions_sorted]

        plt.figure(figsize=(8, 2))
        for xi, c in zip(x, colors):
            plt.axvline(x=xi, color=c, alpha=0.8)
        plt.title("Action Timeline")
        plt.xlabel("Batch Index")
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(outdir / "action_timeline.png")
        plt.close()

    # Plot 3: Cost accumulation
    if plt is not None and costs_sorted:
        x = list(range(1, len(costs_sorted) + 1))
        cumulative = []
        total = 0.0
        for c in costs_sorted:
            total += float(c.get("estimated_usd", 0.0))
            cumulative.append(total)

        plt.figure(figsize=(8, 4))
        plt.plot(x, cumulative, label="cumulative_usd")
        plt.title("Cost Accumulation")
        plt.xlabel("Action Index")
        plt.ylabel("USD")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "cost_curve.png")
        plt.close()

    # Plot 4: Accuracy over time (SVG, no matplotlib dependency)
    _plot_accuracy_over_time(updates, actions_sorted, outdir / "accuracy_over_time.svg")

    # Run summary (stdlib-only)
    summary_path = outdir / "run_report.txt"
    batch_count = len([e for e in events if e.get("event_type") in {"BatchEvent", "batch"}])
    state_count = len(states)
    action_count = len(actions)
    cost_count = len(costs)
    update_count = len(updates)
    fine_tune_count = len([a for a in actions if a.get("action_type") == "fine_tune"])
    total_cost = sum(float(c.get("estimated_usd", 0.0)) for c in costs)
    fine_tune_rate = fine_tune_count / action_count if action_count else 0.0
    cost_per_fine_tune = total_cost / fine_tune_count if fine_tune_count else 0.0

    max_batch = max(
        [_batch_index_from_id(b.get("batch_id", "b_000000")) for b in events if b.get("event_type") in {"BatchEvent", "batch"}],
        default=0,
    )
    shift_batch = max_batch // 2 if max_batch else 0
    pre = [
        s.get("drift_core", 0.0)
        for s in states
        if _batch_index_from_id(s.get("batch_id", "b_000000")) < shift_batch
    ]
    post = [
        s.get("drift_core", 0.0)
        for s in states
        if _batch_index_from_id(s.get("batch_id", "b_000000")) >= shift_batch
    ]
    pre_avg = sum(pre) / len(pre) if pre else 0.0
    post_avg = sum(post) / len(post) if post else 0.0

    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"batch_count={batch_count}\n")
        f.write(f"state_count={state_count}\n")
        f.write(f"action_count={action_count}\n")
        f.write(f"cost_count={cost_count}\n")
        f.write(f"model_update_count={update_count}\n")
        f.write(f"fine_tune_count={fine_tune_count}\n")
        f.write(f"fine_tune_rate={fine_tune_rate:.4f}\n")
        f.write(f"total_cost_usd={total_cost:.4f}\n")
        f.write(f"cost_per_fine_tune={cost_per_fine_tune:.4f}\n")
        f.write(f"shift_batch={shift_batch}\n")
        f.write(f"drift_pre_avg={pre_avg:.6f}\n")
        f.write(f"drift_post_avg={post_avg:.6f}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot a run from events.jsonl")
    parser.add_argument("--events", required=True)
    parser.add_argument("--outdir", required=True)
    return parser


def _plot_accuracy_over_time(updates: List[Dict], actions_sorted: List[Dict], out_path: Path) -> None:
    points = []
    for idx, u in enumerate(updates):
        if not u.get("executed", False):
            continue
        summary = u.get("eval_summary") or {}
        if "accuracy" not in summary:
            continue
        points.append((idx, float(summary.get("accuracy", 0.0))))

    if not points:
        _save_svg(out_path, 640, 360, _svg_text("No accuracy data"))
        return

    width = 640
    height = 360
    margin = 50
    chart_w = width - 2 * margin
    chart_h = height - 2 * margin

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    max_x = max(xs) if xs else 1
    min_y = min(ys) if ys else 0.0
    max_y = max(ys) if ys else 1.0
    if max_y == min_y:
        max_y = min_y + 1.0

    svg = [
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width/2}" y="24" text-anchor="middle" font-size="16" font-family="Arial">Accuracy Over Time</text>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#333"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#333"/>',
    ]

    for i, (xv, yv) in enumerate(points):
        x = margin + (xv / max(max_x, 1)) * chart_w
        y = height - margin - ((yv - min_y) / (max_y - min_y)) * chart_h
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#59a14f"/>')
        if i > 0:
            prev_xv, prev_yv = points[i - 1]
            px = margin + (prev_xv / max(max_x, 1)) * chart_w
            py = height - margin - ((prev_yv - min_y) / (max_y - min_y)) * chart_h
            svg.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{x:.1f}" y2="{y:.1f}" stroke="#59a14f"/>')

    _save_svg(out_path, width, height, svg)


def _save_svg(out_path: Path, width: int, height: int, lines: List[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    svg.extend(lines)
    svg.append("</svg>")
    out_path.write_text("\n".join(svg))


def _svg_text(text: str) -> List[str]:
    return [
        '<rect width="640" height="360" fill="white"/>',
        f'<text x="320" y="180" text-anchor="middle" font-size="16" font-family="Arial">{text}</text>',
    ]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    plot_run(Path(args.events), Path(args.outdir))


if __name__ == "__main__":
    main()
