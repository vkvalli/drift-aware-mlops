import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


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


def _collect_metrics(events: List[Dict]) -> Dict:
    actions = [e for e in events if e.get("event_type") == "action"]
    costs = [e for e in events if e.get("event_type") == "cost"]
    states = {e.get("state_id"): e for e in events if e.get("event_type") == "state"}

    fine_tune = [a for a in actions if a.get("action_type") == "fine_tune"]
    total_cost = sum(float(c.get("estimated_usd", 0.0)) for c in costs)

    # map actions to batch index via state_id -> batch_id
    action_batches = []
    for a in actions:
        state = states.get(a.get("state_id"))
        if state:
            action_batches.append(_batch_index_from_id(state.get("batch_id", "b_000000")))
    return {
        "fine_tune_count": len(fine_tune),
        "total_cost": total_cost,
        "action_batches": action_batches,
        "costs": costs,
        "states": states,
        "actions": actions,
    }


def _svg_header(width: int, height: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'


def _svg_footer() -> str:
    return "</svg>"


def _bar_chart(labels: List[str], values: List[float], title: str, out_path: Path) -> None:
    width, height = 600, 300
    padding = 40
    max_val = max(values) if values else 1.0
    bar_width = (width - 2 * padding) / max(len(values), 1)

    parts = [_svg_header(width, height)]
    parts.append(f'<text x="{width/2}" y="20" text-anchor="middle" font-size="14">{title}</text>')

    for i, (label, val) in enumerate(zip(labels, values)):
        bar_height = (height - 2 * padding) * (val / max_val) if max_val else 0
        x = padding + i * bar_width + 10
        y = height - padding - bar_height
        parts.append(f'<rect x="{x}" y="{y}" width="{bar_width-20}" height="{bar_height}" fill="#4C78A8"/>')
        parts.append(f'<text x="{x + (bar_width-20)/2}" y="{height - padding + 15}" text-anchor="middle" font-size="10">{label}</text>')
        parts.append(f'<text x="{x + (bar_width-20)/2}" y="{y - 5}" text-anchor="middle" font-size="10">{val:.2f}</text>')

    parts.append(_svg_footer())
    out_path.write_text("\n".join(parts))


def _timeline(labels: List[str], batches_list: List[List[int]], out_path: Path) -> None:
    width, height = 800, 200 + 40 * len(labels)
    padding = 40
    max_batch = max((max(b) for b in batches_list if b), default=1)

    parts = [_svg_header(width, height)]
    parts.append(f'<text x="{width/2}" y="20" text-anchor="middle" font-size="14">Action Timeline (fine_tune)</text>')

    for idx, (label, batches) in enumerate(zip(labels, batches_list)):
        y = padding + idx * 30
        parts.append(f'<text x="10" y="{y+5}" font-size="10">{label}</text>')
        for b in batches:
            x = padding + (width - 2 * padding) * (b / max_batch)
            parts.append(f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y+10}" stroke="#E45756" stroke-width="1"/>')

    parts.append(_svg_footer())
    out_path.write_text("\n".join(parts))


def _cost_curves(labels: List[str], costs_list: List[List[float]], out_path: Path) -> None:
    width, height = 800, 300
    padding = 40
    max_len = max((len(c) for c in costs_list), default=1)
    max_val = max((max(c) for c in costs_list if c), default=1.0)

    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]

    parts = [_svg_header(width, height)]
    parts.append(f'<text x="{width/2}" y="20" text-anchor="middle" font-size="14">Cumulative Cost</text>')

    for i, (label, costs) in enumerate(zip(labels, costs_list)):
        if not costs:
            continue
        points = []
        for j, val in enumerate(costs):
            x = padding + (width - 2 * padding) * (j / max(max_len - 1, 1))
            y = height - padding - (height - 2 * padding) * (val / max_val)
            points.append(f"{x},{y}")
        color = colors[i % len(colors)]
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(points)}"/>')
        parts.append(f'<text x="{padding}" y="{padding + 15*i}" font-size="10" fill="{color}">{label}</text>')

    parts.append(_svg_footer())
    out_path.write_text("\n".join(parts))


def plot_compare(runs: List[Tuple[str, Path]], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    labels = []
    fine_counts = []
    costs = []
    timelines = []
    cost_curves = []

    for label, path in runs:
        events = _load_events(path)
        metrics = _collect_metrics(events)
        labels.append(label)
        fine_counts.append(metrics["fine_tune_count"])
        costs.append(metrics["total_cost"])

        # timeline batches for fine_tune only
        fine_batches = []
        for a in metrics["actions"]:
            if a.get("action_type") != "fine_tune":
                continue
            state = metrics["states"].get(a.get("state_id"))
            if state:
                fine_batches.append(_batch_index_from_id(state.get("batch_id", "b_000000")))
        timelines.append(fine_batches)

        # cumulative cost curve
        cumulative = []
        total = 0.0
        for c in metrics["costs"]:
            total += float(c.get("estimated_usd", 0.0))
            cumulative.append(total)
        cost_curves.append(cumulative)

    _bar_chart(labels, fine_counts, "fine_tune_count vs K", outdir / "fine_tune_count_vs_k.svg")
    _bar_chart(labels, costs, "total_cost_usd vs K", outdir / "total_cost_usd_vs_k.svg")
    _timeline(labels, timelines, outdir / "action_timeline_k.svg")
    _cost_curves(labels, cost_curves, outdir / "cost_curve_k.svg")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare K-sweep runs")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--runs", nargs="+", required=True, help="label=path to events.jsonl")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    runs: List[Tuple[str, Path]] = []
    for item in args.runs:
        label, path = item.split("=", 1)
        runs.append((label, Path(path)))

    plot_compare(runs, Path(args.outdir))


if __name__ == "__main__":
    main()
