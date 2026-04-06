"""
Energy-efficiency Pareto plot: one point per policy run.
x = total_cost_usd, y = accuracy_mean, size/color = total_kwh

Usage:
    python tools/plot_energy.py --summary runs/matrix_XXXX/summary.csv \
        --outdir runs/energy_eval/plots
"""
import argparse
import csv
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional


def _load_summary(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _safe_float(val, default: float = 0.0) -> Optional[float]:
    if val is None or str(val).strip() in {"", "NA"}:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _group_by_policy(rows: List[Dict]) -> Dict[str, List[Dict]]:
    groups: Dict[str, List[Dict]] = {}
    for row in rows:
        policy = row.get("policy") or row.get("run_name", "unknown")
        # Normalise policy label: strip seed/param suffixes for grouping
        base = policy.split("_seed")[0] if "_seed" in policy else policy
        groups.setdefault(base, []).append(row)
    return groups


def _aggregate_policy(rows: List[Dict]) -> Dict:
    """Mean across seeds for a policy group."""
    costs = [_safe_float(r.get("total_cost_usd")) for r in rows]
    accs = [_safe_float(r.get("accuracy_mean")) for r in rows]
    kwhs = [_safe_float(r.get("total_kwh")) for r in rows]
    co2s = [_safe_float(r.get("total_co2_kg")) for r in rows]

    def _mean(vals):
        valid = [v for v in vals if v is not None]
        return sum(valid) / len(valid) if valid else 0.0

    return {
        "mean_cost_usd": _mean(costs),
        "mean_accuracy": _mean(accs),
        "mean_kwh": _mean(kwhs),
        "mean_co2_kg": _mean(co2s),
        "n_seeds": len(rows),
    }


def plot_pareto_svg(summary_path: Path, outdir: Path) -> Path:
    """Pure-stdlib SVG Pareto scatter: x=cost, y=accuracy, size=kwh."""
    outdir.mkdir(parents=True, exist_ok=True)
    rows = _load_summary(summary_path)
    groups = _group_by_policy(rows)

    points = []
    for policy, policy_rows in sorted(groups.items()):
        agg = _aggregate_policy(policy_rows)
        points.append({
            "policy": policy,
            **agg,
        })

    if not points:
        out_path = outdir / "energy_pareto.svg"
        out_path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="640" height="400">'
                            '<text x="320" y="200" text-anchor="middle">No data</text></svg>')
        return out_path

    width, height = 700, 480
    margin_l, margin_r, margin_t, margin_b = 80, 40, 50, 80
    chart_w = width - margin_l - margin_r
    chart_h = height - margin_t - margin_b

    costs = [p["mean_cost_usd"] for p in points]
    accs = [p["mean_accuracy"] for p in points]
    kwhs = [p["mean_kwh"] for p in points]

    min_cost, max_cost = min(costs), max(costs)
    min_acc, max_acc = min(accs), max(accs)
    max_kwh = max(kwhs) if max(kwhs) > 0 else 1.0

    # Pad ranges to avoid degenerate axes
    cost_range = max_cost - min_cost if max_cost != min_cost else max(max_cost, 0.01)
    acc_range = max_acc - min_acc if max_acc != min_acc else max(max_acc, 0.01)

    # Colour palette (one per policy, cycling)
    palette = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
               "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]

    def _cx(cost):
        return margin_l + ((cost - min_cost) / cost_range) * chart_w if cost_range else margin_l + chart_w / 2

    def _cy(acc):
        return margin_t + chart_h - ((acc - min_acc) / acc_range) * chart_h if acc_range else margin_t + chart_h / 2

    def _r(kwh):
        # Radius proportional to sqrt(kwh/max_kwh), range 6..22
        ratio = (kwh / max_kwh) ** 0.5 if max_kwh > 0 else 0
        return 6 + ratio * 16

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        # Title
        f'<text x="{width/2:.0f}" y="30" text-anchor="middle" font-size="15" '
        f'font-family="Arial" font-weight="bold">Policy Comparison: Performance vs Cost vs Energy</text>',
        # Axes
        f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t+chart_h}" stroke="#333" stroke-width="1.5"/>',
        f'<line x1="{margin_l}" y1="{margin_t+chart_h}" x2="{margin_l+chart_w}" y2="{margin_t+chart_h}" stroke="#333" stroke-width="1.5"/>',
        # Axis labels
        f'<text x="{margin_l + chart_w/2:.0f}" y="{height - 10}" text-anchor="middle" font-size="12" font-family="Arial">Total Cost (USD)</text>',
        f'<text x="14" y="{margin_t + chart_h/2:.0f}" text-anchor="middle" font-size="12" font-family="Arial" '
        f'transform="rotate(-90,14,{margin_t + chart_h/2:.0f})">Mean Accuracy</text>',
        # Subtitle / legend note
        f'<text x="{margin_l + chart_w/2:.0f}" y="{margin_t + chart_h + 55}" text-anchor="middle" '
        f'font-size="10" font-family="Arial" fill="#666">Circle size ∝ total energy (kWh)</text>',
    ]

    # Y-axis ticks
    for tick_frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        tick_val = min_acc + tick_frac * acc_range
        ty = _cy(tick_val)
        svg_lines.append(
            f'<line x1="{margin_l-4}" y1="{ty:.1f}" x2="{margin_l}" y2="{ty:.1f}" stroke="#333"/>'
        )
        svg_lines.append(
            f'<text x="{margin_l-8}" y="{ty+4:.1f}" text-anchor="end" font-size="10" font-family="Arial">{tick_val:.2f}</text>'
        )

    # X-axis ticks
    for tick_frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        tick_val = min_cost + tick_frac * cost_range
        tx = _cx(tick_val)
        svg_lines.append(
            f'<line x1="{tx:.1f}" y1="{margin_t+chart_h}" x2="{tx:.1f}" y2="{margin_t+chart_h+4}" stroke="#333"/>'
        )
        svg_lines.append(
            f'<text x="{tx:.1f}" y="{margin_t+chart_h+16}" text-anchor="middle" font-size="10" font-family="Arial">{tick_val:.2f}</text>'
        )

    # Data points
    for i, p in enumerate(points):
        cx = _cx(p["mean_cost_usd"])
        cy = _cy(p["mean_accuracy"])
        r = _r(p["mean_kwh"])
        color = palette[i % len(palette)]
        kwh_label = f"{p['mean_kwh']:.5f} kWh"
        svg_lines.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" '
            f'fill="{color}" fill-opacity="0.75" stroke="{color}" stroke-width="1.5">'
            f'<title>{p["policy"]}\nAccuracy: {p["mean_accuracy"]:.3f}\nCost: ${p["mean_cost_usd"]:.2f}\nEnergy: {kwh_label}</title>'
            f'</circle>'
        )
        # Label
        svg_lines.append(
            f'<text x="{cx:.1f}" y="{cy - r - 4:.1f}" text-anchor="middle" '
            f'font-size="10" font-family="Arial" fill="{color}">{p["policy"]}</text>'
        )

    # Legend: colour squares
    legend_x = margin_l + chart_w + 5
    if legend_x + 120 > width:
        legend_x = margin_l
        legend_y_start = margin_t - 15
    else:
        legend_y_start = margin_t

    svg_lines.append("</svg>")

    out_path = outdir / "energy_pareto.svg"
    out_path.write_text("\n".join(svg_lines), encoding="utf-8")
    return out_path


def plot_energy_over_time_svg(summary_path: Path, outdir: Path) -> Path:
    """Bar chart: total_kwh per policy (mean across seeds)."""
    outdir.mkdir(parents=True, exist_ok=True)
    rows = _load_summary(summary_path)
    groups = _group_by_policy(rows)

    bars = []
    for policy, policy_rows in sorted(groups.items()):
        agg = _aggregate_policy(policy_rows)
        bars.append({"policy": policy, "kwh": agg["mean_kwh"], "co2": agg["mean_co2_kg"]})

    if not bars:
        out_path = outdir / "energy_by_policy.svg"
        out_path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="640" height="400">'
                            '<text x="320" y="200" text-anchor="middle">No data</text></svg>')
        return out_path

    width, height = 700, 420
    margin_l, margin_r, margin_t, margin_b = 80, 40, 50, 100
    chart_w = width - margin_l - margin_r
    chart_h = height - margin_t - margin_b

    max_kwh = max(b["kwh"] for b in bars) if bars else 1.0
    if max_kwh == 0:
        max_kwh = 1.0

    bar_w = chart_w / len(bars) * 0.6
    gap = chart_w / len(bars)

    palette = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
               "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width/2:.0f}" y="30" text-anchor="middle" font-size="15" '
        f'font-family="Arial" font-weight="bold">Mean Energy Consumption per Policy (kWh)</text>',
        f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t+chart_h}" stroke="#333" stroke-width="1.5"/>',
        f'<line x1="{margin_l}" y1="{margin_t+chart_h}" x2="{margin_l+chart_w}" y2="{margin_t+chart_h}" stroke="#333" stroke-width="1.5"/>',
        f'<text x="14" y="{margin_t + chart_h/2:.0f}" text-anchor="middle" font-size="12" font-family="Arial" '
        f'transform="rotate(-90,14,{margin_t + chart_h/2:.0f})">kWh</text>',
    ]

    # Y-axis ticks
    for tick_frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        tick_val = tick_frac * max_kwh
        ty = margin_t + chart_h - tick_frac * chart_h
        svg_lines.append(
            f'<line x1="{margin_l-4}" y1="{ty:.1f}" x2="{margin_l}" y2="{ty:.1f}" stroke="#333"/>'
        )
        svg_lines.append(
            f'<text x="{margin_l-8}" y="{ty+4:.1f}" text-anchor="end" font-size="10" font-family="Arial">{tick_val:.5f}</text>'
        )

    for i, b in enumerate(bars):
        bx = margin_l + i * gap + (gap - bar_w) / 2
        bh = (b["kwh"] / max_kwh) * chart_h
        by = margin_t + chart_h - bh
        color = palette[i % len(palette)]
        svg_lines.append(
            f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" '
            f'fill="{color}" fill-opacity="0.8"/>'
        )
        # Value label on top
        svg_lines.append(
            f'<text x="{bx + bar_w/2:.1f}" y="{by - 4:.1f}" text-anchor="middle" '
            f'font-size="9" font-family="Arial">{b["kwh"]:.5f}</text>'
        )
        # X-axis label (rotated)
        label = b["policy"]
        lx = bx + bar_w / 2
        ly = margin_t + chart_h + 14
        svg_lines.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="end" font-size="10" font-family="Arial" '
            f'transform="rotate(-35,{lx:.1f},{ly:.1f})">{label}</text>'
        )

    svg_lines.append("</svg>")

    out_path = outdir / "energy_by_policy.svg"
    out_path.write_text("\n".join(svg_lines), encoding="utf-8")
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate energy evaluation plots")
    parser.add_argument("--summary", required=True, help="Path to summary.csv from aggregate_matrix")
    parser.add_argument("--outdir", required=True, help="Output directory for SVG plots")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary_path = Path(args.summary)
    outdir = Path(args.outdir)
    p1 = plot_pareto_svg(summary_path, outdir)
    p2 = plot_energy_over_time_svg(summary_path, outdir)
    print(f"Saved: {p1}")
    print(f"Saved: {p2}")


if __name__ == "__main__":
    main()
