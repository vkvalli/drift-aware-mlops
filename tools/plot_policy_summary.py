import argparse
import csv
from pathlib import Path
from typing import Dict, List


def _read_summary(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _group_by_policy(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, Dict[str, float]] = {}
    for row in rows:
        policy = row.get("policy", "unknown")
        grouped.setdefault(policy, {"fine_tune_count": [], "total_cost_usd": [], "accuracy_count": []})
        grouped[policy]["fine_tune_count"].append(float(row.get("fine_tune_count", 0)))
        grouped[policy]["total_cost_usd"].append(float(row.get("total_cost_usd", 0)))
        try:
            grouped[policy]["accuracy_count"].append(float(row.get("accuracy_count", 0)))
        except Exception:
            grouped[policy]["accuracy_count"].append(0.0)
    summary: Dict[str, Dict[str, float]] = {}
    for policy, vals in grouped.items():
        summary[policy] = {
            "fine_tune_count": _mean(vals["fine_tune_count"]),
            "total_cost_usd": _mean(vals["total_cost_usd"]),
            "accuracy_count": _mean(vals["accuracy_count"]),
        }
    return summary


def _save_svg_bar_chart(out_path: Path, labels: List[str], values: List[float], title: str, y_label: str) -> None:
    width = 640
    height = 360
    margin = 50
    chart_w = width - 2 * margin
    chart_h = height - 2 * margin
    max_val = max(values) if values else 1.0
    max_val = max_val if max_val > 0 else 1.0

    bar_w = chart_w / max(len(values), 1)
    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width/2}" y="24" text-anchor="middle" font-size="16" font-family="Arial">{title}</text>',
        f'<text x="16" y="{height/2}" text-anchor="middle" font-size="12" font-family="Arial" transform="rotate(-90 16,{height/2})">{y_label}</text>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#333"/>',
    ]

    for i, (label, value) in enumerate(zip(labels, values)):
        bar_h = (value / max_val) * chart_h
        x = margin + i * bar_w + bar_w * 0.15
        y = height - margin - bar_h
        w = bar_w * 0.7
        svg_lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{bar_h:.1f}" fill="#4e79a7"/>')
        svg_lines.append(f'<text x="{x + w/2:.1f}" y="{height - margin + 16}" text-anchor="middle" font-size="10" font-family="Arial">{label}</text>')
        svg_lines.append(f'<text x="{x + w/2:.1f}" y="{y - 4:.1f}" text-anchor="middle" font-size="10" font-family="Arial">{value:.2f}</text>')

    svg_lines.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg_lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot policy summaries from summary.csv")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--outdir", required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    rows = _read_summary(Path(args.summary))
    grouped = _group_by_policy(rows)
    labels = sorted(grouped.keys())
    fine_tunes = [grouped[p]["fine_tune_count"] for p in labels]
    costs = [grouped[p]["total_cost_usd"] for p in labels]
    acc_counts = [grouped[p]["accuracy_count"] for p in labels]

    outdir = Path(args.outdir)
    _save_svg_bar_chart(outdir / "fine_tune_count_vs_policy.svg", labels, fine_tunes, "Fine Tune Count vs Policy", "Fine Tune Count")
    _save_svg_bar_chart(outdir / "total_cost_usd_vs_policy.svg", labels, costs, "Total Cost USD vs Policy", "Total Cost (USD)")
    _save_svg_bar_chart(outdir / "accuracy_count_vs_policy.svg", labels, acc_counts, "Accuracy Count vs Policy", "Accuracy Count")


if __name__ == "__main__":
    main()
