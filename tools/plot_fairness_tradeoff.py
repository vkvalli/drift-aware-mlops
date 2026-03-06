import argparse
import csv
from pathlib import Path
from typing import List, Dict


def _load_summary(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _safe_float(row: Dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0") or 0.0)
    except Exception:
        return 0.0


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _build_svg(points: List[Dict[str, float]], out_path: Path) -> None:
    width, height = 820, 460
    margin = 60
    x_vals = [p["cost"] for p in points] or [0.0]
    y_vals = [p["f1"] for p in points] or [0.0]
    z_vals = [p["dpd"] for p in points] or [0.0]
    min_x, max_x = min(x_vals), max(x_vals)
    min_y, max_y = min(y_vals), max(y_vals)
    min_z, max_z = min(z_vals), max(z_vals)
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)
    span_z = max(max_z - min_z, 1e-9)

    def _x(v: float) -> float:
        return margin + (v - min_x) / span_x * (width - 2 * margin)

    def _y(v: float) -> float:
        return height - margin - (v - min_y) / span_y * (height - 2 * margin)

    def _radius(v: float) -> float:
        return 5 + 12 * (v - min_z) / span_z

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="24" text-anchor="middle" font-size="16" font-family="Arial">Fairness Tradeoff (Cost vs F1, bubble=DPD)</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#333"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#333"/>',
        f'<text x="{width/2}" y="{height-16}" text-anchor="middle" font-size="12" font-family="Arial">Mean Cost (USD)</text>',
        f'<text x="20" y="{height/2}" text-anchor="middle" font-size="12" font-family="Arial" transform="rotate(-90 20,{height/2})">Mean F1</text>',
    ]
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
    for i, point in enumerate(points):
        px, py = _x(point["cost"]), _y(point["f1"])
        r = _radius(point["dpd"])
        color = colors[i % len(colors)]
        lines.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{r:.1f}" fill="{color}" fill-opacity="0.45" stroke="{color}" />'
        )
        lines.append(
            f'<text x="{px+8:.1f}" y="{py-8:.1f}" font-size="11" font-family="Arial">{point["policy"]}</text>'
        )
    lines.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot fairness tradeoff from summary.csv"
    )
    parser.add_argument("--summary", required=True, help="Path to matrix summary.csv")
    parser.add_argument("--out", default="artifacts/plots/fairness_tradeoff.svg")
    args = parser.parse_args()

    rows = _load_summary(Path(args.summary))
    by_policy: Dict[str, Dict[str, List[float]]] = {}
    for row in rows:
        policy = row.get("policy", "unknown")
        slot = by_policy.setdefault(policy, {"cost": [], "f1": [], "dpd": []})
        slot["cost"].append(_safe_float(row, "total_cost_usd"))
        slot["f1"].append(_safe_float(row, "f1_mean"))
        slot["dpd"].append(_safe_float(row, "dpd_mean"))

    points = []
    for policy, vals in sorted(by_policy.items()):
        points.append(
            {
                "policy": policy,
                "cost": _mean(vals["cost"]),
                "f1": _mean(vals["f1"]),
                "dpd": _mean(vals["dpd"]),
            }
        )

    _build_svg(points, Path(args.out))


if __name__ == "__main__":
    main()
