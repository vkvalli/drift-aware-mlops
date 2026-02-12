import argparse
import json
from pathlib import Path
from typing import List, Tuple


def _extract_svg_inner(svg_text: str) -> Tuple[int, int, str]:
    width = 640
    height = 360
    header_end = svg_text.find(">")
    footer_start = svg_text.rfind("</svg>")
    if header_end == -1 or footer_start == -1:
        return width, height, svg_text

    header = svg_text[:header_end + 1]
    inner = svg_text[header_end + 1:footer_start].strip()

    def _attr_value(attr: str, default: int) -> int:
        idx = header.find(f'{attr}="')
        if idx == -1:
            return default
        start = idx + len(attr) + 2
        end = header.find('"', start)
        try:
            return int(float(header[start:end]))
        except Exception:
            return default

    width = _attr_value("width", width)
    height = _attr_value("height", height)
    return width, height, inner


def build_gallery(matrix_dir: Path, out_path: Path, cols: int = 3) -> None:
    run_dirs = sorted([p for p in matrix_dir.iterdir() if p.is_dir()])
    items: List[Tuple[str, int, int, str]] = []
    for run in run_dirs:
        svg_path = run / "plots" / "accuracy_over_time.svg"
        if not svg_path.exists():
            continue
        svg_text = svg_path.read_text()
        width, height, inner = _extract_svg_inner(svg_text)
        items.append((run.name, width, height, inner))

    if not items:
        out_path.write_text("No accuracy plots found\n")
        return

    cell_w = max(w for _, w, _, _ in items)
    cell_h = max(h for _, _, h, _ in items) + 24
    rows = (len(items) + cols - 1) // cols
    canvas_w = cols * cell_w
    canvas_h = rows * cell_h

    svg_lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_w}" height="{canvas_h}">']

    for idx, (name, w, h, inner) in enumerate(items):
        row = idx // cols
        col = idx % cols
        x = col * cell_w
        y = row * cell_h
        svg_lines.append(f'<g transform="translate({x},{y})">')
        svg_lines.append(f'<text x="{w/2}" y="14" text-anchor="middle" font-size="12" font-family="Arial">{name}</text>')
        svg_lines.append(f'<g transform="translate(0,24)">')
        svg_lines.append(inner)
        svg_lines.append("</g></g>")

    svg_lines.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg_lines))


def build_policy_gallery(matrix_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = [p for p in matrix_dir.iterdir() if p.is_dir()]
    policy_map: Dict[str, List[Path]] = {}
    for run in runs:
        events_path = run / "events.jsonl"
        if not events_path.exists():
            continue
        if run.name.startswith("policy="):
            remainder = run.name.replace("policy=", "")
            cut = len(remainder)
            for marker in ["_T", "_K", "_cd", "_seed", "_N"]:
                idx = remainder.find(marker)
                if idx != -1:
                    cut = min(cut, idx)
            policy = remainder[:cut]
        else:
            policy = run.name.split("_", 1)[0]
        policy_map.setdefault(policy, []).append(run)

    for policy, policy_runs in policy_map.items():
        svg_path = out_dir / f"accuracy_policy_{policy}.svg"
        _plot_policy_accuracy(policy, policy_runs, svg_path)


def _plot_policy_accuracy(policy: str, runs: List[Path], out_path: Path) -> None:
    # Collect per-run accuracy sequences
    sequences: List[List[float]] = []
    for run in runs:
        events_path = run / "events.jsonl"
        records = events_path.read_text().splitlines()
        acc_seq: List[float] = []
        for line in records:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("event_type") != "model_update":
                continue
            if not rec.get("executed", False):
                continue
            summary = rec.get("eval_summary") or {}
            if "accuracy" in summary:
                acc_seq.append(float(summary.get("accuracy")))
        if acc_seq:
            sequences.append(acc_seq)

    if not sequences:
        out_path.write_text("No accuracy data\n")
        return

    max_len = max(len(s) for s in sequences)
    mean_seq: List[float] = []
    for i in range(max_len):
        vals = [s[i] for s in sequences if i < len(s)]
        if vals:
            mean_seq.append(sum(vals) / len(vals))

    width, height = 640, 240
    margin = 50
    chart_w = width - 2 * margin
    chart_h = height - 2 * margin
    xs = list(range(len(mean_seq)))
    ys = mean_seq
    max_x = max(xs) if xs else 1
    min_y = min(ys) if ys else 0.0
    max_y = max(ys) if ys else 1.0
    if max_y == min_y:
        max_y = min_y + 1.0

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width/2}" y="24" text-anchor="middle" font-size="16" font-family="Arial">Accuracy Over Time (mean) — {policy}</text>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#333"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#333"/>',
    ]

    for i, (xv, yv) in enumerate(zip(xs, ys)):
        x = margin + (xv / max(max_x, 1)) * chart_w
        y = height - margin - ((yv - min_y) / (max_y - min_y)) * chart_h
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#59a14f"/>')
        if i > 0:
            prev_xv, prev_yv = xs[i - 1], ys[i - 1]
            px = margin + (prev_xv / max(max_x, 1)) * chart_w
            py = height - margin - ((prev_yv - min_y) / (max_y - min_y)) * chart_h
            svg.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{x:.1f}" y2="{y:.1f}" stroke="#59a14f"/>')

    svg.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build accuracy plot gallery SVG")
    parser.add_argument("--matrix-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--policy-only", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    matrix_dir = Path(args.matrix_dir)
    if args.policy_only:
        build_policy_gallery(matrix_dir, Path(args.out).parent)
    else:
        build_gallery(matrix_dir, Path(args.out), cols=args.cols)


if __name__ == "__main__":
    main()
