import argparse
from pathlib import Path


def aggregate(runs_dir: Path, out_path: Path) -> None:
    reports = sorted(runs_dir.glob("run_*/plots/run_report.txt"))

    lines = ["Run Summary", ""]
    for report in reports:
        run_name = report.parents[1].name
        lines.append(run_name)
        lines.append("-" * len(run_name))
        lines.extend(report.read_text().strip().splitlines())
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).rstrip() + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate run_report.txt files into runs/summary.txt")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out", default="runs/summary.txt")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    aggregate(Path(args.runs_dir), Path(args.out))


if __name__ == "__main__":
    main()
