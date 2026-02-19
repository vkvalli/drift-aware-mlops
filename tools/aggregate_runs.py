import argparse
from pathlib import Path


def aggregate(runs_dir: Path, out_path: Path) -> None:
    reports = sorted(runs_dir.glob("**/plots/run_report.txt"))

    single_runs = []
    matrix_runs = []
    for report in reports:
        rel = report.relative_to(runs_dir)
        parts = rel.parts
        if parts and parts[0].startswith("matrix_"):
            matrix_runs.append((rel, report))
        else:
            single_runs.append((rel, report))

    lines = ["Run Summary", ""]

    if single_runs:
        lines.append("Single runs")
        lines.append("-----------")
        for rel, report in single_runs:
            run_name = rel.parents[1].as_posix()
            lines.append(run_name)
            lines.append("-" * len(run_name))
            lines.extend(report.read_text().strip().splitlines())
            lines.append("")

    if matrix_runs:
        lines.append("Matrix runs")
        lines.append("-----------")
        for rel, report in matrix_runs:
            run_name = rel.parents[1].as_posix()
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
