import argparse
from pathlib import Path


def parse_summary(summary_path: Path):
    text = summary_path.read_text().strip().splitlines()
    rows = []
    current = None
    for line in text:
        if not line or line == "Run Summary":
            continue
        if set(line) == {"-"}:
            continue
        if line.startswith("run_"):
            current = {"run": line}
            rows.append(current)
            continue
        if "=" in line and current is not None:
            k, v = line.split("=", 1)
            current[k] = v
    return rows


def build_table(rows):
    cols = [
        "run",
        "batch_count",
        "state_count",
        "action_count",
        "cost_count",
        "model_update_count",
        "fine_tune_count",
        "fine_tune_rate",
        "total_cost_usd",
        "cost_per_fine_tune",
        "shift_batch",
        "drift_pre_avg",
        "drift_post_avg",
    ]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(r.get(c, "") for c in cols) + " |")
    return "\n".join(lines)


def update_log(log_path: Path, table_md: str) -> None:
    header = "## Run Parameters Summary\n\n"
    content = log_path.read_text() if log_path.exists() else "# Experiments Log\n\n"

    if header in content:
        before, _ = content.split(header, 1)
        new_content = before + header + table_md + "\n"
    else:
        new_content = content + "\n" + header + table_md + "\n"

    log_path.write_text(new_content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Update experiments_log.md from runs/summary.txt")
    parser.add_argument("--summary", default="runs/summary.txt")
    parser.add_argument("--log", default="docs/experiments_log.md")
    args = parser.parse_args()

    rows = parse_summary(Path(args.summary))
    table_md = build_table(rows)
    update_log(Path(args.log), table_md)


if __name__ == "__main__":
    main()
