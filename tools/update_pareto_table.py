import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


PARETO_START = "<!-- PARETO_TABLE_START -->"
PARETO_END = "<!-- PARETO_TABLE_END -->"


def _policy_means(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    agg = defaultdict(lambda: {"cost": 0.0, "acc": [], "n": 0})
    for r in rows:
        policy = r.get("policy", "unknown")
        agg[policy]["cost"] += float(r.get("total_cost_usd", 0.0))
        acc = r.get("accuracy_mean", "NA")
        if acc != "NA":
            agg[policy]["acc"].append(float(acc))
        agg[policy]["n"] += 1

    means: Dict[str, Dict[str, float]] = {}
    for policy, v in agg.items():
        mean_cost = v["cost"] / v["n"] if v["n"] else 0.0
        mean_acc = sum(v["acc"]) / len(v["acc"]) if v["acc"] else None
        means[policy] = {"cost": mean_cost, "accuracy": mean_acc}
    return means


def _format_table(means: Dict[str, Dict[str, float]]) -> str:
    lines = [
        "| Policy | Mean Cost (USD) | Mean Accuracy |",
        "| --- | --- | --- |",
    ]
    for policy in sorted(means.keys()):
        cost = means[policy]["cost"]
        acc = means[policy]["accuracy"]
        acc_str = f"{acc:.4f}" if acc is not None else "NA"
        lines.append(f"| {policy} | {cost:.2f} | {acc_str} |")
    return "\n".join(lines)


def _replace_block(text: str, table: str) -> str:
    if PARETO_START not in text or PARETO_END not in text:
        raise ValueError("Missing pareto table markers in target file")
    pre, rest = text.split(PARETO_START, 1)
    _, post = rest.split(PARETO_END, 1)
    return pre + PARETO_START + "\n" + table + "\n" + PARETO_END + post


def update_file(path: Path, table: str) -> None:
    content = path.read_text()
    updated = _replace_block(content, table)
    path.write_text(updated)


def main() -> None:
    parser = argparse.ArgumentParser(description="Update Pareto policy table in docs")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--docs", nargs="+", required=True)
    args = parser.parse_args()

    rows = list(csv.DictReader(Path(args.summary).open()))
    means = _policy_means(rows)
    table = _format_table(means)

    for doc in args.docs:
        update_file(Path(doc), table)


if __name__ == "__main__":
    main()
