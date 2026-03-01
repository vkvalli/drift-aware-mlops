import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def _load_events(path: Path) -> List[Dict]:
    records: List[Dict] = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        records.append(record)
    return records


def _compute_metrics(events: List[Dict]) -> Dict[str, float]:
    records = [e for e in events if e.get("event_type") != "run_meta"]
    batch = [e for e in records if e.get("event_type") in {"BatchEvent", "batch"}]
    action = [e for e in records if e.get("event_type") == "action"]
    cost = [e for e in records if e.get("event_type") == "cost"]
    updates = [e for e in records if e.get("event_type") == "model_update"]
    fine_tune = [a for a in action if a.get("action_type") == "fine_tune"]

    total_cost = sum(float(c.get("estimated_usd", 0.0)) for c in cost)
    action_count = len(action)
    fine_tune_count = len(fine_tune)
    batch_count = len(batch)

    cost_per_update = total_cost / fine_tune_count if fine_tune_count else 0.0
    action_rate = fine_tune_count / action_count if action_count else 0.0

    def _collect_float(series: List[Dict], key: str) -> List[float]:
        vals: List[float] = []
        for item in series:
            value = item.get(key)
            if value is None:
                continue
            try:
                vals.append(float(value))
            except (TypeError, ValueError):
                continue
        return vals

    def _series_stats(values: List[float]):
        if not values:
            return 0, None, None, None, None
        mean = sum(values) / len(values)
        last = values[-1]
        minimum = min(values)
        maximum = max(values)
        if len(values) > 1:
            variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            std = variance ** 0.5
        else:
            std = None
        return len(values), mean, std, minimum, last

    accuracies = _collect_float(cost, "accuracy_pre")
    balanced_accuracies = _collect_float(cost, "balanced_accuracy_pre")
    f1_values = _collect_float(cost, "f1_pre")

    accuracy_count, accuracy_mean, accuracy_std, accuracy_min, accuracy_last = _series_stats(accuracies)
    (
        balanced_accuracy_count,
        balanced_accuracy_mean,
        balanced_accuracy_std,
        balanced_accuracy_min,
        balanced_accuracy_last,
    ) = _series_stats(balanced_accuracies)
    f1_count, f1_mean, f1_std, f1_min, f1_last = _series_stats(f1_values)

    update_accuracies = []
    for u in updates:
        if not u.get("executed", False):
            continue
        summary = u.get("eval_summary") or {}
        if "accuracy" in summary:
            update_accuracies.append(float(summary.get("accuracy")))
    update_accuracy_count = len(update_accuracies)

    if accuracy_count and accuracy_count > 1:
        mean = accuracy_mean or 0.0
        variance = sum((a - mean) ** 2 for a in accuracies) / (accuracy_count - 1)
        accuracy_std = variance ** 0.5

    return {
        "batch_count": batch_count,
        "action_count": action_count,
        "fine_tune_count": fine_tune_count,
        "total_cost_usd": round(total_cost, 4),
        "cost_per_update": round(cost_per_update, 4),
        "action_rate": round(action_rate, 4),
        "accuracy_count": accuracy_count,
        "accuracy_mean": round(accuracy_mean, 4) if accuracy_mean is not None else "NA",
        "accuracy_std": round(accuracy_std, 4) if accuracy_std is not None else "NA",
        "accuracy_min": round(accuracy_min, 4) if accuracy_min is not None else "NA",
        "accuracy_last": round(accuracy_last, 4) if accuracy_last is not None else "NA",
        "balanced_accuracy_count": balanced_accuracy_count,
        "balanced_accuracy_mean": round(balanced_accuracy_mean, 4) if balanced_accuracy_mean is not None else "NA",
        "balanced_accuracy_std": round(balanced_accuracy_std, 4) if balanced_accuracy_std is not None else "NA",
        "balanced_accuracy_min": round(balanced_accuracy_min, 4) if balanced_accuracy_min is not None else "NA",
        "balanced_accuracy_last": round(balanced_accuracy_last, 4) if balanced_accuracy_last is not None else "NA",
        "f1_count": f1_count,
        "f1_mean": round(f1_mean, 4) if f1_mean is not None else "NA",
        "f1_std": round(f1_std, 4) if f1_std is not None else "NA",
        "f1_min": round(f1_min, 4) if f1_min is not None else "NA",
        "f1_last": round(f1_last, 4) if f1_last is not None else "NA",
        "update_accuracy_count": update_accuracy_count,
    }


def aggregate(matrix_dir: Path, out_path: Path) -> None:
    rows: List[Dict] = []
    for run_dir in sorted(matrix_dir.glob("*")):
        events_path = run_dir / "events.jsonl"
        if not events_path.exists():
            continue
        events = _load_events(events_path)
        run_meta = next((e for e in events if e.get("event_type") == "run_meta"), {})
        metrics = _compute_metrics(events)

        row = {
            "run_name": run_dir.name,
            "policy": run_meta.get("policy"),
            "threshold": run_meta.get("threshold"),
            "consecutive_k": run_meta.get("consecutive_k"),
            "fixed_interval": run_meta.get("fixed_interval"),
            "cooldown": run_meta.get("cooldown"),
            "seed": run_meta.get("seed"),
            "window_size": run_meta.get("window_size"),
            "ref_window": run_meta.get("ref_window"),
            "label_delay_batches": run_meta.get("label_delay_batches"),
            "max_rows": run_meta.get("max_rows"),
        }
        row.update(metrics)
        rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "policy",
        "threshold",
        "consecutive_k",
        "fixed_interval",
        "cooldown",
        "seed",
        "window_size",
        "ref_window",
        "label_delay_batches",
        "max_rows",
        "batch_count",
        "action_count",
        "fine_tune_count",
        "total_cost_usd",
        "cost_per_update",
        "action_rate",
        "accuracy_count",
        "accuracy_mean",
        "accuracy_std",
        "accuracy_min",
        "accuracy_last",
        "balanced_accuracy_count",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "balanced_accuracy_min",
        "balanced_accuracy_last",
        "f1_count",
        "f1_mean",
        "f1_std",
        "f1_min",
        "f1_last",
        "update_accuracy_count",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate matrix runs into summary.csv")
    parser.add_argument("--matrix-dir", required=True)
    parser.add_argument("--out", required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    aggregate(Path(args.matrix_dir), Path(args.out))


if __name__ == "__main__":
    main()
