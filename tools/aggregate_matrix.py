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
    states = [e for e in records if e.get("event_type") in {"state", "StateEvent"}]
    action = [e for e in records if e.get("event_type") == "action"]
    cost = [e for e in records if e.get("event_type") == "cost"]
    updates = [e for e in records if e.get("event_type") == "model_update"]
    fine_tune = [a for a in action if a.get("action_type") == "fine_tune"]

    total_cost = sum(float(c.get("estimated_usd", 0.0)) for c in cost)
    total_kwh = sum(float(c.get("energy_kwh", 0.0)) for c in cost)
    total_co2_kg = sum(float(c.get("co2_kg", 0.0)) for c in cost)
    action_count = len(action)
    fine_tune_count = len(fine_tune)
    batch_count = len(batch)
    mean_kwh_per_action = total_kwh / action_count if action_count else 0.0
    mean_co2_per_action = total_co2_kg / action_count if action_count else 0.0

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
        if len(values) > 1:
            variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            std = variance**0.5
        else:
            std = None
        return len(values), mean, std, minimum, last

    accuracies = _collect_float(cost, "accuracy_pre")
    balanced_accuracies = _collect_float(cost, "balanced_accuracy_pre")
    f1_values = _collect_float(cost, "f1_pre")
    mae_values = _collect_float(cost, "mae_pre")
    rmse_values = _collect_float(cost, "rmse_pre")
    r2_values = _collect_float(cost, "r2_pre")
    dpd_values = _collect_float(cost, "dpd_pre")
    eod_values = _collect_float(cost, "eod_pre")

    accuracy_count, accuracy_mean, accuracy_std, accuracy_min, accuracy_last = (
        _series_stats(accuracies)
    )
    (
        balanced_accuracy_count,
        balanced_accuracy_mean,
        balanced_accuracy_std,
        balanced_accuracy_min,
        balanced_accuracy_last,
    ) = _series_stats(balanced_accuracies)
    f1_count, f1_mean, f1_std, f1_min, f1_last = _series_stats(f1_values)
    mae_count, mae_mean, mae_std, mae_min, mae_last = _series_stats(mae_values)
    rmse_count, rmse_mean, rmse_std, rmse_min, rmse_last = _series_stats(rmse_values)
    r2_count, r2_mean, r2_std, r2_min, r2_last = _series_stats(r2_values)
    dpd_count, dpd_mean, dpd_std, dpd_min, dpd_last = _series_stats(dpd_values)
    eod_count, eod_mean, eod_std, eod_min, eod_last = _series_stats(eod_values)

    # Fairness core from state events (label-distribution DPD proxy)
    fairness_core_values = _collect_float(states, "fairness_core")
    fairness_triggered_count = sum(
        1 for a in action if a.get("reason") == "fairness_trigger"
    )
    fc_count, fc_mean, fc_std, fc_min, fc_last = _series_stats(fairness_core_values)
    fc_max = max(fairness_core_values) if fairness_core_values else None

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
        accuracy_std = variance**0.5

    return {
        "batch_count": batch_count,
        "action_count": action_count,
        "fine_tune_count": fine_tune_count,
        "total_cost_usd": round(total_cost, 4),
        "cost_per_update": round(cost_per_update, 4),
        "action_rate": round(action_rate, 4),
        "total_kwh": round(total_kwh, 8),
        "total_co2_kg": round(total_co2_kg, 8),
        "mean_kwh_per_action": round(mean_kwh_per_action, 8),
        "mean_co2_per_action": round(mean_co2_per_action, 8),
        "accuracy_count": accuracy_count,
        "accuracy_mean": round(accuracy_mean, 4) if accuracy_mean is not None else "NA",
        "accuracy_std": round(accuracy_std, 4) if accuracy_std is not None else "NA",
        "accuracy_min": round(accuracy_min, 4) if accuracy_min is not None else "NA",
        "accuracy_last": round(accuracy_last, 4) if accuracy_last is not None else "NA",
        "balanced_accuracy_count": balanced_accuracy_count,
        "balanced_accuracy_mean": (
            round(balanced_accuracy_mean, 4)
            if balanced_accuracy_mean is not None
            else "NA"
        ),
        "balanced_accuracy_std": (
            round(balanced_accuracy_std, 4)
            if balanced_accuracy_std is not None
            else "NA"
        ),
        "balanced_accuracy_min": (
            round(balanced_accuracy_min, 4)
            if balanced_accuracy_min is not None
            else "NA"
        ),
        "balanced_accuracy_last": (
            round(balanced_accuracy_last, 4)
            if balanced_accuracy_last is not None
            else "NA"
        ),
        "f1_count": f1_count,
        "f1_mean": round(f1_mean, 4) if f1_mean is not None else "NA",
        "f1_std": round(f1_std, 4) if f1_std is not None else "NA",
        "f1_min": round(f1_min, 4) if f1_min is not None else "NA",
        "f1_last": round(f1_last, 4) if f1_last is not None else "NA",
        "mae_count": mae_count,
        "mae_mean": round(mae_mean, 4) if mae_mean is not None else "NA",
        "mae_std": round(mae_std, 4) if mae_std is not None else "NA",
        "mae_min": round(mae_min, 4) if mae_min is not None else "NA",
        "mae_last": round(mae_last, 4) if mae_last is not None else "NA",
        "rmse_count": rmse_count,
        "rmse_mean": round(rmse_mean, 4) if rmse_mean is not None else "NA",
        "rmse_std": round(rmse_std, 4) if rmse_std is not None else "NA",
        "rmse_min": round(rmse_min, 4) if rmse_min is not None else "NA",
        "rmse_last": round(rmse_last, 4) if rmse_last is not None else "NA",
        "r2_count": r2_count,
        "r2_mean": round(r2_mean, 4) if r2_mean is not None else "NA",
        "r2_std": round(r2_std, 4) if r2_std is not None else "NA",
        "r2_min": round(r2_min, 4) if r2_min is not None else "NA",
        "r2_last": round(r2_last, 4) if r2_last is not None else "NA",
        "dpd_count": dpd_count,
        "dpd_mean": round(dpd_mean, 4) if dpd_mean is not None else "NA",
        "dpd_std": round(dpd_std, 4) if dpd_std is not None else "NA",
        "dpd_min": round(dpd_min, 4) if dpd_min is not None else "NA",
        "dpd_last": round(dpd_last, 4) if dpd_last is not None else "NA",
        "eod_count": eod_count,
        "eod_mean": round(eod_mean, 4) if eod_mean is not None else "NA",
        "eod_std": round(eod_std, 4) if eod_std is not None else "NA",
        "eod_min": round(eod_min, 4) if eod_min is not None else "NA",
        "eod_last": round(eod_last, 4) if eod_last is not None else "NA",
        "update_accuracy_count": update_accuracy_count,
        "mean_fairness_core": round(fc_mean, 4) if fc_mean is not None else "NA",
        "max_fairness_core": round(fc_max, 4) if fc_max is not None else "NA",
        "fairness_triggered_count": fairness_triggered_count,
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
            "task": run_meta.get("task"),
            "feature_mode": run_meta.get("feature_mode"),
            "target_column": run_meta.get("target_column"),
            "protected_attr": run_meta.get("protected_attr"),
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
        "task",
        "feature_mode",
        "target_column",
        "protected_attr",
        "batch_count",
        "action_count",
        "fine_tune_count",
        "total_cost_usd",
        "cost_per_update",
        "action_rate",
        "total_kwh",
        "total_co2_kg",
        "mean_kwh_per_action",
        "mean_co2_per_action",
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
        "mae_count",
        "mae_mean",
        "mae_std",
        "mae_min",
        "mae_last",
        "rmse_count",
        "rmse_mean",
        "rmse_std",
        "rmse_min",
        "rmse_last",
        "r2_count",
        "r2_mean",
        "r2_std",
        "r2_min",
        "r2_last",
        "dpd_count",
        "dpd_mean",
        "dpd_std",
        "dpd_min",
        "dpd_last",
        "eod_count",
        "eod_mean",
        "eod_std",
        "eod_min",
        "eod_last",
        "update_accuracy_count",
        "mean_fairness_core",
        "max_fairness_core",
        "fairness_triggered_count",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate matrix runs into summary.csv"
    )
    parser.add_argument("--matrix-dir", required=True)
    parser.add_argument("--out", required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    aggregate(Path(args.matrix_dir), Path(args.out))


if __name__ == "__main__":
    main()
