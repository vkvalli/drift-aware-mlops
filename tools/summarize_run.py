import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


def _load_events(path: Path) -> List[Dict]:
    records: List[Dict] = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record.get("event_type") == "run_meta":
            continue
        records.append(record)
    return records


def _batch_index_from_id(batch_id: str) -> int:
    try:
        return int(batch_id.split("_")[-1])
    except Exception:
        return 0


def summarize(events_path: Path, out_path: Path, shift_batch: Optional[int]) -> None:
    events = _load_events(events_path)

    batch = [e for e in events if e.get("event_type") in {"BatchEvent", "batch"}]
    state = [e for e in events if e.get("event_type") == "state"]
    action = [e for e in events if e.get("event_type") == "action"]
    cost = [e for e in events if e.get("event_type") == "cost"]
    updates = [e for e in events if e.get("event_type") == "model_update"]

    total_cost = sum(float(c.get("estimated_usd", 0.0)) for c in cost)
    fine_tune_actions = [a for a in action if a.get("action_type") == "fine_tune"]
    full_retrain_actions = [a for a in action if a.get("action_type") == "full_retrain"]
    update_actions = fine_tune_actions + full_retrain_actions

    max_batch = max([_batch_index_from_id(b.get("batch_id", "b_000000")) for b in batch], default=0)
    if shift_batch is None:
        shift_batch = max_batch // 2

    pre = [
        s.get("drift_core", 0.0)
        for s in state
        if _batch_index_from_id(s.get("batch_id", "b_000000")) < shift_batch
    ]
    post = [
        s.get("drift_core", 0.0)
        for s in state
        if _batch_index_from_id(s.get("batch_id", "b_000000")) >= shift_batch
    ]

    pre_avg = sum(pre) / len(pre) if pre else 0.0
    post_avg = sum(post) / len(post) if post else 0.0

    executed_updates = [u for u in updates if bool(u.get("executed", False))]
    accuracies = []
    balanced_accuracies = []
    f1_values = []
    sklearn_flags = []
    for u in executed_updates:
        summary = u.get("eval_summary") or {}
        if "accuracy" in summary:
            accuracies.append(float(summary.get("accuracy")))
        if "balanced_accuracy" in summary:
            balanced_accuracies.append(float(summary.get("balanced_accuracy")))
        if "f1" in summary:
            f1_values.append(float(summary.get("f1")))
        if "sklearn_available" in summary:
            sklearn_flags.append(bool(summary.get("sklearn_available")))

    accuracy_mean = sum(accuracies) / len(accuracies) if accuracies else None
    accuracy_median = None
    if accuracies:
        sorted_acc = sorted(accuracies)
        mid = len(sorted_acc) // 2
        if len(sorted_acc) % 2 == 1:
            accuracy_median = sorted_acc[mid]
        else:
            accuracy_median = (sorted_acc[mid - 1] + sorted_acc[mid]) / 2.0
    accuracy_min = min(accuracies) if accuracies else None
    accuracy_max = max(accuracies) if accuracies else None
    accuracy_last = accuracies[-1] if accuracies else None
    sklearn_available_rate = None
    if sklearn_flags:
        sklearn_available_rate = sum(1 for v in sklearn_flags if v) / len(sklearn_flags)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"events_total={len(events)}\n")
        f.write(f"batch_count={len(batch)}\n")
        f.write(f"state_count={len(state)}\n")
        f.write(f"action_count={len(action)}\n")
        f.write(f"cost_count={len(cost)}\n")
        f.write(f"model_update_count_total={len(updates)}\n")
        f.write(f"model_update_executed_count={len(executed_updates)}\n")
        f.write(f"fine_tune_count={len(fine_tune_actions)}\n")
        f.write(f"full_retrain_count={len(full_retrain_actions)}\n")
        f.write(f"update_count={len(update_actions)}\n")
        f.write(f"total_cost_usd={total_cost:.4f}\n")
        f.write(f"shift_batch={shift_batch}\n")
        f.write(f"drift_pre_avg={pre_avg:.6f}\n")
        f.write(f"drift_post_avg={post_avg:.6f}\n")
        if accuracies:
            f.write(f"accuracy_count={len(accuracies)}\n")
            f.write(f"accuracy_mean={accuracy_mean:.6f}\n")
            f.write(f"accuracy_median={accuracy_median:.6f}\n")
            f.write(f"accuracy_min={accuracy_min:.6f}\n")
            f.write(f"accuracy_max={accuracy_max:.6f}\n")
            f.write(f"accuracy_last={accuracy_last:.6f}\n")
        else:
            f.write("accuracy_count=0\n")
            f.write("accuracy_mean=NA\n")
            f.write("accuracy_median=NA\n")
            f.write("accuracy_min=NA\n")
            f.write("accuracy_max=NA\n")
            f.write("accuracy_last=NA\n")
        if balanced_accuracies:
            f.write(f"balanced_accuracy_count={len(balanced_accuracies)}\n")
            f.write(f"balanced_accuracy_mean={sum(balanced_accuracies)/len(balanced_accuracies):.6f}\n")
        else:
            f.write("balanced_accuracy_count=0\n")
            f.write("balanced_accuracy_mean=NA\n")
        if f1_values:
            f.write(f"f1_count={len(f1_values)}\n")
            f.write(f"f1_mean={sum(f1_values)/len(f1_values):.6f}\n")
        else:
            f.write("f1_count=0\n")
            f.write("f1_mean=NA\n")
        if sklearn_available_rate is not None:
            f.write(f"sklearn_available_rate={sklearn_available_rate:.4f}\n")
        else:
            f.write("sklearn_available_rate=NA\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize a run from events.jsonl")
    parser.add_argument("--events", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shift-batch", type=int, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    summarize(Path(args.events), Path(args.out), args.shift_batch)


if __name__ == "__main__":
    main()
