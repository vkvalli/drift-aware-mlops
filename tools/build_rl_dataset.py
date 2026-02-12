import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional


FEATURES = ["drift_core", "drift_max", "drift_ready", "cooldown_active"]


def _load_events(path: Path) -> List[Dict]:
    records: List[Dict] = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record.get("event_type") == "run_meta":
            continue
        records.append(record)
    return records


def _safe_float(val: Optional[float]) -> float:
    try:
        return float(val)
    except Exception:
        return 0.0


def build_dataset(events_path: Path, out_path: Path, w_cost: float) -> None:
    events = _load_events(events_path)

    states = {e["state_id"]: e for e in events if e.get("event_type") == "state"}
    actions = [e for e in events if e.get("event_type") == "action"]
    costs = {e["action_id"]: e for e in events if e.get("event_type") == "cost"}
    updates = {e["action_id"]: e for e in events if e.get("event_type") == "model_update"}

    rows: List[Dict] = []
    prev_acc = 0.0
    t = 0
    for action in actions:
        state = states.get(action.get("state_id"))
        cost = costs.get(action.get("action_id"))
        update = updates.get(action.get("action_id"))

        accuracy_pre = None
        if cost:
            accuracy_pre = cost.get("accuracy_pre")
        if accuracy_pre is None:
            delta_acc = 0.0
        else:
            delta_acc = float(accuracy_pre) - prev_acc
            prev_acc = float(accuracy_pre)

        row = {
            "t": t,
            "state_id": action.get("state_id"),
            "action_id": action.get("action_id"),
            "batch_id": state.get("batch_id") if state else "",
            "drift_core": _safe_float(state.get("drift_core") if state else 0.0),
            "drift_max": _safe_float(state.get("drift_max") if state else 0.0),
            "drift_ready": int(bool(state.get("drift_ready", False))) if state else 0,
            "cooldown_active": int(bool(state.get("cooldown_active", False))) if state else 0,
            "action_type": action.get("action_type"),
            "accuracy_pre": accuracy_pre if accuracy_pre is not None else "",
            "cost_usd": _safe_float(cost.get("estimated_usd") if cost else 0.0),
            "model_version": action.get("model_version", ""),
            "new_model_version": update.get("new_model_version", "") if update else "",
            "delta_acc": delta_acc,
        }
        rows.append(row)
        t += 1

    c_max = max((r["cost_usd"] for r in rows), default=0.0)
    for r in rows:
        if c_max == 0:
            r["cost_norm"] = 0.0
        else:
            r["cost_norm"] = r["cost_usd"] / c_max
        r["reward"] = r["delta_acc"] - w_cost * r["cost_norm"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "t",
        "state_id",
        "action_id",
        "batch_id",
        "drift_core",
        "drift_max",
        "drift_ready",
        "cooldown_active",
        "action_type",
        "accuracy_pre",
        "cost_usd",
        "cost_norm",
        "delta_acc",
        "reward",
        "model_version",
        "new_model_version",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    _print_stats(rows, c_max)


def _print_stats(rows: List[Dict], c_max: float) -> None:
    n = len(rows)
    fine_tune = sum(1 for r in rows if r.get("action_type") == "fine_tune")
    acc_vals = [float(r["accuracy_pre"]) for r in rows if r.get("accuracy_pre") not in ("", None)]
    delta_vals = [float(r["delta_acc"]) for r in rows]
    reward_vals = [float(r["reward"]) for r in rows]

    def _mean(vals: List[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    def _std(vals: List[float]) -> float:
        if len(vals) < 2:
            return 0.0
        m = _mean(vals)
        return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5

    print(f"n_rows={n}")
    print(f"fine_tune_rate={fine_tune / n if n else 0.0:.4f}")
    print(f"accuracy_pre_mean={_mean(acc_vals):.4f}")
    print(f"accuracy_pre_std={_std(acc_vals):.4f}")
    print(f"delta_acc_mean={_mean(delta_vals):.4f}")
    print(f"delta_acc_std={_std(delta_vals):.4f}")
    print(f"reward_mean={_mean(reward_vals):.4f}")
    print(f"reward_std={_std(reward_vals):.4f}")
    print(f"C_max={c_max:.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build RL dataset from canonical events")
    parser.add_argument("--events", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--w-cost", type=float, default=0.25)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    build_dataset(Path(args.events), Path(args.out), args.w_cost)


if __name__ == "__main__":
    main()
