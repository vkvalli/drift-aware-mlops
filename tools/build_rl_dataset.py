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


def _rolling_delta(acc_series: List[float], window: int, index: int) -> float:
    if window <= 0:
        return 0.0
    end = index + 1
    start = max(0, end - window)
    cur_window = acc_series[start:end]
    prev_end = start
    prev_start = max(0, prev_end - window)
    prev_window = acc_series[prev_start:prev_end]
    if len(cur_window) < window or len(prev_window) < window:
        return 0.0
    cur_mean = sum(cur_window) / len(cur_window)
    prev_mean = sum(prev_window) / len(prev_window)
    return cur_mean - prev_mean


def build_dataset(
    events_path: Path,
    out_path: Path,
    w_cost: float,
    reward_mode: str,
    perf_window: int,
    w_drift: float,
    drift_threshold: float,
    normalize_perf: bool,
    action_penalty: float,
    defer_bonus: float,
    defer_threshold: float,
    perf_field: str,
    perf_direction: str,
) -> None:
    events = _load_events(events_path)

    states = {e["state_id"]: e for e in events if e.get("event_type") == "state"}
    actions = [e for e in events if e.get("event_type") == "action"]
    costs = {e["action_id"]: e for e in events if e.get("event_type") == "cost"}
    updates = {
        e["action_id"]: e for e in events if e.get("event_type") == "model_update"
    }

    rows: List[Dict] = []
    prev_perf = 0.0
    perf_series: List[float] = []
    raw_perf_series: List[float] = []
    t = 0
    for action in actions:
        state = states.get(action.get("state_id"))
        cost = costs.get(action.get("action_id"))
        update = updates.get(action.get("action_id"))

        accuracy_pre = None
        balanced_accuracy_pre = None
        f1_pre = None
        mae_pre = None
        rmse_pre = None
        r2_pre = None
        if cost:
            accuracy_pre = cost.get("accuracy_pre")
            balanced_accuracy_pre = cost.get("balanced_accuracy_pre")
            f1_pre = cost.get("f1_pre")
            mae_pre = cost.get("mae_pre")
            rmse_pre = cost.get("rmse_pre")
            r2_pre = cost.get("r2_pre")

        perf_pre = None
        if perf_field == "f1_pre":
            perf_pre = f1_pre
        elif perf_field == "balanced_accuracy_pre":
            perf_pre = balanced_accuracy_pre
        elif perf_field == "mae_pre":
            perf_pre = mae_pre
        elif perf_field == "rmse_pre":
            perf_pre = rmse_pre
        elif perf_field == "r2_pre":
            perf_pre = r2_pre
        else:
            perf_pre = accuracy_pre

        if perf_pre is None:
            perf_val = prev_perf
        else:
            perf_val = float(perf_pre)

        raw_perf_series.append(perf_val)
        perf_series.append(perf_val)

        if reward_mode == "shadow":
            delta_perf = 0.0
            if cost and cost.get("delta_shadow") is not None:
                delta_perf = float(cost.get("delta_shadow"))
        elif reward_mode == "rolling":
            delta_perf = _rolling_delta(perf_series, perf_window, t)
        else:
            delta_perf = perf_val - prev_perf
        if perf_direction == "lower_better":
            delta_perf = -delta_perf
        prev_perf = perf_val

        row = {
            "t": t,
            "state_id": action.get("state_id"),
            "action_id": action.get("action_id"),
            "batch_id": state.get("batch_id") if state else "",
            "drift_core": _safe_float(state.get("drift_core") if state else 0.0),
            "drift_max": _safe_float(state.get("drift_max") if state else 0.0),
            "drift_ready": int(bool(state.get("drift_ready", False))) if state else 0,
            "cooldown_active": (
                int(bool(state.get("cooldown_active", False))) if state else 0
            ),
            "action_type": action.get("action_type"),
            "accuracy_pre": accuracy_pre if accuracy_pre is not None else "",
            "balanced_accuracy_pre": (
                balanced_accuracy_pre if balanced_accuracy_pre is not None else ""
            ),
            "f1_pre": f1_pre if f1_pre is not None else "",
            "mae_pre": mae_pre if mae_pre is not None else "",
            "rmse_pre": rmse_pre if rmse_pre is not None else "",
            "r2_pre": r2_pre if r2_pre is not None else "",
            "perf_pre": perf_pre if perf_pre is not None else "",
            "cost_usd": _safe_float(cost.get("estimated_usd") if cost else 0.0),
            "model_version": action.get("model_version", ""),
            "new_model_version": update.get("new_model_version", "") if update else "",
            "delta_perf": delta_perf,
        }
        rows.append(row)
        t += 1

    if normalize_perf and reward_mode != "shadow":
        perf_max = max(raw_perf_series) if raw_perf_series else 0.0
        if perf_max > 0:
            perf_series = [val / perf_max for val in perf_series]
            prev_perf = 0.0
            for idx, r in enumerate(rows):
                if reward_mode == "rolling":
                    delta_perf = _rolling_delta(perf_series, perf_window, idx)
                else:
                    delta_perf = perf_series[idx] - prev_perf
                if perf_direction == "lower_better":
                    delta_perf = -delta_perf
                prev_perf = perf_series[idx]
                r["delta_perf"] = delta_perf
    elif normalize_perf and reward_mode == "shadow":
        # Shadow mode uses update-only delta_shadow; keep that signal unchanged.
        pass

    c_max = max((r["cost_usd"] for r in rows), default=0.0)
    for r in rows:
        if c_max == 0:
            r["cost_norm"] = 0.0
        else:
            r["cost_norm"] = r["cost_usd"] / c_max
        reward = r["delta_perf"] - w_cost * r["cost_norm"]
        if r.get("action_type") == "defer" and w_drift > 0.0:
            drift_val = float(r.get("drift_core", 0.0))
            if drift_val >= drift_threshold:
                reward -= w_drift * drift_val
        if r.get("action_type") == "fine_tune" and action_penalty > 0.0:
            reward -= action_penalty
        if r.get("action_type") == "defer" and defer_bonus > 0.0:
            drift_val = float(r.get("drift_core", 0.0))
            if drift_val < defer_threshold:
                reward += defer_bonus
        r["reward"] = reward

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
        "balanced_accuracy_pre",
        "f1_pre",
        "mae_pre",
        "rmse_pre",
        "r2_pre",
        "perf_pre",
        "cost_usd",
        "cost_norm",
        "delta_perf",
        "reward",
        "model_version",
        "new_model_version",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    _print_stats(rows, c_max, perf_field)


def _print_stats(rows: List[Dict], c_max: float, perf_field: str) -> None:
    n = len(rows)
    fine_tune = sum(1 for r in rows if r.get("action_type") == "fine_tune")
    acc_vals = [
        float(r["accuracy_pre"])
        for r in rows
        if r.get("accuracy_pre") not in ("", None)
    ]
    perf_vals = [
        float(r["perf_pre"]) for r in rows if r.get("perf_pre") not in ("", None)
    ]
    delta_vals = [float(r["delta_perf"]) for r in rows]
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
    print(f"{perf_field}_mean={_mean(perf_vals):.4f}")
    print(f"{perf_field}_std={_std(perf_vals):.4f}")
    print(f"delta_perf_mean={_mean(delta_vals):.4f}")
    print(f"delta_perf_std={_std(delta_vals):.4f}")
    print(f"reward_mean={_mean(reward_vals):.4f}")
    print(f"reward_std={_std(reward_vals):.4f}")
    print(f"C_max={c_max:.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build RL dataset from canonical events"
    )
    parser.add_argument("--events", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--w-cost", type=float, default=0.25)
    parser.add_argument(
        "--reward-mode", choices=["raw", "rolling", "shadow"], default="rolling"
    )
    parser.add_argument("--perf-window", type=int, default=10)
    parser.add_argument("--w-drift", type=float, default=0.0)
    parser.add_argument("--drift-threshold", type=float, default=0.0)
    parser.add_argument("--normalize-perf", action="store_true")
    parser.add_argument("--action-penalty", type=float, default=0.0)
    parser.add_argument("--defer-bonus", type=float, default=0.0)
    parser.add_argument("--defer-threshold", type=float, default=0.0)
    parser.add_argument(
        "--perf-field",
        choices=[
            "accuracy_pre",
            "balanced_accuracy_pre",
            "f1_pre",
            "mae_pre",
            "rmse_pre",
            "r2_pre",
        ],
        default="accuracy_pre",
    )
    parser.add_argument(
        "--perf-direction",
        choices=["higher_better", "lower_better"],
        default="higher_better",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    build_dataset(
        Path(args.events),
        Path(args.out),
        args.w_cost,
        args.reward_mode,
        args.perf_window,
        args.w_drift,
        args.drift_threshold,
        args.normalize_perf,
        args.action_penalty,
        args.defer_bonus,
        args.defer_threshold,
        args.perf_field,
        args.perf_direction,
    )


if __name__ == "__main__":
    main()
