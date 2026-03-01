import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def _latest_matrix(runs_dir: Path) -> Path:
    matrices = sorted(runs_dir.glob("matrix_*"), key=lambda p: p.name)
    if not matrices:
        raise RuntimeError("No matrix directory found after run.")
    return matrices[-1]


def _run_matrix(
    csv_path: Path,
    model_path: Path,
    outdir: Path,
    dataset_id: str,
    label_column: str,
    positive_label: str,
    max_rows: int,
    window_size: int,
    ref_window: int,
    seeds: str,
    label_delay: int,
    warm_start_n: int,
    warm_start_k: int,
    epsilon1: float,
    epsilon1_steps: int,
    epsilon2: float,
) -> Path:
    cmd = [
        sys.executable,
        "-m",
        "tools.run_matrix",
        "--csv",
        str(csv_path),
        "--window-size",
        str(window_size),
        "--max-rows",
        str(max_rows),
        "--label-delay-batches",
        str(label_delay),
        "--ref-window",
        str(ref_window),
        "--seeds",
        seeds,
        "--policies",
        "linucb",
        "--linucb-model",
        str(model_path),
        "--warm-start-n",
        str(warm_start_n),
        "--warm-start-k",
        str(warm_start_k),
        "--epsilon1",
        str(epsilon1),
        "--epsilon1-steps",
        str(epsilon1_steps),
        "--epsilon2",
        str(epsilon2),
        "--dataset-id",
        dataset_id,
        "--label-column",
        label_column,
        "--positive-label",
        positive_label,
        "--outdir",
        str(outdir),
    ]
    subprocess.check_call(cmd)
    return _latest_matrix(outdir)


def _reward_mean(events_path: Path, perf_field: str) -> float:
    out_tmp = Path("/tmp/rl_delay_tmp.csv")
    cmd = [
        sys.executable,
        "tools/build_rl_dataset.py",
        "--events",
        str(events_path),
        "--out",
        str(out_tmp),
        "--reward-mode",
        "rolling",
        "--perf-window",
        "10",
        "--perf-field",
        perf_field,
        "--w-cost",
        "0.25",
        "--normalize-perf",
        "--action-penalty",
        "0.02",
        "--defer-bonus",
        "0.01",
        "--defer-threshold",
        "0.2",
    ]
    out = subprocess.check_output(cmd, text=True)
    for line in out.splitlines():
        if line.startswith("reward_mean="):
            return float(line.split("=", 1)[1].strip())
    return 0.0


def _proxy_gap(events_path: Path, perf_key: str) -> float:
    states: Dict[str, Dict] = {}
    actions: Dict[str, str] = {}
    costs: List[Dict] = []
    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            et = e.get("event_type")
            if et == "state":
                states[e.get("state_id")] = e
            elif et == "action":
                actions[e.get("action_id")] = e.get("state_id")
            elif et == "cost":
                costs.append(e)

    paired: List[Tuple[float, float]] = []
    for c in costs:
        perf = c.get(perf_key)
        if perf is None:
            continue
        state_id = actions.get(c.get("action_id"))
        if not state_id:
            continue
        state = states.get(state_id)
        if not state:
            continue
        paired.append((float(state.get("drift_core", 0.0)), float(perf)))

    if not paired:
        return 0.0

    drifts = [p[0] for p in paired]
    d_min = min(drifts)
    d_max = max(drifts)
    denom = (d_max - d_min) if d_max > d_min else 1.0
    gaps = []
    for d, perf in paired:
        proxy_perf = 1.0 - ((d - d_min) / denom)
        gaps.append(abs(proxy_perf - perf))
    return sum(gaps) / len(gaps)


def _summarize_matrix(matrix_dir: Path, perf_key: str, out_csv: Path, dataset_name: str, delay: int) -> Dict[str, float]:
    summary_path = matrix_dir / "summary.csv"
    subprocess.check_call(
        [sys.executable, "tools/aggregate_matrix.py", "--matrix-dir", str(matrix_dir), "--out", str(summary_path)]
    )

    rows = list(csv.DictReader(summary_path.open("r", encoding="utf-8")))
    if not rows:
        raise RuntimeError(f"No summary rows for {matrix_dir}")

    reward_means = []
    gaps = []
    for run_row in rows:
        run_name = run_row["run_name"]
        events_path = matrix_dir / run_name / "events.jsonl"
        reward_means.append(_reward_mean(events_path, "f1_pre" if perf_key == "f1_mean" else "accuracy_pre"))
        gaps.append(_proxy_gap(events_path, "f1_pre" if perf_key == "f1_mean" else "accuracy_pre"))

    def _mean(col: str) -> float:
        vals = [float(r[col]) for r in rows if r[col] not in {"NA", ""}]
        return sum(vals) / len(vals) if vals else 0.0

    metric = {
        "dataset": dataset_name,
        "label_delay_batches": delay,
        "action_rate_mean": _mean("action_rate"),
        "total_cost_usd_mean": _mean("total_cost_usd"),
        "update_count_mean": _mean("update_accuracy_count"),
        "perf_mean": _mean(perf_key),
        "reward_mean": sum(reward_means) / len(reward_means) if reward_means else 0.0,
        "proxy_groundtruth_gap": sum(gaps) / len(gaps) if gaps else 0.0,
    }

    write_header = not out_csv.exists()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(metric)

    return metric


def main() -> None:
    parser = argparse.ArgumentParser(description="Run delayed-label stress suite for Electricity + Airline.")
    parser.add_argument("--out-csv", default="runs/delay_stress_summary.csv")
    parser.add_argument("--seeds", default="111,222,333")
    parser.add_argument("--delays", default="0,2,5,10")
    args = parser.parse_args()

    out_csv = Path(args.out_csv)
    if out_csv.exists():
        out_csv.unlink()

    delays = [int(x.strip()) for x in args.delays.split(",") if x.strip()]

    runs_root = Path("runs")
    elec_csv = Path("data/electricity-normalized.csv")
    elec_model = Path("artifacts/policies/linucb_electricity_conservative.pkl")

    airline_csv = Path("data/airline_2018_2019_model_ready_120k_labeled.csv")
    airline_model = Path("artifacts/policies/linucb_airline_f1_alpha025_pen002.pkl")

    for d in delays:
        elec_matrix = _run_matrix(
            csv_path=elec_csv,
            model_path=elec_model,
            outdir=runs_root,
            dataset_id="electricity",
            label_column="class",
            positive_label="UP",
            max_rows=20000,
            window_size=200,
            ref_window=3,
            seeds=args.seeds,
            label_delay=d,
            warm_start_n=20,
            warm_start_k=8,
            epsilon1=0.02,
            epsilon1_steps=30,
            epsilon2=0.02,
        )
        metric = _summarize_matrix(elec_matrix, "accuracy_mean", out_csv, "electricity", d)
        print("electricity", d, metric)

        airline_matrix = _run_matrix(
            csv_path=airline_csv,
            model_path=airline_model,
            outdir=runs_root,
            dataset_id="airline",
            label_column="ArrDel15",
            positive_label="1",
            max_rows=120000,
            window_size=200,
            ref_window=3,
            seeds=args.seeds,
            label_delay=d,
            warm_start_n=20,
            warm_start_k=8,
            epsilon1=0.02,
            epsilon1_steps=30,
            epsilon2=0.02,
        )
        metric = _summarize_matrix(airline_matrix, "f1_mean", out_csv, "airline", d)
        print("airline", d, metric)

    print(f"Wrote summary: {out_csv}")


if __name__ == "__main__":
    main()
