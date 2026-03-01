import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from src.logger.jsonl_logger import write_event
from src.schemas.batch import BatchEvent
from src.schemas.state import StateEvent
from src.schemas.validate import validate_or_dlq
from src.utils.ids import IdCounter


EXCLUDED_COLUMNS = {
    "label",
    "class",
    "ArrDel15",
    "arrdel15",
    "timestamp_start",
    "timestamp_end",
    "row_id",
    "Date",
    "Day",
    "Period",
    "date",
    "day",
    "period",
}


def _load_batch_events(events_path: Path, dlq_path: Path) -> List[Dict]:
    records: List[Dict] = []
    for line in events_path.read_text().splitlines():
        record = json.loads(line)
        event_type = record.get("event_type", "")
        if event_type in {"BatchEvent", "batch"}:
            validated = validate_or_dlq(record, BatchEvent, dlq_path)
            if validated:
                records.append(validated.dict())
    return records


def _load_numeric_rows(csv_path: Path) -> List[Dict[str, float]]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, float]] = []
        for row in reader:
            cleaned: Dict[str, float] = {}
            for key, value in row.items():
                if key in EXCLUDED_COLUMNS:
                    continue
                try:
                    cleaned[key] = float(value)
                except (TypeError, ValueError):
                    continue
            rows.append(cleaned)
    return rows


def _mean_vector(rows: List[Dict[str, float]], feature_keys: List[str]) -> List[float]:
    if not feature_keys:
        return []
    sums = [0.0] * len(feature_keys)
    for row in rows:
        for i, key in enumerate(feature_keys):
            sums[i] += row.get(key, 0.0)
    count = max(len(rows), 1)
    return [s / count for s in sums]


def _mean_of_vectors(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    sums = [0.0] * dim
    for vec in vectors:
        for i in range(dim):
            sums[i] += vec[i]
    count = len(vectors)
    return [s / count for s in sums]


def _drift_stats(current: List[float], ref: List[float]) -> tuple[float, float]:
    if not current or not ref:
        return 0.0, 0.0
    diffs = [abs(c - r) for c, r in zip(current, ref)]
    drift_core = sum(diffs) / len(diffs)
    drift_max = max(diffs)
    return drift_core, drift_max


def run_monitor(
    events_path: Path,
    csv_path: Path,
    output_path: Path,
    window_size: int = 10,
    ref_window: int = 3,
    dlq_path: Path = Path("artifacts/dlq.jsonl"),
) -> int:
    batch_events = _load_batch_events(events_path, dlq_path)
    rows = _load_numeric_rows(csv_path)

    feature_keys: List[str] = []
    if rows:
        feature_keys = [k for k in rows[0].keys() if k not in EXCLUDED_COLUMNS]

    num_batches = (len(rows) + window_size - 1) // window_size
    batch_means: List[List[float]] = []
    for i in range(num_batches):
        start = i * window_size
        end = min(start + window_size, len(rows))
        batch_rows = rows[start:end]
        batch_means.append(_mean_vector(batch_rows, feature_keys))

    state_ids = IdCounter("s_", start=1)
    prior_means: List[List[float]] = []

    for i, batch in enumerate(batch_events):
        current_mean = batch_means[i] if i < len(batch_means) else []
        has_features = len(current_mean) > 0

        if len(prior_means) < ref_window or not has_features:
            drift_core = 0.0
            drift_max = 0.0
            drift_ready = False
        else:
            ref = _mean_of_vectors(prior_means[-ref_window:])
            drift_core, drift_max = _drift_stats(current_mean, ref)
            drift_ready = True

        state_event = StateEvent(
            state_id=state_ids.next(),
            batch_id=batch["batch_id"],
            window_id=batch["window_id"],
            timestamp=float(batch["timestamp_end"]),
            model_version=batch["model_version"],
            label_available=bool(batch["label_available"]),
            drift_core=drift_core,
            drift_max=drift_max,
            drift_ready=drift_ready,
            perf_core=0.0,
            fairness_core=0.0,
        )
        validated_state = validate_or_dlq(state_event.dict(), StateEvent, dlq_path)
        if validated_state:
            write_event(validated_state, output_path, event_type="state")

        prior_means.append(current_mean)

    return len(batch_events)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor JSONL events and emit StateEvents")
    parser.add_argument("--events", required=True, help="Path to input events.jsonl")
    parser.add_argument("--csv", required=True, help="Path to source CSV")
    parser.add_argument("--output", default="artifacts/events.jsonl")
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--ref-window", type=int, default=3)
    parser.add_argument("--dlq", default="artifacts/dlq.jsonl")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_monitor(
        events_path=Path(args.events),
        csv_path=Path(args.csv),
        output_path=Path(args.output),
        window_size=args.window_size,
        ref_window=args.ref_window,
        dlq_path=Path(args.dlq),
    )


if __name__ == "__main__":
    main()
