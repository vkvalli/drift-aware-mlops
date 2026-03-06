import argparse
import csv
import json
import math
from collections import Counter
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


def _load_numeric_rows(
    csv_path: Path, label_column: str, target_column: str, protected_attr: str
) -> List[Dict[str, float]]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, float]] = []
        for row in reader:
            cleaned: Dict[str, float] = {}
            for key, value in row.items():
                if key in EXCLUDED_COLUMNS or key in {
                    label_column,
                    target_column,
                    protected_attr,
                }:
                    continue
                try:
                    cleaned[key] = float(value)
                except (TypeError, ValueError):
                    continue
            rows.append(cleaned)
    return rows


def _load_text_rows(
    csv_path: Path, label_column: str, target_column: str, protected_attr: str
) -> List[str]:
    rows: List[str] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            preferred = ["title", "description", "text", "headline", "content"]
            parts: List[str] = []
            for key in preferred:
                val = row.get(key)
                if val:
                    parts.append(str(val).strip())
            if not parts:
                for key, value in row.items():
                    if key in {label_column, target_column, protected_attr}:
                        continue
                    if value is None:
                        continue
                    txt = str(value).strip()
                    if txt:
                        parts.append(txt)
            rows.append(" ".join(parts))
    return rows


def _text_summary(texts: List[str]) -> tuple[float, float]:
    if not texts:
        return 0.0, 0.0
    token_counts: Counter[str] = Counter()
    lengths: List[int] = []
    for text in texts:
        toks = [t for t in text.lower().split() if t]
        lengths.append(len(toks))
        token_counts.update(toks)
    total_tokens = sum(token_counts.values())
    if total_tokens <= 0:
        return float(sum(lengths) / len(lengths)), 0.0
    entropy = 0.0
    for count in token_counts.values():
        p = count / total_tokens
        entropy -= p * math.log(p + 1e-12)
    return float(sum(lengths) / len(lengths)), float(entropy)


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
    task: str = "classification",
    label_column: str = "class",
    target_column: str = "class",
    protected_attr: str = "",
    dlq_path: Path = Path("artifacts/dlq.jsonl"),
) -> int:
    batch_events = _load_batch_events(events_path, dlq_path)
    rows = _load_numeric_rows(
        csv_path,
        label_column=label_column,
        target_column=target_column,
        protected_attr=protected_attr,
    )
    text_rows = _load_text_rows(
        csv_path,
        label_column=label_column,
        target_column=target_column,
        protected_attr=protected_attr,
    )

    feature_keys: List[str] = []
    if rows:
        feature_keys = [k for k in rows[0].keys() if k not in EXCLUDED_COLUMNS]

    num_batches = (max(len(rows), len(text_rows)) + window_size - 1) // window_size
    batch_means: List[List[float]] = []
    text_stats: List[tuple[float, float]] = []
    for i in range(num_batches):
        start = i * window_size
        end = min(start + window_size, len(rows))
        batch_rows = rows[start:end]
        batch_means.append(_mean_vector(batch_rows, feature_keys))
        text_start = i * window_size
        text_end = min(text_start + window_size, len(text_rows))
        text_stats.append(_text_summary(text_rows[text_start:text_end]))

    state_ids = IdCounter("s_", start=1)
    prior_means: List[List[float]] = []

    for i, batch in enumerate(batch_events):
        current_mean = batch_means[i] if i < len(batch_means) else []
        has_features = len(current_mean) > 0

        doc_len_mean = text_stats[i][0] if i < len(text_stats) else 0.0
        vocab_entropy = text_stats[i][1] if i < len(text_stats) else 0.0
        if len(prior_means) < ref_window or (
            not has_features and task != "text_classification"
        ):
            drift_core = 0.0
            drift_max = 0.0
            drift_ready = False
        else:
            if task == "text_classification" and not has_features:
                ref_docs = [text_stats[j][0] for j in range(max(0, i - ref_window), i)]
                ref_entropy = [
                    text_stats[j][1] for j in range(max(0, i - ref_window), i)
                ]
                if ref_docs and ref_entropy:
                    doc_diff = abs(doc_len_mean - (sum(ref_docs) / len(ref_docs)))
                    ent_diff = abs(
                        vocab_entropy - (sum(ref_entropy) / len(ref_entropy))
                    )
                    # For text batches, use both summary signals so drift_core is sensitive
                    # even when average document length is stable.
                    drift_core = (doc_diff + ent_diff) / 2.0
                    drift_max = max(doc_diff, ent_diff)
                else:
                    drift_core = 0.0
                    drift_max = 0.0
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
            rec = validated_state.dict()
            if task == "text_classification":
                rec["doc_len_mean"] = doc_len_mean
                rec["vocab_entropy_proxy"] = vocab_entropy
            write_event(rec, output_path, event_type="state")

        prior_means.append(current_mean)

    return len(batch_events)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Monitor JSONL events and emit StateEvents"
    )
    parser.add_argument("--events", required=True, help="Path to input events.jsonl")
    parser.add_argument("--csv", required=True, help="Path to source CSV")
    parser.add_argument("--output", default="artifacts/events.jsonl")
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--ref-window", type=int, default=3)
    parser.add_argument(
        "--task",
        choices=["classification", "regression", "text_classification"],
        default="classification",
    )
    parser.add_argument("--label-column", default="class")
    parser.add_argument("--target-column", default="class")
    parser.add_argument("--protected-attr", default="")
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
        task=args.task,
        label_column=args.label_column,
        target_column=args.target_column,
        protected_attr=args.protected_attr,
        dlq_path=Path(args.dlq),
    )


if __name__ == "__main__":
    main()
