import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

from src.logger.jsonl_logger import write_event
from src.schemas.batch import BatchEvent
from src.schemas.label import LabelEvent
from src.schemas.validate import validate_or_dlq
from src.utils.ids import IdCounter


def _count_rows(csv_path: Path, max_rows: Optional[int]) -> int:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return 0
        if max_rows is None:
            return sum(1 for _ in reader)
        count = 0
        for _ in reader:
            count += 1
            if count >= max_rows:
                break
        return count


def run_replay(
    csv_path: Path,
    window_size: int = 10,
    label_delay_batches: int = 2,
    flush_pending_labels: bool = True,
    dataset_id: str = "electricity",
    model_version: str = "m_000001",
    output_path: Path = Path("artifacts/events.jsonl"),
    max_rows: Optional[int] = None,
    dlq_path: Path = Path("artifacts/dlq.jsonl"),
) -> int:
    num_rows = _count_rows(csv_path, max_rows)
    if num_rows == 0:
        return 0

    num_batches = (num_rows + window_size - 1) // window_size

    batch_ids = IdCounter("b_", start=1)
    label_ids = IdCounter("l_", start=1)

    scheduled: Dict[int, List[str]] = {}

    for i in range(num_batches):
        start = i * window_size
        end = min(start + window_size, num_rows)

        batch_id = batch_ids.next()
        label_available = i >= label_delay_batches

        batch_event = BatchEvent(
            batch_id=batch_id,
            window_id=f"w_{i + 1:06d}",
            timestamp_start=float(start),
            timestamp_end=float(end),
            label_available=label_available,
            label_delay_batches=label_delay_batches,
            dataset_id=dataset_id,
            model_version=model_version,
        )
        validated_batch = validate_or_dlq(batch_event.dict(), BatchEvent, dlq_path)
        if validated_batch:
            write_event(validated_batch, output_path)

        due_index = i + label_delay_batches
        scheduled.setdefault(due_index, []).append(batch_id)

        if i in scheduled:
            for target_batch_id in scheduled.pop(i):
                label_event = LabelEvent(
                    label_id=label_ids.next(),
                    batch_id=target_batch_id,
                    timestamp=float(i),
                )
                validated_label = validate_or_dlq(label_event.dict(), LabelEvent, dlq_path)
                if validated_label:
                    write_event(validated_label, output_path)

    if flush_pending_labels:
        for due_index in sorted(scheduled.keys()):
            for target_batch_id in scheduled[due_index]:
                label_event = LabelEvent(
                    label_id=label_ids.next(),
                    batch_id=target_batch_id,
                    timestamp=float(due_index),
                )
                validated_label = validate_or_dlq(label_event.dict(), LabelEvent, dlq_path)
                if validated_label:
                    write_event(validated_label, output_path)

    return num_batches


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a CSV into BatchEvent windows")
    parser.add_argument("--csv", required=True, help="Path to input CSV")
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--label-delay-batches", type=int, default=2)
    parser.add_argument("--flush-pending-labels", type=str, default="true")
    parser.add_argument("--dataset-id", type=str, default="electricity")
    parser.add_argument("--model-version", type=str, default="m_000001")
    parser.add_argument("--output", type=str, default="artifacts/events.jsonl")
    parser.add_argument("--dlq", type=str, default="artifacts/dlq.jsonl")
    parser.add_argument("--max-rows", type=int, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_path = Path(args.output)
    csv_path = Path(args.csv)
    flush = str(args.flush_pending_labels).lower() in {"true", "1", "yes"}

    run_replay(
        csv_path=csv_path,
        window_size=args.window_size,
        label_delay_batches=args.label_delay_batches,
        flush_pending_labels=flush,
        dataset_id=args.dataset_id,
        model_version=args.model_version,
        output_path=output_path,
        max_rows=args.max_rows,
        dlq_path=Path(args.dlq),
    )


if __name__ == "__main__":
    main()
