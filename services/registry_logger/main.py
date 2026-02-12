import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

from src.bus.file_bus import FileBus
from services.registry_logger.canonical_id import canonical_id
from services.registry_logger.validate import expected_schema, validate_event


def _load_checkpoint(path: Path, source_file: Optional[str]) -> int:
    if not path.exists():
        return 0
    data = json.loads(path.read_text())
    last_offset = int(data.get("last_byte_offset", 0))
    if source_file:
        if data.get("source_file") and data.get("source_file") != source_file:
            print("warning: checkpoint source_file differs from current input")
        if "source_size_bytes" in data:
            try:
                current_size = Path(source_file).stat().st_size
                if current_size != int(data.get("source_size_bytes", current_size)):
                    print("warning: checkpoint source_size_bytes differs from current input size")
            except Exception:
                pass
    return last_offset


def _save_checkpoint(path: Path, offset: int, source_file: Optional[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"last_byte_offset": offset}
    if source_file:
        data["source_file"] = source_file
        try:
            data["source_size_bytes"] = Path(source_file).stat().st_size
        except Exception:
            data["source_size_bytes"] = None
    path.write_text(json.dumps(data))


def _load_seen_ids(canonical_path: Path, seen_index: Optional[Path] = None) -> set[str]:
    seen: set[str] = set()
    if seen_index and seen_index.exists():
        for line in seen_index.read_text().splitlines():
            if line.strip():
                seen.add(line.strip())
        return seen

    if not canonical_path.exists():
        return seen
    if canonical_path.stat().st_size > 200 * 1024 * 1024:
        print("warning: canonical log > 200MB; consider --seen-index for faster startup")
    for line in canonical_path.read_text().splitlines():
        try:
            record = json.loads(line)
            cid = canonical_id(record)
            seen.add(cid)
        except Exception:
            continue
    return seen


def _write_dlq(
    dlq_path: Path,
    error_type: str,
    error_msg: str,
    raw_event: Dict,
    source_offset: int,
    expected: str,
) -> None:
    dlq_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "error_type": error_type,
        "error_msg": error_msg,
        "raw_event": raw_event,
        "source_offset": source_offset,
        "expected_schema": expected,
        "logged_at": time.time(),
    }
    with dlq_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def run_logger(
    input_path: Path,
    canonical_path: Path,
    checkpoint_path: Path,
    dlq_path: Path,
    source_file: Optional[str] = None,
    seen_index: Optional[Path] = None,
) -> int:
    bus = FileBus(input_path)
    start_offset = _load_checkpoint(checkpoint_path, source_file)
    seen_ids = _load_seen_ids(canonical_path, seen_index=seen_index)

    processed = 0
    for event, offset in bus.read(start_offset=start_offset):
        try:
            validate_event(event)
            cid = canonical_id(event)
        except Exception as e:
            _write_dlq(
                dlq_path,
                type(e).__name__,
                str(e),
                event,
                offset,
                expected_schema(str(event.get("event_type"))),
            )
            _save_checkpoint(checkpoint_path, offset, source_file)
            continue

        if cid in seen_ids:
            _save_checkpoint(checkpoint_path, offset, source_file)
            continue

        record = dict(event)
        if "logged_at" not in record:
            record["logged_at"] = time.time()
        record["source_offset"] = offset
        if source_file:
            record["source_file"] = source_file

        with canonical_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        seen_ids.add(cid)
        if seen_index:
            seen_index.parent.mkdir(parents=True, exist_ok=True)
            with seen_index.open("a", encoding="utf-8") as sf:
                sf.write(cid + "\n")
        processed += 1
        _save_checkpoint(checkpoint_path, offset, source_file)

    return processed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Registry logger (FileBus)")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--canonical", default="artifacts/events.jsonl")
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/registry_logger.json")
    parser.add_argument("--dlq", default="artifacts/dlq.jsonl")
    parser.add_argument("--seen-index", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_logger(
        input_path=Path(args.input),
        canonical_path=Path(args.canonical),
        checkpoint_path=Path(args.checkpoint),
        dlq_path=Path(args.dlq),
        source_file=str(args.input),
        seen_index=Path(args.seen_index) if args.seen_index else None,
    )


if __name__ == "__main__":
    main()
