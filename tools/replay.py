import argparse
import json
import time
from pathlib import Path
from typing import Optional, Tuple

from src.bus.file_bus import FileBus
from services.registry_logger.canonical_id import canonical_id


def _load_checkpoint_offset(checkpoint_path: Path) -> int:
    if not checkpoint_path.exists():
        return 0
    try:
        data = json.loads(checkpoint_path.read_text())
    except Exception:
        return 0
    return int(data.get("last_byte_offset", 0) or 0)


def _iter_events_with_offsets(source_path: Path, start_offset: int) -> Tuple[int, dict]:
    with source_path.open("rb") as f:
        if start_offset > 0:
            f.seek(start_offset)
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            try:
                event = json.loads(line.decode("utf-8"))
            except Exception:
                continue
            yield offset, event


def replay(
    source_path: Path,
    out_path: Path,
    from_id: Optional[str],
    to_id: Optional[str],
    rate: Optional[float],
    deterministic: bool,
    checkpoint_path: Optional[Path],
) -> int:
    bus = FileBus(out_path)
    count = 0
    # deterministic replay is file order (byte offset order)
    start_offset = _load_checkpoint_offset(checkpoint_path) if checkpoint_path else 0

    started = from_id is None
    for _, event in _iter_events_with_offsets(source_path, start_offset):
        try:
            cid = canonical_id(event)
        except Exception:
            continue

        if not started:
            if cid == from_id:
                started = True
            else:
                continue

        bus.publish(event)
        count += 1

        if to_id and cid == to_id:
            break

        if rate:
            time.sleep(rate)

    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay canonical log into a bus")
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--from-id", default=None, help="canonical id like state:s_000123")
    parser.add_argument("--to-id", default=None, help="canonical id like state:s_000123")
    parser.add_argument("--checkpoint", default=None, help="registry logger checkpoint json")
    parser.add_argument("--rate", type=float, default=None)
    parser.add_argument("--deterministic", action="store_true", help="replay in file order")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    replay(
        source_path=Path(args.source),
        out_path=Path(args.out),
        from_id=args.from_id,
        to_id=args.to_id,
        rate=args.rate,
        deterministic=args.deterministic,
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
    )


if __name__ == "__main__":
    main()
