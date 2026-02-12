import json
import time
from pathlib import Path
from typing import Any, Optional

from src.schemas.compat import BaseModelLike


DEFAULT_PATH = Path("artifacts/events.jsonl")


def _to_record(event: Any) -> dict:
    if isinstance(event, BaseModelLike):
        if hasattr(event, "model_dump"):
            return event.model_dump()
        return event.dict()
    if isinstance(event, dict):
        return dict(event)
    raise TypeError("write_event expects a Pydantic model or dict")


def write_event(event: Any, path: Path = DEFAULT_PATH, event_type: Optional[str] = None) -> None:
    record = _to_record(event)

    path.parent.mkdir(parents=True, exist_ok=True)

    record["event_type"] = event_type or record.get("event_type") or event.__class__.__name__
    record.setdefault("logged_at", time.time())

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
