import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Type

from src.schemas.action import ActionEvent
from src.schemas.batch import BatchEvent
from src.schemas.compat import parse_obj_as_event
from src.schemas.cost import CostEvent
from src.schemas.label import LabelEvent
from src.schemas.model_update import ModelUpdateEvent
from src.schemas.state import StateEvent


EVENT_MODEL_MAP = {
    "batch": BatchEvent,
    "BatchEvent": BatchEvent,
    "state": StateEvent,
    "StateEvent": StateEvent,
    "action": ActionEvent,
    "ActionEvent": ActionEvent,
    "cost": CostEvent,
    "CostEvent": CostEvent,
    "model_update": ModelUpdateEvent,
    "ModelUpdateEvent": ModelUpdateEvent,
    "label": LabelEvent,
    "LabelEvent": LabelEvent,
}


SCHEMA_NAME_BY_MODEL: Dict[Type, str] = {
    BatchEvent: "BatchEvent:v1",
    StateEvent: "StateEvent:v1",
    ActionEvent: "ActionEvent:v1",
    CostEvent: "CostEvent:v1",
    ModelUpdateEvent: "ModelUpdateEvent:v1",
    LabelEvent: "LabelEvent:v1",
}


def expected_schema_for_model(model_cls: Type) -> str:
    return SCHEMA_NAME_BY_MODEL.get(model_cls, "unknown")


def expected_schema_for_event_type(event_type: str) -> str:
    model = EVENT_MODEL_MAP.get(event_type, None)
    if model is None:
        return "unknown"
    return expected_schema_for_model(model)


def validate_or_dlq(
    event: Dict[str, Any],
    model_cls: Type,
    dlq_path: Path,
    expected_schema: Optional[str] = None,
    source_offset: Optional[int] = None,
) -> Optional[Any]:
    try:
        return parse_obj_as_event(model_cls, event)
    except Exception as e:
        _write_dlq(
            dlq_path=dlq_path,
            error_type="ValidationError",
            error_msg=str(e),
            raw_event=event,
            expected_schema=expected_schema or expected_schema_for_model(model_cls),
            source_offset=source_offset,
        )
        return None


def _write_dlq(
    dlq_path: Path,
    error_type: str,
    error_msg: str,
    raw_event: Dict[str, Any],
    expected_schema: str,
    source_offset: Optional[int],
) -> None:
    dlq_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "error_type": error_type,
        "error_msg": error_msg,
        "raw_event": raw_event,
        "expected_schema": expected_schema,
        "source_offset": source_offset,
        "logged_at": time.time(),
    }
    with dlq_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
