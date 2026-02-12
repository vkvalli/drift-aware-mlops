from typing import Dict

from src.schemas.compat import parse_obj_as_event
from src.schemas.validate import EVENT_MODEL_MAP, expected_schema_for_event_type


def expected_schema(event_type: str) -> str:
    return expected_schema_for_event_type(event_type)


def validate_event(event: Dict) -> None:
    event_type = event.get("event_type")
    model = EVENT_MODEL_MAP.get(event_type)
    if model is None:
        raise ValueError("unknown event_type")
    try:
        parse_obj_as_event(model, event)
    except Exception as e:
        raise ValueError(str(e)) from e
