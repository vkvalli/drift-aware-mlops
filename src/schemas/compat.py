from __future__ import annotations

from typing import Any, Dict, Tuple, get_args, get_origin, Union

try:
    from pydantic import BaseModel as PydanticBaseModel
    from pydantic import ValidationError as PydanticValidationError
    from pydantic import StrictBool, StrictFloat, StrictInt, StrictStr

    PYDANTIC_AVAILABLE = True
except Exception:
    PydanticBaseModel = None
    PydanticValidationError = Exception
    PYDANTIC_AVAILABLE = False
    StrictBool = bool
    StrictFloat = float
    StrictInt = int
    StrictStr = str


if PYDANTIC_AVAILABLE and PydanticBaseModel is not None:
    BaseModelLike = PydanticBaseModel
else:
    class BaseModelLike:
        def __init__(self, **data: Any) -> None:
            defaults = _collect_defaults(self.__class__)
            merged = {**defaults, **data}
            for key, value in merged.items():
                setattr(self, key, value)

        def dict(self) -> Dict[str, Any]:
            return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

        def model_dump(self) -> Dict[str, Any]:
            return self.dict()


def _collect_annotations(cls: type) -> Dict[str, Any]:
    annotations: Dict[str, Any] = {}
    for base in reversed(cls.__mro__):
        annotations.update(getattr(base, "__annotations__", {}))
    return annotations


def _collect_defaults(cls: type) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}
    for base in reversed(cls.__mro__):
        for key, value in base.__dict__.items():
            if key.startswith("_"):
                continue
            if callable(value):
                continue
            if isinstance(value, type):
                continue
            defaults[key] = value
    return defaults


def _is_allowed_type(value: Any, expected_type: Any) -> bool:
    if expected_type is Any:
        return True

    origin = get_origin(expected_type)
    if origin is Union:
        args = get_args(expected_type)
        if value is None:
            return type(None) in args
        return any(_is_allowed_type(value, a) for a in args if a is not type(None))

    if expected_type is bool:
        return type(value) is bool
    if expected_type is int:
        return type(value) is int
    if expected_type is float:
        return type(value) in (int, float)
    if expected_type is str:
        return isinstance(value, str)
    if expected_type is dict:
        return isinstance(value, dict)
    if expected_type is list:
        return isinstance(value, list)

    return True


def _validate_payload(model_cls: type, payload: Dict[str, Any]) -> Tuple[bool, str]:
    annotations = _collect_annotations(model_cls)
    defaults = _collect_defaults(model_cls)
    for field, field_type in annotations.items():
        if field not in payload:
            if field in defaults:
                continue
            return False, f"missing required field: {field}"
        value = payload.get(field)
        if value is None:
            if field in defaults:
                continue
            return False, f"field {field} is None"
        if not _is_allowed_type(value, field_type):
            return False, f"type mismatch for {field}"
    return True, ""


def parse_obj_as_event(model_cls: type, payload: Dict[str, Any]) -> Any:
    if PYDANTIC_AVAILABLE and PydanticBaseModel and issubclass(model_cls, PydanticBaseModel):
        return model_cls(**payload)

    ok, msg = _validate_payload(model_cls, payload)
    if not ok:
        raise ValueError(msg)
    return model_cls(**payload)


__all__ = [
    "BaseModelLike",
    "PYDANTIC_AVAILABLE",
    "PydanticBaseModel",
    "PydanticValidationError",
    "parse_obj_as_event",
    "StrictBool",
    "StrictFloat",
    "StrictInt",
    "StrictStr",
]
