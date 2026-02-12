from typing import Any, Dict

from src.schemas.base import BaseEvent
from src.schemas.compat import StrictBool, StrictFloat, StrictStr


class ModelUpdateEvent(BaseEvent):
    update_id: StrictStr
    action_id: StrictStr
    timestamp: StrictFloat
    new_model_version: StrictStr
    artifact_path: StrictStr
    eval_summary: Dict[str, Any]
    executed: StrictBool
