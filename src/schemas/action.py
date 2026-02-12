from typing import Any, Dict

from src.schemas.base import BaseEvent
from src.schemas.compat import StrictFloat, StrictStr


class ActionEvent(BaseEvent):
    """
    ActionEvent.state_id references exactly one state.
    At most one action per state is a policy/consumer rule.
    """
    action_id: StrictStr
    state_id: StrictStr
    action_type: StrictStr
    model_version: StrictStr
    timestamp: StrictFloat
    reason: StrictStr
    action_params: Dict[str, Any]
    requested_at: StrictFloat
