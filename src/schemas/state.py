from src.schemas.base import BaseEvent
from src.schemas.compat import StrictBool, StrictFloat, StrictInt, StrictStr


class StateEvent(BaseEvent):
    """
    StateEvent.state_id must be unique per state.
    """
    state_id: StrictStr
    batch_id: StrictStr
    window_id: StrictStr
    timestamp: StrictFloat
    model_version: StrictStr
    label_available: StrictBool
    drift_core: StrictFloat
    drift_max: StrictFloat
    drift_ready: StrictBool
    perf_core: StrictFloat
    fairness_core: StrictFloat
    fairness_ready: StrictBool = False
    fairness_support: StrictInt = 0
