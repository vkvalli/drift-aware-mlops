from typing import Optional

from src.schemas.base import BaseEvent
from src.schemas.compat import StrictBool, StrictFloat, StrictStr


class CostEvent(BaseEvent):
    cost_id: StrictStr
    action_id: StrictStr
    timestamp: StrictFloat
    train_time_s: StrictFloat
    estimated_usd: StrictFloat
    energy_kwh: StrictFloat
    co2_kg: StrictFloat
    executed: StrictBool
    reason_code: StrictStr
    accuracy_pre: Optional[StrictFloat] = None
    shadow_accuracy_pre: Optional[StrictFloat] = None
    shadow_accuracy_post: Optional[StrictFloat] = None
    delta_shadow: Optional[StrictFloat] = None
