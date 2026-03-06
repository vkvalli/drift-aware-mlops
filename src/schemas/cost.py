from typing import Optional

from src.schemas.base import BaseEvent
from src.schemas.compat import StrictBool, StrictFloat, StrictInt, StrictStr


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
    balanced_accuracy_pre: Optional[StrictFloat] = None
    f1_pre: Optional[StrictFloat] = None
    mae_pre: Optional[StrictFloat] = None
    rmse_pre: Optional[StrictFloat] = None
    r2_pre: Optional[StrictFloat] = None
    precision_pre: Optional[StrictFloat] = None
    recall_pre: Optional[StrictFloat] = None
    dpd_pre: Optional[StrictFloat] = None
    eod_pre: Optional[StrictFloat] = None
    tp_pre: Optional[StrictInt] = None
    fp_pre: Optional[StrictInt] = None
    tn_pre: Optional[StrictInt] = None
    fn_pre: Optional[StrictInt] = None
    shadow_accuracy_pre: Optional[StrictFloat] = None
    shadow_accuracy_post: Optional[StrictFloat] = None
    delta_shadow: Optional[StrictFloat] = None
