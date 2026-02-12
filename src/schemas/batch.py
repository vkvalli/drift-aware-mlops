from src.schemas.base import BaseEvent
from src.schemas.compat import StrictBool, StrictFloat, StrictInt, StrictStr


class BatchEvent(BaseEvent):
    batch_id: StrictStr
    window_id: StrictStr
    timestamp_start: StrictFloat
    timestamp_end: StrictFloat
    label_available: StrictBool
    label_delay_batches: StrictInt
    dataset_id: StrictStr
    model_version: StrictStr
