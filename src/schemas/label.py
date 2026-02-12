from src.schemas.base import BaseEvent
from src.schemas.compat import StrictFloat, StrictStr


class LabelEvent(BaseEvent):
    label_id: StrictStr
    batch_id: StrictStr
    timestamp: StrictFloat
