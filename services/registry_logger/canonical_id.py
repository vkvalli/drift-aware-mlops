from typing import Dict


def canonical_id(event: Dict) -> str:
    event_type = event.get("event_type")
    if event_type in {"BatchEvent", "batch"}:
        key = event.get("batch_id")
        if not key:
            raise ValueError("missing batch_id")
        return f"batch:{key}"
    if event_type in {"LabelEvent", "label"}:
        key = event.get("label_id")
        if not key:
            raise ValueError("missing label_id")
        return f"label:{key}"
    if event_type in {"StateEvent", "state"}:
        key = event.get("state_id")
        if not key:
            raise ValueError("missing state_id")
        return f"state:{key}"
    if event_type in {"ActionEvent", "action"}:
        key = event.get("action_id")
        if not key:
            raise ValueError("missing action_id")
        return f"action:{key}"
    if event_type in {"CostEvent", "cost"}:
        key = event.get("cost_id")
        if not key:
            raise ValueError("missing cost_id")
        return f"cost:{key}"
    if event_type in {"ModelUpdateEvent", "model_update"}:
        key = event.get("update_id")
        if not key:
            raise ValueError("missing update_id")
        return f"model_update:{key}"

    raise ValueError("unknown event_type")
