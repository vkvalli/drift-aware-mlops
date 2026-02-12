import argparse
import csv
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.logger.jsonl_logger import write_event
from src.schemas.action import ActionEvent
from src.schemas.cost import CostEvent
from src.schemas.model_update import ModelUpdateEvent
from src.schemas.validate import validate_or_dlq
from src.trainer.compat import SKLEARN_AVAILABLE, create_model
from src.utils.ids import IdCounter


EXCLUDED_COLUMNS = {"Date", "Day", "Period", "date", "day", "period"}
LABEL_COLUMN = "class"


def _load_actions(path: Path, dlq_path: Path) -> List[Dict]:
    records: List[Dict] = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        event_type = record.get("event_type", "")
        if event_type in {"action", "ActionEvent"}:
            validated = validate_or_dlq(record, ActionEvent, dlq_path)
            if validated:
                records.append(validated.dict())
    return records


def _load_states(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        return {}
    states: Dict[str, Dict] = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record.get("event_type") in {"state", "StateEvent"}:
            state_id = record.get("state_id")
            if state_id:
                states[state_id] = record
    return states


def _load_csv_rows(csv_path: Path) -> Tuple[List[List[float]], List[int]]:
    rows_x: List[List[float]] = []
    rows_y: List[int] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get(LABEL_COLUMN)
            if label is None:
                continue
            y_val = 1 if str(label).upper() == "UP" else 0

            features: List[float] = []
            for key, value in row.items():
                if key == LABEL_COLUMN or key in EXCLUDED_COLUMNS:
                    continue
                try:
                    features.append(float(value))
                except (TypeError, ValueError):
                    continue
            if not features:
                continue
            rows_x.append(features)
            rows_y.append(y_val)
    return rows_x, rows_y


def _batch_slice(batch_id: str, window_size: int) -> slice:
    try:
        idx = int(batch_id.split("_")[-1])
    except Exception:
        idx = 1
    start = (idx - 1) * window_size
    end = start + window_size
    return slice(start, end)


def _load_model(path: Path):
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def _save_model(path: Path, model) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(model, f)


def _load_processed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    processed: set[str] = set()
    for line in path.read_text().splitlines():
        record = json.loads(line)
        action_id = record.get("action_id")
        if action_id:
            processed.add(action_id)
    return processed


def _append_processed(path: Path, action_id: str, outcome: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "action_id": action_id,
        "processed_at": time.time(),
        "outcome": outcome,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def run_trainer(
    actions_path: Path,
    events_path: Path,
    processed_index_path: Path,
    seed: Optional[int] = None,
    dlq_path: Path = Path("artifacts/dlq.jsonl"),
    states_path: Optional[Path] = None,
    csv_path: Optional[Path] = None,
    window_size: int = 10,
) -> int:
    actions = _load_actions(actions_path, dlq_path)
    processed = _load_processed(processed_index_path)
    states = _load_states(states_path) if states_path else {}
    rows_x: List[List[float]] = []
    rows_y: List[int] = []
    if csv_path:
        rows_x, rows_y = _load_csv_rows(csv_path)

    start = 1 if seed is None else max(1, int(seed))
    cost_ids = IdCounter("c_", start=start)
    update_ids = IdCounter("u_", start=start)
    model_ids = IdCounter("m_", start=start)

    processed_count = 0
    current_model_version: Optional[str] = None

    for action in actions:
        if current_model_version is None:
            current_model_version = action.get("model_version", "m_000000")

        action_id = action["action_id"]
        if action_id in processed:
            continue

        action_type = action.get("action_type", "defer")
        executed = action_type == "fine_tune"
        reason_code = "ok" if executed else "defer"
        state = states.get(action.get("state_id", ""))
        label_available = bool(state.get("label_available", True)) if state else True
        if executed and not label_available:
            executed = False
            reason_code = "insufficient_data"

        accuracy_pre = None
        X_batch: List[List[float]] = []
        y_batch: List[int] = []
        model = None

        if csv_path and rows_x and label_available:
            batch_id = state.get("batch_id") if state else None
            if batch_id:
                sl = _batch_slice(batch_id, window_size)
                X_batch = rows_x[sl]
                y_batch = rows_y[sl]
            else:
                X_batch = rows_x
                y_batch = rows_y

            model_path = Path(f"artifacts/models/{current_model_version or 'm_000000'}.pkl")
            model = _load_model(model_path)
            if model is None:
                model = create_model(seed)

            if model is not None and hasattr(model, "classes_") and X_batch:
                preds = model.predict(X_batch)
                correct = sum(int(p == y) for p, y in zip(preds, y_batch))
                accuracy_pre = correct / max(len(y_batch), 1)

        if executed:
            cost = CostEvent(
                cost_id=cost_ids.next(),
                action_id=action_id,
                timestamp=float(action.get("timestamp", time.time())),
                train_time_s=3.0,
                estimated_usd=1.50,
                energy_kwh=0.02,
                co2_kg=0.0005,
                executed=True,
                reason_code=reason_code,
                accuracy_pre=accuracy_pre,
            )
        else:
            cost = CostEvent(
                cost_id=cost_ids.next(),
                action_id=action_id,
                timestamp=float(action.get("timestamp", time.time())),
                train_time_s=0.0,
                estimated_usd=0.0,
                energy_kwh=0.0,
                co2_kg=0.0,
                executed=False,
                reason_code=reason_code,
                accuracy_pre=accuracy_pre,
            )

        validated_cost = validate_or_dlq(cost.dict(), CostEvent, dlq_path)
        if validated_cost:
            write_event(validated_cost, events_path, event_type="cost")

        if executed:
            new_model_version = model_ids.next()
            artifact_path = f"artifacts/models/{new_model_version}.pkl"
            eval_summary: Dict[str, float] = {}

            if SKLEARN_AVAILABLE and csv_path and rows_x and label_available:
                if model is None:
                    model = create_model(seed)

                if model is not None and X_batch:
                    if accuracy_pre is not None:
                        eval_summary = {"accuracy": float(accuracy_pre), "n_samples": len(y_batch)}
                    else:
                        eval_summary = {"n_samples": len(y_batch)}

                    if not hasattr(model, "classes_"):
                        model.partial_fit(X_batch, y_batch, classes=[0, 1])
                    else:
                        model.partial_fit(X_batch, y_batch)

                    _save_model(Path(artifact_path), model)
            else:
                eval_summary = {"sklearn_available": False}

            update = ModelUpdateEvent(
                update_id=update_ids.next(),
                action_id=action_id,
                timestamp=float(action.get("timestamp", time.time())),
                new_model_version=new_model_version,
                artifact_path=artifact_path,
                eval_summary=eval_summary,
                executed=True,
            )
            validated_update = validate_or_dlq(update.dict(), ModelUpdateEvent, dlq_path)
            if validated_update:
                write_event(validated_update, events_path, event_type="model_update")
            current_model_version = new_model_version

        _append_processed(processed_index_path, action_id, reason_code)
        processed.add(action_id)
        processed_count += 1

    return processed_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline trainer stub")
    parser.add_argument("--actions-file", default="artifacts/actions.jsonl")
    parser.add_argument("--events-file", default="artifacts/trainer_output.jsonl")
    parser.add_argument("--processed-index-file", default="artifacts/processed_actions.jsonl")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--states-file", default=None)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--dlq", default="artifacts/dlq.jsonl")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_trainer(
        actions_path=Path(args.actions_file),
        events_path=Path(args.events_file),
        processed_index_path=Path(args.processed_index_file),
        seed=args.seed,
        dlq_path=Path(args.dlq),
        states_path=Path(args.states_file) if args.states_file else None,
        csv_path=Path(args.csv) if args.csv else None,
        window_size=args.window_size,
    )


if __name__ == "__main__":
    main()
