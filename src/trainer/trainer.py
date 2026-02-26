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


def _batch_slice_from_id(batch_id: str, window_size: int) -> slice:
    return _batch_slice(batch_id, window_size)


def _filter_shadow_rows_from_slice(
    x_rows: List[List[float]],
    y_rows: List[int],
    start_idx: int,
    window_size: int,
    shadow_batch_ids: set[str],
) -> Tuple[List[List[float]], List[int]]:
    if not shadow_batch_ids:
        return x_rows, y_rows
    kept_x: List[List[float]] = []
    kept_y: List[int] = []
    for offset, (xv, yv) in enumerate(zip(x_rows, y_rows)):
        global_idx = start_idx + offset
        batch_num = (global_idx // window_size) + 1
        batch_id = f"b_{batch_num:06d}"
        if batch_id in shadow_batch_ids:
            continue
        kept_x.append(xv)
        kept_y.append(yv)
    return kept_x, kept_y


def _predict_accuracy(model, X: List[List[float]], y: List[int]) -> Optional[float]:
    if model is None or not X or not y:
        return None
    if not hasattr(model, "classes_"):
        return None
    preds = model.predict(X)
    correct = sum(int(p == yy) for p, yy in zip(preds, y))
    return correct / max(len(y), 1)


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
    shadow_batch_ids: Optional[List[str]] = None,
    retrain_window_batches: int = 5,
) -> int:
    actions = _load_actions(actions_path, dlq_path)
    processed = _load_processed(processed_index_path)
    states = _load_states(states_path) if states_path else {}
    rows_x: List[List[float]] = []
    rows_y: List[int] = []
    if csv_path:
        rows_x, rows_y = _load_csv_rows(csv_path)
    shadow_batch_ids_set = set(shadow_batch_ids or [])
    shadow_x: List[List[float]] = []
    shadow_y: List[int] = []
    if rows_x and shadow_batch_ids_set:
        for shadow_batch_id in sorted(shadow_batch_ids_set):
            sl = _batch_slice_from_id(shadow_batch_id, window_size)
            shadow_x.extend(rows_x[sl])
            shadow_y.extend(rows_y[sl])

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
        executed = action_type in {"fine_tune", "full_retrain"}
        reason_code = "ok" if executed else "defer"
        state = states.get(action.get("state_id", ""))
        label_available = bool(state.get("label_available", True)) if state else True
        batch_id = state.get("batch_id") if state else None
        if executed and not label_available:
            executed = False
            reason_code = "insufficient_data"
        if executed and batch_id in shadow_batch_ids_set:
            # Held-out shadow batches are evaluation-only and never used for training.
            executed = False
            reason_code = "shadow_batch"

        accuracy_pre = None
        shadow_accuracy_pre = None
        shadow_accuracy_post = None
        delta_shadow = 0.0
        X_batch: List[List[float]] = []
        y_batch: List[int] = []
        model = None
        pending_update: Optional[ModelUpdateEvent] = None

        if csv_path and rows_x and label_available:
            if batch_id:
                if action_type == "full_retrain":
                    cur_sl = _batch_slice(batch_id, window_size)
                    end_row = cur_sl.stop or len(rows_x)
                    retrain_batches = max(1, int(retrain_window_batches))
                    start_row = max(0, end_row - (retrain_batches * window_size))
                    X_batch = rows_x[start_row:end_row]
                    y_batch = rows_y[start_row:end_row]
                    slice_start = start_row
                else:
                    sl = _batch_slice(batch_id, window_size)
                    X_batch = rows_x[sl]
                    y_batch = rows_y[sl]
                    slice_start = sl.start or 0
                X_batch, y_batch = _filter_shadow_rows_from_slice(
                    X_batch,
                    y_batch,
                    slice_start,
                    window_size,
                    shadow_batch_ids_set,
                )
            else:
                X_batch = rows_x
                y_batch = rows_y
                X_batch, y_batch = _filter_shadow_rows_from_slice(
                    X_batch,
                    y_batch,
                    0,
                    window_size,
                    shadow_batch_ids_set,
                )

            model_path = Path(f"artifacts/models/{current_model_version or 'm_000000'}.pkl")
            model = _load_model(model_path)
            if model is None:
                model = create_model(seed)

            accuracy_pre = _predict_accuracy(model, X_batch, y_batch)
            if executed and shadow_x and shadow_y:
                shadow_accuracy_pre = _predict_accuracy(model, shadow_x, shadow_y)

        if executed:
            if action_type == "full_retrain":
                train_time_s = 10.0
                estimated_usd = 4.5
                energy_kwh = 0.06
                co2_kg = 0.0015
            else:
                train_time_s = 3.0
                estimated_usd = 1.50
                energy_kwh = 0.02
                co2_kg = 0.0005
            cost = CostEvent(
                cost_id=cost_ids.next(),
                action_id=action_id,
                timestamp=float(action.get("timestamp", time.time())),
                train_time_s=train_time_s,
                estimated_usd=estimated_usd,
                energy_kwh=energy_kwh,
                co2_kg=co2_kg,
                executed=True,
                reason_code=reason_code,
                accuracy_pre=accuracy_pre,
                shadow_accuracy_pre=shadow_accuracy_pre,
                shadow_accuracy_post=shadow_accuracy_post,
                delta_shadow=delta_shadow,
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
                shadow_accuracy_pre=shadow_accuracy_pre,
                shadow_accuracy_post=shadow_accuracy_post,
                delta_shadow=delta_shadow,
            )

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

                    train_model = model
                    if action_type == "full_retrain":
                        train_model = create_model(seed)

                    if train_model is not None:
                        if not hasattr(train_model, "classes_"):
                            train_model.partial_fit(X_batch, y_batch, classes=[0, 1])
                        else:
                            train_model.partial_fit(X_batch, y_batch)

                    if shadow_x and shadow_y and train_model is not None:
                        shadow_accuracy_post = _predict_accuracy(train_model, shadow_x, shadow_y)
                        if shadow_accuracy_pre is not None and shadow_accuracy_post is not None:
                            delta_shadow = float(shadow_accuracy_post - shadow_accuracy_pre)

                    if train_model is not None:
                        _save_model(Path(artifact_path), train_model)
            else:
                eval_summary = {"sklearn_available": False}

            pending_update = ModelUpdateEvent(
                update_id=update_ids.next(),
                action_id=action_id,
                timestamp=float(action.get("timestamp", time.time())),
                new_model_version=new_model_version,
                artifact_path=artifact_path,
                eval_summary=eval_summary,
                executed=True,
            )
            current_model_version = new_model_version

        if executed:
            cost = CostEvent(
                cost_id=cost.cost_id,
                action_id=cost.action_id,
                timestamp=cost.timestamp,
                train_time_s=cost.train_time_s,
                estimated_usd=cost.estimated_usd,
                energy_kwh=cost.energy_kwh,
                co2_kg=cost.co2_kg,
                executed=cost.executed,
                reason_code=cost.reason_code,
                accuracy_pre=cost.accuracy_pre,
                shadow_accuracy_pre=shadow_accuracy_pre,
                shadow_accuracy_post=shadow_accuracy_post,
                delta_shadow=delta_shadow,
            )

        validated_cost = validate_or_dlq(cost.dict(), CostEvent, dlq_path)
        if validated_cost:
            write_event(validated_cost, events_path, event_type="cost")

        if pending_update is not None:
            validated_update = validate_or_dlq(pending_update.dict(), ModelUpdateEvent, dlq_path)
            if validated_update:
                write_event(validated_update, events_path, event_type="model_update")

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
    parser.add_argument("--shadow-batch-ids", default="")
    parser.add_argument("--retrain-window-batches", type=int, default=5)
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
        shadow_batch_ids=[x for x in args.shadow_batch_ids.split(",") if x],
        retrain_window_batches=args.retrain_window_batches,
    )


if __name__ == "__main__":
    main()
