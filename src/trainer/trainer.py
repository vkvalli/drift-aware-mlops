import argparse
import csv
import json
import math
import pickle
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from src.logger.jsonl_logger import write_event
from src.schemas.action import ActionEvent
from src.schemas.cost import CostEvent
from src.schemas.model_update import ModelUpdateEvent
from src.schemas.validate import validate_or_dlq
from src.trainer.compat import SKLEARN_AVAILABLE, create_model
from src.utils.ids import IdCounter

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:
    TfidfVectorizer = None


EXCLUDED_COLUMNS = {
    "Date",
    "Day",
    "Period",
    "date",
    "day",
    "period",
    "ArrDel15",
    "arrdel15",
}
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


def _label_to_binary(
    label_value: str, label_column: str, positive_label: str
) -> Optional[int]:
    raw = str(label_value).strip()
    if raw == "":
        return None
    lowered = raw.lower()
    if lowered in {"1", "1.0", "true", "yes"}:
        return 1
    if lowered in {"0", "0.0", "false", "no"}:
        return 0
    if label_column.lower() == "arrdel15":
        try:
            return 1 if float(raw) >= 1.0 else 0
        except ValueError:
            return None
    return 1 if raw.upper() == positive_label.upper() else 0


def _parse_label(
    raw: str, task: str, label_column: str, positive_label: str
) -> Optional[float]:
    if raw is None:
        return None
    if task == "regression":
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None
    if task == "text_classification":
        value = str(raw).strip()
        if value == "":
            return None
        try:
            val = int(float(value))
            return float(val - 1) if val > 0 else float(val)
        except ValueError:
            raise ValueError(
                f"Non-numeric label '{value}' for text_classification in column '{label_column}'. "
                "Expected numeric class IDs (e.g., 1..N)."
            )
    return float(_label_to_binary(str(raw), label_column, positive_label) or 0)


def _make_text(
    row: Dict[str, str],
    label_column: str,
    target_column: str,
    protected_attr: Optional[str],
) -> str:
    preferred = ["title", "description", "text", "headline", "content"]
    parts: List[str] = []
    for key in preferred:
        val = row.get(key)
        if val:
            parts.append(str(val).strip())
    if parts:
        return " ".join(parts)
    for key, value in row.items():
        if key in {label_column, target_column}:
            continue
        if protected_attr and key == protected_attr:
            continue
        if value is None:
            continue
        txt = str(value).strip()
        if txt:
            parts.append(txt)
    return " ".join(parts)


def _load_csv_rows(
    csv_path: Path,
    task: str,
    label_column: str = LABEL_COLUMN,
    target_column: Optional[str] = None,
    positive_label: str = "UP",
    feature_mode: str = "tabular",
    protected_attr: Optional[str] = None,
) -> Tuple[List, List[float], List[str]]:
    rows_x: List = []
    rows_y: List[float] = []
    groups: List[str] = []
    target_col = target_column or label_column

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=2):
            label = row.get(target_col)
            try:
                y_val = _parse_label(label, task, target_col, positive_label)
            except ValueError as exc:
                raise ValueError(f"{csv_path}:{row_idx}: {exc}") from exc
            if y_val is None:
                continue

            if feature_mode == "tfidf" or task == "text_classification":
                text_value = _make_text(row, label_column, target_col, protected_attr)
                if not text_value:
                    continue
                rows_x.append(text_value)
            else:
                features: List[float] = []
                for key, value in row.items():
                    if key in {target_col, label_column} or key in EXCLUDED_COLUMNS:
                        continue
                    if protected_attr and key == protected_attr:
                        continue
                    try:
                        features.append(float(value))
                    except (TypeError, ValueError):
                        continue
                if not features:
                    continue
                rows_x.append(features)

            rows_y.append(y_val)
            if protected_attr:
                groups.append(str(row.get(protected_attr, "")).strip())
            else:
                groups.append("")

    return rows_x, rows_y, groups


def _batch_slice(batch_id: str, window_size: int) -> slice:
    try:
        idx = int(batch_id.split("_")[-1])
    except Exception:
        idx = 1
    start = (idx - 1) * window_size
    end = start + window_size
    return slice(start, end)


def _filter_shadow_rows_from_slice(
    x_rows: Sequence,
    y_rows: Sequence[float],
    group_rows: Sequence[str],
    start_idx: int,
    window_size: int,
    shadow_batch_ids: set[str],
) -> Tuple[List, List[float], List[str]]:
    if not shadow_batch_ids:
        return list(x_rows), list(y_rows), list(group_rows)
    kept_x: List = []
    kept_y: List[float] = []
    kept_g: List[str] = []
    for offset, (xv, yv, gv) in enumerate(zip(x_rows, y_rows, group_rows)):
        global_idx = start_idx + offset
        batch_num = (global_idx // window_size) + 1
        batch_id = f"b_{batch_num:06d}"
        if batch_id in shadow_batch_ids:
            continue
        kept_x.append(xv)
        kept_y.append(float(yv))
        kept_g.append(gv)
    return kept_x, kept_y, kept_g


def _binary_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    n = len(y_true)
    if n == 0:
        return {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
        }
    tp = tn = fp = fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 1 and yp == 0:
            fn += 1

    accuracy = (tp + tn) / n
    recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    recall_neg = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_accuracy = (recall_pos + recall_neg) / 2.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        2 * precision * recall_pos / (precision + recall_pos)
        if (precision + recall_pos) > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall_pos,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _multiclass_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    if not y_true:
        return {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
    classes = sorted(set(y_true) | set(y_pred))
    accuracy = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / len(y_true)
    recalls: List[float] = []
    precisions: List[float] = []
    f1s: List[float] = []
    for c in classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": sum(recalls) / len(recalls) if recalls else 0.0,
        "precision": sum(precisions) / len(precisions) if precisions else 0.0,
        "recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "f1": sum(f1s) / len(f1s) if f1s else 0.0,
    }


def _regression_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    if not y_true:
        return {"mae": 0.0, "rmse": 0.0, "r2": 0.0}
    n = len(y_true)
    err = [yt - yp for yt, yp in zip(y_true, y_pred)]
    mae = sum(abs(e) for e in err) / n
    mse = sum(e * e for e in err) / n
    rmse = math.sqrt(mse)
    mean_y = sum(y_true) / n
    sst = sum((yt - mean_y) ** 2 for yt in y_true)
    ssr = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    r2 = 1.0 - (ssr / sst) if sst > 0 else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _fairness_metrics(
    y_true: List[int], y_pred: List[int], groups: List[str]
) -> Dict[str, float]:
    valid_groups = [g for g in sorted(set(groups)) if g]
    if len(valid_groups) < 2:
        return {"dpd": 0.0, "eod": 0.0}
    g0, g1 = valid_groups[0], valid_groups[1]

    def _indices(group: str) -> List[int]:
        return [i for i, g in enumerate(groups) if g == group]

    idx0 = _indices(g0)
    idx1 = _indices(g1)
    if not idx0 or not idx1:
        return {"dpd": 0.0, "eod": 0.0}

    def _positive_rate(indices: List[int]) -> float:
        return sum(1 for i in indices if y_pred[i] == 1) / len(indices)

    def _tpr(indices: List[int]) -> float:
        positives = [i for i in indices if y_true[i] == 1]
        if not positives:
            return 0.0
        return sum(1 for i in positives if y_pred[i] == 1) / len(positives)

    dpd = abs(_positive_rate(idx0) - _positive_rate(idx1))
    eod = abs(_tpr(idx0) - _tpr(idx1))
    return {"dpd": dpd, "eod": eod}


def _predict_metrics(
    model, X, y, task: str, groups: Optional[List[str]] = None
) -> Optional[Dict[str, float]]:
    if _row_count(X) == 0 or len(y) == 0:
        return None
    if model is None:
        if task == "regression":
            preds_raw = [0.0 for _ in y]
        else:
            preds_raw = [0 for _ in y]
    elif not hasattr(model, "predict"):
        return None
    else:
        try:
            preds_raw = model.predict(X)
        except Exception:
            if task == "regression":
                preds_raw = [0.0 for _ in y]
            else:
                preds_raw = [0 for _ in y]
    if task == "regression":
        preds = [float(v) for v in preds_raw]
        y_true = [float(v) for v in y]
        return _regression_metrics(y_true, preds)

    y_true = [int(v) for v in y]
    y_pred = [int(v) for v in preds_raw]
    if len(set(y_true) | set(y_pred)) > 2:
        metrics = _multiclass_metrics(y_true, y_pred)
    else:
        metrics = _binary_metrics(y_true, y_pred)
        if groups:
            fair = _fairness_metrics(y_true, y_pred, groups)
            metrics["dpd"] = fair["dpd"]
            metrics["eod"] = fair["eod"]
    return metrics


def _compute_sample_weights(y: List[float], task: str) -> List[float]:
    if not y or task == "regression":
        return [1.0 for _ in y]
    counts = Counter(int(v) for v in y)
    total = len(y)
    classes = max(len(counts), 1)
    weights: Dict[int, float] = {}
    for klass, c in counts.items():
        if c <= 0:
            weights[klass] = 1.0
        else:
            weights[klass] = total / (classes * c)
    return [weights[int(v)] for v in y]


def _load_model(path: Path):
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _save_model(path: Path, model) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(model, f)


def _vectorizer_path(model_version: str) -> Path:
    return Path(f"artifacts/models/{model_version}.vectorizer.pkl")


def _load_vectorizer(path: Path):
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def _save_vectorizer(path: Path, vectorizer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(vectorizer, f)


def _feature_dim(X) -> Optional[int]:
    if X is None:
        return None
    if hasattr(X, "shape"):
        try:
            return int(X.shape[1])
        except Exception:
            return None
    if isinstance(X, list) and X:
        if isinstance(X[0], list):
            return len(X[0])
    return None


def _row_count(X) -> int:
    if X is None:
        return 0
    if hasattr(X, "shape"):
        try:
            return int(X.shape[0])
        except Exception:
            return 0
    try:
        return len(X)
    except Exception:
        return 0


def _compatible_model_for_batch(model, X_batch, seed: Optional[int], task: str):
    if model is None:
        return model
    dim = _feature_dim(X_batch)
    if dim is None:
        return model
    try:
        expected = getattr(model, "n_features_in_", None)
        if expected is not None and int(expected) != dim:
            return create_model(seed, task=task)
    except Exception:
        return create_model(seed, task=task)
    return model


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
    label_column: str = LABEL_COLUMN,
    target_column: Optional[str] = None,
    positive_label: str = "UP",
    window_size: int = 10,
    shadow_batch_ids: Optional[List[str]] = None,
    retrain_window_batches: int = 5,
    task: str = "classification",
    feature_mode: str = "tabular",
    protected_attr: Optional[str] = None,
) -> int:
    actions = _load_actions(actions_path, dlq_path)
    processed = _load_processed(processed_index_path)
    states = _load_states(states_path) if states_path else {}
    rows_x: List = []
    rows_y: List[float] = []
    groups: List[str] = []
    if csv_path:
        rows_x, rows_y, groups = _load_csv_rows(
            csv_path,
            task=task,
            label_column=label_column,
            target_column=target_column,
            positive_label=positive_label,
            feature_mode=feature_mode,
            protected_attr=protected_attr,
        )
    shadow_batch_ids_set = set(shadow_batch_ids or [])
    shadow_x: List = []
    shadow_y: List[float] = []
    if rows_x and shadow_batch_ids_set:
        for shadow_batch_id in sorted(shadow_batch_ids_set):
            sl = _batch_slice(shadow_batch_id, window_size)
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
            executed = False
            reason_code = "shadow_batch"

        accuracy_pre = None
        balanced_accuracy_pre = None
        f1_pre = None
        precision_pre = None
        recall_pre = None
        mae_pre = None
        rmse_pre = None
        r2_pre = None
        dpd_pre = None
        eod_pre = None
        tp_pre = None
        fp_pre = None
        tn_pre = None
        fn_pre = None
        shadow_accuracy_pre = None
        shadow_accuracy_post = None
        delta_shadow = 0.0
        X_batch_raw: List = []
        y_batch: List[float] = []
        g_batch: List[str] = []
        model = None
        vectorizer = None
        pending_update: Optional[ModelUpdateEvent] = None

        if csv_path and rows_x and label_available:
            if batch_id:
                if action_type == "full_retrain":
                    cur_sl = _batch_slice(batch_id, window_size)
                    end_row = cur_sl.stop or len(rows_x)
                    retrain_batches = max(1, int(retrain_window_batches))
                    start_row = max(0, end_row - (retrain_batches * window_size))
                    X_batch_raw = rows_x[start_row:end_row]
                    y_batch = rows_y[start_row:end_row]
                    g_batch = groups[start_row:end_row]
                    slice_start = start_row
                else:
                    sl = _batch_slice(batch_id, window_size)
                    X_batch_raw = rows_x[sl]
                    y_batch = rows_y[sl]
                    g_batch = groups[sl]
                    slice_start = sl.start or 0
                X_batch_raw, y_batch, g_batch = _filter_shadow_rows_from_slice(
                    X_batch_raw,
                    y_batch,
                    g_batch,
                    slice_start,
                    window_size,
                    shadow_batch_ids_set,
                )
            else:
                X_batch_raw = list(rows_x)
                y_batch = list(rows_y)
                g_batch = list(groups)
                X_batch_raw, y_batch, g_batch = _filter_shadow_rows_from_slice(
                    X_batch_raw,
                    y_batch,
                    g_batch,
                    0,
                    window_size,
                    shadow_batch_ids_set,
                )

            model_path = Path(
                f"artifacts/models/{current_model_version or 'm_000000'}.pkl"
            )
            model = _load_model(model_path)
            if model is None:
                model = create_model(seed, task=task)

            X_batch = X_batch_raw
            if feature_mode == "tfidf" or task == "text_classification":
                vec_path = _vectorizer_path(current_model_version or "m_000000")
                vectorizer = _load_vectorizer(vec_path)
                if vectorizer is None and TfidfVectorizer is not None:
                    vectorizer = TfidfVectorizer(max_features=5000)
                    X_batch = (
                        vectorizer.fit_transform([str(x) for x in X_batch_raw])
                        if X_batch_raw
                        else X_batch_raw
                    )
                elif vectorizer is not None:
                    X_batch = (
                        vectorizer.transform([str(x) for x in X_batch_raw])
                        if X_batch_raw
                        else X_batch_raw
                    )
            model = _compatible_model_for_batch(model, X_batch, seed, task)

            pre_metrics = _predict_metrics(model, X_batch, y_batch, task, g_batch)
            if pre_metrics is not None:
                accuracy_pre = pre_metrics.get("accuracy")
                balanced_accuracy_pre = pre_metrics.get("balanced_accuracy")
                f1_pre = pre_metrics.get("f1")
                precision_pre = pre_metrics.get("precision")
                recall_pre = pre_metrics.get("recall")
                mae_pre = pre_metrics.get("mae")
                rmse_pre = pre_metrics.get("rmse")
                r2_pre = pre_metrics.get("r2")
                dpd_pre = pre_metrics.get("dpd")
                eod_pre = pre_metrics.get("eod")
                tp_pre = int(pre_metrics["tp"]) if "tp" in pre_metrics else None
                fp_pre = int(pre_metrics["fp"]) if "fp" in pre_metrics else None
                tn_pre = int(pre_metrics["tn"]) if "tn" in pre_metrics else None
                fn_pre = int(pre_metrics["fn"]) if "fn" in pre_metrics else None
            if executed and shadow_x and shadow_y and task != "regression":
                shadow_x_eval = shadow_x
                if feature_mode == "tfidf" or task == "text_classification":
                    if vectorizer is not None:
                        shadow_x_eval = vectorizer.transform([str(x) for x in shadow_x])
                shadow_metrics_pre = _predict_metrics(
                    model, shadow_x_eval, shadow_y, task
                )
                if shadow_metrics_pre is not None:
                    shadow_accuracy_pre = shadow_metrics_pre.get("accuracy")

        if executed:
            # Energy model: proxy-only estimate using CPU TDP
            # kwh = train_time_s * TDP_W / 3_600_000  (TDP_W = 65W CPU proxy)
            # co2_kg = kwh * 0.233  (UK grid average carbon intensity, BEIS 2023)
            _TDP_W = 65.0
            _CO2_INTENSITY = 0.233
            if action_type == "full_retrain":
                train_time_s = 10.0
                estimated_usd = 4.5
            else:
                train_time_s = 3.0
                estimated_usd = 1.50
            energy_kwh = train_time_s * _TDP_W / 3_600_000
            co2_kg = energy_kwh * _CO2_INTENSITY
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
                balanced_accuracy_pre=balanced_accuracy_pre,
                f1_pre=f1_pre,
                mae_pre=mae_pre,
                rmse_pre=rmse_pre,
                r2_pre=r2_pre,
                precision_pre=precision_pre,
                recall_pre=recall_pre,
                dpd_pre=dpd_pre,
                eod_pre=eod_pre,
                tp_pre=tp_pre,
                fp_pre=fp_pre,
                tn_pre=tn_pre,
                fn_pre=fn_pre,
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
                balanced_accuracy_pre=balanced_accuracy_pre,
                f1_pre=f1_pre,
                mae_pre=mae_pre,
                rmse_pre=rmse_pre,
                r2_pre=r2_pre,
                precision_pre=precision_pre,
                recall_pre=recall_pre,
                dpd_pre=dpd_pre,
                eod_pre=eod_pre,
                tp_pre=tp_pre,
                fp_pre=fp_pre,
                tn_pre=tn_pre,
                fn_pre=fn_pre,
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
                    model = create_model(seed, task=task)

                X_batch_train = X_batch_raw
                if feature_mode == "tfidf" or task == "text_classification":
                    if vectorizer is None and TfidfVectorizer is not None:
                        vectorizer = TfidfVectorizer(max_features=5000)
                        X_batch_train = (
                            vectorizer.fit_transform([str(x) for x in X_batch_raw])
                            if X_batch_raw
                            else X_batch_raw
                        )
                    elif vectorizer is not None:
                        X_batch_train = (
                            vectorizer.transform([str(x) for x in X_batch_raw])
                            if X_batch_raw
                            else X_batch_raw
                        )

                if model is not None and X_batch_train is not None and len(y_batch) > 0:
                    eval_summary = {"n_samples": len(y_batch)}
                    if accuracy_pre is not None:
                        eval_summary["accuracy"] = float(accuracy_pre)
                    if balanced_accuracy_pre is not None:
                        eval_summary["balanced_accuracy"] = float(balanced_accuracy_pre)
                    if f1_pre is not None:
                        eval_summary["f1"] = float(f1_pre)
                    if precision_pre is not None:
                        eval_summary["precision"] = float(precision_pre)
                    if recall_pre is not None:
                        eval_summary["recall"] = float(recall_pre)
                    if mae_pre is not None:
                        eval_summary["mae"] = float(mae_pre)
                    if rmse_pre is not None:
                        eval_summary["rmse"] = float(rmse_pre)
                    if r2_pre is not None:
                        eval_summary["r2"] = float(r2_pre)
                    if dpd_pre is not None:
                        eval_summary["dpd"] = float(dpd_pre)
                    if eod_pre is not None:
                        eval_summary["eod"] = float(eod_pre)

                    train_model = model
                    if action_type == "full_retrain":
                        train_model = create_model(seed, task=task)

                    if train_model is not None:
                        sample_weight = _compute_sample_weights(y_batch, task)
                        if task == "regression":
                            train_model.partial_fit(
                                X_batch_train, y_batch, sample_weight=sample_weight
                            )
                        else:
                            class_values = (
                                sorted({int(v) for v in rows_y}) if rows_y else [0, 1]
                            )
                            if not hasattr(train_model, "classes_"):
                                train_model.partial_fit(
                                    X_batch_train,
                                    [int(v) for v in y_batch],
                                    classes=class_values,
                                    sample_weight=sample_weight,
                                )
                            else:
                                train_model.partial_fit(
                                    X_batch_train,
                                    [int(v) for v in y_batch],
                                    sample_weight=sample_weight,
                                )

                    if (
                        shadow_x
                        and shadow_y
                        and train_model is not None
                        and task != "regression"
                    ):
                        shadow_x_eval = shadow_x
                        if feature_mode == "tfidf" or task == "text_classification":
                            if vectorizer is not None:
                                shadow_x_eval = vectorizer.transform(
                                    [str(x) for x in shadow_x]
                                )
                        shadow_metrics_post = _predict_metrics(
                            train_model, shadow_x_eval, shadow_y, task
                        )
                        shadow_accuracy_post = (
                            shadow_metrics_post.get("accuracy")
                            if shadow_metrics_post
                            else None
                        )
                        if (
                            shadow_accuracy_pre is not None
                            and shadow_accuracy_post is not None
                        ):
                            delta_shadow = float(
                                shadow_accuracy_post - shadow_accuracy_pre
                            )

                    if train_model is not None:
                        _save_model(Path(artifact_path), train_model)
                    if (
                        feature_mode == "tfidf" or task == "text_classification"
                    ) and vectorizer is not None:
                        _save_vectorizer(
                            _vectorizer_path(new_model_version), vectorizer
                        )
            else:
                eval_summary = {"sklearn_available": False}
                if accuracy_pre is not None:
                    eval_summary["accuracy"] = float(accuracy_pre)
                if balanced_accuracy_pre is not None:
                    eval_summary["balanced_accuracy"] = float(balanced_accuracy_pre)
                if f1_pre is not None:
                    eval_summary["f1"] = float(f1_pre)
                if mae_pre is not None:
                    eval_summary["mae"] = float(mae_pre)
                if rmse_pre is not None:
                    eval_summary["rmse"] = float(rmse_pre)
                if r2_pre is not None:
                    eval_summary["r2"] = float(r2_pre)

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
                balanced_accuracy_pre=cost.balanced_accuracy_pre,
                f1_pre=cost.f1_pre,
                mae_pre=cost.mae_pre,
                rmse_pre=cost.rmse_pre,
                r2_pre=cost.r2_pre,
                precision_pre=cost.precision_pre,
                recall_pre=cost.recall_pre,
                dpd_pre=cost.dpd_pre,
                eod_pre=cost.eod_pre,
                tp_pre=cost.tp_pre,
                fp_pre=cost.fp_pre,
                tn_pre=cost.tn_pre,
                fn_pre=cost.fn_pre,
                shadow_accuracy_pre=shadow_accuracy_pre,
                shadow_accuracy_post=shadow_accuracy_post,
                delta_shadow=delta_shadow,
            )

        validated_cost = validate_or_dlq(cost.dict(), CostEvent, dlq_path)
        if validated_cost:
            write_event(validated_cost, events_path, event_type="cost")

        if pending_update is not None:
            validated_update = validate_or_dlq(
                pending_update.dict(), ModelUpdateEvent, dlq_path
            )
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
    parser.add_argument(
        "--processed-index-file", default="artifacts/processed_actions.jsonl"
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--states-file", default=None)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--label-column", default=LABEL_COLUMN)
    parser.add_argument("--target-column", default=None)
    parser.add_argument("--positive-label", default="UP")
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--shadow-batch-ids", default="")
    parser.add_argument("--retrain-window-batches", type=int, default=5)
    parser.add_argument(
        "--task",
        choices=["classification", "regression", "text_classification"],
        default="classification",
    )
    parser.add_argument(
        "--feature-mode", choices=["tabular", "tfidf"], default="tabular"
    )
    parser.add_argument("--protected-attr", default=None)
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
        label_column=args.label_column,
        target_column=args.target_column,
        positive_label=args.positive_label,
        window_size=args.window_size,
        shadow_batch_ids=[x for x in args.shadow_batch_ids.split(",") if x],
        retrain_window_batches=args.retrain_window_batches,
        task=args.task,
        feature_mode=args.feature_mode,
        protected_attr=args.protected_attr,
    )


if __name__ == "__main__":
    main()
