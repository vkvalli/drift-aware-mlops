from typing import Any, Optional

try:
    from sklearn.linear_model import SGDClassifier, SGDRegressor

    SKLEARN_AVAILABLE = True
except Exception:
    SGDClassifier = None
    SGDRegressor = None
    SKLEARN_AVAILABLE = False


def create_model(seed: Optional[int], task: str = "classification") -> Any:
    if not SKLEARN_AVAILABLE:
        return None
    if task == "regression":
        if SGDRegressor is None:
            return None
        return SGDRegressor(random_state=seed)
    if SGDClassifier is None:
        return None
    return SGDClassifier(loss="log_loss", random_state=seed)


__all__ = ["SKLEARN_AVAILABLE", "create_model"]
