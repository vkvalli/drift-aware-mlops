from typing import Any, Optional

try:
    from sklearn.linear_model import SGDClassifier

    SKLEARN_AVAILABLE = True
except Exception:
    SGDClassifier = None
    SKLEARN_AVAILABLE = False


def create_model(seed: Optional[int]) -> Any:
    if not SKLEARN_AVAILABLE or SGDClassifier is None:
        return None
    return SGDClassifier(loss="log_loss", random_state=seed)


__all__ = ["SKLEARN_AVAILABLE", "create_model"]
