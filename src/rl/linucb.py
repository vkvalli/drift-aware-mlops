from typing import Dict

import numpy as np


ACTIONS = ["defer", "fine_tune"]


class LinUCB:
    def __init__(self, n_features: int, alpha: float = 1.0):
        self.n_features = n_features
        self.alpha = alpha
        self.A: Dict[str, np.ndarray] = {a: np.identity(n_features) for a in ACTIONS}
        self.b: Dict[str, np.ndarray] = {a: np.zeros((n_features, 1)) for a in ACTIONS}

    def update(self, action: str, x: np.ndarray, reward: float) -> None:
        x = x.reshape(-1, 1)
        self.A[action] += x @ x.T
        self.b[action] += reward * x

    def choose_action(self, x: np.ndarray) -> str:
        x = x.reshape(-1, 1)
        scores = {}
        for action in ACTIONS:
            A_inv = np.linalg.inv(self.A[action])
            theta = A_inv @ self.b[action]
            mean = float(theta.T @ x)
            bonus = self.alpha * float(np.sqrt(x.T @ A_inv @ x))
            scores[action] = mean + bonus
        return max(scores, key=scores.get)
