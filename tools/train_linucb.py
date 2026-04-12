import argparse
import csv
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.rl.linucb import ACTIONS, LinUCB


FEATURES = [
    "drift_core",
    "drift_max",
    "drift_ready",
    "cooldown_active",
    "fairness_core",
    "fairness_ready",
]


def _load_rows(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def train_linucb(dataset_path: Path, out_path: Path, alpha: float) -> None:
    rows = _load_rows(dataset_path)
    model = LinUCB(n_features=len(FEATURES), alpha=alpha)

    for row in rows:
        action = row.get("action_type")
        if action not in ACTIONS:
            continue
        x = np.array([float(row.get(f, 0.0)) for f in FEATURES], dtype=float)
        reward = float(row.get("reward", 0.0))
        model.update(action, x, reward)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(model, f)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train LinUCB on RL dataset CSV")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", default="artifacts/policies/linucb.pkl")
    parser.add_argument("--alpha", type=float, default=1.0)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    train_linucb(Path(args.dataset), Path(args.out), args.alpha)


if __name__ == "__main__":
    main()
