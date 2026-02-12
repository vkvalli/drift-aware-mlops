import argparse
import csv
import pickle
import sys
from pathlib import Path
from typing import Dict, List

# Ensure repo root is on sys.path when running as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.rl.linucb import LinUCB, ACTIONS


FEATURES = ["drift_core", "drift_max", "drift_ready", "cooldown_active"]


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
