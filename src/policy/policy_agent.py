import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

from src.logger.jsonl_logger import write_event
from src.schemas.action import ActionEvent
from src.schemas.state import StateEvent
from src.schemas.validate import validate_or_dlq
from src.utils.ids import IdCounter

import pickle


def _load_state_events(path: Path, dlq_path: Path) -> List[Dict]:
    records: List[Dict] = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        event_type = record.get("event_type", "")
        if event_type in {"state", "StateEvent"}:
            validated = validate_or_dlq(record, StateEvent, dlq_path)
            if validated:
                records.append(validated.dict())
    return records


def run_policy(
    states_path: Path,
    actions_path: Path,
    threshold: float = 1.0,
    threshold_full: Optional[float] = None,
    action_start: int = 1,
    seed: Optional[int] = None,
    consecutive_k: int = 3,
    policy: str = "threshold_persistence",
    fixed_interval: int = 10,
    cooldown: int = 0,
    linucb_model_path: Optional[Path] = None,
    warm_start_n: int = 30,
    warm_start_k: int = 6,
    epsilon1: float = 0.05,
    epsilon1_steps: int = 50,
    epsilon2: float = 0.01,
    fairness_threshold: float = 0.1,
    dlq_path: Path = Path("artifacts/dlq.jsonl"),
) -> int:
    states = _load_state_events(states_path, dlq_path)

    start = action_start
    if seed is not None:
        start = max(1, int(seed))

    action_ids = IdCounter("a_", start=start)
    rng = random.Random(seed or 0)

    consecutive = 0
    cooldown_remaining = 0
    action_index = 0
    linucb = None
    if policy == "linucb" and linucb_model_path and linucb_model_path.exists():
        from src.rl.linucb import LinUCB  # noqa: F401
        with linucb_model_path.open("rb") as f:
            linucb = pickle.load(f)

    for state in states:
        action_index += 1
        drift_ready = bool(state.get("drift_ready", False))
        drift_core = float(state.get("drift_core", 0.0))
        fairness_core = float(state.get("fairness_core", 0.0))

        if policy == "linucb" and linucb is not None:
            if warm_start_n > 0 and action_index <= warm_start_n:
                if action_index % warm_start_k == 0:
                    action_type = "fine_tune"
                    reason = "linucb_warm_start"
                else:
                    action_type = "defer"
                    reason = "linucb_warm_start"
            else:
                post_warm_step = action_index - warm_start_n
                epsilon = epsilon2
                if epsilon1_steps > 0 and post_warm_step <= epsilon1_steps:
                    epsilon = epsilon1
                if rng.random() < epsilon:
                    action_type = rng.choice(["defer", "fine_tune"])
                    reason = "linucb_epsilon"
                else:
                    try:
                        import numpy as np

                        features = np.array(
                            [
                                float(state.get("drift_core", 0.0)),
                                float(state.get("drift_max", 0.0)),
                                1.0 if state.get("drift_ready", False) else 0.0,
                                1.0 if state.get("cooldown_active", False) else 0.0,
                            ],
                            dtype=float,
                        )
                        action_type = linucb.choose_action(features)
                        reason = "linucb_policy"
                    except Exception:
                        action_type = "defer"
                        reason = "linucb_unavailable"
        elif policy == "no_retrain":
            action_type = "defer"
            reason = "no_retrain"
        elif policy == "fairness_trigger":
            # Trigger fine_tune if fairness_core exceeds fairness_threshold
            # OR if drift_core exceeds the drift threshold.
            fairness_triggered = fairness_core > fairness_threshold
            drift_triggered = drift_ready and drift_core >= threshold
            if fairness_triggered:
                action_type = "fine_tune"
                reason = "fairness_trigger"
            elif drift_triggered:
                action_type = "fine_tune"
                reason = "drift_threshold"
            else:
                action_type = "defer"
                reason = "below_threshold"
        elif policy == "fixed_interval":
            if fixed_interval <= 0:
                action_type = "defer"
                reason = "fixed_interval_invalid"
            elif action_index % fixed_interval == 0:
                action_type = "fine_tune"
                reason = "fixed_interval"
            else:
                action_type = "defer"
                reason = "fixed_interval_wait"
        else:
            if drift_ready and drift_core >= threshold:
                consecutive += 1
            else:
                consecutive = 0

            if (
                drift_ready
                and threshold_full is not None
                and drift_core >= threshold_full
                and consecutive >= consecutive_k
            ):
                action_type = "full_retrain"
                reason = "drift_full_threshold"
            elif drift_ready and consecutive >= consecutive_k:
                action_type = "fine_tune"
                reason = "drift_threshold"
            else:
                action_type = "defer"
                reason = "drift_below_threshold"

        if cooldown > 0:
            if cooldown_remaining > 0:
                action_type = "defer"
                reason = "cooldown"
                cooldown_remaining -= 1
            elif action_type == "fine_tune":
                cooldown_remaining = cooldown

        requested_at = float(state.get("timestamp", time.time()))

        action_event = ActionEvent(
            action_id=action_ids.next(),
            state_id=state["state_id"],
            action_type=action_type,
            model_version=state.get("model_version", "m_000001"),
            timestamp=requested_at,
            reason=reason,
            action_params={
                "threshold": threshold,
                "threshold_full": threshold_full,
                "consecutive_k": consecutive_k,
                "policy": policy,
                "fixed_interval": fixed_interval,
                "cooldown": cooldown,
                "warm_start_n": warm_start_n,
                "warm_start_k": warm_start_k,
                "epsilon1": epsilon1,
                "epsilon1_steps": epsilon1_steps,
                "epsilon2": epsilon2,
                "fairness_threshold": fairness_threshold,
            },
            requested_at=requested_at,
        )
        validated_action = validate_or_dlq(action_event.dict(), ActionEvent, dlq_path)
        if validated_action:
            write_event(validated_action, actions_path, event_type="action")

    return len(states)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline policy agent")
    parser.add_argument("--states", default="artifacts/states.jsonl")
    parser.add_argument("--actions", default="artifacts/actions.jsonl")
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--threshold-full", type=float, default=None)
    parser.add_argument("--consecutive-k", type=int, default=3)
    parser.add_argument("--policy", type=str, default="threshold_persistence")
    parser.add_argument("--fixed-interval", type=int, default=10)
    parser.add_argument("--cooldown", type=int, default=0)
    parser.add_argument("--linucb-model", type=str, default="artifacts/policies/linucb.pkl")
    parser.add_argument("--warm-start-n", type=int, default=30)
    parser.add_argument("--warm-start-k", type=int, default=6)
    parser.add_argument("--epsilon1", type=float, default=0.05)
    parser.add_argument("--epsilon1-steps", type=int, default=50)
    parser.add_argument("--epsilon2", type=float, default=0.01)
    parser.add_argument("--action-start", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dlq", default="artifacts/dlq.jsonl")
    parser.add_argument("--fairness-threshold", type=float, default=0.1,
                        help="DPD proxy threshold for fairness_trigger policy (default: 0.1)")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_policy(
        states_path=Path(args.states),
        actions_path=Path(args.actions),
        threshold=args.threshold,
        threshold_full=args.threshold_full,
        action_start=args.action_start,
        seed=args.seed,
        consecutive_k=args.consecutive_k,
        policy=args.policy,
        fixed_interval=args.fixed_interval,
        cooldown=args.cooldown,
        linucb_model_path=Path(args.linucb_model) if args.linucb_model else None,
        warm_start_n=args.warm_start_n,
        warm_start_k=args.warm_start_k,
        epsilon1=args.epsilon1,
        epsilon1_steps=args.epsilon1_steps,
        epsilon2=args.epsilon2,
        fairness_threshold=args.fairness_threshold,
        dlq_path=Path(args.dlq),
    )


if __name__ == "__main__":
    main()
