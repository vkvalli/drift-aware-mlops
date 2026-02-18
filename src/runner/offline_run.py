import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from src.ingestion.replay import run_replay
from src.monitor.monitor import run_monitor
from src.policy.policy_agent import run_policy
from src.runner.merge_utils import append_file_to_canonical
from src.trainer.trainer import run_trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run offline ingestion + monitor")
    parser.add_argument("--csv", required=True, help="Path to source CSV")
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--label-delay-batches", type=int, default=2)
    parser.add_argument("--flush-pending-labels", type=str, default="true")
    parser.add_argument("--ref-window", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=1.0)
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
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dataset-id", type=str, default="electricity")
    parser.add_argument("--model-version", type=str, default="m_000001")
    parser.add_argument("--replay-output", type=str, default="artifacts/replay.jsonl")
    parser.add_argument("--state-output", type=str, default="artifacts/state.jsonl")
    parser.add_argument("--actions-output", type=str, default="artifacts/actions.jsonl")
    parser.add_argument("--trainer-output", type=str, default="artifacts/trainer_output.jsonl")
    parser.add_argument("--canonical-output", type=str, default="artifacts/events.jsonl")
    parser.add_argument("--processed-index-file", type=str, default="artifacts/processed_actions.jsonl")
    parser.add_argument("--dlq", type=str, default="artifacts/dlq.jsonl")
    parser.add_argument("--clean-artifacts", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    replay_output = Path(args.replay_output)
    state_output = Path(args.state_output)
    actions_output = Path(args.actions_output)
    trainer_output = Path(args.trainer_output)
    canonical_output = Path(args.canonical_output)
    processed_index_path = Path(args.processed_index_file)
    dlq_path = Path(args.dlq)
    csv_path = Path(args.csv)
    flush = str(args.flush_pending_labels).lower() in {"true", "1", "yes"}

    if args.clean_artifacts:
        for path in [
            replay_output,
            state_output,
            actions_output,
            trainer_output,
            canonical_output,
            processed_index_path,
            dlq_path,
        ]:
            if path.exists():
                path.unlink()

    canonical_output.parent.mkdir(parents=True, exist_ok=True)
    canonical_output.write_text("")

    repo_root = Path(__file__).resolve().parents[2]
    try:
        git_commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(repo_root),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        git_commit = "unknown"

    run_meta = {
        "event_type": "run_meta",
        "dataset_id": args.dataset_id,
        "window_size": args.window_size,
        "max_rows": args.max_rows,
        "label_delay_batches": args.label_delay_batches,
        "ref_window": args.ref_window,
        "threshold": args.threshold,
        "policy": args.policy,
        "consecutive_k": args.consecutive_k,
        "fixed_interval": args.fixed_interval,
        "cooldown": args.cooldown,
        "seed": args.seed,
        "git_commit": git_commit,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with canonical_output.open("a", encoding="utf-8") as f:
        f.write(json.dumps(run_meta) + "\n")

    run_replay(
        csv_path=csv_path,
        window_size=args.window_size,
        label_delay_batches=args.label_delay_batches,
        flush_pending_labels=flush,
        dataset_id=args.dataset_id,
        model_version=args.model_version,
        output_path=replay_output,
        max_rows=args.max_rows,
        dlq_path=dlq_path,
    )
    append_file_to_canonical(replay_output, canonical_output)

    run_monitor(
        events_path=replay_output,
        csv_path=csv_path,
        output_path=state_output,
        window_size=args.window_size,
        ref_window=args.ref_window,
        dlq_path=dlq_path,
    )
    append_file_to_canonical(state_output, canonical_output)

    run_policy(
        states_path=state_output,
        actions_path=actions_output,
        threshold=args.threshold,
        action_start=1,
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
        dlq_path=dlq_path,
    )
    append_file_to_canonical(actions_output, canonical_output)

    run_trainer(
        actions_path=actions_output,
        events_path=trainer_output,
        processed_index_path=processed_index_path,
        seed=args.seed,
        dlq_path=dlq_path,
        states_path=state_output,
        csv_path=csv_path,
        window_size=args.window_size,
    )
    append_file_to_canonical(trainer_output, canonical_output)


if __name__ == "__main__":
    main()
