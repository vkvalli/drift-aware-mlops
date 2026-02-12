import argparse
from datetime import datetime
from pathlib import Path

from src.runner.offline_run import main as offline_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run offline experiment into a new timestamped folder")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--label-delay-batches", type=int, default=2)
    parser.add_argument("--ref-window", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--consecutive-k", type=int, default=3)
    parser.add_argument("--policy", type=str, default="threshold_persistence")
    parser.add_argument("--fixed-interval", type=int, default=10)
    parser.add_argument("--cooldown", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dataset-id", type=str, default="electricity")
    parser.add_argument("--model-version", type=str, default="m_000001")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    import sys

    old_argv = sys.argv
    try:
        sys.argv = [
            "offline_run.py",
            "--csv",
            args.csv,
            "--window-size",
            str(args.window_size),
            "--label-delay-batches",
            str(args.label_delay_batches),
            "--ref-window",
            str(args.ref_window),
            "--threshold",
            str(args.threshold),
            "--consecutive-k",
            str(args.consecutive_k),
            "--policy",
            args.policy,
            "--fixed-interval",
            str(args.fixed_interval),
            "--cooldown",
            str(args.cooldown),
            "--seed",
            str(args.seed),
            "--dataset-id",
            args.dataset_id,
            "--model-version",
            args.model_version,
            "--replay-output",
            str(run_dir / "replay.jsonl"),
            "--state-output",
            str(run_dir / "state.jsonl"),
            "--actions-output",
            str(run_dir / "actions.jsonl"),
            "--trainer-output",
            str(run_dir / "trainer_output.jsonl"),
            "--canonical-output",
            str(run_dir / "events.jsonl"),
            "--processed-index-file",
            str(run_dir / "processed_actions.jsonl"),
            "--clean-artifacts",
        ]
        offline_main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
