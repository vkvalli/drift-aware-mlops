import argparse
from datetime import datetime
from pathlib import Path
from typing import List

from src.runner.offline_run import main as offline_main


def _parse_int_list(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _parse_float_list(value: str) -> List[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a matrix of offline experiments")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--label-delay-batches", type=int, default=2)
    parser.add_argument("--ref-window", type=int, default=3)
    parser.add_argument("--seeds", type=str, default="1,2,3")
    parser.add_argument("--thresholds", type=str, default="0.1")
    parser.add_argument("--threshold-full", type=float, default=None)
    parser.add_argument("--ks", type=str, default="1,3,5")
    parser.add_argument("--fixed-intervals", type=str, default="10")
    parser.add_argument("--cooldowns", type=str, default="0")
    parser.add_argument(
        "--policies",
        type=str,
        default="no_retrain,fixed_interval,threshold_persistence",
    )
    parser.add_argument("--dataset-id", type=str, default="electricity")
    parser.add_argument(
        "--task",
        choices=["classification", "regression", "text_classification"],
        default="classification",
    )
    parser.add_argument(
        "--feature-mode", choices=["tabular", "tfidf"], default="tabular"
    )
    parser.add_argument("--model-version", type=str, default="m_000001")
    parser.add_argument("--label-column", type=str, default="class")
    parser.add_argument("--target-column", type=str, default=None)
    parser.add_argument("--positive-label", type=str, default="UP")
    parser.add_argument("--protected-attr", type=str, default=None)
    parser.add_argument(
        "--perf-direction", choices=["higher_better", "lower_better"], default=None
    )
    parser.add_argument("--outdir", type=str, default="runs")
    parser.add_argument(
        "--linucb-model", type=str, default="artifacts/policies/linucb.pkl"
    )
    parser.add_argument("--warm-start-n", type=int, default=30)
    parser.add_argument("--warm-start-k", type=int, default=6)
    parser.add_argument("--epsilon1", type=float, default=0.05)
    parser.add_argument("--epsilon1-steps", type=int, default=50)
    parser.add_argument("--epsilon2", type=float, default=0.01)
    parser.add_argument("--enable-shadow-buffer", action="store_true")
    parser.add_argument("--shadow-batches", type=int, default=2)
    parser.add_argument("--fine-tune-window-batches", type=int, default=1)
    parser.add_argument("--retrain-window-batches", type=int, default=5)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    seeds = _parse_int_list(args.seeds)
    thresholds = _parse_float_list(args.thresholds)
    ks = _parse_int_list(args.ks)
    fixed_intervals = _parse_int_list(args.fixed_intervals)
    cooldowns = _parse_int_list(args.cooldowns)
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) / f"matrix_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)

    for policy in policies:
        for seed in seeds:
            if policy == "threshold_persistence":
                for threshold in thresholds:
                    for k in ks:
                        for cooldown in cooldowns:
                            run_id = f"policy=threshold_persistence_T{threshold}_K{k}_cd{cooldown}_seed{seed}"
                            run_dir = outdir / run_id
                            _run_single(
                                args,
                                run_dir,
                                policy,
                                seed,
                                threshold,
                                k,
                                fixed_intervals[0],
                                cooldown,
                            )
            elif policy == "fixed_interval":
                for interval in fixed_intervals:
                    for cooldown in cooldowns:
                        run_id = (
                            f"policy=fixed_interval_N{interval}_cd{cooldown}_seed{seed}"
                        )
                        run_dir = outdir / run_id
                        _run_single(
                            args,
                            run_dir,
                            policy,
                            seed,
                            thresholds[0],
                            ks[0],
                            interval,
                            cooldown,
                        )
            elif policy == "no_retrain":
                run_id = f"policy=no_retrain_seed{seed}"
                run_dir = outdir / run_id
                _run_single(
                    args,
                    run_dir,
                    policy,
                    seed,
                    thresholds[0],
                    ks[0],
                    fixed_intervals[0],
                    cooldowns[0],
                )
            elif policy == "linucb":
                run_id = f"policy=linucb_seed{seed}"
                run_dir = outdir / run_id
                _run_single(
                    args,
                    run_dir,
                    policy,
                    seed,
                    thresholds[0],
                    ks[0],
                    fixed_intervals[0],
                    cooldowns[0],
                )


def _run_single(
    args,
    run_dir: Path,
    policy: str,
    seed: int,
    threshold: float,
    k: int,
    fixed_interval: int,
    cooldown: int,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    import sys

    old_argv = sys.argv
    try:
        argv = [
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
            str(threshold),
            "--consecutive-k",
            str(k),
            "--policy",
            policy,
            "--fixed-interval",
            str(fixed_interval),
            "--cooldown",
            str(cooldown),
            "--linucb-model",
            str(args.linucb_model),
            "--warm-start-n",
            str(args.warm_start_n),
            "--warm-start-k",
            str(args.warm_start_k),
            "--epsilon1",
            str(args.epsilon1),
            "--epsilon1-steps",
            str(args.epsilon1_steps),
            "--epsilon2",
            str(args.epsilon2),
            "--shadow-batches",
            str(args.shadow_batches),
            "--fine-tune-window-batches",
            str(args.fine_tune_window_batches),
            "--retrain-window-batches",
            str(args.retrain_window_batches),
            "--seed",
            str(seed),
            "--dataset-id",
            args.dataset_id,
            "--task",
            args.task,
            "--feature-mode",
            args.feature_mode,
            "--model-version",
            args.model_version,
            "--label-column",
            args.label_column,
            "--target-column",
            args.target_column or args.label_column,
            "--positive-label",
            args.positive_label,
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
            "--dlq",
            str(run_dir / "dlq.jsonl"),
            "--clean-artifacts",
        ]
        if args.max_rows is not None:
            argv.extend(["--max-rows", str(args.max_rows)])
        if args.protected_attr:
            argv.extend(["--protected-attr", args.protected_attr])
        if args.perf_direction:
            argv.extend(["--perf-direction", args.perf_direction])
        if args.threshold_full is not None:
            argv.extend(["--threshold-full", str(args.threshold_full)])
        if args.enable_shadow_buffer:
            argv.append("--enable-shadow-buffer")
        sys.argv = argv
        offline_main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
