"""Run all 4 fairness benchmark policies on Adult dataset."""
import os
import sys

# Force the workspace root to be the drift-fairness directory
WORKSPACE = r"C:\Users\KIIT\OneDrive\Documents\Playground\drift-fairness"
os.chdir(WORKSPACE)
# Remove any existing src paths from sys.path and prepend ours
sys.path = [p for p in sys.path if "drift-aware-mlops" not in p]
sys.path.insert(0, WORKSPACE)

from pathlib import Path
from src.runner.offline_run import main as offline_main

BASE_ARGS = [
    "--csv", "data/adult.csv",
    "--task", "classification",
    "--label-column", "income",
    "--target-column", "income",
    "--protected-attr", "sex",
    "--dataset-id", "adult_openml",
    "--window-size", "200",
    "--max-rows", "40000",
    "--label-delay-batches", "2",
    "--seed", "111",
]

POLICIES = {
    "no_retrain": [],
    "fixed_interval": [],
    "threshold_persistence": [],
    "fairness_trigger": ["--fairness-threshold", "0.05"],
}

for policy, extra_args in POLICIES.items():
    run_dir = f"runs/fairness_eval/{policy}"
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    sys.argv = ["offline_run.py"] + BASE_ARGS + [
        "--replay-output", f"{run_dir}/replay.jsonl",
        "--state-output", f"{run_dir}/state.jsonl",
        "--actions-output", f"{run_dir}/actions.jsonl",
        "--trainer-output", f"{run_dir}/trainer_output.jsonl",
        "--canonical-output", f"{run_dir}/events.jsonl",
        "--processed-index-file", f"{run_dir}/processed_actions.jsonl",
        "--dlq", f"{run_dir}/dlq.jsonl",
        "--policy", policy,
    ] + extra_args
    print(f"Running: {policy}")
    try:
        offline_main()
        print(f"  Done: {policy}")
    except SystemExit as e:
        if e.code != 0:
            print(f"  FAILED: {policy} exit={e.code}")
        else:
            print(f"  Done: {policy}")
    except Exception as e:
        print(f"  ERROR: {policy}: {e}")

print("All policies complete.")
