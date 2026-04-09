"""Run fairness_trigger policy on Adult dataset."""
import sys
import os

# Ensure we're in the right directory
os.chdir(r"C:\Users\KIIT\OneDrive\Documents\Playground\drift-fairness")
sys.path.insert(0, r"C:\Users\KIIT\OneDrive\Documents\Playground\drift-fairness")

sys.argv = [
    "offline_run.py",
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
    "--replay-output", "runs/fairness_eval/fairness_trigger/replay.jsonl",
    "--state-output", "runs/fairness_eval/fairness_trigger/state.jsonl",
    "--actions-output", "runs/fairness_eval/fairness_trigger/actions.jsonl",
    "--trainer-output", "runs/fairness_eval/fairness_trigger/trainer_output.jsonl",
    "--canonical-output", "runs/fairness_eval/fairness_trigger/events.jsonl",
    "--processed-index-file", "runs/fairness_eval/fairness_trigger/processed_actions.jsonl",
    "--dlq", "runs/fairness_eval/fairness_trigger/dlq.jsonl",
    "--policy", "fairness_trigger",
    "--fairness-threshold", "0.05",
]

from src.runner.offline_run import main
main()
print("Done: fairness_trigger")
