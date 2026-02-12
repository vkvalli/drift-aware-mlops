import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.runner.offline_run import main as offline_main


def _run_runner(tmp: Path, seed: int) -> Path:
    csv_path = Path("tests/data/sample.csv").resolve()
    replay_path = tmp / "replay.jsonl"
    state_path = tmp / "state.jsonl"
    actions_path = tmp / "actions.jsonl"
    trainer_path = tmp / "trainer.jsonl"
    canonical_path = tmp / "events.jsonl"
    processed_path = tmp / "processed.jsonl"

    args = [
        "--csv",
        str(csv_path),
        "--window-size",
        "2",
        "--label-delay-batches",
        "2",
        "--ref-window",
        "2",
        "--threshold",
        "1.0",
        "--seed",
        str(seed),
        "--replay-output",
        str(replay_path),
        "--state-output",
        str(state_path),
        "--actions-output",
        str(actions_path),
        "--trainer-output",
        str(trainer_path),
        "--canonical-output",
        str(canonical_path),
        "--processed-index-file",
        str(processed_path),
        "--clean-artifacts",
    ]

    import sys

    old_argv = sys.argv
    try:
        sys.argv = ["offline_run.py"] + args
        offline_main()
    finally:
        sys.argv = old_argv

    return canonical_path


class TestOfflineE2E(unittest.TestCase):
    def test_counts_and_ordering(self):
        with TemporaryDirectory() as tmpdir:
            canonical_path = _run_runner(Path(tmpdir), seed=1)

            records = [json.loads(line) for line in canonical_path.read_text().splitlines()]
            events = [r for r in records if r.get("event_type") != "run_meta"]

            batch = [r for r in events if r["event_type"] in {"BatchEvent", "batch"}]
            state = [r for r in events if r["event_type"] == "state"]
            action = [r for r in events if r["event_type"] == "action"]
            cost = [r for r in events if r["event_type"] == "cost"]
            update = [r for r in events if r["event_type"] == "model_update"]

            expected_batches = 6
            self.assertEqual(len(batch), expected_batches)
            self.assertEqual(len(state), len(batch))
            self.assertEqual(len(action), len(state))
            self.assertEqual(len(cost), len(action))

            fine_tune_actions = [a for a in action if a["action_type"] == "fine_tune"]
            self.assertEqual(len(update), len(fine_tune_actions))

            index = {}
            for i, r in enumerate(events):
                if r["event_type"] in {"BatchEvent", "batch"}:
                    index[("batch", r["batch_id"])] = i
                if r["event_type"] == "state":
                    index[("state", r["state_id"])] = i
                if r["event_type"] == "action":
                    index[("action", r["action_id"])] = i
                if r["event_type"] == "cost":
                    index[("cost", r["action_id"])] = i

            for s in state:
                self.assertGreater(index[("state", s["state_id"])], index[("batch", s["batch_id"])])

            for a in action:
                self.assertGreater(index[("action", a["action_id"])], index[("state", a["state_id"])])

            for c in cost:
                self.assertGreater(
                    index[("cost", c["action_id"])],
                    index[("action", c["action_id"])],
                )

    def test_determinism(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            path1 = _run_runner(tmp / "run1", seed=1)
            path2 = _run_runner(tmp / "run2", seed=1)

            def _normalize(path: Path):
                records = [json.loads(line) for line in path.read_text().splitlines()]
                events = [r for r in records if r.get("event_type") != "run_meta"]
                seq = []
                for r in events:
                    if r["event_type"] == "state":
                        seq.append((r["event_type"], r["batch_id"], r["state_id"], r["drift_core"]))
                    elif r["event_type"] == "action":
                        seq.append((r["event_type"], r["action_id"], r["state_id"]))
                    elif r["event_type"] == "cost":
                        seq.append((r["event_type"], r["action_id"], r["executed"]))
                    elif r["event_type"] == "model_update":
                        seq.append((r["event_type"], r["action_id"], r["new_model_version"]))
                    elif r["event_type"] in {"BatchEvent", "batch"}:
                        seq.append((r["event_type"], r["batch_id"]))
                return seq

            self.assertEqual(_normalize(path1), _normalize(path2))


if __name__ == "__main__":
    unittest.main()
