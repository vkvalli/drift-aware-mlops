import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.trainer.trainer import run_trainer


class TestFairnessMetrics(unittest.TestCase):
    def test_dpd_and_eod_are_logged(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            csv_path = root / "adult.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["age", "hours", "sex", "class"])
                for i in range(20):
                    sex = "Male" if i % 2 == 0 else "Female"
                    label = 1 if i % 3 == 0 else 0
                    w.writerow([25 + i, 30 + (i % 10), sex, label])

            states = root / "state.jsonl"
            actions = root / "actions.jsonl"
            trainer_out = root / "trainer.jsonl"
            processed = root / "processed.jsonl"
            dlq = root / "dlq.jsonl"

            states.write_text(
                json.dumps(
                    {
                        "event_type": "state",
                        "state_id": "s_000001",
                        "batch_id": "b_000001",
                        "window_id": "w1",
                        "timestamp": 1.0,
                        "model_version": "m_000001",
                        "label_available": True,
                        "drift_core": 0.1,
                        "drift_max": 0.1,
                        "drift_ready": True,
                        "perf_core": 0.0,
                        "fairness_core": 0.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            actions.write_text(
                json.dumps(
                    {
                        "event_type": "action",
                        "action_id": "a_000001",
                        "state_id": "s_000001",
                        "action_type": "fine_tune",
                        "model_version": "m_000001",
                        "timestamp": 1.0,
                        "reason": "test",
                        "action_params": {},
                        "requested_at": 1.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            run_trainer(
                actions_path=actions,
                events_path=trainer_out,
                processed_index_path=processed,
                states_path=states,
                csv_path=csv_path,
                label_column="class",
                target_column="class",
                task="classification",
                window_size=20,
                protected_attr="sex",
                dlq_path=dlq,
                seed=3,
            )

            events = [
                json.loads(x)
                for x in trainer_out.read_text(encoding="utf-8").splitlines()
            ]
            cost = next(e for e in events if e.get("event_type") == "cost")
            self.assertIn("dpd_pre", cost)
            self.assertIn("eod_pre", cost)


if __name__ == "__main__":
    unittest.main()
