import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.trainer.trainer import run_trainer


class TestRegressionTrainer(unittest.TestCase):
    def test_regression_cost_fields_present(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            csv_path = root / "reg.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["x1", "x2", "target"])
                for i in range(20):
                    w.writerow([i, i * 2, i * 0.5 + 1.0])

            states = root / "state.jsonl"
            actions = root / "actions.jsonl"
            trainer_out = root / "trainer.jsonl"
            processed = root / "processed.jsonl"
            dlq = root / "dlq.jsonl"

            states.write_text(
                "\n".join(
                    [
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
                        ),
                        json.dumps(
                            {
                                "event_type": "state",
                                "state_id": "s_000002",
                                "batch_id": "b_000002",
                                "window_id": "w2",
                                "timestamp": 2.0,
                                "model_version": "m_000001",
                                "label_available": True,
                                "drift_core": 0.2,
                                "drift_max": 0.2,
                                "drift_ready": True,
                                "perf_core": 0.0,
                                "fairness_core": 0.0,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            actions.write_text(
                "\n".join(
                    [
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
                        ),
                        json.dumps(
                            {
                                "event_type": "action",
                                "action_id": "a_000002",
                                "state_id": "s_000002",
                                "action_type": "fine_tune",
                                "model_version": "m_000001",
                                "timestamp": 2.0,
                                "reason": "test",
                                "action_params": {},
                                "requested_at": 2.0,
                            }
                        ),
                    ]
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
                label_column="target",
                target_column="target",
                task="regression",
                window_size=10,
                dlq_path=dlq,
                seed=1,
            )

            events = [
                json.loads(x)
                for x in trainer_out.read_text(encoding="utf-8").splitlines()
            ]
            costs = [e for e in events if e.get("event_type") == "cost"]
            self.assertGreaterEqual(len(costs), 2)
            self.assertIn("mae_pre", costs[0])
            self.assertIn("rmse_pre", costs[0])
            self.assertIn("r2_pre", costs[0])


if __name__ == "__main__":
    unittest.main()
