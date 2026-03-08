import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.trainer.trainer import run_trainer


class TestTextTrainer(unittest.TestCase):
    def test_text_classification_emits_f1(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            csv_path = root / "ag.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["title", "description", "class"])
                for i in range(24):
                    cls = (i % 4) + 1
                    w.writerow(
                        [f"title {i}", f"description topic {cls} token {i}", cls]
                    )

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
                label_column="class",
                target_column="class",
                task="text_classification",
                feature_mode="tfidf",
                window_size=12,
                dlq_path=dlq,
                seed=7,
            )

            rows = [
                json.loads(x)
                for x in trainer_out.read_text(encoding="utf-8").splitlines()
            ]
            costs = [e for e in rows if e.get("event_type") == "cost"]
            self.assertGreaterEqual(len(costs), 2)
            self.assertIn("f1_pre", costs[0])

    def test_text_classification_rejects_non_numeric_labels(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            csv_path = root / "ag_bad.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["title", "description", "class"])
                w.writerow(["t1", "d1", "World"])
                w.writerow(["t2", "d2", "Sports"])

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

            with self.assertRaisesRegex(ValueError, "Non-numeric label"):
                run_trainer(
                    actions_path=actions,
                    events_path=trainer_out,
                    processed_index_path=processed,
                    states_path=states,
                    csv_path=csv_path,
                    label_column="class",
                    target_column="class",
                    task="text_classification",
                    feature_mode="tfidf",
                    window_size=12,
                    dlq_path=dlq,
                    seed=7,
                )


if __name__ == "__main__":
    unittest.main()
