import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.policy.policy_agent import run_policy


class TestPolicyAgent(unittest.TestCase):
    def test_threshold_logic_and_ids(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            states_path = tmp / "states.jsonl"
            actions_path = tmp / "actions.jsonl"

            states = [
                {
                    "event_type": "state",
                    "state_id": "s_000001",
                    "batch_id": "b_000001",
                    "window_id": "w_000001",
                    "timestamp": 1.0,
                    "model_version": "m_000001",
                    "label_available": False,
                    "drift_ready": True,
                    "drift_core": 0.5,
                    "drift_max": 0.5,
                    "perf_core": 0.0,
                    "fairness_core": 0.0,
                },
                {
                    "event_type": "state",
                    "state_id": "s_000002",
                    "batch_id": "b_000002",
                    "window_id": "w_000002",
                    "timestamp": 2.0,
                    "model_version": "m_000001",
                    "label_available": False,
                    "drift_ready": True,
                    "drift_core": 2.5,
                    "drift_max": 2.5,
                    "perf_core": 0.0,
                    "fairness_core": 0.0,
                },
                {
                    "event_type": "state",
                    "state_id": "s_000003",
                    "batch_id": "b_000003",
                    "window_id": "w_000003",
                    "timestamp": 3.0,
                    "model_version": "m_000001",
                    "label_available": False,
                    "drift_ready": True,
                    "drift_core": 2.0,
                    "drift_max": 2.0,
                    "perf_core": 0.0,
                    "fairness_core": 0.0,
                },
            ]
            states_path.write_text("\n".join(json.dumps(s) for s in states))

            run_policy(
                states_path=states_path,
                actions_path=actions_path,
                threshold=1.0,
                action_start=1,
                seed=1,
                consecutive_k=2,
            )

            records = [json.loads(line) for line in actions_path.read_text().splitlines()]
            self.assertEqual(len(records), 3)

            types = [r["action_type"] for r in records]
            self.assertEqual(types, ["defer", "defer", "fine_tune"])

            for r in records:
                self.assertTrue(r["action_id"].startswith("a_"))
                self.assertTrue(r["state_id"].startswith("s_"))
                self.assertIn("reason", r)
                self.assertIn("action_params", r)
                self.assertIn("requested_at", r)

    def test_no_retrain_policy(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            states_path = tmp / "states.jsonl"
            actions_path = tmp / "actions.jsonl"

            states = [
                {
                    "event_type": "state",
                    "state_id": "s_000001",
                    "batch_id": "b_000001",
                    "window_id": "w_000001",
                    "timestamp": 1.0,
                    "model_version": "m_000001",
                    "label_available": False,
                    "drift_ready": True,
                    "drift_core": 2.5,
                    "drift_max": 2.5,
                    "perf_core": 0.0,
                    "fairness_core": 0.0,
                },
            ]
            states_path.write_text("\n".join(json.dumps(s) for s in states))

            run_policy(
                states_path=states_path,
                actions_path=actions_path,
                policy="no_retrain",
            )

            records = [json.loads(line) for line in actions_path.read_text().splitlines()]
            self.assertEqual(records[0]["action_type"], "defer")

    def test_fixed_interval_policy(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            states_path = tmp / "states.jsonl"
            actions_path = tmp / "actions.jsonl"

            states = []
            for i in range(1, 5):
                states.append(
                    {
                        "event_type": "state",
                        "state_id": f"s_{i:06d}",
                        "batch_id": f"b_{i:06d}",
                        "window_id": f"w_{i:06d}",
                        "timestamp": float(i),
                        "model_version": "m_000001",
                        "label_available": False,
                        "drift_ready": True,
                        "drift_core": 0.1,
                        "drift_max": 0.1,
                        "perf_core": 0.0,
                        "fairness_core": 0.0,
                    }
                )
            states_path.write_text("\n".join(json.dumps(s) for s in states))

            run_policy(
                states_path=states_path,
                actions_path=actions_path,
                policy="fixed_interval",
                fixed_interval=2,
            )

            records = [json.loads(line) for line in actions_path.read_text().splitlines()]
            types = [r["action_type"] for r in records]
            self.assertEqual(types, ["defer", "fine_tune", "defer", "fine_tune"])


if __name__ == "__main__":
    unittest.main()
