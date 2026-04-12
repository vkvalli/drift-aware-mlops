import json
import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.policy.policy_agent import run_policy
from src.rl.linucb import LinUCB


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

            records = [
                json.loads(line) for line in actions_path.read_text().splitlines()
            ]
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

            records = [
                json.loads(line) for line in actions_path.read_text().splitlines()
            ]
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

            records = [
                json.loads(line) for line in actions_path.read_text().splitlines()
            ]
            types = [r["action_type"] for r in records]
            self.assertEqual(types, ["defer", "fine_tune", "defer", "fine_tune"])

    def test_fairness_trigger_policy_uses_fairness_core(self):
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
                    "drift_core": 0.2,
                    "drift_max": 0.2,
                    "perf_core": 0.0,
                    "fairness_core": 0.15,
                    "fairness_ready": False,
                    "fairness_support": 40,
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
                    "drift_core": 2.0,
                    "drift_max": 2.0,
                    "perf_core": 0.0,
                    "fairness_core": 0.15,
                    "fairness_ready": True,
                    "fairness_support": 200,
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
                    "drift_core": 0.3,
                    "drift_max": 0.3,
                    "perf_core": 0.0,
                    "fairness_core": 0.05,
                    "fairness_ready": True,
                    "fairness_support": 200,
                },
            ]
            states_path.write_text("\n".join(json.dumps(s) for s in states))

            run_policy(
                states_path=states_path,
                actions_path=actions_path,
                policy="fairness_trigger",
                threshold=1.0,
                fairness_threshold=0.1,
                consecutive_k=2,
            )

            records = [
                json.loads(line) for line in actions_path.read_text().splitlines()
            ]
            self.assertEqual(records[0]["action_type"], "defer")
            self.assertEqual(records[0]["reason"], "fairness_not_ready")
            self.assertEqual(records[1]["action_type"], "fine_tune")
            self.assertEqual(records[1]["reason"], "fairness_trigger")
            self.assertEqual(records[2]["action_type"], "defer")
            self.assertEqual(records[2]["reason"], "below_fairness_threshold")

    def test_fairness_trigger_can_escalate_to_full_retrain(self):
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
                    "drift_core": 0.1,
                    "drift_max": 0.1,
                    "perf_core": 0.0,
                    "fairness_core": 0.18,
                    "fairness_ready": True,
                    "fairness_support": 250,
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
                    "drift_core": 0.1,
                    "drift_max": 0.1,
                    "perf_core": 0.0,
                    "fairness_core": 0.18,
                    "fairness_ready": True,
                    "fairness_support": 250,
                },
            ]
            states_path.write_text("\n".join(json.dumps(s) for s in states))

            run_policy(
                states_path=states_path,
                actions_path=actions_path,
                policy="fairness_trigger",
                fairness_threshold=0.10,
                fairness_threshold_full=0.15,
                consecutive_k=2,
            )

            records = [
                json.loads(line) for line in actions_path.read_text().splitlines()
            ]
            self.assertEqual(records[0]["action_type"], "defer")
            self.assertEqual(records[1]["action_type"], "full_retrain")
            self.assertEqual(records[1]["reason"], "fairness_full_retrain")

    def test_linucb_can_use_fairness_features(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            states_path = tmp / "states.jsonl"
            actions_path = tmp / "actions.jsonl"
            model_path = tmp / "linucb.pkl"

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
                    "drift_core": 0.1,
                    "drift_max": 0.1,
                    "perf_core": 0.0,
                    "cooldown_active": False,
                    "fairness_core": 0.0,
                    "fairness_ready": True,
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
                    "drift_core": 0.1,
                    "drift_max": 0.1,
                    "perf_core": 0.0,
                    "cooldown_active": False,
                    "fairness_core": 1.0,
                    "fairness_ready": True,
                },
            ]
            states_path.write_text("\n".join(json.dumps(s) for s in states))

            # Build a 6-feature LinUCB where fairness_core (index 4) drives action.
            model = LinUCB(n_features=6, alpha=0.0)
            model.b["fine_tune"][4, 0] = 10.0
            with model_path.open("wb") as f:
                pickle.dump(model, f)

            run_policy(
                states_path=states_path,
                actions_path=actions_path,
                policy="linucb",
                linucb_model_path=model_path,
                warm_start_n=0,
                epsilon1=0.0,
                epsilon1_steps=0,
                epsilon2=0.0,
                seed=1,
            )

            records = [
                json.loads(line) for line in actions_path.read_text().splitlines()
            ]
            self.assertEqual(records[0]["action_type"], "defer")
            self.assertEqual(records[1]["action_type"], "fine_tune")


if __name__ == "__main__":
    unittest.main()
