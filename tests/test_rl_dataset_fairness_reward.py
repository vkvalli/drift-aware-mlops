import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.build_rl_dataset import build_dataset


class TestRlDatasetFairnessReward(unittest.TestCase):
    def test_reward_includes_fairness_penalty(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            events = root / "events.jsonl"
            out = root / "rl.csv"
            rows = [
                {
                    "event_type": "state",
                    "state_id": "s_000001",
                    "batch_id": "b_000001",
                    "drift_core": 0.1,
                    "drift_max": 0.2,
                    "drift_ready": True,
                    "cooldown_active": False,
                },
                {
                    "event_type": "state",
                    "state_id": "s_000002",
                    "batch_id": "b_000002",
                    "drift_core": 0.1,
                    "drift_max": 0.2,
                    "drift_ready": True,
                    "cooldown_active": False,
                },
                {
                    "event_type": "action",
                    "action_id": "a_000001",
                    "state_id": "s_000001",
                    "action_type": "defer",
                    "model_version": "m_000001",
                },
                {
                    "event_type": "action",
                    "action_id": "a_000002",
                    "state_id": "s_000002",
                    "action_type": "defer",
                    "model_version": "m_000001",
                },
                {
                    "event_type": "cost",
                    "action_id": "a_000001",
                    "estimated_usd": 0.0,
                    "dpd_pre": 0.1,
                    "eod_pre": 0.2,
                },
                {
                    "event_type": "cost",
                    "action_id": "a_000002",
                    "estimated_usd": 0.0,
                    "dpd_pre": 0.0,
                    "eod_pre": 0.0,
                },
            ]
            events.write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
            )

            build_dataset(
                events_path=events,
                out_path=out,
                w_cost=0.25,
                reward_mode="raw",
                perf_window=2,
                w_drift=0.0,
                drift_threshold=0.0,
                normalize_perf=False,
                action_penalty=0.0,
                defer_bonus=0.0,
                defer_threshold=0.0,
                perf_field="accuracy_pre",
                perf_direction="higher_better",
                w_fair=1.0,
                fairness_field="max_gap",
            )

            with out.open("r", encoding="utf-8") as f:
                data = list(csv.DictReader(f))
            self.assertEqual(len(data), 2)
            self.assertAlmostEqual(float(data[0]["fairness_gap"]), 0.2, places=6)
            self.assertAlmostEqual(float(data[1]["fairness_gap"]), 0.0, places=6)
            self.assertAlmostEqual(float(data[0]["fair_norm"]), 1.0, places=6)
            self.assertAlmostEqual(float(data[1]["fair_norm"]), 0.0, places=6)
            self.assertLess(float(data[0]["reward"]), float(data[1]["reward"]))


if __name__ == "__main__":
    unittest.main()
