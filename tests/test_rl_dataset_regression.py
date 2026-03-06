import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.build_rl_dataset import build_dataset


class TestRlDatasetRegression(unittest.TestCase):
    def test_lower_better_direction_for_mae(self):
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
                    "drift_core": 0.2,
                    "drift_max": 0.3,
                    "drift_ready": True,
                    "cooldown_active": False,
                },
                {
                    "event_type": "action",
                    "action_id": "a_000001",
                    "state_id": "s_000001",
                    "action_type": "fine_tune",
                    "model_version": "m_000001",
                },
                {
                    "event_type": "action",
                    "action_id": "a_000002",
                    "state_id": "s_000002",
                    "action_type": "fine_tune",
                    "model_version": "m_000001",
                },
                {
                    "event_type": "cost",
                    "action_id": "a_000001",
                    "estimated_usd": 1.5,
                    "mae_pre": 2.0,
                },
                {
                    "event_type": "cost",
                    "action_id": "a_000002",
                    "estimated_usd": 1.5,
                    "mae_pre": 1.0,
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
                perf_field="mae_pre",
                perf_direction="lower_better",
            )

            with out.open("r", encoding="utf-8") as f:
                data = list(csv.DictReader(f))
            self.assertEqual(len(data), 2)
            # MAE drops from 2.0 to 1.0, so lower_better should produce positive delta on row2.
            self.assertGreater(float(data[1]["delta_perf"]), 0.0)


if __name__ == "__main__":
    unittest.main()
