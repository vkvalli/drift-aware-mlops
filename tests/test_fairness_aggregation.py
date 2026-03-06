import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.aggregate_matrix import aggregate


class TestFairnessAggregation(unittest.TestCase):
    def test_summary_includes_fairness_columns(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_dir = root / "matrix_1" / "policy=test_seed1"
            run_dir.mkdir(parents=True)
            events = run_dir / "events.jsonl"
            rows = [
                {
                    "event_type": "run_meta",
                    "policy": "test",
                    "seed": 1,
                    "window_size": 2,
                    "ref_window": 2,
                    "label_delay_batches": 0,
                    "max_rows": 10,
                },
                {"event_type": "batch", "batch_id": "b_000001"},
                {
                    "event_type": "action",
                    "action_id": "a_000001",
                    "action_type": "fine_tune",
                },
                {
                    "event_type": "cost",
                    "action_id": "a_000001",
                    "estimated_usd": 1.5,
                    "f1_pre": 0.4,
                    "dpd_pre": 0.1,
                    "eod_pre": 0.2,
                    "mae_pre": 1.2,
                },
            ]
            events.write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
            )

            out = root / "summary.csv"
            aggregate(root / "matrix_1", out)

            with out.open("r", encoding="utf-8") as f:
                row = next(csv.DictReader(f))
            self.assertIn("dpd_mean", row)
            self.assertIn("eod_mean", row)
            self.assertIn("mae_mean", row)


if __name__ == "__main__":
    unittest.main()
