import json
import unittest
from pathlib import Path

from src.logger.jsonl_logger import write_event
from src.schemas.batch import BatchEvent
from src.schemas.action import ActionEvent
from src.schemas.cost import CostEvent


class TestJsonlLogger(unittest.TestCase):
    def test_write_events(self):
        path = Path("artifacts/events.jsonl")
        if path.exists():
            path.unlink()

        events = [
            BatchEvent(
                batch_id="b_000001",
                window_id="w1",
                timestamp_start=0.0,
                timestamp_end=1.0,
                label_available=True,
                label_delay_batches=2,
                dataset_id="electricity",
                model_version="m_000001",
            ),
            ActionEvent(
                action_id="a_000001",
                state_id="s_000001",
                action_type="defer",
                model_version="m_000001",
                timestamp=2.0,
                reason="drift_below_threshold",
                action_params={"threshold": 1.0},
                requested_at=2.0,
            ),
            CostEvent(
                cost_id="c1",
                action_id="a_000001",
                timestamp=3.0,
                train_time_s=3.0,
                estimated_usd=1.23,
                energy_kwh=0.45,
                co2_kg=0.0005,
                executed=True,
                reason_code="ok",
            ),
        ]

        for event in events:
            write_event(event, path)

        lines = path.read_text().strip().splitlines()
        self.assertEqual(len(lines), 3)

        parsed = [json.loads(line) for line in lines]
        for record in parsed:
            self.assertIn("event_type", record)
            self.assertIn("logged_at", record)


if __name__ == "__main__":
    unittest.main()
