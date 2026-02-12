import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.schemas.batch import BatchEvent
from src.schemas.compat import PYDANTIC_AVAILABLE
from src.schemas.validate import validate_or_dlq


class TestValidateOrDlq(unittest.TestCase):
    def test_valid_batch_event(self):
        with TemporaryDirectory() as tmpdir:
            dlq_path = Path(tmpdir) / "dlq.jsonl"
            event = {
                "batch_id": "b_000001",
                "window_id": "w1",
                "timestamp_start": 0.0,
                "timestamp_end": 1.0,
                "label_available": True,
                "label_delay_batches": 2,
                "dataset_id": "electricity",
                "model_version": "m_000001",
            }
            validated = validate_or_dlq(event, BatchEvent, dlq_path)
            self.assertIsNotNone(validated)
            self.assertFalse(dlq_path.exists())

    def test_invalid_event_goes_to_dlq(self):
        with TemporaryDirectory() as tmpdir:
            dlq_path = Path(tmpdir) / "dlq.jsonl"
            event = {
                "batch_id": None,
                "window_id": "w1",
                "timestamp_start": 0.0,
                "timestamp_end": 1.0,
                "label_available": True,
                "label_delay_batches": 2,
                "dataset_id": "electricity",
                "model_version": "m_000001",
            }
            validated = validate_or_dlq(event, BatchEvent, dlq_path)
            self.assertIsNone(validated)
            self.assertTrue(dlq_path.exists())
            record = json.loads(dlq_path.read_text().splitlines()[0])
            self.assertIn("error_type", record)
            self.assertIn("expected_schema", record)

    @unittest.skipUnless(PYDANTIC_AVAILABLE, "pydantic not installed")
    def test_strict_type_validation_pydantic(self):
        with TemporaryDirectory() as tmpdir:
            dlq_path = Path(tmpdir) / "dlq.jsonl"
            event = {
                "batch_id": 123,
                "window_id": "w1",
                "timestamp_start": 0.0,
                "timestamp_end": 1.0,
                "label_available": True,
                "label_delay_batches": 2,
                "dataset_id": "electricity",
                "model_version": "m_000001",
            }
            validated = validate_or_dlq(event, BatchEvent, dlq_path)
            self.assertIsNone(validated)


if __name__ == "__main__":
    unittest.main()
