import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from services.registry_logger.main import run_logger


class TestRegistryLogger(unittest.TestCase):
    def test_append_only_and_idempotent(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "input.jsonl"
            canonical_path = tmp / "events.jsonl"
            checkpoint_path = tmp / "checkpoints.json"
            dlq_path = tmp / "dlq.jsonl"

            events = [
                {"event_type": "batch", "batch_id": "b_000001", "window_id": "w1", "timestamp_start": 0.0, "timestamp_end": 1.0, "label_available": True, "label_delay_batches": 2, "dataset_id": "x", "model_version": "m_000001"},
                {"event_type": "batch", "batch_id": "b_000002", "window_id": "w2", "timestamp_start": 1.0, "timestamp_end": 2.0, "label_available": True, "label_delay_batches": 2, "dataset_id": "x", "model_version": "m_000001"},
            ]
            input_path.write_text("\n".join(json.dumps(e) for e in events))

            run_logger(input_path, canonical_path, checkpoint_path, dlq_path, source_file=str(input_path))
            first = canonical_path.read_text().splitlines()
            self.assertEqual(len(first), 2)

            # re-run should not append duplicates
            run_logger(input_path, canonical_path, checkpoint_path, dlq_path, source_file=str(input_path))
            second = canonical_path.read_text().splitlines()
            self.assertEqual(len(second), 2)

    def test_dlq_on_invalid(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "input.jsonl"
            canonical_path = tmp / "events.jsonl"
            checkpoint_path = tmp / "checkpoints.json"
            dlq_path = tmp / "dlq.jsonl"

            events = [
                {"event_type": "batch", "batch_id": "b_000001"},
            ]
            input_path.write_text("\n".join(json.dumps(e) for e in events))

            run_logger(input_path, canonical_path, checkpoint_path, dlq_path, source_file=str(input_path))
            self.assertFalse(canonical_path.exists())
            self.assertTrue(dlq_path.exists())
            dlq = json.loads(dlq_path.read_text().splitlines()[0])
            self.assertIn("error_type", dlq)
            self.assertIn("raw_event", dlq)
            self.assertIn("expected_schema", dlq)


if __name__ == "__main__":
    unittest.main()
