import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.ingestion.replay import run_replay


class TestReplay(unittest.TestCase):
    def _run_and_load(self, flush: bool):
        sample = Path("tests/data/sample.csv")
        with TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "events.jsonl"
            num_batches = run_replay(
                csv_path=sample,
                window_size=2,
                label_delay_batches=2,
                flush_pending_labels=flush,
                dataset_id="electricity",
                model_version="m_000001",
                output_path=out_path,
            )
            records = [json.loads(line) for line in out_path.read_text().splitlines()]
            return num_batches, records

    def test_replay_without_flush(self):
        num_batches, records = self._run_and_load(flush=False)
        batch_records = [r for r in records if r["event_type"] == "BatchEvent"]
        label_records = [r for r in records if r["event_type"] == "LabelEvent"]

        self.assertEqual(num_batches, 6)
        self.assertEqual(len(batch_records), 6)
        self.assertEqual(len(label_records), num_batches - 2)

        for idx, record in enumerate(batch_records):
            if idx < 2:
                self.assertFalse(record["label_available"])
            else:
                self.assertTrue(record["label_available"])
            self.assertEqual(record["label_delay_batches"], 2)
            self.assertTrue(record["batch_id"].startswith("b_"))

        timestamps = [r["timestamp_start"] for r in batch_records]
        self.assertEqual(timestamps, sorted(timestamps))

        label_batch_ids = {r["batch_id"] for r in label_records}
        expected_labels = {f"b_{i:06d}" for i in range(1, 5)}
        self.assertEqual(label_batch_ids, expected_labels)

        index_by_batch = {}
        for i, r in enumerate(records):
            if r["event_type"] == "BatchEvent":
                index_by_batch[r["batch_id"]] = i

        for i, r in enumerate(records):
            if r["event_type"] == "LabelEvent":
                self.assertIn(r["batch_id"], index_by_batch)
                self.assertGreater(i, index_by_batch[r["batch_id"]])

    def test_replay_with_flush(self):
        num_batches, records = self._run_and_load(flush=True)
        batch_records = [r for r in records if r["event_type"] == "BatchEvent"]
        label_records = [r for r in records if r["event_type"] == "LabelEvent"]

        self.assertEqual(num_batches, 6)
        self.assertEqual(len(batch_records), 6)
        self.assertEqual(len(label_records), num_batches)

        label_batch_ids = {r["batch_id"] for r in label_records}
        expected_labels = {f"b_{i:06d}" for i in range(1, 7)}
        self.assertEqual(label_batch_ids, expected_labels)

        index_by_batch = {}
        for i, r in enumerate(records):
            if r["event_type"] == "BatchEvent":
                index_by_batch[r["batch_id"]] = i

        for i, r in enumerate(records):
            if r["event_type"] == "LabelEvent":
                self.assertGreater(i, index_by_batch[r["batch_id"]])


if __name__ == "__main__":
    unittest.main()
