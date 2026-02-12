import unittest
from src.schemas.batch import BatchEvent


class TestBatchEvent(unittest.TestCase):
    def test_batch_event_creation(self):
        event = BatchEvent(
            batch_id="b_000001",
            window_id="w1",
            timestamp_start=0.0,
            timestamp_end=1.0,
            label_available=True,
            label_delay_batches=2,
            dataset_id="electricity",
            model_version="m_000001",
        )
        self.assertEqual(event.batch_id, "b_000001")
        self.assertTrue(event.label_available)


if __name__ == "__main__":
    unittest.main()
