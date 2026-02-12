import unittest
from src.schemas.label import LabelEvent


class TestLabelEvent(unittest.TestCase):
    def test_label_event_creation(self):
        event = LabelEvent(
            label_id="l_000001",
            batch_id="b_000001",
            timestamp=5.0,
        )
        self.assertEqual(event.label_id, "l_000001")
        self.assertEqual(event.batch_id, "b_000001")
        self.assertEqual(event.schema_version, "v1")


if __name__ == "__main__":
    unittest.main()
