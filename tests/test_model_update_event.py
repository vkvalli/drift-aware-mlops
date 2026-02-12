import unittest
from src.schemas.model_update import ModelUpdateEvent


class TestModelUpdateEvent(unittest.TestCase):
    def test_model_update_event_creation(self):
        event = ModelUpdateEvent(
            update_id="u1",
            action_id="a_000001",
            timestamp=4.0,
            new_model_version="m_000001",
            artifact_path="/tmp/model.pt",
            eval_summary={},
            executed=True,
        )
        self.assertEqual(event.update_id, "u1")
        self.assertEqual(event.new_model_version, "m_000001")
        self.assertEqual(event.schema_version, "v1")


if __name__ == "__main__":
    unittest.main()
