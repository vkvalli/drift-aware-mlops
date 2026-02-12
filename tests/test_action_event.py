import unittest
from src.schemas.action import ActionEvent


class TestActionEvent(unittest.TestCase):
    def test_action_event_creation(self):
        event = ActionEvent(
            action_id="a_000001",
            state_id="s_000001",
            action_type="defer",
            model_version="m_000001",
            timestamp=2.0,
            reason="drift_below_threshold",
            action_params={"threshold": 1.0},
            requested_at=2.0,
        )
        self.assertEqual(event.action_id, "a_000001")
        self.assertEqual(event.action_type, "defer")
        self.assertEqual(event.schema_version, "v1")


if __name__ == "__main__":
    unittest.main()
