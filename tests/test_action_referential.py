import unittest
from src.schemas.action import ActionEvent


class TestActionReferential(unittest.TestCase):
    def test_action_id_prefixes(self):
        event = ActionEvent(
            action_id="a_000001",
            state_id="s_000001",
            action_type="defer",
            model_version="m_000001",
            timestamp=0.0,
            reason="drift_below_threshold",
            action_params={"threshold": 1.0},
            requested_at=0.0,
        )
        self.assertTrue(event.state_id.startswith("s_"))
        self.assertTrue(event.action_id.startswith("a_"))


if __name__ == "__main__":
    unittest.main()
