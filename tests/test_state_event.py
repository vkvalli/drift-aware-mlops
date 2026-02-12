import unittest
from src.schemas.state import StateEvent


class TestStateEvent(unittest.TestCase):
    def test_state_event_creation(self):
        event = StateEvent(
            state_id="s_000001",
            batch_id="b_000001",
            window_id="w1",
            timestamp=1.0,
            model_version="m_000001",
            label_available=False,
            drift_core=0.1,
            drift_max=0.2,
            drift_ready=True,
            perf_core=0.9,
            fairness_core=0.05,
        )
        self.assertEqual(event.state_id, "s_000001")
        self.assertEqual(event.batch_id, "b_000001")
        self.assertEqual(event.schema_version, "v1")


if __name__ == "__main__":
    unittest.main()
