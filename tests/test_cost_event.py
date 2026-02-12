import unittest
from src.schemas.cost import CostEvent


class TestCostEvent(unittest.TestCase):
    def test_cost_event_creation(self):
        event = CostEvent(
            cost_id="c1",
            action_id="a_000001",
            timestamp=3.0,
            train_time_s=3.0,
            estimated_usd=1.23,
            energy_kwh=0.45,
            co2_kg=0.0005,
            executed=True,
            reason_code="ok",
        )
        self.assertEqual(event.cost_id, "c1")
        self.assertEqual(event.action_id, "a_000001")
        self.assertEqual(event.schema_version, "v1")


if __name__ == "__main__":
    unittest.main()
