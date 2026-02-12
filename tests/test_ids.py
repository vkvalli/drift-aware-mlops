import unittest
from src.utils.ids import IdCounter, next_id


class TestIds(unittest.TestCase):
    def test_next_id_format(self):
        self.assertEqual(next_id("b_", 1), "b_000001")

    def test_id_counter_increments(self):
        counter = IdCounter("s_", start=1)
        self.assertEqual(counter.next(), "s_000001")
        self.assertEqual(counter.next(), "s_000002")


if __name__ == "__main__":
    unittest.main()
