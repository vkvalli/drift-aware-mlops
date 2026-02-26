import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.trainer.trainer import run_trainer


class TestTrainer(unittest.TestCase):
    def _write_actions(self, path: Path) -> None:
        actions = [
            {
                "event_type": "action",
                "action_id": "a_000001",
                "state_id": "s_000001",
                "action_type": "defer",
                "model_version": "m_000001",
                "timestamp": 1.0,
                "reason": "drift_below_threshold",
                "action_params": {"threshold": 1.0, "consecutive_k": 2},
                "requested_at": 1.0,
            },
            {
                "event_type": "action",
                "action_id": "a_000002",
                "state_id": "s_000002",
                "action_type": "fine_tune",
                "model_version": "m_000001",
                "timestamp": 2.0,
                "reason": "drift_threshold",
                "action_params": {"threshold": 1.0, "consecutive_k": 2},
                "requested_at": 2.0,
            },
            {
                "event_type": "action",
                "action_id": "a_000003",
                "state_id": "s_000003",
                "action_type": "full_retrain",
                "model_version": "m_000001",
                "timestamp": 3.0,
                "reason": "drift_full_threshold",
                "action_params": {"threshold": 1.0, "threshold_full": 2.0, "consecutive_k": 2},
                "requested_at": 3.0,
            },
        ]
        path.write_text("\n".join(json.dumps(a) for a in actions))

    def test_basic_execution(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            actions_path = tmp / "actions.jsonl"
            events_path = tmp / "events.jsonl"
            processed_path = tmp / "processed.jsonl"

            self._write_actions(actions_path)

            run_trainer(
                actions_path=actions_path,
                events_path=events_path,
                processed_index_path=processed_path,
                seed=1,
            )

            records = [json.loads(line) for line in events_path.read_text().splitlines()]
            cost_records = [r for r in records if r["event_type"] == "cost"]
            update_records = [r for r in records if r["event_type"] == "model_update"]

            self.assertEqual(len(cost_records), 3)
            self.assertEqual(len(update_records), 2)

            defer = cost_records[0]
            fine_tune = cost_records[1]
            full_retrain = cost_records[2]
            self.assertFalse(defer["executed"])
            self.assertEqual(defer["reason_code"], "defer")
            self.assertTrue(fine_tune["executed"])
            self.assertEqual(fine_tune["reason_code"], "ok")
            self.assertTrue(full_retrain["executed"])
            self.assertGreater(full_retrain["estimated_usd"], fine_tune["estimated_usd"])

    def test_idempotency(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            actions_path = tmp / "actions.jsonl"
            events_path = tmp / "events.jsonl"
            processed_path = tmp / "processed.jsonl"

            self._write_actions(actions_path)
            processed_path.write_text(
                json.dumps({"action_id": "a_000002", "processed_at": 1.0, "outcome": "ok"})
                + "\n"
            )

            run_trainer(
                actions_path=actions_path,
                events_path=events_path,
                processed_index_path=processed_path,
                seed=1,
            )

            records = [json.loads(line) for line in events_path.read_text().splitlines()]
            cost_records = [r for r in records if r["event_type"] == "cost"]
            update_records = [r for r in records if r["event_type"] == "model_update"]

            self.assertEqual(len(cost_records), 2)
            self.assertEqual(len(update_records), 1)
            self.assertEqual(cost_records[0]["action_id"], "a_000001")

    def test_determinism(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            actions_path = tmp / "actions.jsonl"
            processed_path = tmp / "processed.jsonl"

            self._write_actions(actions_path)

            events_path_1 = tmp / "events1.jsonl"
            events_path_2 = tmp / "events2.jsonl"

            run_trainer(
                actions_path=actions_path,
                events_path=events_path_1,
                processed_index_path=processed_path,
                seed=1,
            )

            processed_path.unlink()

            run_trainer(
                actions_path=actions_path,
                events_path=events_path_2,
                processed_index_path=processed_path,
                seed=1,
            )

            def _normalize(lines):
                records = []
                for line in lines:
                    record = json.loads(line)
                    record.pop("logged_at", None)
                    records.append(record)
                return records

            seq1 = _normalize(events_path_1.read_text().splitlines())
            seq2 = _normalize(events_path_2.read_text().splitlines())
            self.assertEqual(seq1, seq2)


if __name__ == "__main__":
    unittest.main()
