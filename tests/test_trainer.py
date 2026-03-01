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

    def _write_states(self, path: Path) -> None:
        states = [
            {
                "event_type": "state",
                "state_id": "s_000001",
                "batch_id": "b_000001",
                "window_id": "w_000001",
                "timestamp": 1.0,
                "model_version": "m_000001",
                "label_available": True,
                "drift_core": 0.1,
                "drift_max": 0.1,
                "drift_ready": True,
                "perf_core": 0.0,
                "fairness_core": 0.0,
            },
            {
                "event_type": "state",
                "state_id": "s_000002",
                "batch_id": "b_000002",
                "window_id": "w_000002",
                "timestamp": 2.0,
                "model_version": "m_000001",
                "label_available": True,
                "drift_core": 0.2,
                "drift_max": 0.2,
                "drift_ready": True,
                "perf_core": 0.0,
                "fairness_core": 0.0,
            },
            {
                "event_type": "state",
                "state_id": "s_000003",
                "batch_id": "b_000003",
                "window_id": "w_000003",
                "timestamp": 3.0,
                "model_version": "m_000001",
                "label_available": True,
                "drift_core": 0.3,
                "drift_max": 0.3,
                "drift_ready": True,
                "perf_core": 0.0,
                "fairness_core": 0.0,
            },
        ]
        path.write_text("\n".join(json.dumps(s) for s in states))

    def _write_airline_csv(self, path: Path) -> None:
        path.write_text(
            "\n".join(
                [
                    "Year,Month,DayOfWeek,CRSDepTime,CRSArrTime,CRSElapsedTime,Distance,Cancelled,Diverted,ArrDel15",
                    "2018,1,1,300,420,120,500,0,0,0",
                    "2018,1,2,310,430,120,510,0,0,1",
                    "2018,1,3,320,440,120,520,0,0,0",
                    "2018,1,4,330,450,120,530,0,0,1",
                    "2018,1,5,340,460,120,540,0,0,0",
                    "2018,1,6,350,470,120,550,0,0,1",
                ]
            )
            + "\n"
        )

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

    def test_logs_balanced_metrics_for_defer_and_updates(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            actions_path = tmp / "actions.jsonl"
            states_path = tmp / "states.jsonl"
            csv_path = tmp / "airline.csv"
            events_path = tmp / "events.jsonl"
            processed_path = tmp / "processed.jsonl"

            self._write_actions(actions_path)
            self._write_states(states_path)
            self._write_airline_csv(csv_path)

            run_trainer(
                actions_path=actions_path,
                events_path=events_path,
                processed_index_path=processed_path,
                seed=1,
                states_path=states_path,
                csv_path=csv_path,
                label_column="ArrDel15",
                positive_label="1",
                window_size=2,
            )

            records = [json.loads(line) for line in events_path.read_text().splitlines()]
            cost_records = [r for r in records if r["event_type"] == "cost"]
            update_records = [r for r in records if r["event_type"] == "model_update"]

            # defer event should still carry pre-eval metrics
            self.assertIsNotNone(cost_records[0].get("accuracy_pre"))
            self.assertIsNotNone(cost_records[0].get("balanced_accuracy_pre"))
            self.assertIsNotNone(cost_records[0].get("f1_pre"))
            self.assertIsNotNone(cost_records[0].get("tp_pre"))
            self.assertIsNotNone(cost_records[0].get("fp_pre"))
            self.assertIsNotNone(cost_records[0].get("tn_pre"))
            self.assertIsNotNone(cost_records[0].get("fn_pre"))

            # executed updates should publish richer eval summary metrics
            self.assertTrue(update_records)
            eval_summary = update_records[0].get("eval_summary", {})
            self.assertIn("accuracy", eval_summary)
            self.assertIn("balanced_accuracy", eval_summary)
            self.assertIn("f1", eval_summary)


if __name__ == "__main__":
    unittest.main()
