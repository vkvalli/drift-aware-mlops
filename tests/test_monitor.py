import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.ingestion.replay import run_replay
from src.monitor.monitor import run_monitor


class TestMonitor(unittest.TestCase):
    def _write_shift_csv(self, path: Path) -> None:
        # 12 rows -> 6 batches with window_size=2
        # First 6 rows mean ~0, next 6 rows mean ~10
        rows = [
            (0.0, 0),
            (0.5, 1),
            (-0.2, 0),
            (0.1, 1),
            (0.0, 0),
            (0.2, 1),
            (10.0, 0),
            (10.5, 1),
            (9.8, 0),
            (10.1, 1),
            (10.2, 0),
            (9.9, 1),
        ]
        with path.open("w", encoding="utf-8") as f:
            f.write("value,label\n")
            for v, label in rows:
                f.write(f"{v},{label}\n")

    def _run_pipeline(self, tmp: Path) -> list[dict]:
        csv_path = tmp / "shift.csv"
        events_path = tmp / "events.jsonl"
        state_path = tmp / "state.jsonl"

        self._write_shift_csv(csv_path)
        if events_path.exists():
            events_path.unlink()
        if state_path.exists():
            state_path.unlink()
        run_replay(
            csv_path=csv_path,
            window_size=2,
            label_delay_batches=2,
            flush_pending_labels=True,
            dataset_id="electricity",
            model_version="m_000001",
            output_path=events_path,
        )

        run_monitor(
            events_path=events_path,
            csv_path=csv_path,
            output_path=state_path,
            window_size=2,
            ref_window=2,
        )

        return [json.loads(line) for line in state_path.read_text().splitlines()]

    def test_drift_increases_after_shift(self):
        with TemporaryDirectory() as tmpdir:
            records = self._run_pipeline(Path(tmpdir))

        drift_ready_records = [r for r in records if r.get("drift_ready")]
        self.assertTrue(drift_ready_records)

        k_shift_batch = 3
        pre = [r["drift_core"] for r in records[:k_shift_batch] if r.get("drift_ready")]
        post = [r["drift_core"] for r in records[k_shift_batch:] if r.get("drift_ready")]

        self.assertTrue(pre)
        self.assertTrue(post)

        epsilon = 1.0
        self.assertGreater(max(post), max(pre) + epsilon)
        self.assertGreater(sum(post) / len(post), sum(pre) / len(pre) + epsilon)

        state_ids = [r["state_id"] for r in records]
        self.assertEqual(state_ids, sorted(state_ids))
        for sid in state_ids:
            self.assertTrue(sid.startswith("s_"))

    def test_determinism(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            first = self._run_pipeline(path)
            second = self._run_pipeline(path)

        seq1 = [(r["batch_id"], r["state_id"], r["drift_core"]) for r in first]
        seq2 = [(r["batch_id"], r["state_id"], r["drift_core"]) for r in second]
        self.assertEqual(seq1, seq2)

    def test_fairness_core_uses_only_arrived_labels(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = tmp / "adult_like.csv"
            events_path = tmp / "events.jsonl"
            state_path = tmp / "state.jsonl"

            rows = [
                ("Male", 1, 0.0),
                ("Male", 1, 0.1),
                ("Female", 0, 0.2),
                ("Female", 0, 0.3),
                ("Male", 0, 0.4),
                ("Female", 1, 0.5),
            ]
            with csv_path.open("w", encoding="utf-8") as f:
                f.write("sex,class,value\n")
                for sex, label, value in rows:
                    f.write(f"{sex},{label},{value}\n")

            run_replay(
                csv_path=csv_path,
                window_size=2,
                label_delay_batches=2,
                flush_pending_labels=True,
                dataset_id="adult_openml",
                model_version="m_000001",
                output_path=events_path,
            )
            run_monitor(
                events_path=events_path,
                csv_path=csv_path,
                output_path=state_path,
                window_size=2,
                ref_window=2,
                task="classification",
                label_column="class",
                target_column="class",
                protected_attr="sex",
            )

            states = [json.loads(line) for line in state_path.read_text().splitlines()]
            fairness_seq = [round(float(s["fairness_core"]), 4) for s in states]
            ready_seq = [bool(s["fairness_ready"]) for s in states]
            support_seq = [int(s["fairness_support"]) for s in states]

            # With label_delay_batches=2, the first two batches cannot use any
            # fairness signal yet. The third batch can only use batch 1 labels.
            self.assertEqual(fairness_seq[0], 0.0)
            self.assertEqual(fairness_seq[1], 0.0)
            self.assertFalse(ready_seq[0])
            self.assertFalse(ready_seq[1])
            self.assertEqual(support_seq[2], 2)
            self.assertEqual(fairness_seq[2], 0.0)


if __name__ == "__main__":
    unittest.main()
