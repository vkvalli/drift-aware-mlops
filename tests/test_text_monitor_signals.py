import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.monitor.monitor import run_monitor


class TestTextMonitorSignals(unittest.TestCase):
    def test_text_summary_fields_are_emitted(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            csv_path = root / "ag.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["title", "description", "class"])
                for i in range(12):
                    w.writerow([f"t{i}", f"desc token{i} token{i%3}", (i % 4) + 1])

            replay = root / "replay.jsonl"
            state = root / "state.jsonl"
            dlq = root / "dlq.jsonl"
            replay.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "event_type": "batch",
                            "schema_version": "v1",
                            "batch_id": f"b_{i+1:06d}",
                            "window_id": f"w{i+1}",
                            "timestamp_start": float(i),
                            "timestamp_end": float(i + 1),
                            "label_available": True,
                            "label_delay_batches": 0,
                            "dataset_id": "ag_news",
                            "model_version": "m_000001",
                        }
                    )
                    for i in range(3)
                )
                + "\n",
                encoding="utf-8",
            )

            run_monitor(
                events_path=replay,
                csv_path=csv_path,
                output_path=state,
                window_size=4,
                ref_window=2,
                task="text_classification",
                label_column="class",
                target_column="class",
                dlq_path=dlq,
            )

            states = [
                json.loads(x) for x in state.read_text(encoding="utf-8").splitlines()
            ]
            self.assertGreaterEqual(len(states), 1)
            self.assertIn("doc_len_mean", states[0])
            self.assertIn("vocab_entropy_proxy", states[0])


if __name__ == "__main__":
    unittest.main()
