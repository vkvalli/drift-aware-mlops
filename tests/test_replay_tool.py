import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib

from tools.replay import replay


class TestReplayTool(unittest.TestCase):
    def test_deterministic_replay(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = tmp / "events.jsonl"
            out1 = tmp / "out1.jsonl"
            out2 = tmp / "out2.jsonl"

            events = [
                {"event_type": "batch", "batch_id": "b_000001", "window_id": "w1", "timestamp_start": 0.0, "timestamp_end": 1.0, "label_available": True, "dataset_id": "x", "model_version": "m_000001"},
                {"event_type": "action", "action_id": "a_000001", "state_id": "s_000001", "action_type": "defer", "model_version": "m_000001", "timestamp": 1.0},
            ]
            src.write_text("\n".join(json.dumps(e) for e in events))

            replay(src, out1, from_id=None, to_id=None, rate=None, deterministic=True, checkpoint_path=None)
            replay(src, out2, from_id=None, to_id=None, rate=None, deterministic=True, checkpoint_path=None)

            h1 = hashlib.sha256(out1.read_bytes()).hexdigest()
            h2 = hashlib.sha256(out2.read_bytes()).hexdigest()
            self.assertEqual(h1, h2)

    def test_from_to_id_file_order(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = tmp / "events.jsonl"
            out = tmp / "out.jsonl"

            events = [
                {"event_type": "batch", "batch_id": "b_000001", "window_id": "w1", "timestamp_start": 0.0, "timestamp_end": 1.0, "label_available": True, "dataset_id": "x", "model_version": "m_000001"},
                {"event_type": "state", "state_id": "s_000001", "batch_id": "b_000001", "drift_core": 0.0, "drift_ready": True, "model_version": "m_000001"},
                {"event_type": "action", "action_id": "a_000001", "state_id": "s_000001", "action_type": "defer", "model_version": "m_000001", "timestamp": 1.0},
            ]
            src.write_text("\n".join(json.dumps(e) for e in events))

            replay(src, out, from_id="state:s_000001", to_id="action:a_000001", rate=None, deterministic=True, checkpoint_path=None)
            lines = out.read_text().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["event_type"], "state")
            self.assertEqual(json.loads(lines[1])["event_type"], "action")

    def test_checkpoint_determinism(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = tmp / "events.jsonl"
            out1 = tmp / "out1.jsonl"
            out2 = tmp / "out2.jsonl"
            checkpoint = tmp / "checkpoint.json"

            events = [
                {"event_type": "batch", "batch_id": "b_000001", "window_id": "w1", "timestamp_start": 0.0, "timestamp_end": 1.0, "label_available": True, "dataset_id": "x", "model_version": "m_000001"},
                {"event_type": "batch", "batch_id": "b_000002", "window_id": "w2", "timestamp_start": 1.0, "timestamp_end": 2.0, "label_available": True, "dataset_id": "x", "model_version": "m_000001"},
            ]
            src.write_text("\n".join(json.dumps(e) for e in events) + "\n")

            # simulate checkpoint after first line
            first_line = (json.dumps(events[0]) + "\n").encode("utf-8")
            checkpoint.write_text(json.dumps({"last_byte_offset": len(first_line)}))

            replay(src, out1, from_id=None, to_id=None, rate=None, deterministic=True, checkpoint_path=checkpoint)
            replay(src, out2, from_id=None, to_id=None, rate=None, deterministic=True, checkpoint_path=checkpoint)

            h1 = hashlib.sha256(out1.read_bytes()).hexdigest()
            h2 = hashlib.sha256(out2.read_bytes()).hexdigest()
            self.assertEqual(h1, h2)
            self.assertEqual(out1.read_text().count("\n"), 1)


if __name__ == "__main__":
    unittest.main()
