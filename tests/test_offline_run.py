import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

class TestOfflineRun(unittest.TestCase):
    def test_offline_run_outputs(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = tmp / "data.csv"
            csv_path.write_text("value,label\n1,0\n2,1\n3,0\n4,1\n")

            replay_path = tmp / "replay.jsonl"
            state_path = tmp / "state.jsonl"
            actions_path = tmp / "actions.jsonl"
            trainer_path = tmp / "trainer.jsonl"
            canonical_path = tmp / "events.jsonl"

            # Run via functions directly to avoid sys.argv manip
            from src.ingestion.replay import run_replay
            from src.monitor.monitor import run_monitor

            run_replay(
                csv_path=csv_path,
                window_size=2,
                label_delay_batches=1,
                flush_pending_labels=True,
                dataset_id="electricity",
                model_version="m_000001",
                output_path=replay_path,
            )
            run_monitor(
                events_path=replay_path,
                csv_path=csv_path,
                output_path=state_path,
                window_size=2,
                ref_window=1,
            )
            from src.policy.policy_agent import run_policy
            from src.runner.merge_utils import append_file_to_canonical
            from src.trainer.trainer import run_trainer

            canonical_path.write_text("")
            append_file_to_canonical(replay_path, canonical_path)
            append_file_to_canonical(state_path, canonical_path)

            run_policy(
                states_path=state_path,
                actions_path=actions_path,
                threshold=1.0,
                action_start=1,
                seed=1,
            )
            append_file_to_canonical(actions_path, canonical_path)

            run_trainer(
                actions_path=actions_path,
                events_path=trainer_path,
                processed_index_path=tmp / "processed.jsonl",
                seed=1,
            )
            append_file_to_canonical(trainer_path, canonical_path)

            self.assertTrue(replay_path.exists())
            self.assertTrue(state_path.exists())
            self.assertTrue(actions_path.exists())
            self.assertTrue(trainer_path.exists())
            self.assertTrue(canonical_path.exists())

            replay_records = [json.loads(l) for l in replay_path.read_text().splitlines()]
            state_records = [json.loads(l) for l in state_path.read_text().splitlines()]
            canonical_records = [json.loads(l) for l in canonical_path.read_text().splitlines()]

            self.assertTrue(replay_records)
            self.assertTrue(state_records)
            self.assertTrue(canonical_records)


if __name__ == "__main__":
    unittest.main()
