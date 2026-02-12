import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.plot_compare_k import plot_compare


class TestPlotCompareK(unittest.TestCase):
    def test_compare_outputs(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            outdir = tmp / "plots"

            # create tiny synthetic events.jsonl for three runs
            runs = []
            for label in ["K1", "K3", "K5"]:
                path = tmp / f"{label}.jsonl"
                path.write_text(
                    "\n".join(
                        [
                            '{"event_type":"state","batch_id":"b_000001","state_id":"s_000001","drift_core":0.1}',
                            '{"event_type":"action","action_id":"a_000001","state_id":"s_000001","action_type":"fine_tune"}',
                            '{"event_type":"cost","action_id":"a_000001","estimated_usd":1.0}',
                        ]
                    )
                    + "\n"
                )
                runs.append((label, path))

            plot_compare(runs, outdir)

            self.assertTrue((outdir / "fine_tune_count_vs_k.svg").exists())
            self.assertTrue((outdir / "total_cost_usd_vs_k.svg").exists())
            self.assertTrue((outdir / "action_timeline_k.svg").exists())
            self.assertTrue((outdir / "cost_curve_k.svg").exists())


if __name__ == "__main__":
    unittest.main()
