import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.runner.offline_run import main as offline_main

try:
    from tools import plot_run as plot_run_mod
    plot_run = plot_run_mod.plot_run
except Exception:  # pragma: no cover
    plot_run = None
    plot_run_mod = None


class TestPlotRun(unittest.TestCase):
    def test_plot_run_outputs(self):
        if plot_run is None:
            self.skipTest("matplotlib not available")
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            events_path = tmp / "events.jsonl"
            outdir = tmp / "plots"

            csv_path = Path("tests/data/sample.csv").resolve()

            import sys

            old_argv = sys.argv
            try:
                sys.argv = [
                    "offline_run.py",
                    "--csv",
                    str(csv_path),
                    "--window-size",
                    "2",
                    "--label-delay-batches",
                    "2",
                    "--ref-window",
                    "2",
                    "--threshold",
                    "1.0",
                    "--seed",
                    "1",
                    "--canonical-output",
                    str(events_path),
                    "--replay-output",
                    str(tmp / "replay.jsonl"),
                    "--state-output",
                    str(tmp / "state.jsonl"),
                    "--actions-output",
                    str(tmp / "actions.jsonl"),
                    "--trainer-output",
                    str(tmp / "trainer.jsonl"),
                    "--processed-index-file",
                    str(tmp / "processed.jsonl"),
                    "--clean-artifacts",
                ]
                offline_main()
            finally:
                sys.argv = old_argv

            plot_run(events_path, outdir)

            self.assertTrue((outdir / "run_report.txt").exists())

            if plot_run_mod is None or plot_run_mod.plt is None:
                self.skipTest("matplotlib not available")

            self.assertTrue((outdir / "drift_timeseries.png").exists())
            self.assertTrue((outdir / "action_timeline.png").exists())
            self.assertTrue((outdir / "cost_curve.png").exists())


if __name__ == "__main__":
    unittest.main()
