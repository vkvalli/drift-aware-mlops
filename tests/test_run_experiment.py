import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.run_experiment import main as run_experiment_main


class TestRunExperiment(unittest.TestCase):
    def test_creates_timestamped_folder(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = tmp / "data.csv"
            csv_path.write_text("value,label\n1,0\n2,1\n")

            import sys

            old_argv = sys.argv
            try:
                sys.argv = [
                    "run_experiment.py",
                    "--csv",
                    str(csv_path),
                    "--window-size",
                    "1",
                    "--label-delay-batches",
                    "1",
                    "--ref-window",
                    "1",
                    "--threshold",
                    "1.0",
                    "--seed",
                    "1",
                ]

                cwd = Path.cwd()
                try:
                    # run in temp dir to avoid polluting real runs/
                    import os

                    os.chdir(tmp)
                    run_experiment_main()
                finally:
                    os.chdir(cwd)
            finally:
                sys.argv = old_argv

            runs_dir = tmp / "runs"
            self.assertTrue(runs_dir.exists())
            run_dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
            self.assertEqual(len(run_dirs), 1)
            self.assertTrue(run_dirs[0].name.startswith("run_"))


if __name__ == "__main__":
    unittest.main()
