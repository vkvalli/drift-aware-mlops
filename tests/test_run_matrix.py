import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.run_matrix import main as run_matrix_main


class TestRunMatrix(unittest.TestCase):
    def test_single_run_matrix(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = Path("tests/data/sample.csv")

            import sys

            old_argv = sys.argv
            try:
                sys.argv = [
                    "run_matrix.py",
                    "--csv",
                    str(csv_path),
                    "--window-size",
                    "2",
                    "--max-rows",
                    "4",
                    "--seeds",
                    "1",
                    "--policies",
                    "no_retrain",
                    "--outdir",
                    str(tmp),
                ]
                run_matrix_main()
            finally:
                sys.argv = old_argv

            matrix_dirs = list(tmp.glob("matrix_*"))
            self.assertEqual(len(matrix_dirs), 1)

            run_dirs = [p for p in matrix_dirs[0].iterdir() if p.is_dir()]
            self.assertEqual(len(run_dirs), 1)
            self.assertTrue((run_dirs[0] / "events.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
