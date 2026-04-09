"""
Unit tests for the fairness extension.

Tests:
  1. fairness_core > 0 when protected groups have different label distributions
  2. fairness_core == 0.0 when no protected_attr is provided
  3. fairness_trigger policy fires fine_tune when fairness_core > threshold
  4. fairness_trigger policy defers when fairness_core < threshold and drift < threshold

All tests use synthetic in-memory data — no file I/O required for unit tests
(temp files are used only where the function signature requires a Path).
"""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.monitor.monitor import _compute_fairness_core
from src.policy.policy_agent import run_policy


class TestFairnessCoreMetric(unittest.TestCase):
    """Tests for the label-distribution DPD proxy computed in the monitor."""

    def _write_csv(self, root: Path, rows) -> Path:
        """Helper: write a CSV with columns feature1, label, sex."""
        p = root / "data.csv"
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["feature1", "label", "sex"])
            for row in rows:
                w.writerow(row)
        return p

    def test_fairness_core_positive_when_groups_differ(self):
        """DPD proxy > 0 when Male group has higher positive-label rate than Female."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # Male: all label=1, Female: all label=0 → DPD proxy = 1.0
            rows = [[1.0, 1, "Male"]] * 50 + [[1.0, 0, "Female"]] * 50
            csv_path = self._write_csv(root, rows)

            result = _compute_fairness_core(
                csv_path=csv_path,
                label_column="label",
                target_column="label",
                protected_attr="sex",
                start=0,
                end=100,
            )
            self.assertGreater(result, 0.0, "Expected DPD proxy > 0 for imbalanced groups")

    def test_fairness_core_zero_when_groups_equal(self):
        """DPD proxy == 0 when both groups have identical label distributions."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # Both groups: 50% label=1 → DPD proxy = 0.0
            rows = []
            for i in range(50):
                rows.append([1.0, i % 2, "Male"])
                rows.append([1.0, i % 2, "Female"])
            csv_path = self._write_csv(root, rows)

            result = _compute_fairness_core(
                csv_path=csv_path,
                label_column="label",
                target_column="label",
                protected_attr="sex",
                start=0,
                end=100,
            )
            self.assertAlmostEqual(result, 0.0, places=6,
                                   msg="Expected DPD proxy ≈ 0 for balanced groups")

    def test_fairness_core_zero_when_no_protected_attr(self):
        """fairness_core must be 0.0 when protected_attr is empty string."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows = [[1.0, 1, "Male"]] * 50 + [[1.0, 0, "Female"]] * 50
            csv_path = self._write_csv(root, rows)

            result = _compute_fairness_core(
                csv_path=csv_path,
                label_column="label",
                target_column="label",
                protected_attr="",   # no protected attr
                start=0,
                end=100,
            )
            self.assertEqual(result, 0.0, "Expected 0.0 when protected_attr is empty")

    def test_fairness_core_partial_window(self):
        """DPD proxy uses only the rows in [start, end) window."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # First 50 rows: balanced (DPD=0), next 50: imbalanced (DPD>0)
            rows = []
            for i in range(25):
                rows.append([1.0, 1, "Male"])
                rows.append([1.0, 1, "Female"])
            for i in range(25):
                rows.append([1.0, 1, "Male"])
                rows.append([1.0, 0, "Female"])
            csv_path = self._write_csv(root, rows)

            balanced = _compute_fairness_core(
                csv_path=csv_path,
                label_column="label",
                target_column="label",
                protected_attr="sex",
                start=0,
                end=50,
            )
            imbalanced = _compute_fairness_core(
                csv_path=csv_path,
                label_column="label",
                target_column="label",
                protected_attr="sex",
                start=50,
                end=100,
            )
            self.assertAlmostEqual(balanced, 0.0, places=6)
            self.assertGreater(imbalanced, 0.0)


class TestFairnessTriggerPolicy(unittest.TestCase):
    """Tests for the fairness_trigger policy in policy_agent."""

    def _write_state(self, root: Path, fairness_core: float, drift_core: float,
                     drift_ready: bool = True) -> Path:
        """Write a single StateEvent JSONL."""
        p = root / "states.jsonl"
        state = {
            "event_type": "state",
            "state_id": "s_000001",
            "batch_id": "b_000001",
            "window_id": "w1",
            "timestamp": 1.0,
            "model_version": "m_000001",
            "label_available": True,
            "drift_core": drift_core,
            "drift_max": drift_core,
            "drift_ready": drift_ready,
            "perf_core": 0.0,
            "fairness_core": fairness_core,
        }
        p.write_text(json.dumps(state) + "\n", encoding="utf-8")
        return p

    def _run_policy(self, root: Path, states_path: Path,
                    fairness_threshold: float = 0.1,
                    drift_threshold: float = 1.0) -> dict:
        actions_path = root / "actions.jsonl"
        dlq_path = root / "dlq.jsonl"
        run_policy(
            states_path=states_path,
            actions_path=actions_path,
            threshold=drift_threshold,
            policy="fairness_trigger",
            fairness_threshold=fairness_threshold,
            dlq_path=dlq_path,
        )
        lines = actions_path.read_text(encoding="utf-8").splitlines()
        actions = [json.loads(l) for l in lines if l.strip()]
        return actions[0] if actions else {}

    def test_fairness_trigger_fires_when_fairness_above_threshold(self):
        """fairness_trigger emits fine_tune when fairness_core > fairness_threshold."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            states_path = self._write_state(root, fairness_core=0.35, drift_core=0.0,
                                            drift_ready=False)
            action = self._run_policy(root, states_path, fairness_threshold=0.1)
            self.assertEqual(action.get("action_type"), "fine_tune")
            self.assertEqual(action.get("reason"), "fairness_trigger")

    def test_fairness_trigger_defers_when_both_below_threshold(self):
        """fairness_trigger defers when fairness_core < threshold AND drift < threshold."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            states_path = self._write_state(root, fairness_core=0.05, drift_core=0.3,
                                            drift_ready=True)
            action = self._run_policy(root, states_path,
                                      fairness_threshold=0.1, drift_threshold=1.0)
            self.assertEqual(action.get("action_type"), "defer")

    def test_fairness_trigger_fires_on_drift_when_fairness_ok(self):
        """fairness_trigger also fires fine_tune when drift_core >= threshold."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            states_path = self._write_state(root, fairness_core=0.02, drift_core=2.0,
                                            drift_ready=True)
            action = self._run_policy(root, states_path,
                                      fairness_threshold=0.1, drift_threshold=1.0)
            self.assertEqual(action.get("action_type"), "fine_tune")
            self.assertEqual(action.get("reason"), "drift_threshold")

    def test_existing_policies_unaffected(self):
        """Existing no_retrain policy still works after fairness_trigger addition."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            states_path = self._write_state(root, fairness_core=0.9, drift_core=5.0,
                                            drift_ready=True)
            actions_path = root / "actions.jsonl"
            dlq_path = root / "dlq.jsonl"
            run_policy(
                states_path=states_path,
                actions_path=actions_path,
                policy="no_retrain",
                dlq_path=dlq_path,
            )
            lines = actions_path.read_text(encoding="utf-8").splitlines()
            actions = [json.loads(l) for l in lines if l.strip()]
            self.assertEqual(actions[0].get("action_type"), "defer")
            self.assertEqual(actions[0].get("reason"), "no_retrain")


if __name__ == "__main__":
    unittest.main()
