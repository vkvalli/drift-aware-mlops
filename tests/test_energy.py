"""
Tests for the energy-efficiency evaluation path.

Covers:
- CostEvent for fine_tune has kwh > 0 and co2_kg > 0
- CostEvent for defer has kwh == 0.0 and co2_kg == 0.0
- kwh and co2_kg scale proportionally with train_time_s
- aggregate_matrix output includes energy columns
"""
import csv
import json
import tempfile
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers to compute energy using the documented formula
# ---------------------------------------------------------------------------
_TDP_W = 65.0
_CO2_INTENSITY = 0.233  # kg CO2 per kWh (UK grid average, BEIS 2023)


def _compute_kwh(train_time_s: float) -> float:
    return train_time_s * _TDP_W / 3_600_000


def _compute_co2(kwh: float) -> float:
    return kwh * _CO2_INTENSITY


# ---------------------------------------------------------------------------
# Unit tests for the energy formula
# ---------------------------------------------------------------------------
class TestEnergyFormula(unittest.TestCase):
    def test_fine_tune_kwh_positive(self):
        """fine_tune train_time_s=3.0 should yield kwh > 0."""
        kwh = _compute_kwh(3.0)
        self.assertGreater(kwh, 0.0)

    def test_fine_tune_co2_positive(self):
        """fine_tune should yield co2_kg > 0."""
        kwh = _compute_kwh(3.0)
        co2 = _compute_co2(kwh)
        self.assertGreater(co2, 0.0)

    def test_full_retrain_kwh_positive(self):
        """full_retrain train_time_s=10.0 should yield kwh > 0."""
        kwh = _compute_kwh(10.0)
        self.assertGreater(kwh, 0.0)

    def test_defer_kwh_zero(self):
        """defer has train_time_s=0.0, so kwh must be 0.0."""
        kwh = _compute_kwh(0.0)
        self.assertEqual(kwh, 0.0)

    def test_defer_co2_zero(self):
        """defer has train_time_s=0.0, so co2_kg must be 0.0."""
        kwh = _compute_kwh(0.0)
        co2 = _compute_co2(kwh)
        self.assertEqual(co2, 0.0)

    def test_proportional_scaling(self):
        """kwh and co2_kg must scale linearly with train_time_s."""
        kwh_3 = _compute_kwh(3.0)
        kwh_6 = _compute_kwh(6.0)
        self.assertAlmostEqual(kwh_6, 2 * kwh_3, places=12)

        co2_3 = _compute_co2(kwh_3)
        co2_6 = _compute_co2(kwh_6)
        self.assertAlmostEqual(co2_6, 2 * co2_3, places=12)

    def test_full_retrain_greater_than_fine_tune(self):
        """full_retrain (10s) should consume more energy than fine_tune (3s)."""
        kwh_ft = _compute_kwh(3.0)
        kwh_fr = _compute_kwh(10.0)
        self.assertGreater(kwh_fr, kwh_ft)

    def test_known_values(self):
        """Spot-check formula against hand-computed values."""
        # fine_tune: 3s * 65W / 3_600_000 = 5.4167e-5 kWh
        expected_kwh = 3.0 * 65.0 / 3_600_000
        self.assertAlmostEqual(_compute_kwh(3.0), expected_kwh, places=12)
        # co2: 5.4167e-5 * 0.233
        expected_co2 = expected_kwh * 0.233
        self.assertAlmostEqual(_compute_co2(expected_kwh), expected_co2, places=12)


# ---------------------------------------------------------------------------
# Integration test: trainer emits correct energy values in CostEvent
# ---------------------------------------------------------------------------
class TestTrainerEnergyIntegration(unittest.TestCase):
    """Run the trainer on a minimal synthetic dataset and check CostEvent fields."""

    def _make_minimal_csv(self, tmp: Path) -> Path:
        """Create a tiny CSV with 30 rows for classification."""
        csv_path = tmp / "data.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["feat1", "feat2", "class"])
            for i in range(30):
                writer.writerow([float(i), float(i % 3), i % 2])
        return csv_path

    def _make_actions_jsonl(self, tmp: Path, action_type: str) -> Path:
        """Create a minimal actions.jsonl with one action."""
        path = tmp / "actions.jsonl"
        record = {
            "event_type": "action",
            "action_id": "a_000001",
            "state_id": "s_000001",
            "action_type": action_type,
            "model_version": "m_000001",
            "timestamp": 1.0,
            "reason": "test",
            "action_params": {},
            "requested_at": 1.0,
        }
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        return path

    def _run_trainer(self, tmp: Path, action_type: str):
        from src.trainer.trainer import run_trainer

        actions_path = self._make_actions_jsonl(tmp, action_type)
        events_path = tmp / "events.jsonl"
        processed_path = tmp / "processed.jsonl"
        csv_path = self._make_minimal_csv(tmp)
        dlq_path = tmp / "dlq.jsonl"

        run_trainer(
            actions_path=actions_path,
            events_path=events_path,
            processed_index_path=processed_path,
            seed=1,
            dlq_path=dlq_path,
            csv_path=csv_path,
            label_column="class",
            window_size=10,
        )

        cost_events = []
        if events_path.exists():
            for line in events_path.read_text().splitlines():
                rec = json.loads(line)
                if rec.get("event_type") == "cost":
                    cost_events.append(rec)
        return cost_events

    def test_fine_tune_energy_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            costs = self._run_trainer(tmp, "fine_tune")
            self.assertEqual(len(costs), 1)
            self.assertGreater(costs[0]["energy_kwh"], 0.0)
            self.assertGreater(costs[0]["co2_kg"], 0.0)

    def test_defer_energy_zero(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            costs = self._run_trainer(tmp, "defer")
            self.assertEqual(len(costs), 1)
            self.assertEqual(costs[0]["energy_kwh"], 0.0)
            self.assertEqual(costs[0]["co2_kg"], 0.0)

    def test_fine_tune_formula_matches(self):
        """Trainer-emitted kwh must match the documented formula."""
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            costs = self._run_trainer(tmp, "fine_tune")
            self.assertEqual(len(costs), 1)
            train_time_s = costs[0]["train_time_s"]
            expected_kwh = _compute_kwh(train_time_s)
            expected_co2 = _compute_co2(expected_kwh)
            self.assertAlmostEqual(costs[0]["energy_kwh"], expected_kwh, places=10)
            self.assertAlmostEqual(costs[0]["co2_kg"], expected_co2, places=10)


# ---------------------------------------------------------------------------
# Test: aggregate_matrix output includes energy columns
# ---------------------------------------------------------------------------
class TestAggregateMatrixEnergyColumns(unittest.TestCase):
    def _make_events_jsonl(self, tmp: Path, n_fine_tune: int = 2) -> Path:
        """Create a minimal events.jsonl with cost events."""
        path = tmp / "events.jsonl"
        lines = []
        # run_meta
        lines.append(json.dumps({
            "event_type": "run_meta",
            "policy": "test_policy",
            "seed": 1,
            "window_size": 10,
        }))
        for i in range(5):
            action_type = "fine_tune" if i < n_fine_tune else "defer"
            executed = action_type == "fine_tune"
            train_time_s = 3.0 if executed else 0.0
            kwh = _compute_kwh(train_time_s)
            co2 = _compute_co2(kwh)
            lines.append(json.dumps({
                "event_type": "action",
                "action_id": f"a_{i:06d}",
                "action_type": action_type,
                "state_id": f"s_{i:06d}",
                "timestamp": float(i),
            }))
            lines.append(json.dumps({
                "event_type": "cost",
                "cost_id": f"c_{i:06d}",
                "action_id": f"a_{i:06d}",
                "timestamp": float(i),
                "train_time_s": train_time_s,
                "estimated_usd": 1.5 if executed else 0.0,
                "energy_kwh": kwh,
                "co2_kg": co2,
                "executed": executed,
                "reason_code": "ok" if executed else "defer",
            }))
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def test_energy_columns_present(self):
        """aggregate_matrix summary CSV must contain energy columns."""
        from tools.aggregate_matrix import aggregate

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            run_dir = tmp / "run1"
            run_dir.mkdir()
            self._make_events_jsonl(run_dir)

            out_path = tmp / "summary.csv"
            aggregate(tmp, out_path)

            self.assertTrue(out_path.exists())
            with out_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertIn("total_kwh", row)
            self.assertIn("total_co2_kg", row)
            self.assertIn("mean_kwh_per_action", row)
            self.assertIn("mean_co2_per_action", row)

    def test_energy_values_correct(self):
        """aggregate_matrix energy totals must match sum of cost events."""
        from tools.aggregate_matrix import aggregate

        n_fine_tune = 3
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            run_dir = tmp / "run1"
            run_dir.mkdir()
            self._make_events_jsonl(run_dir, n_fine_tune=n_fine_tune)

            out_path = tmp / "summary.csv"
            aggregate(tmp, out_path)

            with out_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            row = rows[0]
            expected_kwh = n_fine_tune * _compute_kwh(3.0)
            expected_co2 = n_fine_tune * _compute_co2(_compute_kwh(3.0))
            self.assertAlmostEqual(float(row["total_kwh"]), expected_kwh, places=8)
            self.assertAlmostEqual(float(row["total_co2_kg"]), expected_co2, places=8)


if __name__ == "__main__":
    unittest.main()
