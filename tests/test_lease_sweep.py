import json
import tempfile
import unittest
from pathlib import Path

from slackmaint.lease_simulator import LeaseExperimentSpec
from slackmaint.lease_sweep import build_sweep_cases, run_sweep_to_directory


class LeaseSweepTest(unittest.TestCase):
    def _config(self) -> dict[str, object]:
        return {
            "name": "sweep-test",
            "tick_ms": 10,
            "retention_capacity_mb": 5000,
            "fixed_kv_ttl_ms": 1000,
            "encoder_ttl_ms": 5000,
            "max_time_ms": 20000,
            "workflows": [
                {
                    "workflow_id": "w0",
                    "start_ms": 0,
                    "model_segments_ms": [100, 100],
                    "tool_waits_ms": [500],
                    "expected_tool_waits_ms": [400],
                    "kv_size_mb": 4000,
                    "encoder_size_mb": 400,
                    "prefill_ms": 200,
                    "encoder_ms": 100,
                }
            ],
        }

    def test_sweep_cases_are_unique_and_cover_expected_axes(self) -> None:
        spec = LeaseExperimentSpec.from_dict(self._config())
        cases = build_sweep_cases(spec)

        self.assertEqual(len(cases), 78)
        self.assertEqual(len({case.case_id for case in cases}), 78)
        self.assertIn("capacity_ttl:1800:250", {case.case_id for case in cases})
        self.assertIn("prediction_scale:1", {case.case_id for case in cases})

    def test_sweep_writes_self_contained_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(json.dumps(self._config()), encoding="utf-8")

            summary = run_sweep_to_directory(config, root / "output")

            self.assertEqual(summary["case_count"], 78)
            self.assertEqual(summary["metric_row_count"], 390)
            self.assertTrue((root / "output" / "metrics.json").is_file())
            self.assertTrue((root / "output" / "comparisons.csv").is_file())
            self.assertTrue((root / "output" / "summary.md").is_file())


if __name__ == "__main__":
    unittest.main()
