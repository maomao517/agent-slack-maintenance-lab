import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_d2_results.py"
SPEC = importlib.util.spec_from_file_location("audit_d2_results", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class D2ResultAuditTest(unittest.TestCase):
    def test_flags_incomplete_result_evidence(self) -> None:
        report = MODULE.build_audit(
            {
                "meta": {"rounds": 4},
                "summary": [
                    {
                        "answer_consistency": 0.0,
                        "vision_state_mb": 0.625,
                        "cpu_to_gpu_transfer_ms": 0.0,
                        "duplicate_encoder_calls": 0.0,
                        "avg_jct_ms": 5150.0,
                        "p95_jct_ms": 5300.0,
                    }
                ],
            },
            expected_low_state_mb=2.5,
        )
        codes = {item["code"] for item in report["findings"]}
        self.assertFalse(report["report_ready"])
        self.assertIn("possible-image-embeds-only-state", codes)
        self.assertIn("workflow-jct-boundaries-missing", codes)

    def test_accepts_complete_instrumented_schema(self) -> None:
        report = MODULE.build_audit(
            {
                "summary": [
                    {
                        "answer_consistency": 1.0,
                        "vision_state_mb": 2.5,
                        "cpu_to_gpu_transfer_ms": 12.0,
                        "duplicate_encoder_calls": 2.0,
                        "workflow_start_ns": 10,
                        "workflow_end_ns": 20,
                        "workflow_jct_ms": 10.0,
                    }
                ]
            },
            expected_low_state_mb=2.5,
        )
        self.assertTrue(report["report_ready"])
        self.assertEqual(report["findings"], [])


if __name__ == "__main__":
    unittest.main()
