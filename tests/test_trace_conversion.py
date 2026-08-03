import unittest

from slackmaint.trace_conversion import events_to_experiment


class TraceConversionTest(unittest.TestCase):
    def test_converts_llm_calls_and_inter_call_gaps(self) -> None:
        events = [
            {
                "event": "llm_call",
                "scenario": "s01",
                "arm": "Direct",
                "trial": 0,
                "start_unix_ns": 1_000_000_000,
                "end_unix_ns": 1_100_000_000,
            },
            {
                "event": "llm_call",
                "scenario": "s01",
                "arm": "Direct",
                "trial": 0,
                "start_unix_ns": 1_600_000_000,
                "end_unix_ns": 1_800_000_000,
            },
            {
                "event": "llm_call",
                "scenario": "s01",
                "arm": "CP",
                "trial": 0,
                "start_unix_ns": 2_000_000_000,
                "end_unix_ns": 2_050_000_000,
            },
            {
                "event": "llm_call",
                "scenario": "s01",
                "arm": "Direct",
                "trial": 0,
                "status": 502,
                "error": "upstream failed",
                "start_unix_ns": 2_100_000_000,
                "end_unix_ns": 2_200_000_000,
            },
        ]

        result = events_to_experiment(
            events, arm="Direct", maintenance_ms=25, tick_ms=5
        )

        self.assertEqual(result["metadata"]["llm_calls"], 2)
        self.assertEqual(len(result["workflows"]), 1)
        self.assertEqual(result["workflows"][0]["model_segments_ms"], [100, 200])
        self.assertEqual(result["workflows"][0]["tool_waits_ms"], [500])
        self.assertEqual(len(result["maintenance_tasks"]), 1)
        self.assertEqual(result["maintenance_tasks"][0]["work_ms"], 25)

    def test_rejects_trace_without_matching_calls(self) -> None:
        with self.assertRaises(ValueError):
            events_to_experiment([], arm="Direct", maintenance_ms=10)


if __name__ == "__main__":
    unittest.main()
