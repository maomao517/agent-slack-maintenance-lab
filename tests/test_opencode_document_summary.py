import unittest

from scripts.summarize_opencode_document_results import (
    classify_visual_reuse,
    summarize,
)


class OpenCodeDocumentSummaryTest(unittest.TestCase):
    def test_classifies_cold_within_and_cross_workflow_accesses(self) -> None:
        rows = [
            {
                "start_unix_ns": 1,
                "image_sha256": "a",
                "document_version": "v1",
                "workflow_id": "w1",
            },
            {
                "start_unix_ns": 2,
                "image_sha256": "a",
                "document_version": "v1",
                "workflow_id": "w1",
            },
            {
                "start_unix_ns": 3,
                "image_sha256": "a",
                "document_version": "v1",
                "workflow_id": "w2",
            },
            {
                "start_unix_ns": 4,
                "image_sha256": "a",
                "document_version": "v2",
                "workflow_id": "w2",
            },
        ]

        result = classify_visual_reuse(rows)

        self.assertEqual(result["cold_accesses"], 2)
        self.assertEqual(result["within_workflow_reuses"], 1)
        self.assertEqual(result["cross_workflow_reuses"], 1)

    def test_summarizes_jct_and_cache_metrics(self) -> None:
        tasks = [
            {
                "run_id": "r",
                "task_id": "t1",
                "workflow_id": "w1",
                "wall_ms": 100,
                "start_unix_ns": 1_000_000_000,
                "end_unix_ns": 1_100_000_000,
                "success": True,
            },
            {
                "run_id": "r",
                "task_id": "t2",
                "workflow_id": "w2",
                "wall_ms": 200,
                "start_unix_ns": 1_000_000_000,
                "end_unix_ns": 1_200_000_000,
                "success": True,
            },
        ]
        tools = [
            {"run_id": "r", "duration_ms": 10},
            {"run_id": "r", "duration_ms": 20},
        ]
        visual = [
            {
                "run_id": "r",
                "start_unix_ns": 1,
                "workflow_id": "w1",
                "image_sha256": "a",
                "document_version": "v1",
                "total_ms": 50,
                "cache_hit": False,
                "encoder_called": True,
            },
            {
                "run_id": "r",
                "start_unix_ns": 2,
                "workflow_id": "w2",
                "image_sha256": "a",
                "document_version": "v1",
                "total_ms": 5,
                "cache_hit": True,
                "encoder_called": False,
                "peak_gpu_memory_mb": 1024,
                "h2d_state_transfer_mb": 9.375,
            },
        ]

        result = summarize(tasks, tools, visual, run_id="r")

        self.assertEqual(result["task_jct_ms"]["median"], 150.0)
        self.assertEqual(result["task_jct_ms"]["p95"], 200.0)
        self.assertEqual(result["cache_hit_rate"], 0.5)
        self.assertEqual(result["cross_workflow_reuses"], 1)
        self.assertEqual(result["throughput_tasks_per_second"], 10.0)
        self.assertEqual(result["peak_gpu_memory_mb"], 1024.0)
        self.assertEqual(result["h2d_state_transfer_mb"], 9.375)


if __name__ == "__main__":
    unittest.main()
