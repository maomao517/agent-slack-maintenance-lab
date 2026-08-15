import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.doc_tool import identifiers, safe_calculate, search_manifest


class DocumentToolTest(unittest.TestCase):
    def test_safe_calculate_allows_arithmetic(self) -> None:
        self.assertEqual(safe_calculate("(1115 * 2) + 3"), 2233)
        self.assertAlmostEqual(safe_calculate("10 / 4"), 2.5)

    def test_safe_calculate_rejects_code(self) -> None:
        with self.assertRaises(ValueError):
            safe_calculate("__import__('os').system('id')")

    def test_search_manifest_returns_ranked_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "manifest.csv"
            manifest.write_text(
                "document_id,page_id,title,keywords,image\n"
                "d1,p1,Annual report,revenue profit,p1.png\n"
                "d1,p2,Annual report,employees offices,p2.png\n",
                encoding="utf-8",
            )

            result = search_manifest(manifest, "revenue", 2)

            self.assertTrue(result["ok"])
            self.assertEqual(len(result["matches"]), 1)
            self.assertEqual(result["matches"][0]["page_id"], "p1")

    def test_identifiers_require_experiment_context(self) -> None:
        args = argparse.Namespace(
            run_id=None,
            workflow_id=None,
            task_id=None,
            require_context=True,
        )
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                identifiers(args)

    def test_identifiers_read_environment(self) -> None:
        args = argparse.Namespace(
            run_id=None,
            workflow_id=None,
            task_id=None,
            require_context=True,
        )
        with patch.dict(
            os.environ,
            {"RUN_ID": "r", "WORKFLOW_ID": "w", "TASK_ID": "t"},
            clear=True,
        ):
            self.assertEqual(
                identifiers(args),
                {"run_id": "r", "workflow_id": "w", "task_id": "t"},
            )


if __name__ == "__main__":
    unittest.main()
