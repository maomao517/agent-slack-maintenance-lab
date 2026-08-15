import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.run_opencode_document_tasks import (
    clear_server_cache,
    extract_session_id,
    load_tasks,
)


class OpenCodeDocumentRunnerTest(unittest.TestCase):
    def test_extracts_nested_session_id(self) -> None:
        output = (
            '{"type":"message","properties":{"sessionID":"session-123"}}\n'
            '{"type":"done"}\n'
        )

        self.assertEqual(extract_session_id(output), "session-123")

    def test_ignores_non_json_output_when_extracting_session(self) -> None:
        output = "starting\n{\"session_id\":\"session-456\"}\n"

        self.assertEqual(extract_session_id(output), "session-456")

    def test_load_tasks_validates_unique_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tasks.jsonl"
            path.write_text(
                '{"task_id":"t1","prompt":"first"}\n'
                '{"task_id":"t2","prompt":"second"}\n',
                encoding="utf-8",
            )

            tasks = load_tasks(path)

            self.assertEqual([task["task_id"] for task in tasks], ["t1", "t2"])

    def test_load_tasks_rejects_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tasks.jsonl"
            path.write_text(
                '{"task_id":"t1","prompt":"first"}\n'
                '{"task_id":"t1","prompt":"second"}\n',
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_tasks(path)

    def test_load_tasks_rejects_unsafe_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tasks.jsonl"
            path.write_text(
                '{"task_id":"../outside","prompt":"first"}\n',
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_tasks(path)

    @patch("scripts.run_opencode_document_tasks.urllib.request.build_opener")
    def test_clear_server_cache_uses_direct_opener(self, build_opener) -> None:
        response = MagicMock()
        response.read.return_value = b'{"ok":true,"cleared_entries":1}'
        build_opener.return_value.open.return_value.__enter__.return_value = response

        clear_server_cache("http://127.0.0.1:30200", 10)

        build_opener.assert_called_once()


if __name__ == "__main__":
    unittest.main()
