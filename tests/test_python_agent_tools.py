import tempfile
import unittest
from pathlib import Path

from scripts.run_python_agent_trace import (
    execute_tool,
    resolve_workspace_path,
    run_turn,
)


class _Object:
    def __init__(self, **values):
        self.__dict__.update(values)


class _Message(_Object):
    def model_dump(self, exclude_none=True):
        result = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            result["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in self.tool_calls
            ]
        return result


class _Completions:
    def __init__(self, responses):
        self.responses = iter(responses)

    def create(self, **_kwargs):
        return next(self.responses)


class PythonAgentToolsTest(unittest.TestCase):
    def test_read_file_is_wrapped_for_contextpilot_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contracts = root / "contracts"
            contracts.mkdir()
            (contracts / "sample.txt").write_text("contract body", encoding="utf-8")

            result = execute_tool(
                "read_file",
                '{"path":"contracts/sample.txt"}',
                root,
                1000,
            )

            self.assertIn('<files><file path="contracts/sample.txt">', result)
            self.assertIn("contract body", result)

    def test_rejects_workspace_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                resolve_workspace_path(Path(temporary), "../outside.txt")

    def test_list_files_returns_only_workspace_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contracts = root / "contracts"
            contracts.mkdir()
            (contracts / "a.txt").write_text("abc", encoding="utf-8")

            result = execute_tool(
                "list_files", '{"path":"contracts"}', root, 1000
            )

            self.assertIn("contracts/a.txt", result)
            self.assertIn('"size_bytes": 3', result)

    def test_agent_loop_executes_tool_then_returns_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contracts = root / "contracts"
            contracts.mkdir()
            (contracts / "a.txt").write_text("evidence", encoding="utf-8")
            tool_call = _Object(
                id="call-1",
                function=_Object(
                    name="read_file",
                    arguments='{"path":"contracts/a.txt"}',
                ),
            )
            responses = [
                _Object(
                    usage=_Object(prompt_tokens=10, completion_tokens=2),
                    choices=[
                        _Object(
                            message=_Message(content=None, tool_calls=[tool_call])
                        )
                    ],
                ),
                _Object(
                    usage=_Object(prompt_tokens=20, completion_tokens=4),
                    choices=[
                        _Object(message=_Message(content="answer", tool_calls=[]))
                    ],
                ),
            ]
            client = _Object(chat=_Object(completions=_Completions(responses)))
            messages = [{"role": "user", "content": "read it"}]

            result = run_turn(
                client,
                model="test",
                messages=messages,
                workspace=root,
                max_steps=3,
                max_tokens=128,
                max_file_chars=1000,
            )

            self.assertEqual(result["output"], "answer")
            self.assertEqual(result["llm_calls"], 2)
            self.assertEqual(result["tool_calls"], 1)
            self.assertEqual(result["prompt_tokens"], 30)
            self.assertEqual(messages[-2]["role"], "tool")


if __name__ == "__main__":
    unittest.main()
