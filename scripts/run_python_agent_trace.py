#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any


DOCUMENT_CATEGORIES = ("commercial", "legal", "compliance", "strategic")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory under the isolated workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative directory, normally contracts",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file under the isolated workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative file such as contracts/example.txt",
                    }
                },
                "required": ["path"],
            },
        },
    },
]


def resolve_workspace_path(root: Path, requested: str) -> Path:
    relative = Path(requested)
    if relative.is_absolute():
        raise ValueError("absolute paths are not allowed")
    root = root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("path escapes the isolated workspace") from exc
    return candidate


def list_files(root: Path, requested: str) -> str:
    directory = resolve_workspace_path(root, requested)
    if not directory.is_dir():
        raise ValueError(f"not a directory: {requested}")
    files = [
        {
            "path": str(path.relative_to(root.resolve())),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(directory.iterdir())
        if path.is_file()
    ]
    return json.dumps({"files": files}, ensure_ascii=False)


def read_file(root: Path, requested: str, max_file_chars: int) -> str:
    path = resolve_workspace_path(root, requested)
    if not path.is_file():
        raise ValueError(f"not a file: {requested}")
    content = path.read_text(encoding="utf-8", errors="replace")
    if len(content) > max_file_chars:
        raise ValueError(
            f"file exceeds max_file_chars: {len(content)} > {max_file_chars}"
        )
    escaped_path = html.escape(str(path.relative_to(root.resolve())), quote=True)
    return f'<files><file path="{escaped_path}">\n{content}\n</file></files>'


def execute_tool(
    name: str,
    arguments_json: str,
    workspace: Path,
    max_file_chars: int,
) -> str:
    try:
        arguments = json.loads(arguments_json or "{}")
        requested = str(arguments.get("path", ""))
        if name == "list_files":
            return list_files(workspace, requested)
        if name == "read_file":
            return read_file(workspace, requested, max_file_chars)
        return json.dumps({"error": f"unknown tool: {name}"})
    except Exception as exc:
        return json.dumps(
            {"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False
        )


def post_context(url: str, payload: dict[str, object]) -> None:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        if response.status != 200:
            raise RuntimeError(f"trace proxy returned HTTP {response.status}")


def load_tasks(
    root: Path, categories: list[str], scenarios: set[str]
) -> list[dict[str, Any]]:
    tasks = []
    for category in categories:
        path = root / "claw-tasks" / category / "tasks.json"
        for task in json.loads(path.read_text(encoding="utf-8")):
            if scenarios and task["name"] not in scenarios:
                continue
            tasks.append({**task, "category": category})
    return tasks


def prepare_workspace(clawtasks_root: Path, workspace: Path) -> None:
    contracts = workspace / "contracts"
    contracts.mkdir(parents=True, exist_ok=True)
    for path in (clawtasks_root / "data" / "workspace").iterdir():
        if path.is_file():
            shutil.copy2(path, contracts / path.name)


def run_turn(
    client,
    *,
    model: str,
    messages: list[dict[str, Any]],
    workspace: Path,
    max_steps: int,
    max_tokens: int,
    max_file_chars: int,
) -> dict[str, Any]:
    total_prompt_tokens = 0
    total_completion_tokens = 0
    llm_calls = 0
    tool_calls = 0
    final_content = ""

    for _step in range(max_steps):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0,
            max_tokens=max_tokens,
        )
        llm_calls += 1
        if response.usage is not None:
            total_prompt_tokens += response.usage.prompt_tokens or 0
            total_completion_tokens += response.usage.completion_tokens or 0

        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))
        if not message.tool_calls:
            final_content = message.content or ""
            return {
                "output": final_content,
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "llm_calls": llm_calls,
                "tool_calls": tool_calls,
                "error": None,
            }

        for tool_call in message.tool_calls:
            tool_calls += 1
            result = execute_tool(
                tool_call.function.name,
                tool_call.function.arguments,
                workspace,
                max_file_chars,
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": result,
                }
            )

    return {
        "output": final_content,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "llm_calls": llm_calls,
        "tool_calls": tool_calls,
        "error": "max_steps_exceeded",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ClawTasks with a minimal Python tool-calling agent"
    )
    parser.add_argument("--clawtasks-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30100/v1")
    parser.add_argument("--trace-control-url", default="http://127.0.0.1:30100/_trace/context")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--arm", choices=("Direct", "CP"), required=True)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--category", nargs="+", choices=DOCUMENT_CATEGORIES, default=list(DOCUMENT_CATEGORIES))
    parser.add_argument("--scenario", nargs="*", default=[])
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-file-chars", type=int, default=500_000)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--store-output", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from openai import OpenAI

    tasks = load_tasks(args.clawtasks_root, args.category, set(args.scenario))
    if not tasks:
        raise SystemExit("no matching document-analysis scenarios")
    prepare_workspace(args.clawtasks_root, args.workspace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.request_timeout)
    system_prompt = (
        "You are a document-analysis agent. Use list_files and read_file to inspect "
        "the isolated workspace. Never use outside knowledge or invent file content. "
        "Preserve relevant evidence across conversation turns and answer the user directly."
    )

    with args.output.open("w", encoding="utf-8") as output:
        for task in tasks:
            session_id = f"python-{task['name']}-{args.arm}-t{args.trial}-{uuid.uuid4().hex[:10]}"
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt}
            ]
            for turn, prompt in enumerate(task["turns"]):
                labels = {
                    "scenario": task["name"],
                    "arm": args.arm,
                    "trial": args.trial,
                    "turn": turn,
                    "session_id": session_id,
                }
                post_context(args.trace_control_url, labels)
                messages.append({"role": "user", "content": prompt})
                start_ns = time.time_ns()
                result = run_turn(
                    client,
                    model=args.model,
                    messages=messages,
                    workspace=args.workspace,
                    max_steps=args.max_steps,
                    max_tokens=args.max_tokens,
                    max_file_chars=args.max_file_chars,
                )
                end_ns = time.time_ns()
                content = result.pop("output")
                record = {
                    "schema": "slackmaint.python-agent-turn.v1",
                    "event": "agent_turn",
                    **labels,
                    "start_unix_ns": start_ns,
                    "end_unix_ns": end_ns,
                    "wall_ms": round((end_ns - start_ns) / 1_000_000, 3),
                    "output_chars": len(content),
                    "output_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    **result,
                }
                if args.store_output:
                    record["output"] = content
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                print(
                    f"{task['name']} turn={turn} arm={args.arm} "
                    f"wall={record['wall_ms'] / 1000:.2f}s "
                    f"llm_calls={record['llm_calls']} tools={record['tool_calls']} "
                    f"error={record['error']}"
                )


if __name__ == "__main__":
    main()
