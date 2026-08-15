#!/usr/bin/env python3
"""Run a JSONL document workload through OpenCode and preserve raw events."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


SESSION_KEYS = {
    "session_id",
    "sessionid",
    "sessionID",
    "sessionId",
}
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


def validate_id(value: str, field: str) -> str:
    if not SAFE_ID.fullmatch(value):
        raise ValueError(
            f"{field} must match {SAFE_ID.pattern!r}; received {value!r}"
        )
    return value


def load_tasks(path: Path) -> list[dict[str, Any]]:
    tasks = []
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                task = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(task, dict):
                raise ValueError(f"line {line_number} must contain a JSON object")
            task_id = validate_id(str(task.get("task_id", "")).strip(), "task_id")
            prompt = str(task.get("prompt", "")).strip()
            if not task_id or not prompt:
                raise ValueError(
                    f"line {line_number} requires non-empty task_id and prompt"
                )
            if task_id in seen:
                raise ValueError(f"duplicate task_id: {task_id}")
            seen.add(task_id)
            tasks.append(task)
    if not tasks:
        raise ValueError("task file is empty")
    return tasks


def find_session_id(value: Any) -> str | None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in SESSION_KEYS and isinstance(item, str) and item:
                return item
        for item in value.values():
            found = find_session_id(item)
            if found:
                return found
    if isinstance(value, list):
        for item in value:
            found = find_session_id(item)
            if found:
                return found
    return None


def extract_session_id(stdout: str) -> str | None:
    for line in stdout.splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        found = find_session_id(payload)
        if found:
            return found
    return None


def append_summary(path: Path, record: dict[str, Any]) -> None:
    payload = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, payload)
    finally:
        os.close(descriptor)


def clear_server_cache(server: str, timeout: float) -> None:
    request = urllib.request.Request(
        f"{server.rstrip('/')}/cache/clear",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"cache clear returned HTTP {exc.code}: {detail}") from exc
    if not isinstance(payload, dict) or not payload.get("ok"):
        raise RuntimeError(f"cache clear failed: {payload}")


def run_task(
    task: dict[str, Any],
    *,
    repeat: int,
    args: argparse.Namespace,
    instructions: str,
    run_dir: Path,
) -> dict[str, Any]:
    task_id = str(task["task_id"])
    workflow_id = str(task.get("workflow_id") or f"{task_id}-r{repeat}")
    title = str(task.get("title") or f"{workflow_id}-{task_id}")
    prompt = (
        f"{instructions.rstrip()}\n\n任务：\n{task['prompt']}"
        if instructions
        else str(task["prompt"])
    )
    stem = f"{task_id}.r{repeat}"
    event_path = run_dir / f"{stem}.events.jsonl"
    stderr_path = run_dir / f"{stem}.stderr.log"
    session_path = run_dir / f"{stem}.session.json"
    command = [
        args.opencode,
        "run",
        "--auto",
        "--format",
        "json",
        "--title",
        title,
        *args.opencode_arg,
        prompt,
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "RUN_ID": args.run_id,
            "WORKFLOW_ID": workflow_id,
            "TASK_ID": task_id,
            "DOC_TOOL_TRACE": str(run_dir / "tool-events.jsonl"),
            "DOC_PAGE_MANIFEST": str(args.manifest.resolve()),
        }
    )
    if args.server:
        environment["DOC_VLM_SERVER_URL"] = args.server

    start_unix_ns = time.time_ns()
    started = time.perf_counter_ns()
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            cwd=args.workspace,
            env=environment,
            text=True,
            capture_output=True,
            timeout=args.timeout,
            check=False,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        exit_code = completed.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        exit_code = 124
    end_unix_ns = time.time_ns()
    event_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    session_id = extract_session_id(stdout)
    export_error = None

    if args.export_sessions and session_id and exit_code == 0:
        try:
            exported = subprocess.run(
                [args.opencode, "export", session_id],
                cwd=args.workspace,
                env=environment,
                text=True,
                capture_output=True,
                timeout=args.timeout,
                check=False,
            )
            if exported.returncode == 0:
                session_path.write_text(exported.stdout, encoding="utf-8")
            else:
                export_error = exported.stderr.strip() or f"exit {exported.returncode}"
        except subprocess.TimeoutExpired:
            export_error = "session export timed out"

    record = {
        "schema": "slackmaint.opencode-task.v1",
        "event": "opencode_task",
        "run_id": args.run_id,
        "workflow_id": workflow_id,
        "task_id": task_id,
        "repeat": repeat,
        "title": title,
        "opencode_session_id": session_id,
        "start_unix_ns": start_unix_ns,
        "end_unix_ns": end_unix_ns,
        "wall_ms": round((time.perf_counter_ns() - started) / 1_000_000, 3),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "success": exit_code == 0 and not timed_out,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "event_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        "event_bytes": len(stdout.encode("utf-8")),
        "stderr_bytes": len(stderr.encode("utf-8")),
        "export_error": export_error,
        "event_path": str(event_path),
        "session_path": str(session_path) if session_path.is_file() else None,
    }
    append_summary(run_dir / "task-summary.jsonl", record)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run document-analysis tasks through OpenCode"
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instructions", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--opencode", default="opencode")
    parser.add_argument("--opencode-arg", action="append", default=[])
    parser.add_argument(
        "--server",
        default=os.environ.get("DOC_VLM_SERVER_URL", "http://127.0.0.1:30200"),
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--export-sessions", action="store_true")
    parser.add_argument("--preserve-cache-between-repeats", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.tasks.is_file():
        raise FileNotFoundError(args.tasks)
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    if not args.workspace.is_dir():
        raise FileNotFoundError(args.workspace)
    if args.instructions is not None and not args.instructions.is_file():
        raise FileNotFoundError(args.instructions)
    if args.concurrency <= 0 or args.repeats <= 0 or args.timeout <= 0:
        raise ValueError("concurrency, repeats and timeout must be positive")
    resolved = shutil.which(args.opencode)
    if resolved is None:
        raise FileNotFoundError(f"OpenCode executable not found: {args.opencode}")
    args.opencode = resolved
    args.run_id = args.run_id or f"opencode-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    args.run_id = validate_id(args.run_id, "run_id")
    run_dir = args.output_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    tasks = load_tasks(args.tasks)
    instructions = (
        args.instructions.read_text(encoding="utf-8")
        if args.instructions is not None
        else ""
    )
    records = []
    for repeat in range(args.repeats):
        if not args.preserve_cache_between_repeats:
            clear_server_cache(args.server, min(args.timeout, 60))
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as executor:
            futures = [
                executor.submit(
                    run_task,
                    task,
                    repeat=repeat,
                    args=args,
                    instructions=instructions,
                    run_dir=run_dir,
                )
                for task in tasks
            ]
            for future in concurrent.futures.as_completed(futures):
                record = future.result()
                records.append(record)
                print(
                    f"{record['task_id']} repeat={record['repeat']} "
                    f"wall={record['wall_ms'] / 1000:.2f}s "
                    f"exit={record['exit_code']} session={record['opencode_session_id']}",
                    flush=True,
                )
    success_count = sum(bool(record["success"]) for record in records)
    summary = {
        "schema": "slackmaint.opencode-run.v1",
        "run_id": args.run_id,
        "task_count": len(tasks),
        "repeats": args.repeats,
        "concurrency": args.concurrency,
        "execution_count": len(records),
        "success_count": success_count,
        "failure_count": len(records) - success_count,
        "run_dir": str(run_dir),
    }
    (run_dir / "run-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if success_count != len(records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
