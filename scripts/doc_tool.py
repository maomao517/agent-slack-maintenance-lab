#!/usr/bin/env python3
"""Shell-friendly tools for the OpenCode document-agent workload."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


def append_jsonl(path: Path | None, event: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, payload)
    finally:
        os.close(descriptor)


def identifiers(args: argparse.Namespace) -> dict[str, str]:
    values = {
        "run_id": args.run_id or os.environ.get("RUN_ID", "manual"),
        "workflow_id": args.workflow_id
        or os.environ.get("WORKFLOW_ID", "manual"),
        "task_id": args.task_id or os.environ.get("TASK_ID", "manual"),
    }
    if args.require_context and "manual" in values.values():
        raise ValueError(
            "RUN_ID, WORKFLOW_ID and TASK_ID are required in experiment mode"
        )
    return values


def request_json(
    method: str, url: str, payload: dict[str, Any] | None, timeout: float
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"document service returned HTTP {exc.code}: {detail}") from exc
    if not isinstance(result, dict):
        raise RuntimeError("document service returned a non-object JSON value")
    return result


def tokenize_query(text: str) -> list[str]:
    lowered = text.casefold()
    tokens = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]{2,}", lowered)
    return list(dict.fromkeys(tokens))


def search_manifest(path: Path, query: str, top_k: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    tokens = tokenize_query(query)
    scored = []
    for row in rows:
        haystack = " ".join(str(value) for value in row.values()).casefold()
        score = sum(haystack.count(token) for token in tokens)
        if query.casefold() in haystack:
            score += 3
        if score:
            scored.append((score, row))
    scored.sort(key=lambda item: (-item[0], str(item[1].get("page_id", ""))))
    return {
        "ok": True,
        "query": query,
        "matches": [dict(row, score=score) for score, row in scored[:top_k]],
    }


ALLOWED_BINARY = {
    ast.Add: lambda left, right: left + right,
    ast.Sub: lambda left, right: left - right,
    ast.Mult: lambda left, right: left * right,
    ast.Div: lambda left, right: left / right,
    ast.FloorDiv: lambda left, right: left // right,
    ast.Mod: lambda left, right: left % right,
    ast.Pow: lambda left, right: left**right,
}
ALLOWED_UNARY = {ast.UAdd: lambda value: value, ast.USub: lambda value: -value}


def safe_calculate(expression: str) -> int | float:
    if len(expression) > 200:
        raise ValueError("expression is too long")
    tree = ast.parse(expression, mode="eval")

    def evaluate(node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if isinstance(node.value, bool):
                raise ValueError("booleans are not allowed")
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_BINARY:
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 12:
                raise ValueError("exponent is too large")
            result = ALLOWED_BINARY[type(node.op)](left, right)
            if not math.isfinite(float(result)) or abs(float(result)) > 1e18:
                raise ValueError("result is outside the allowed range")
            return result
        if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_UNARY:
            return ALLOWED_UNARY[type(node.op)](evaluate(node.operand))
        raise ValueError(f"unsupported expression node: {type(node).__name__}")

    return evaluate(tree)


def execute(args: argparse.Namespace, ids: dict[str, str]) -> dict[str, Any]:
    if args.command == "search":
        return search_manifest(args.manifest, args.query, args.top_k)
    if args.command == "analyze":
        payload = {
            **ids,
            "turn_id": args.turn_id,
            "document_id": args.document_id,
            "page_id": args.page_id,
            "document_version": args.version,
            "image": str(args.image),
            "question": args.question,
            "max_new_tokens": args.max_new_tokens,
        }
        return request_json(
            "POST", f"{args.server.rstrip('/')}/analyze", payload, args.timeout
        )
    if args.command == "calculate":
        return {
            "ok": True,
            "expression": args.expression,
            "result": safe_calculate(args.expression),
        }
    if args.command == "submit":
        evidence = json.loads(args.evidence_json)
        if not isinstance(evidence, list) or not evidence:
            raise ValueError("evidence-json must be a non-empty JSON list")
        return {"ok": True, "answer": args.answer, "evidence": evidence}
    if args.command == "health":
        return request_json("GET", f"{args.server.rstrip('/')}/health", None, args.timeout)
    if args.command == "metrics":
        return request_json("GET", f"{args.server.rstrip('/')}/metrics", None, args.timeout)
    if args.command == "clear-cache":
        return request_json(
            "POST", f"{args.server.rstrip('/')}/cache/clear", {}, args.timeout
        )
    raise ValueError(f"unknown command: {args.command}")


def add_server_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--server",
        default=os.environ.get("DOC_VLM_SERVER_URL", "http://127.0.0.1:30200"),
    )
    parser.add_argument("--timeout", type=float, default=600)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenCode tools for visual document analysis"
    )
    parser.add_argument("--trace-output", type=Path, default=os.environ.get("DOC_TOOL_TRACE"))
    parser.add_argument("--run-id")
    parser.add_argument("--workflow-id")
    parser.add_argument("--task-id")
    parser.add_argument("--require-context", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    search = subparsers.add_parser("search")
    search.add_argument("--manifest", type=Path, required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--top-k", type=int, default=5)

    analyze = subparsers.add_parser("analyze")
    add_server_args(analyze)
    analyze.add_argument("--turn-id", type=int, default=0)
    analyze.add_argument("--document-id", required=True)
    analyze.add_argument("--page-id", required=True)
    analyze.add_argument("--version", required=True)
    analyze.add_argument("--image", type=Path, required=True)
    analyze.add_argument("--question", required=True)
    analyze.add_argument("--max-new-tokens", type=int, default=128)

    calculate = subparsers.add_parser("calculate")
    calculate.add_argument("--expression", required=True)

    submit = subparsers.add_parser("submit")
    submit.add_argument("--answer", required=True)
    submit.add_argument("--evidence-json", required=True)

    for name in ("health", "metrics", "clear-cache"):
        command = subparsers.add_parser(name)
        add_server_args(command)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    event_id = uuid.uuid4().hex
    start_unix_ns = time.time_ns()
    started = time.perf_counter_ns()
    success = False
    error = None
    ids = {
        "run_id": args.run_id or os.environ.get("RUN_ID", "manual"),
        "workflow_id": args.workflow_id or os.environ.get("WORKFLOW_ID", "manual"),
        "task_id": args.task_id or os.environ.get("TASK_ID", "manual"),
    }
    try:
        ids = identifiers(args)
        result = execute(args, ids)
        success = bool(result.get("ok", True))
        if not success:
            raise RuntimeError(str(result.get("error", "tool returned ok=false")))
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(json.dumps({"ok": False, "error": error}, ensure_ascii=False), file=sys.stdout)
    finally:
        end_unix_ns = time.time_ns()
        append_jsonl(
            args.trace_output,
            {
                "schema": "slackmaint.opencode-tool.v1",
                "event": "tool_call",
                "event_id": event_id,
                "tool_name": args.command,
                "start_unix_ns": start_unix_ns,
                "end_unix_ns": end_unix_ns,
                "duration_ms": round((time.perf_counter_ns() - started) / 1_000_000, 3),
                "success": success,
                "error": error,
                **ids,
            },
        )
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
