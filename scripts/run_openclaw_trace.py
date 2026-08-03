#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path


CATEGORIES = ("commercial", "legal", "compliance", "strategic", "coding")


def post_json(url: str, payload: dict[str, object]) -> None:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        if response.status != 200:
            raise RuntimeError(f"trace proxy returned HTTP {response.status}")


def load_tasks(root: Path, categories: list[str], scenarios: set[str]) -> list[dict]:
    tasks = []
    for category in categories:
        path = root / "claw-tasks" / category / "tasks.json"
        for task in json.loads(path.read_text(encoding="utf-8")):
            if scenarios and task["name"] not in scenarios:
                continue
            tasks.append({**task, "category": category})
    return tasks


def set_provider_url(config: dict, provider: str, base_url: str) -> None:
    try:
        config["models"]["providers"][provider]["baseUrl"] = base_url
    except KeyError as exc:
        raise KeyError(
            f"OpenClaw config has no models.providers.{provider}.baseUrl"
        ) from exc


def parse_agent_output(stdout: str) -> dict[str, object]:
    try:
        payload = json.loads(stdout[stdout.index("{") :])
        meta = payload.get("meta", {}).get("agentMeta", {})
        usage = meta.get("lastCallUsage", meta.get("usage", {}))
        texts = [item.get("text", "") for item in payload.get("payloads", [])]
        content = "\n".join(texts)
        return {
            "prompt_tokens": usage.get("input", 0),
            "completion_tokens": usage.get("output", 0),
            "output_chars": len(content),
            "output_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        }
    except (ValueError, json.JSONDecodeError):
        return {"error": "parse_failed"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ClawTasks through a trace proxy")
    parser.add_argument("--clawtasks-root", type=Path, required=True)
    parser.add_argument("--openclaw", type=Path, required=True)
    parser.add_argument("--node", default="node")
    parser.add_argument("--config", type=Path, default=Path("~/.openclaw/openclaw.json"))
    parser.add_argument("--workspace", type=Path, default=Path("~/.openclaw/workspace/contracts"))
    parser.add_argument("--provider", default="sglang")
    parser.add_argument("--base-url", default="http://127.0.0.1:30100/v1")
    parser.add_argument("--trace-control-url", default="http://127.0.0.1:30100/_trace/context")
    parser.add_argument("--arm", choices=("Direct", "CP"), required=True)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--category", nargs="+", choices=CATEGORIES, default=list(CATEGORIES))
    parser.add_argument("--scenario", nargs="*", default=[])
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.config = args.config.expanduser()
    args.workspace = args.workspace.expanduser()
    args.openclaw = args.openclaw.expanduser()
    tasks = load_tasks(args.clawtasks_root, args.category, set(args.scenario))
    if not tasks:
        raise SystemExit("no matching ClawTasks scenarios")

    args.workspace.mkdir(parents=True, exist_ok=True)
    source = args.clawtasks_root / "data" / "workspace"
    for path in source.iterdir():
        if path.is_file():
            shutil.copy2(path, args.workspace / path.name)

    original_text = args.config.read_text(encoding="utf-8")
    config = json.loads(original_text)
    set_provider_url(config, args.provider, args.base_url)
    args.config.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        with args.output.open("w", encoding="utf-8") as output:
            for task in tasks:
                session_id = f"trace-{task['name']}-{args.arm}-t{args.trial}-{time.time_ns()}"
                for turn, message in enumerate(task["turns"]):
                    labels = {
                        "scenario": task["name"],
                        "arm": args.arm,
                        "trial": args.trial,
                        "turn": turn,
                        "session_id": session_id,
                    }
                    post_json(args.trace_control_url, labels)
                    start_ns = time.time_ns()
                    completed = subprocess.run(
                        [
                            args.node,
                            str(args.openclaw),
                            "agent",
                            "--local",
                            "--session-id",
                            session_id,
                            "--message",
                            message,
                            "--json",
                            "--timeout",
                            str(args.timeout),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=args.timeout + 30,
                    )
                    end_ns = time.time_ns()
                    record = {
                        "schema": "slackmaint.agent-turn.v1",
                        "event": "agent_turn",
                        **labels,
                        "start_unix_ns": start_ns,
                        "end_unix_ns": end_ns,
                        "wall_ms": round((end_ns - start_ns) / 1_000_000, 3),
                        "returncode": completed.returncode,
                        **parse_agent_output(completed.stdout),
                    }
                    if completed.returncode != 0:
                        record["stderr_sha256"] = hashlib.sha256(
                            completed.stderr.encode("utf-8")
                        ).hexdigest()
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    print(
                        f"{task['name']} turn={turn} arm={args.arm} "
                        f"wall={record['wall_ms'] / 1000:.2f}s "
                        f"prompt_tokens={record.get('prompt_tokens', 0)}"
                    )
    finally:
        args.config.write_text(original_text, encoding="utf-8")


if __name__ == "__main__":
    main()
