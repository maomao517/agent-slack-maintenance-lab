#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import re
import time
from collections import defaultdict, deque
from pathlib import Path

from contextpilot.server.live_index import ContextPilot


FILE_PATTERN = re.compile(r"contracts/([A-Za-z0-9_.-]+\.(?:txt|py))")


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def timed(function, *args):
    start = time.perf_counter_ns()
    result = function(*args)
    return result, (time.perf_counter_ns() - start) / 1_000_000


def load_contexts(root: Path, repeats: int) -> list[list[str]]:
    base_contexts = []
    for task_file in sorted((root / "claw-tasks").glob("*/tasks.json")):
        for task in json.loads(task_file.read_text(encoding="utf-8")):
            seen = []
            for turn, message in enumerate(task["turns"]):
                for filename in FILE_PATTERN.findall(message):
                    if filename not in seen:
                        seen.append(filename)
                current = seen or [f"directory-listing:{task['name']}:{turn}"]
                base_contexts.append(list(current))
    return [list(context) for _ in range(repeats) for context in base_contexts]


def summarize(operation_ms: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    return {
        operation: {
            "count": len(values),
            "average_ms": sum(values) / len(values) if values else 0.0,
            "p50_ms": percentile(values, 0.50),
            "p95_ms": percentile(values, 0.95),
            "p99_ms": percentile(values, 0.99),
            "max_ms": max(values, default=0.0),
        }
        for operation, values in sorted(operation_ms.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure ContextPilot index maintenance")
    parser.add_argument("--clawtasks-root", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--initial-batch", type=int, default=8)
    parser.add_argument("--active-requests", type=int, default=128)
    parser.add_argument("--rebuild-every", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    contexts = load_contexts(args.clawtasks_root, args.repeats)
    if len(contexts) < args.initial_batch:
        raise SystemExit("not enough contexts for initial batch")

    operation_ms: dict[str, list[float]] = defaultdict(list)
    records = []
    active_ids: deque[str] = deque()
    active_contexts: deque[list[str]] = deque()
    engine = ContextPilot(use_gpu=False)
    output_stream = None if args.verbose else io.StringIO()

    with contextlib.redirect_stdout(output_stream or io.StringIO()) if not args.verbose else contextlib.nullcontext():
        initial = contexts[: args.initial_batch]
        result, elapsed = timed(engine.build_incremental, initial)
        operation_ms["initial_build"].append(elapsed)
        active_ids.extend(result["request_ids"])
        active_contexts.extend(initial)
        records.append({"operation": "initial_build", "elapsed_ms": elapsed, "batch": len(initial)})

        for index, context in enumerate(contexts[args.initial_batch :], start=args.initial_batch):
            result, elapsed = timed(engine.build_incremental, [context])
            operation_ms["incremental_update"].append(elapsed)
            active_ids.extend(result["request_ids"])
            active_contexts.append(context)
            records.append(
                {
                    "operation": "incremental_update",
                    "elapsed_ms": elapsed,
                    "position": index,
                    "matched": result["matched_count"],
                    "merged": result["merged_count"],
                }
            )

            while len(active_ids) > args.active_requests:
                request_id = active_ids.popleft()
                active_contexts.popleft()
                removal, removal_ms = timed(engine.remove_requests, {request_id})
                operation_ms["eviction_remove"].append(removal_ms)
                records.append(
                    {
                        "operation": "eviction_remove",
                        "elapsed_ms": removal_ms,
                        "removed": removal["removed_count"],
                    }
                )

            if args.rebuild_every and (index + 1) % args.rebuild_every == 0:
                fresh = ContextPilot(use_gpu=False)
                _, rebuild_ms = timed(fresh.build_incremental, list(active_contexts))
                operation_ms["fresh_rebuild"].append(rebuild_ms)
                records.append(
                    {
                        "operation": "fresh_rebuild",
                        "elapsed_ms": rebuild_ms,
                        "active_contexts": len(active_contexts),
                    }
                )

    report = {
        "schema": "slackmaint.contextpilot-probe.v1",
        "contextpilot_commit": "1fa0a143fdeda344585666648ab2b30cb7fea77f",
        "workload": {
            "source": "EfficientContext/ClawTasks",
            "contexts": len(contexts),
            "repeats": args.repeats,
            "active_requests": args.active_requests,
            "rebuild_every": args.rebuild_every,
        },
        "summary": summarize(operation_ms),
        "final_index_stats": engine.get_stats(),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
