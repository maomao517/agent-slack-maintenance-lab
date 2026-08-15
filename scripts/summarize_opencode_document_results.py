#!/usr/bin/env python3
"""Summarize OpenCode task, tool, and visual-access JSONL traces."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * fraction + 0.999) - 1))
    return ordered[index]


def distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 3),
        "median": round(statistics.median(values), 3),
        "p95": round(percentile(values, 0.95) or 0.0, 3),
    }


def classify_visual_reuse(rows: list[dict[str, Any]]) -> dict[str, int]:
    seen: dict[tuple[str, str], set[str]] = {}
    cold = 0
    within = 0
    cross = 0
    for row in sorted(rows, key=lambda item: int(item.get("start_unix_ns", 0))):
        image = str(row.get("image_sha256", ""))
        version = str(row.get("document_version", ""))
        workflow = str(row.get("workflow_id", ""))
        key = (image, version)
        workflows = seen.setdefault(key, set())
        if not workflows:
            cold += 1
        elif workflow in workflows:
            within += 1
        else:
            cross += 1
        workflows.add(workflow)
    return {
        "cold_accesses": cold,
        "within_workflow_reuses": within,
        "cross_workflow_reuses": cross,
    }


def summarize(
    tasks: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    visual: list[dict[str, Any]],
    *,
    run_id: str | None,
) -> dict[str, Any]:
    if run_id:
        tasks = [row for row in tasks if row.get("run_id") == run_id]
        tools = [row for row in tools if row.get("run_id") == run_id]
        visual = [row for row in visual if row.get("run_id") == run_id]
    task_jct = [float(row["wall_ms"]) for row in tasks if "wall_ms" in row]
    tool_ms = [float(row["duration_ms"]) for row in tools if "duration_ms" in row]
    visual_total_ms = [
        float(row["total_ms"]) for row in visual if "total_ms" in row
    ]
    reuse = classify_visual_reuse(visual)
    visual_count = len(visual)
    cache_hits = sum(bool(row.get("cache_hit")) for row in visual)
    encoder_calls = sum(bool(row.get("encoder_called")) for row in visual)
    start_times = [int(row["start_unix_ns"]) for row in tasks if "start_unix_ns" in row]
    end_times = [int(row["end_unix_ns"]) for row in tasks if "end_unix_ns" in row]
    run_span_ms = (
        (max(end_times) - min(start_times)) / 1_000_000
        if start_times and end_times
        else None
    )
    successful_tasks = sum(bool(row.get("success")) for row in tasks)
    peak_memory = [
        float(row["peak_gpu_memory_mb"])
        for row in visual
        if "peak_gpu_memory_mb" in row
    ]
    h2d_transfer_mb = sum(float(row.get("h2d_state_transfer_mb", 0)) for row in visual)
    workflows = {str(row.get("workflow_id")) for row in tasks if row.get("workflow_id")}
    task_ids = {str(row.get("task_id")) for row in tasks if row.get("task_id")}
    return {
        "schema": "slackmaint.opencode-summary.v1",
        "run_id": run_id,
        "task_execution_count": len(tasks),
        "unique_task_count": len(task_ids),
        "workflow_count": len(workflows),
        "successful_tasks": successful_tasks,
        "failed_tasks": sum(not bool(row.get("success")) for row in tasks),
        "run_span_ms": round(run_span_ms, 3) if run_span_ms is not None else None,
        "throughput_tasks_per_second": round(successful_tasks / (run_span_ms / 1000), 6)
        if run_span_ms
        else None,
        "task_jct_ms": distribution(task_jct),
        "tool_call_count": len(tools),
        "tool_duration_ms": distribution(tool_ms),
        "visual_access_count": visual_count,
        "visual_access_ms": distribution(visual_total_ms),
        "cache_hits": cache_hits,
        "cache_hit_rate": round(cache_hits / visual_count, 6) if visual_count else 0.0,
        "encoder_calls": encoder_calls,
        "encoder_avoidance_rate": round(1 - encoder_calls / visual_count, 6)
        if visual_count
        else 0.0,
        "peak_gpu_memory_mb": max(peak_memory) if peak_memory else None,
        "h2d_state_transfer_mb": round(h2d_transfer_mb, 6),
        **reuse,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize an OpenCode document run")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--visual-events", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_path = args.run_dir / "task-summary.jsonl"
    tool_path = args.run_dir / "tool-events.jsonl"
    for path in (task_path, tool_path, args.visual_events):
        if not path.is_file():
            raise FileNotFoundError(path)
    run_id = args.run_id or args.run_dir.name
    result = summarize(
        read_jsonl(task_path),
        read_jsonl(tool_path),
        read_jsonl(args.visual_events),
        run_id=run_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
