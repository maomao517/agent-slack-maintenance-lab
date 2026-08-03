from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _rounded_ms(nanoseconds: int, tick_ms: int, *, allow_zero: bool) -> int:
    milliseconds = round(nanoseconds / 1_000_000)
    if allow_zero and milliseconds <= 0:
        return 0
    milliseconds = max(tick_ms, milliseconds)
    return ((milliseconds + tick_ms - 1) // tick_ms) * tick_ms


def events_to_experiment(
    events: Iterable[dict[str, Any]],
    *,
    arm: str | None,
    maintenance_ms: int,
    tick_ms: int = 1,
) -> dict[str, Any]:
    if maintenance_ms <= 0:
        raise ValueError("maintenance_ms must be positive")
    if tick_ms <= 0:
        raise ValueError("tick_ms must be positive")

    calls = [
        event
        for event in events
        if event.get("event") == "llm_call"
        and (arm is None or event.get("arm") == arm)
        and event.get("error") in (None, "")
        and (
            event.get("status") is None
            or 200 <= int(event["status"]) < 300
        )
        and isinstance(event.get("start_unix_ns"), int)
        and isinstance(event.get("end_unix_ns"), int)
        and event["end_unix_ns"] >= event["start_unix_ns"]
    ]
    if not calls:
        raise ValueError("trace contains no matching llm_call events")

    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for event in calls:
        key = (
            str(event.get("scenario", "unknown")),
            str(event.get("arm", "unknown")),
            int(event.get("trial", 0)),
        )
        groups[key].append(event)

    trace_start_ns = min(event["start_unix_ns"] for event in calls)
    workflows = []
    tasks = []

    for key in sorted(groups):
        scenario, event_arm, trial = key
        ordered = sorted(groups[key], key=lambda item: item["start_unix_ns"])
        workflow_id = f"{scenario}-{event_arm}-trial-{trial}"
        model_segments = [
            _rounded_ms(
                event["end_unix_ns"] - event["start_unix_ns"],
                tick_ms,
                allow_zero=False,
            )
            for event in ordered
        ]
        tool_waits = [
            _rounded_ms(
                max(0, right["start_unix_ns"] - left["end_unix_ns"]),
                tick_ms,
                allow_zero=True,
            )
            for left, right in zip(ordered, ordered[1:])
        ]
        workflows.append(
            {
                "workflow_id": workflow_id,
                "tenant_id": scenario,
                "start_ms": _rounded_ms(
                    ordered[0]["start_unix_ns"] - trace_start_ns,
                    tick_ms,
                    allow_zero=True,
                ),
                "model_segments_ms": model_segments,
                "tool_waits_ms": tool_waits,
            }
        )

        for segment_index in range(len(ordered) - 1):
            tasks.append(
                {
                    "task_id": f"maintenance-{workflow_id}-{segment_index}",
                    "owner_workflow_id": workflow_id,
                    "task_type": "context_index_update",
                    "work_ms": maintenance_ms,
                    "trigger_after_segment": segment_index,
                    "required_before_segment": segment_index + 1,
                    "version": segment_index + 1,
                }
            )

    return {
        "name": f"agent-trace-{arm or 'all'}",
        "tick_ms": tick_ms,
        "periodic_interval_ms": 200,
        "periodic_budget_ms": 50,
        "workflows": workflows,
        "maintenance_tasks": tasks,
        "metadata": {
            "source": "openai_trace_proxy",
            "arm": arm,
            "llm_calls": len(calls),
            "maintenance_ms": maintenance_ms,
        },
    }


def convert_trace_file(
    path: Path,
    *,
    arm: str | None,
    maintenance_ms: int,
    tick_ms: int = 1,
) -> dict[str, Any]:
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return events_to_experiment(
        events,
        arm=arm,
        maintenance_ms=maintenance_ms,
        tick_ms=tick_ms,
    )
