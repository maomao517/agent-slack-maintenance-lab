#!/usr/bin/env python3
"""Audit D2 result JSON for reporting blockers without requiring a GPU."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find correctness and metric-definition blockers in D2 result JSON."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--expected-low-state-mb",
        type=float,
        default=7.875,
        help="Measured complete state size for the current low-resolution input.",
    )
    return parser.parse_args()


def walk(value: Any, path: str = "$") -> Iterator[tuple[str, str, Any]]:
    """Yield every mapping field in a nested JSON document."""
    if isinstance(value, dict):
        for key, item in value.items():
            item_path = f"{path}.{key}"
            yield item_path, str(key), item
            yield from walk(item, item_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from walk(item, f"{path}[{index}]")


def numeric_fields(payload: Any, field: str) -> list[float]:
    values: list[float] = []
    for _path, key, value in walk(payload):
        if key != field or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def field_paths(payload: Any, field: str) -> list[str]:
    return [path for path, key, _value in walk(payload) if key == field]


def summarize(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "zero_count": sum(value == 0 for value in values),
    }


def finding(priority: str, code: str, evidence: str, action: str) -> dict[str, str]:
    return {
        "priority": priority,
        "code": code,
        "evidence": evidence,
        "action": action,
    }


def build_audit(payload: Any, expected_low_state_mb: float) -> dict[str, Any]:
    """Build a schema-tolerant audit report from a D2 JSON result file."""
    findings: list[dict[str, str]] = []
    tracked_fields = (
        "answer_consistency",
        "vision_state_mb",
        "state_size_mb",
        "cpu_to_gpu_transfer_ms",
        "gpu_to_cpu_transfer_ms",
        "duplicate_encoder_calls",
        "avg_jct_ms",
        "p95_jct_ms",
        "total_ms",
        "encoder_ms",
        "workflow_jct_ms",
        "workflow_start_ns",
        "workflow_end_ns",
    )
    summaries = {
        field: summarize(numeric_fields(payload, field)) for field in tracked_fields
    }

    answer_values = numeric_fields(payload, "answer_consistency")
    if answer_values and all(value == 0 for value in answer_values):
        findings.append(
            finding(
                "P0",
                "answer-consistency-not-collected",
                "All answer_consistency values are 0, which may represent an "
                "unpopulated placeholder rather than output equality.",
                "Run a cold-versus-hit output check with logits and greedy generation "
                "for each cache tier before reporting correctness.",
            )
        )

    state_values = numeric_fields(payload, "vision_state_mb")
    positive_state_values = [value for value in state_values if value > 0]
    if positive_state_values and any(
        value < expected_low_state_mb for value in positive_state_values
    ):
        findings.append(
            finding(
                "P0",
                "possible-image-embeds-only-state",
                f"vision_state_mb includes a positive value below the measured "
                f"complete low-resolution state ({expected_low_state_mb:g}MB). "
                f"The field may contain image_embeds only or use a mismatched "
                f"input-size definition.",
                "Inspect the D2 save/load/inject path and verify that all three "
                "DeepStack tensors are saved and restored, then report tensor "
                "shapes and image_grid_thw with the size value.",
            )
        )

    h2d_values = numeric_fields(payload, "cpu_to_gpu_transfer_ms")
    if h2d_values and all(value == 0 for value in h2d_values):
        findings.append(
            finding(
                "P1",
                "cpu-to-gpu-transfer-uninstrumented",
                "All cpu_to_gpu_transfer_ms values are 0.",
                "Measure CPU cache lookup, CPU-to-GPU restore, state injection, "
                "cached forward, and queue wait in synchronized wall-clock windows.",
            )
        )

    duplicate_values = numeric_fields(payload, "duplicate_encoder_calls")
    if duplicate_values and all(value == 0 for value in duplicate_values):
        findings.append(
            finding(
                "P2",
                "no-coalescing-collision-pressure",
                "All duplicate_encoder_calls values are 0.",
                "Add a synchronized same-page cold-miss workload before claiming "
                "request coalescing suppresses duplicate encodes.",
            )
        )

    has_jct = bool(field_paths(payload, "avg_jct_ms") or field_paths(payload, "p95_jct_ms"))
    has_workflow_boundaries = bool(
        field_paths(payload, "workflow_start_ns")
        and field_paths(payload, "workflow_end_ns")
    )
    if has_jct and not has_workflow_boundaries and not field_paths(payload, "workflow_jct_ms"):
        findings.append(
            finding(
                "P0",
                "workflow-jct-boundaries-missing",
                "JCT fields exist but workflow_start_ns/workflow_end_ns and "
                "workflow_jct_ms are absent from the result JSON.",
                "Call the existing metric request latency until workflow boundaries "
                "are recorded and JCT can be recomputed per workflow.",
            )
        )

    meta = payload.get("meta") if isinstance(payload, dict) else None
    return {
        "schema": "slackmaint.d2-result-audit.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "meta": meta if isinstance(meta, dict) else {},
        "field_summaries": summaries,
        "finding_counts": dict(Counter(item["priority"] for item in findings)),
        "findings": findings,
        "report_ready": not any(item["priority"] == "P0" for item in findings),
    }


def main() -> None:
    args = parse_args()
    with args.input.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    report = build_audit(payload, args.expected_low_state_mb)
    report["input"] = str(args.input.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
