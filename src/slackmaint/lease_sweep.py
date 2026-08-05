from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

from .lease_simulator import (
    LeaseExperimentSpec,
    LeaseMetrics,
    LeasePolicy,
    LeaseSimulator,
    LeaseWorkflowSpec,
)


CAPACITY_VALUES_MB = (1800, 2400, 3600, 5400, 7200, 10800, 14400)
KV_TTL_VALUES_MS = (250, 500, 1000, 2000, 4000, 8000)
WAIT_SCALE_VALUES = (0.5, 1.0, 2.0, 4.0)
ENCODER_RATIO_VALUES = (0.05, 0.10, 0.15, 0.25, 0.40, 0.70)
ENCODER_COST_SCALE_VALUES = (0.25, 0.5, 1.0, 2.0, 4.0)
CONCURRENCY_MULTIPLIERS = (1, 2, 4)
PREDICTION_SCALE_VALUES = (0.25, 0.5, 1.0, 2.0)


@dataclass(frozen=True)
class SweepCase:
    axis: str
    value: str
    spec: LeaseExperimentSpec

    @property
    def case_id(self) -> str:
        return f"{self.axis}:{self.value}"


def _scale(values: Iterable[int], factor: float) -> tuple[int, ...]:
    return tuple(max(0, round(value * factor)) for value in values)


def _replace_workflows(
    spec: LeaseExperimentSpec,
    workflows: Iterable[LeaseWorkflowSpec],
    *,
    name: str,
    max_time_ms: int | None = None,
) -> LeaseExperimentSpec:
    return replace(
        spec,
        name=name,
        workflows=tuple(workflows),
        max_time_ms=max_time_ms or spec.max_time_ms,
    )


def build_sweep_cases(spec: LeaseExperimentSpec) -> list[SweepCase]:
    cases = [SweepCase("baseline", "configured", spec)]

    for capacity_mb in CAPACITY_VALUES_MB:
        cases.append(
            SweepCase(
                "capacity_mb",
                str(capacity_mb),
                replace(spec, retention_capacity_mb=capacity_mb),
            )
        )

    for ttl_ms in KV_TTL_VALUES_MS:
        cases.append(
            SweepCase(
                "kv_ttl_ms",
                str(ttl_ms),
                replace(spec, fixed_kv_ttl_ms=ttl_ms),
            )
        )

    for capacity_mb in CAPACITY_VALUES_MB:
        for ttl_ms in KV_TTL_VALUES_MS:
            cases.append(
                SweepCase(
                    "capacity_ttl",
                    f"{capacity_mb}:{ttl_ms}",
                    replace(
                        spec,
                        retention_capacity_mb=capacity_mb,
                        fixed_kv_ttl_ms=ttl_ms,
                    ),
                )
            )

    for factor in WAIT_SCALE_VALUES:
        workflows = (
            replace(
                workflow,
                tool_waits_ms=_scale(workflow.tool_waits_ms, factor),
                expected_tool_waits_ms=_scale(
                    workflow.expected_tool_waits_ms, factor
                ),
            )
            for workflow in spec.workflows
        )
        cases.append(
            SweepCase(
                "tool_wait_scale",
                f"{factor:g}",
                _replace_workflows(
                    spec, workflows, name=f"{spec.name}-wait-{factor:g}"
                ),
            )
        )

    for ratio in ENCODER_RATIO_VALUES:
        workflows = (
            replace(
                workflow,
                encoder_size_mb=max(1, round(workflow.kv_size_mb * ratio)),
            )
            for workflow in spec.workflows
        )
        cases.append(
            SweepCase(
                "encoder_kv_ratio",
                f"{ratio:g}",
                _replace_workflows(
                    spec, workflows, name=f"{spec.name}-ratio-{ratio:g}"
                ),
            )
        )

    for factor in ENCODER_COST_SCALE_VALUES:
        workflows = (
            replace(
                workflow,
                encoder_ms=max(1, round(workflow.encoder_ms * factor)),
            )
            for workflow in spec.workflows
        )
        cases.append(
            SweepCase(
                "encoder_cost_scale",
                f"{factor:g}",
                _replace_workflows(
                    spec, workflows, name=f"{spec.name}-cost-{factor:g}"
                ),
            )
        )

    for multiplier in CONCURRENCY_MULTIPLIERS:
        workflows = []
        for replica in range(multiplier):
            for workflow in spec.workflows:
                workflows.append(
                    replace(
                        workflow,
                        workflow_id=f"{workflow.workflow_id}-r{replica}",
                        start_ms=workflow.start_ms + replica * 100,
                    )
                )
        cases.append(
            SweepCase(
                "concurrency_multiplier",
                str(multiplier),
                _replace_workflows(
                    spec,
                    workflows,
                    name=f"{spec.name}-concurrency-{multiplier}",
                    max_time_ms=spec.max_time_ms * multiplier,
                ),
            )
        )

    for factor in PREDICTION_SCALE_VALUES:
        workflows = (
            replace(
                workflow,
                expected_tool_waits_ms=_scale(workflow.tool_waits_ms, factor),
            )
            for workflow in spec.workflows
        )
        cases.append(
            SweepCase(
                "prediction_scale",
                f"{factor:g}",
                _replace_workflows(
                    spec, workflows, name=f"{spec.name}-prediction-{factor:g}"
                ),
            )
        )

    return cases


def _percent_gain(reference: float, candidate: float) -> float:
    if reference == 0:
        return 0.0
    return (reference - candidate) / reference * 100.0


def _comparison(case: SweepCase, metrics: list[LeaseMetrics]) -> dict[str, object]:
    by_policy = {metric.policy: metric for metric in metrics}
    joint = by_policy[LeasePolicy.JOINT_LEASE.value]
    fixed = by_policy[LeasePolicy.FIXED_KV_LEASE.value]
    encoder = by_policy[LeasePolicy.ENCODER_LRU.value]
    oracle = by_policy[LeasePolicy.ORACLE.value]
    baselines = (
        by_policy[LeasePolicy.NO_CACHE.value],
        encoder,
        fixed,
    )
    best_baseline = min(
        baselines,
        key=lambda metric: (
            metric.average_jct_ms,
            metric.p95_jct_ms,
            metric.retained_memory_time_mb_ms,
        ),
    )
    non_oracle = (*baselines, joint)
    winner = min(
        non_oracle,
        key=lambda metric: (
            metric.average_jct_ms,
            metric.p95_jct_ms,
            metric.retained_memory_time_mb_ms,
        ),
    )
    return {
        "case_id": case.case_id,
        "axis": case.axis,
        "value": case.value,
        "workflow_count": len(case.spec.workflows),
        "retention_capacity_mb": case.spec.retention_capacity_mb,
        "fixed_kv_ttl_ms": case.spec.fixed_kv_ttl_ms,
        "winner": winner.policy,
        "best_baseline": best_baseline.policy,
        "joint_average_jct_ms": joint.average_jct_ms,
        "joint_p95_jct_ms": joint.p95_jct_ms,
        "joint_recompute_ms": joint.total_recompute_ms,
        "joint_average_retained_mb": (
            joint.retained_memory_time_mb_ms / joint.makespan_ms
            if joint.makespan_ms
            else 0.0
        ),
        "joint_vs_best_jct_gain_pct": _percent_gain(
            best_baseline.average_jct_ms, joint.average_jct_ms
        ),
        "joint_vs_fixed_jct_gain_pct": _percent_gain(
            fixed.average_jct_ms, joint.average_jct_ms
        ),
        "joint_vs_encoder_jct_gain_pct": _percent_gain(
            encoder.average_jct_ms, joint.average_jct_ms
        ),
        "joint_vs_best_p95_gain_pct": _percent_gain(
            best_baseline.p95_jct_ms, joint.p95_jct_ms
        ),
        "joint_oracle_jct_gap_pct": _percent_gain(
            joint.average_jct_ms, oracle.average_jct_ms
        ),
    }


def run_sweep(spec: LeaseExperimentSpec) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metric_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    for case in build_sweep_cases(spec):
        metrics = [LeaseSimulator(case.spec, policy).run() for policy in LeasePolicy]
        comparison_rows.append(_comparison(case, metrics))
        for metric in metrics:
            metric_rows.append(
                {"case_id": case.case_id, "axis": case.axis, "value": case.value, **metric.to_dict()}
            )
    return metric_rows, comparison_rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary_markdown(comparisons: list[dict[str, object]]) -> str:
    wins: dict[str, int] = {}
    for row in comparisons:
        winner = str(row["winner"])
        wins[winner] = wins.get(winner, 0) + 1
    robust_joint = [
        row
        for row in comparisons
        if float(row["joint_vs_best_jct_gain_pct"]) > 0
        and float(row["joint_vs_best_p95_gain_pct"]) >= 0
    ]
    best = max(
        comparisons,
        key=lambda row: float(row["joint_vs_best_jct_gain_pct"]),
    )
    worst = min(
        comparisons,
        key=lambda row: float(row["joint_vs_best_jct_gain_pct"]),
    )
    lines = [
        "# Synthetic lease sweep summary",
        "",
        "> These results are generated from synthetic profiles. They validate simulator behavior, not real GPU performance.",
        "",
        f"- Cases: {len(comparisons)}",
        f"- Joint lease improves both average JCT and P95 over the best non-oracle baseline in {len(robust_joint)} cases.",
        f"- Best joint case: `{best['case_id']}` ({float(best['joint_vs_best_jct_gain_pct']):.2f}% average JCT gain).",
        f"- Worst joint case: `{worst['case_id']}` ({float(worst['joint_vs_best_jct_gain_pct']):.2f}% average JCT gain).",
        "",
        "## Non-oracle winner counts",
        "",
    ]
    for policy, count in sorted(wins.items()):
        lines.append(f"- `{policy}`: {count}")
    lines.extend(
        [
            "",
            "See `comparisons.csv` for derived gains and `metrics.json` for raw policy metrics.",
            "",
        ]
    )
    return "\n".join(lines)


def run_sweep_to_directory(config_path: Path, output_dir: Path) -> dict[str, object]:
    spec = LeaseExperimentSpec.from_dict(
        json.loads(config_path.read_text(encoding="utf-8"))
    )
    metrics, comparisons = run_sweep(spec)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "comparisons.csv", comparisons)
    (output_dir / "summary.md").write_text(
        _summary_markdown(comparisons), encoding="utf-8"
    )
    return {
        "case_count": len(comparisons),
        "metric_row_count": len(metrics),
        "output_dir": str(output_dir),
    }
