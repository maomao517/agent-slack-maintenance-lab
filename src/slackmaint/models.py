from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PolicyKind(str, Enum):
    NONE = "none"
    SYNC = "sync"
    PERIODIC = "periodic"
    BACKGROUND = "background"
    RESOURCE_AWARE = "resource_aware"
    AGENT_AWARE = "agent_aware"
    DUAL_AWARE = "dual_aware"
    ORACLE = "oracle"


@dataclass(frozen=True)
class WorkflowSpec:
    workflow_id: str
    tenant_id: str
    start_ms: int
    model_segments_ms: tuple[int, ...]
    tool_waits_ms: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.model_segments_ms) != len(self.tool_waits_ms) + 1:
            raise ValueError(
                f"{self.workflow_id}: model_segments_ms must contain exactly "
                "one more item than tool_waits_ms"
            )
        if any(value <= 0 for value in self.model_segments_ms):
            raise ValueError("model segment durations must be positive")
        if any(value < 0 for value in self.tool_waits_ms):
            raise ValueError("tool wait durations cannot be negative")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowSpec":
        return cls(
            workflow_id=str(data["workflow_id"]),
            tenant_id=str(data.get("tenant_id", "default")),
            start_ms=int(data.get("start_ms", 0)),
            model_segments_ms=tuple(int(v) for v in data["model_segments_ms"]),
            tool_waits_ms=tuple(int(v) for v in data["tool_waits_ms"]),
        )


@dataclass(frozen=True)
class MaintenanceTaskSpec:
    task_id: str
    owner_workflow_id: str
    work_ms: int
    required_before_segment: int
    task_type: str = "index_update"
    trigger_after_segment: int | None = None
    release_ms: int | None = None
    freshness_deadline_ms: int | None = None
    resource: str = "cpu"
    version: int = 1
    preemptible: bool = True

    def __post_init__(self) -> None:
        if self.work_ms <= 0:
            raise ValueError("maintenance work must be positive")
        if self.trigger_after_segment is None and self.release_ms is None:
            raise ValueError("maintenance task needs a trigger or absolute release time")
        if not self.preemptible:
            raise ValueError(
                "the current simulator requires maintenance to be split into preemptible quanta"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MaintenanceTaskSpec":
        return cls(
            task_id=str(data["task_id"]),
            owner_workflow_id=str(data["owner_workflow_id"]),
            work_ms=int(data["work_ms"]),
            required_before_segment=int(data["required_before_segment"]),
            task_type=str(data.get("task_type", "index_update")),
            trigger_after_segment=(
                int(data["trigger_after_segment"])
                if data.get("trigger_after_segment") is not None
                else None
            ),
            release_ms=(
                int(data["release_ms"])
                if data.get("release_ms") is not None
                else None
            ),
            freshness_deadline_ms=(
                int(data["freshness_deadline_ms"])
                if data.get("freshness_deadline_ms") is not None
                else None
            ),
            resource=str(data.get("resource", "cpu")),
            version=int(data.get("version", 1)),
            preemptible=bool(data.get("preemptible", True)),
        )


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    tick_ms: int
    workflows: tuple[WorkflowSpec, ...]
    maintenance_tasks: tuple[MaintenanceTaskSpec, ...]
    periodic_interval_ms: int = 200
    periodic_budget_ms: int = 50
    max_time_ms: int = 600_000
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tick_ms <= 0:
            raise ValueError("tick_ms must be positive")
        workflow_ids = {workflow.workflow_id for workflow in self.workflows}
        if len(workflow_ids) != len(self.workflows):
            raise ValueError("workflow_id values must be unique")
        for task in self.maintenance_tasks:
            if task.owner_workflow_id not in workflow_ids:
                raise ValueError(f"unknown task owner: {task.owner_workflow_id}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentSpec":
        return cls(
            name=str(data.get("name", "experiment")),
            tick_ms=int(data.get("tick_ms", 10)),
            workflows=tuple(WorkflowSpec.from_dict(v) for v in data["workflows"]),
            maintenance_tasks=tuple(
                MaintenanceTaskSpec.from_dict(v)
                for v in data.get("maintenance_tasks", [])
            ),
            periodic_interval_ms=int(data.get("periodic_interval_ms", 200)),
            periodic_budget_ms=int(data.get("periodic_budget_ms", 50)),
            max_time_ms=int(data.get("max_time_ms", 600_000)),
            metadata=dict(data.get("metadata", {})),
        )

