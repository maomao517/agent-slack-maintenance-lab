from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean

from .models import ExperimentSpec, MaintenanceTaskSpec, PolicyKind, WorkflowSpec
from .policies import PolicyView, choose_action, enforces_freshness


@dataclass
class _WorkflowRuntime:
    spec: WorkflowSpec
    state: str = "pending"
    segment_index: int = 0
    remaining_model_ms: int = 0
    wait_until_ms: int | None = None
    completion_ms: int | None = None
    freshness_checked_segments: set[int] | None = None

    def __post_init__(self) -> None:
        self.remaining_model_ms = self.spec.model_segments_ms[0]
        self.freshness_checked_segments = set()


@dataclass
class _TaskRuntime:
    spec: MaintenanceTaskSpec
    remaining_ms: int
    release_time_ms: int | None = None
    completion_ms: int | None = None

    @property
    def released(self) -> bool:
        return self.release_time_ms is not None

    @property
    def complete(self) -> bool:
        return self.remaining_ms <= 0


@dataclass(frozen=True)
class SimulationMetrics:
    experiment: str
    policy: str
    workflow_count: int
    task_count: int
    average_jct_ms: float
    p95_jct_ms: float
    makespan_ms: int
    freshness_violations: int
    freshness_block_ms: int
    foreground_interference_ms: int
    maintenance_total_work_ms: int
    maintenance_completed_work_ms: int
    maintenance_overlap_ms: int
    maintenance_overlap_ratio: float
    maintenance_backlog_ms: int
    completed_tasks: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _p95(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, int(0.95 * len(ordered) + 0.999999) - 1)
    return float(ordered[index])


class Simulator:
    """Single-resource, trace-driven simulator for preemptible maintenance."""

    def __init__(self, spec: ExperimentSpec, policy: PolicyKind):
        self.spec = spec
        self.policy = policy
        self.workflows = {
            item.workflow_id: _WorkflowRuntime(item) for item in spec.workflows
        }
        self.tasks = {
            item.task_id: _TaskRuntime(item, item.work_ms)
            for item in spec.maintenance_tasks
        }
        self.now_ms = 0
        self.freshness_violations = 0
        self.freshness_block_ms = 0
        self.foreground_interference_ms = 0
        self.maintenance_overlap_ms = 0

    def run(self) -> SimulationMetrics:
        tick = self.spec.tick_ms
        while self.now_ms <= self.spec.max_time_ms:
            self._release_absolute_tasks()
            self._activate_workflows()
            self._wake_tools()

            if all(runtime.state == "done" for runtime in self.workflows.values()):
                break

            view = self._policy_view()
            decision = choose_action(self.policy, view)
            slice_ms = tick

            if decision.action == "maintenance" and decision.task_id is not None:
                task = self.tasks[decision.task_id]
                slice_ms = min(tick, task.remaining_ms)
                if view.ready_workflow_ids:
                    self.foreground_interference_ms += slice_ms
                if view.waiting_until_ms:
                    self.maintenance_overlap_ms += slice_ms
                if decision.task_id in view.critical_task_ids:
                    self.freshness_block_ms += slice_ms
                task.remaining_ms -= slice_ms
                if task.complete:
                    task.completion_ms = self.now_ms + slice_ms

            elif decision.action == "model" and decision.workflow_id is not None:
                workflow = self.workflows[decision.workflow_id]
                slice_ms = min(tick, workflow.remaining_model_ms)
                workflow.remaining_model_ms -= slice_ms
                if workflow.remaining_model_ms <= 0:
                    self._finish_model_segment(workflow, self.now_ms + slice_ms)

            self.now_ms += slice_ms

        else:
            raise RuntimeError("simulation exceeded max_time_ms")

        if not all(runtime.state == "done" for runtime in self.workflows.values()):
            raise RuntimeError("simulation ended before all workflows completed")

        return self._metrics()

    def _activate_workflows(self) -> None:
        for runtime in self.workflows.values():
            if runtime.state == "pending" and runtime.spec.start_ms <= self.now_ms:
                runtime.state = "ready"

    def _wake_tools(self) -> None:
        for runtime in self.workflows.values():
            if (
                runtime.state == "waiting"
                and runtime.wait_until_ms is not None
                and runtime.wait_until_ms <= self.now_ms
            ):
                runtime.state = "ready"
                runtime.wait_until_ms = None

    def _release_absolute_tasks(self) -> None:
        for task in self.tasks.values():
            if (
                not task.released
                and task.spec.release_ms is not None
                and task.spec.release_ms <= self.now_ms
            ):
                task.release_time_ms = self.now_ms

    def _finish_model_segment(
        self, runtime: _WorkflowRuntime, completion_time_ms: int
    ) -> None:
        finished_segment = runtime.segment_index
        for task in self.tasks.values():
            if (
                not task.released
                and task.spec.owner_workflow_id == runtime.spec.workflow_id
                and task.spec.trigger_after_segment == finished_segment
            ):
                task.release_time_ms = completion_time_ms

        if finished_segment == len(runtime.spec.model_segments_ms) - 1:
            runtime.state = "done"
            runtime.completion_ms = completion_time_ms
            return

        wait_ms = runtime.spec.tool_waits_ms[finished_segment]
        runtime.segment_index += 1
        runtime.remaining_model_ms = runtime.spec.model_segments_ms[runtime.segment_index]
        runtime.wait_until_ms = completion_time_ms + wait_ms
        runtime.state = "waiting"

    def _pending_required_tasks(self, workflow_id: str, segment_index: int) -> list[str]:
        return [
            task_id
            for task_id, task in self.tasks.items()
            if task.released
            and not task.complete
            and task.spec.owner_workflow_id == workflow_id
            and task.spec.required_before_segment <= segment_index
        ]

    def _policy_view(self) -> PolicyView:
        raw_ready = sorted(
            runtime.spec.workflow_id
            for runtime in self.workflows.values()
            if runtime.state == "ready"
        )
        critical: list[str] = []
        ready: list[str] = []

        for workflow_id in raw_ready:
            runtime = self.workflows[workflow_id]
            pending = self._pending_required_tasks(workflow_id, runtime.segment_index)
            if pending and enforces_freshness(self.policy):
                critical.extend(pending)
                continue
            if (
                pending
                and runtime.freshness_checked_segments is not None
                and runtime.segment_index not in runtime.freshness_checked_segments
            ):
                self.freshness_violations += 1
                runtime.freshness_checked_segments.add(runtime.segment_index)
            ready.append(workflow_id)

        runnable = tuple(
            sorted(
                task_id
                for task_id, task in self.tasks.items()
                if task.released and not task.complete
            )
        )
        waiting_until = tuple(
            sorted(
                runtime.wait_until_ms
                for runtime in self.workflows.values()
                if runtime.state == "waiting" and runtime.wait_until_ms is not None
            )
        )

        return PolicyView(
            now_ms=self.now_ms,
            ready_workflow_ids=tuple(ready),
            waiting_until_ms=waiting_until,
            runnable_task_ids=runnable,
            critical_task_ids=tuple(sorted(set(critical))),
            task_remaining_ms={
                task_id: self.tasks[task_id].remaining_ms for task_id in runnable
            },
            task_deadlines_ms={
                task_id: self.tasks[task_id].spec.freshness_deadline_ms
                for task_id in runnable
            },
            periodic_interval_ms=self.spec.periodic_interval_ms,
            periodic_budget_ms=self.spec.periodic_budget_ms,
        )

    def _metrics(self) -> SimulationMetrics:
        jcts = [
            runtime.completion_ms - runtime.spec.start_ms
            for runtime in self.workflows.values()
            if runtime.completion_ms is not None
        ]
        total_work = sum(task.spec.work_ms for task in self.tasks.values())
        backlog = sum(task.remaining_ms for task in self.tasks.values())
        completed_work = total_work - backlog
        overlap_ratio = (
            self.maintenance_overlap_ms / completed_work if completed_work else 0.0
        )

        return SimulationMetrics(
            experiment=self.spec.name,
            policy=self.policy.value,
            workflow_count=len(self.workflows),
            task_count=len(self.tasks),
            average_jct_ms=mean(jcts) if jcts else 0.0,
            p95_jct_ms=_p95(jcts),
            makespan_ms=max(
                runtime.completion_ms or 0 for runtime in self.workflows.values()
            ),
            freshness_violations=self.freshness_violations,
            freshness_block_ms=self.freshness_block_ms,
            foreground_interference_ms=self.foreground_interference_ms,
            maintenance_total_work_ms=total_work,
            maintenance_completed_work_ms=completed_work,
            maintenance_overlap_ms=self.maintenance_overlap_ms,
            maintenance_overlap_ratio=overlap_ratio,
            maintenance_backlog_ms=backlog,
            completed_tasks=sum(task.complete for task in self.tasks.values()),
        )

