from __future__ import annotations

from dataclasses import dataclass

from .models import PolicyKind


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    task_id: str | None = None
    workflow_id: str | None = None


@dataclass(frozen=True)
class PolicyView:
    now_ms: int
    ready_workflow_ids: tuple[str, ...]
    waiting_until_ms: tuple[int, ...]
    runnable_task_ids: tuple[str, ...]
    critical_task_ids: tuple[str, ...]
    task_remaining_ms: dict[str, int]
    task_deadlines_ms: dict[str, int | None]
    periodic_interval_ms: int
    periodic_budget_ms: int


def enforces_freshness(policy: PolicyKind) -> bool:
    return policy is not PolicyKind.NONE


def _earliest_deadline_task(view: PolicyView, task_ids: tuple[str, ...]) -> str | None:
    if not task_ids:
        return None
    return min(
        task_ids,
        key=lambda task_id: (
            view.task_deadlines_ms[task_id]
            if view.task_deadlines_ms[task_id] is not None
            else float("inf"),
            view.task_remaining_ms[task_id],
            task_id,
        ),
    )


def choose_action(policy: PolicyKind, view: PolicyView) -> PolicyDecision:
    critical = _earliest_deadline_task(view, view.critical_task_ids)
    if critical is not None:
        return PolicyDecision("maintenance", task_id=critical)

    ready = view.ready_workflow_ids[0] if view.ready_workflow_ids else None
    task = _earliest_deadline_task(view, view.runnable_task_ids)
    has_waiting_agent = bool(view.waiting_until_ms)

    if policy is PolicyKind.NONE:
        return PolicyDecision("model", workflow_id=ready) if ready else PolicyDecision("idle")

    if policy is PolicyKind.SYNC:
        return PolicyDecision("model", workflow_id=ready) if ready else PolicyDecision("idle")

    if policy is PolicyKind.BACKGROUND:
        if task is not None:
            return PolicyDecision("maintenance", task_id=task)
        return PolicyDecision("model", workflow_id=ready) if ready else PolicyDecision("idle")

    if policy is PolicyKind.PERIODIC:
        in_budget = (
            view.now_ms % view.periodic_interval_ms < view.periodic_budget_ms
        )
        if task is not None and in_budget:
            return PolicyDecision("maintenance", task_id=task)
        return PolicyDecision("model", workflow_id=ready) if ready else PolicyDecision("idle")

    if policy is PolicyKind.AGENT_AWARE:
        if task is not None and has_waiting_agent:
            return PolicyDecision("maintenance", task_id=task)
        return PolicyDecision("model", workflow_id=ready) if ready else PolicyDecision("idle")

    if policy is PolicyKind.RESOURCE_AWARE:
        if ready is not None:
            return PolicyDecision("model", workflow_id=ready)
        if task is not None:
            return PolicyDecision("maintenance", task_id=task)
        return PolicyDecision("idle")

    if policy is PolicyKind.DUAL_AWARE:
        if ready is not None:
            return PolicyDecision("model", workflow_id=ready)
        if task is not None and has_waiting_agent:
            return PolicyDecision("maintenance", task_id=task)
        return PolicyDecision("idle")

    if policy is PolicyKind.ORACLE:
        if ready is not None:
            return PolicyDecision("model", workflow_id=ready)
        if task is None or not has_waiting_agent:
            return PolicyDecision("idle")
        slack_ms = min(view.waiting_until_ms) - view.now_ms
        fitting = tuple(
            task_id
            for task_id in view.runnable_task_ids
            if view.task_remaining_ms[task_id] <= slack_ms
        )
        selected = _earliest_deadline_task(view, fitting) or task
        return PolicyDecision("maintenance", task_id=selected)

    raise ValueError(f"unsupported policy: {policy}")

