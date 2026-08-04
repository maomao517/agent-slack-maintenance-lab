from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from statistics import mean
from typing import Any


class LeasePolicy(str, Enum):
    NO_CACHE = "no_cache"
    ENCODER_LRU = "encoder_lru"
    FIXED_KV_LEASE = "fixed_kv_lease"
    JOINT_LEASE = "joint_lease"
    ORACLE = "oracle"


@dataclass(frozen=True)
class LeaseWorkflowSpec:
    workflow_id: str
    start_ms: int
    model_segments_ms: tuple[int, ...]
    tool_waits_ms: tuple[int, ...]
    expected_tool_waits_ms: tuple[int, ...]
    kv_size_mb: int
    encoder_size_mb: int
    prefill_ms: int
    encoder_ms: int

    def __post_init__(self) -> None:
        waits = len(self.tool_waits_ms)
        if len(self.model_segments_ms) != waits + 1:
            raise ValueError(
                f"{self.workflow_id}: model_segments_ms must contain one "
                "more item than tool_waits_ms"
            )
        if len(self.expected_tool_waits_ms) != waits:
            raise ValueError(
                f"{self.workflow_id}: expected_tool_waits_ms must match "
                "tool_waits_ms"
            )
        positive = (
            *self.model_segments_ms,
            self.kv_size_mb,
            self.encoder_size_mb,
            self.prefill_ms,
            self.encoder_ms,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("durations and state sizes must be positive")
        if self.encoder_size_mb >= self.kv_size_mb:
            raise ValueError(
                "encoder_size_mb must be smaller than kv_size_mb for tier demotion"
            )
        if any(value < 0 for value in self.tool_waits_ms):
            raise ValueError("tool waits cannot be negative")
        if any(value < 0 for value in self.expected_tool_waits_ms):
            raise ValueError("expected tool waits cannot be negative")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LeaseWorkflowSpec":
        actual_waits = tuple(int(value) for value in data["tool_waits_ms"])
        expected_waits = tuple(
            int(value)
            for value in data.get("expected_tool_waits_ms", actual_waits)
        )
        return cls(
            workflow_id=str(data["workflow_id"]),
            start_ms=int(data.get("start_ms", 0)),
            model_segments_ms=tuple(
                int(value) for value in data["model_segments_ms"]
            ),
            tool_waits_ms=actual_waits,
            expected_tool_waits_ms=expected_waits,
            kv_size_mb=int(data["kv_size_mb"]),
            encoder_size_mb=int(data["encoder_size_mb"]),
            prefill_ms=int(data["prefill_ms"]),
            encoder_ms=int(data["encoder_ms"]),
        )


@dataclass(frozen=True)
class LeaseExperimentSpec:
    name: str
    tick_ms: int
    retention_capacity_mb: int
    fixed_kv_ttl_ms: int
    encoder_ttl_ms: int
    max_time_ms: int
    workflows: tuple[LeaseWorkflowSpec, ...]

    def __post_init__(self) -> None:
        if self.tick_ms <= 0:
            raise ValueError("tick_ms must be positive")
        if self.retention_capacity_mb <= 0:
            raise ValueError("retention_capacity_mb must be positive")
        if self.fixed_kv_ttl_ms <= 0 or self.encoder_ttl_ms <= 0:
            raise ValueError("lease TTL values must be positive")
        workflow_ids = {item.workflow_id for item in self.workflows}
        if len(workflow_ids) != len(self.workflows):
            raise ValueError("workflow_id values must be unique")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LeaseExperimentSpec":
        return cls(
            name=str(data.get("name", "lease-experiment")),
            tick_ms=int(data.get("tick_ms", 10)),
            retention_capacity_mb=int(data["retention_capacity_mb"]),
            fixed_kv_ttl_ms=int(data.get("fixed_kv_ttl_ms", 2000)),
            encoder_ttl_ms=int(data.get("encoder_ttl_ms", 10000)),
            max_time_ms=int(data.get("max_time_ms", 600_000)),
            workflows=tuple(
                LeaseWorkflowSpec.from_dict(item) for item in data["workflows"]
            ),
        )


@dataclass
class _WorkflowRuntime:
    spec: LeaseWorkflowSpec
    state: str = "pending"
    segment_index: int = 0
    remaining_model_ms: int = 0
    ready_since_ms: int | None = None
    wait_until_ms: int | None = None
    completion_ms: int | None = None
    restoration_pending: bool = True


@dataclass
class _RetainedState:
    workflow_id: str
    target_segment: int
    tier: str
    size_mb: int
    created_ms: int
    expires_ms: int
    next_use_ms: int
    saved_ms: int

    @property
    def density(self) -> float:
        return self.saved_ms / self.size_mb


@dataclass(frozen=True)
class LeaseMetrics:
    experiment: str
    policy: str
    workflow_count: int
    retention_capacity_mb: int
    fixed_kv_ttl_ms: int
    encoder_ttl_ms: int
    average_jct_ms: float
    p95_jct_ms: float
    makespan_ms: int
    total_recompute_ms: int
    avoided_recompute_ms: int
    kv_hits: int
    encoder_hits: int
    cache_misses: int
    lease_expirations: int
    demotions: int
    forced_evictions: int
    peak_retained_mb: int
    retained_memory_time_mb_ms: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _p95(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, int(0.95 * len(ordered) + 0.999999) - 1)
    return float(ordered[index])


class LeaseSimulator:
    """Trace-driven simulator for KV and multimodal encoder state leases.

    The configured retention capacity is a budget reserved for states held
    between turns. Active request memory is intentionally excluded so every
    policy receives the same foreground execution capacity. The oracle policy
    knows actual tool return times but still uses greedy admission, so it is a
    clairvoyant reference rather than a mathematically optimal upper bound.
    """

    def __init__(self, spec: LeaseExperimentSpec, policy: LeasePolicy):
        self.spec = spec
        self.policy = policy
        self.workflows = {
            item.workflow_id: _WorkflowRuntime(item) for item in spec.workflows
        }
        self.retained: dict[str, _RetainedState] = {}
        self.now_ms = 0
        self.running_workflow_id: str | None = None
        self.total_recompute_ms = 0
        self.avoided_recompute_ms = 0
        self.kv_hits = 0
        self.encoder_hits = 0
        self.cache_misses = 0
        self.lease_expirations = 0
        self.demotions = 0
        self.forced_evictions = 0
        self.peak_retained_mb = 0
        self.retained_memory_time_mb_ms = 0

    def run(self) -> LeaseMetrics:
        tick = self.spec.tick_ms
        while self.now_ms <= self.spec.max_time_ms:
            self._expire_states()
            self._activate_workflows()
            self._wake_tools()

            if all(item.state == "done" for item in self.workflows.values()):
                break

            if self.running_workflow_id is None:
                self._start_next_ready_workflow()

            slice_ms = tick
            if self.running_workflow_id is not None:
                runtime = self.workflows[self.running_workflow_id]
                slice_ms = min(tick, runtime.remaining_model_ms)
                runtime.remaining_model_ms -= slice_ms
                if runtime.remaining_model_ms == 0:
                    self._finish_segment(runtime, self.now_ms + slice_ms)

            self.retained_memory_time_mb_ms += self._retained_mb() * slice_ms
            self.now_ms += slice_ms
        else:
            raise RuntimeError("lease simulation exceeded max_time_ms")

        if not all(item.state == "done" for item in self.workflows.values()):
            raise RuntimeError("lease simulation ended before completion")
        return self._metrics()

    def _activate_workflows(self) -> None:
        for runtime in self.workflows.values():
            if runtime.state == "pending" and runtime.spec.start_ms <= self.now_ms:
                runtime.state = "ready"
                runtime.ready_since_ms = runtime.spec.start_ms

    def _wake_tools(self) -> None:
        for runtime in self.workflows.values():
            if (
                runtime.state == "waiting"
                and runtime.wait_until_ms is not None
                and runtime.wait_until_ms <= self.now_ms
            ):
                runtime.state = "ready"
                runtime.ready_since_ms = runtime.wait_until_ms
                runtime.wait_until_ms = None

    def _start_next_ready_workflow(self) -> None:
        ready = [item for item in self.workflows.values() if item.state == "ready"]
        if not ready:
            return

        def priority(runtime: _WorkflowRuntime) -> tuple[int, int, str]:
            retained = self.retained.get(runtime.spec.workflow_id)
            has_kv_lease = int(
                retained is not None
                and retained.target_segment == runtime.segment_index
                and retained.tier == "kv"
            )
            lease_rank = -has_kv_lease if self.policy in {
                LeasePolicy.FIXED_KV_LEASE,
                LeasePolicy.JOINT_LEASE,
                LeasePolicy.ORACLE,
            } else 0
            return (
                lease_rank,
                runtime.ready_since_ms or 0,
                runtime.spec.workflow_id,
            )

        runtime = min(ready, key=priority)
        if runtime.restoration_pending:
            penalty = self._consume_retained_state(runtime)
            runtime.remaining_model_ms = (
                runtime.spec.model_segments_ms[runtime.segment_index] + penalty
            )
            runtime.restoration_pending = False
        runtime.state = "running"
        self.running_workflow_id = runtime.spec.workflow_id

    def _consume_retained_state(self, runtime: _WorkflowRuntime) -> int:
        spec = runtime.spec
        cold_cost = spec.prefill_ms + spec.encoder_ms
        if runtime.segment_index == 0:
            self.total_recompute_ms += cold_cost
            return cold_cost

        retained = self.retained.pop(spec.workflow_id, None)
        if retained is None or retained.target_segment != runtime.segment_index:
            self.cache_misses += 1
            self.total_recompute_ms += cold_cost
            return cold_cost
        if retained.tier == "kv":
            self.kv_hits += 1
            self.avoided_recompute_ms += cold_cost
            return 0

        self.encoder_hits += 1
        self.total_recompute_ms += spec.prefill_ms
        self.avoided_recompute_ms += spec.encoder_ms
        return spec.prefill_ms

    def _finish_segment(
        self, runtime: _WorkflowRuntime, completion_time_ms: int
    ) -> None:
        self.running_workflow_id = None
        if runtime.segment_index == len(runtime.spec.model_segments_ms) - 1:
            runtime.state = "done"
            runtime.completion_ms = completion_time_ms
            self.retained.pop(runtime.spec.workflow_id, None)
            return

        wait_index = runtime.segment_index
        actual_wait = runtime.spec.tool_waits_ms[wait_index]
        expected_wait = runtime.spec.expected_tool_waits_ms[wait_index]
        runtime.segment_index += 1
        runtime.restoration_pending = True
        runtime.wait_until_ms = completion_time_ms + actual_wait
        runtime.state = "waiting"
        self._retain_after_turn(
            runtime,
            created_ms=completion_time_ms,
            actual_wait_ms=actual_wait,
            expected_wait_ms=expected_wait,
        )

    def _retain_after_turn(
        self,
        runtime: _WorkflowRuntime,
        *,
        created_ms: int,
        actual_wait_ms: int,
        expected_wait_ms: int,
    ) -> None:
        spec = runtime.spec
        if self.policy is LeasePolicy.NO_CACHE:
            return
        if self.policy is LeasePolicy.ENCODER_LRU:
            tier = "encoder"
        elif self.policy is LeasePolicy.FIXED_KV_LEASE:
            tier = "kv"
        elif self.policy is LeasePolicy.ORACLE:
            tier = (
                "kv"
                if actual_wait_ms <= self.spec.fixed_kv_ttl_ms
                else "encoder"
            )
        else:
            tier = (
                "kv"
                if expected_wait_ms <= self.spec.fixed_kv_ttl_ms
                else "encoder"
            )

        state = self._make_state(
            runtime,
            tier=tier,
            created_ms=created_ms,
            actual_wait_ms=actual_wait_ms,
        )
        self._admit(state)

    def _make_state(
        self,
        runtime: _WorkflowRuntime,
        *,
        tier: str,
        created_ms: int,
        actual_wait_ms: int,
    ) -> _RetainedState:
        if tier == "kv":
            size_mb = runtime.spec.kv_size_mb
            ttl_ms = self.spec.fixed_kv_ttl_ms
            saved_ms = runtime.spec.prefill_ms + runtime.spec.encoder_ms
        else:
            size_mb = runtime.spec.encoder_size_mb
            ttl_ms = self.spec.encoder_ttl_ms
            saved_ms = runtime.spec.encoder_ms
        return _RetainedState(
            workflow_id=runtime.spec.workflow_id,
            target_segment=runtime.segment_index,
            tier=tier,
            size_mb=size_mb,
            created_ms=created_ms,
            expires_ms=created_ms + ttl_ms,
            next_use_ms=created_ms + actual_wait_ms,
            saved_ms=saved_ms,
        )

    def _admit(self, state: _RetainedState) -> None:
        self.retained.pop(state.workflow_id, None)
        if state.size_mb > self.spec.retention_capacity_mb:
            if state.tier == "kv" and self.policy in {
                LeasePolicy.JOINT_LEASE,
                LeasePolicy.ORACLE,
            }:
                state = self._demote_state(state)
            if state.size_mb > self.spec.retention_capacity_mb:
                self.forced_evictions += 1
                return

        while self._retained_mb() + state.size_mb > self.spec.retention_capacity_mb:
            if self.policy in {LeasePolicy.JOINT_LEASE, LeasePolicy.ORACLE}:
                candidate = self._lowest_density_kv()
                if candidate is not None:
                    self.retained[candidate.workflow_id] = self._demote_state(candidate)
                    if self._retained_mb() + state.size_mb <= self.spec.retention_capacity_mb:
                        break
            victim = self._eviction_victim(state)
            if victim is None:
                self.forced_evictions += 1
                return
            del self.retained[victim.workflow_id]
            self.forced_evictions += 1

        self.retained[state.workflow_id] = state
        self.peak_retained_mb = max(self.peak_retained_mb, self._retained_mb())

    def _lowest_density_kv(self) -> _RetainedState | None:
        candidates = [item for item in self.retained.values() if item.tier == "kv"]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda item: (item.density, -item.next_use_ms, item.workflow_id),
        )

    def _demote_state(self, state: _RetainedState) -> _RetainedState:
        runtime = self.workflows[state.workflow_id]
        self.demotions += 1
        return _RetainedState(
            workflow_id=state.workflow_id,
            target_segment=state.target_segment,
            tier="encoder",
            size_mb=runtime.spec.encoder_size_mb,
            created_ms=self.now_ms,
            expires_ms=self.now_ms + self.spec.encoder_ttl_ms,
            next_use_ms=state.next_use_ms,
            saved_ms=runtime.spec.encoder_ms,
        )

    def _eviction_victim(
        self, incoming: _RetainedState
    ) -> _RetainedState | None:
        if not self.retained:
            return None
        if self.policy is LeasePolicy.ENCODER_LRU:
            return min(
                self.retained.values(),
                key=lambda item: (item.created_ms, item.workflow_id),
            )
        candidates = list(self.retained.values()) + [incoming]
        victim = min(
            candidates,
            key=lambda item: (item.density, -item.next_use_ms, item.workflow_id),
        )
        return None if victim is incoming else victim

    def _expire_states(self) -> None:
        for workflow_id, state in list(self.retained.items()):
            if self.now_ms < state.expires_ms:
                continue
            del self.retained[workflow_id]
            self.lease_expirations += 1
            if state.tier == "kv" and self.policy in {
                LeasePolicy.JOINT_LEASE,
                LeasePolicy.ORACLE,
            }:
                self._admit(self._demote_state(state))

    def _retained_mb(self) -> int:
        return sum(item.size_mb for item in self.retained.values())

    def _metrics(self) -> LeaseMetrics:
        jcts = [
            runtime.completion_ms - runtime.spec.start_ms
            for runtime in self.workflows.values()
            if runtime.completion_ms is not None
        ]
        return LeaseMetrics(
            experiment=self.spec.name,
            policy=self.policy.value,
            workflow_count=len(self.workflows),
            retention_capacity_mb=self.spec.retention_capacity_mb,
            fixed_kv_ttl_ms=self.spec.fixed_kv_ttl_ms,
            encoder_ttl_ms=self.spec.encoder_ttl_ms,
            average_jct_ms=mean(jcts) if jcts else 0.0,
            p95_jct_ms=_p95(jcts),
            makespan_ms=max(
                runtime.completion_ms or 0 for runtime in self.workflows.values()
            ),
            total_recompute_ms=self.total_recompute_ms,
            avoided_recompute_ms=self.avoided_recompute_ms,
            kv_hits=self.kv_hits,
            encoder_hits=self.encoder_hits,
            cache_misses=self.cache_misses,
            lease_expirations=self.lease_expirations,
            demotions=self.demotions,
            forced_evictions=self.forced_evictions,
            peak_retained_mb=self.peak_retained_mb,
            retained_memory_time_mb_ms=self.retained_memory_time_mb_ms,
        )
