from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class MaintenanceWork:
    task_id: str
    task_type: str
    estimated_work_ms: int
    required_generation: int
    freshness_deadline_ms: int | None
    resource: str = "cpu"


class MaintenanceAdapter(Protocol):
    """Control-plane contract implemented by each target data system."""

    def discover_work(self) -> list[MaintenanceWork]: ...

    def run_quantum(self, task_id: str, budget_ms: int) -> int:
        """Run at most budget_ms and return consumed milliseconds."""

    def pause(self, task_id: str) -> None: ...

    def commit(self, task_id: str, required_generation: int) -> bool:
        """Atomically publish completed work if the generation is current."""

    def abort(self, task_id: str) -> None: ...

