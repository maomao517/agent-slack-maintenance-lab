from __future__ import annotations

import random


def generate_experiment(seed: int, workflows: int, turns: int) -> dict[str, object]:
    if workflows <= 0:
        raise ValueError("workflows must be positive")
    if turns < 2:
        raise ValueError("turns must be at least 2")

    rng = random.Random(seed)
    workflow_specs = []
    task_specs = []

    for workflow_index in range(workflows):
        workflow_id = f"workflow-{workflow_index:03d}"
        model_segments = [rng.randrange(80, 181, 10) for _ in range(turns)]
        tool_waits = [rng.randrange(150, 901, 10) for _ in range(turns - 1)]
        workflow_specs.append(
            {
                "workflow_id": workflow_id,
                "tenant_id": f"tenant-{workflow_index % 4}",
                "start_ms": workflow_index * 20,
                "model_segments_ms": model_segments,
                "tool_waits_ms": tool_waits,
            }
        )

        for segment in range(turns - 1):
            task_specs.append(
                {
                    "task_id": f"task-{workflow_index:03d}-{segment:02d}",
                    "owner_workflow_id": workflow_id,
                    "task_type": rng.choice(
                        ["index_update", "local_rebuild", "compact"]
                    ),
                    "work_ms": rng.randrange(50, 351, 10),
                    "trigger_after_segment": segment,
                    "required_before_segment": segment + 1,
                    "version": segment + 1,
                }
            )

    return {
        "name": f"generated-seed-{seed}",
        "tick_ms": 10,
        "periodic_interval_ms": 200,
        "periodic_budget_ms": 50,
        "workflows": workflow_specs,
        "maintenance_tasks": task_specs,
        "metadata": {"seed": seed, "generator": "slackmaint"},
    }

