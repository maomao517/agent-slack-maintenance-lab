import unittest

from slackmaint.generator import generate_experiment
from slackmaint.models import ExperimentSpec, PolicyKind
from slackmaint.simulator import Simulator


class SimulatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = ExperimentSpec.from_dict(
            {
                "name": "one-workflow",
                "tick_ms": 10,
                "workflows": [
                    {
                        "workflow_id": "w0",
                        "tenant_id": "t0",
                        "start_ms": 0,
                        "model_segments_ms": [100, 100],
                        "tool_waits_ms": [500],
                    }
                ],
                "maintenance_tasks": [
                    {
                        "task_id": "m0",
                        "owner_workflow_id": "w0",
                        "work_ms": 200,
                        "trigger_after_segment": 0,
                        "required_before_segment": 1,
                    }
                ],
            }
        )

    def test_dual_aware_hides_maintenance_in_tool_wait(self) -> None:
        sync = Simulator(self.spec, PolicyKind.SYNC).run()
        dual = Simulator(self.spec, PolicyKind.DUAL_AWARE).run()

        self.assertEqual(sync.average_jct_ms, 900)
        self.assertEqual(dual.average_jct_ms, 700)
        self.assertEqual(dual.freshness_violations, 0)
        self.assertEqual(dual.maintenance_overlap_ratio, 1.0)

    def test_none_records_freshness_violation(self) -> None:
        result = Simulator(self.spec, PolicyKind.NONE).run()

        self.assertEqual(result.average_jct_ms, 700)
        self.assertEqual(result.freshness_violations, 1)
        self.assertEqual(result.maintenance_backlog_ms, 200)

    def test_freshness_enforcing_policies_do_not_read_stale_state(self) -> None:
        policies = [policy for policy in PolicyKind if policy is not PolicyKind.NONE]

        for policy in policies:
            with self.subTest(policy=policy.value):
                result = Simulator(self.spec, policy).run()
                self.assertEqual(result.freshness_violations, 0)
                self.assertEqual(result.maintenance_backlog_ms, 0)

    def test_generator_is_deterministic(self) -> None:
        first = generate_experiment(seed=7, workflows=3, turns=3)
        second = generate_experiment(seed=7, workflows=3, turns=3)

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
