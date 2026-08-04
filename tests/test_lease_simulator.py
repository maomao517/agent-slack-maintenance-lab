import unittest

from slackmaint.lease_simulator import (
    LeaseExperimentSpec,
    LeasePolicy,
    LeaseSimulator,
)


class LeaseSimulatorTest(unittest.TestCase):
    def _spec(self, capacity_mb: int = 5000) -> LeaseExperimentSpec:
        return LeaseExperimentSpec.from_dict(
            {
                "name": "lease-test",
                "tick_ms": 10,
                "retention_capacity_mb": capacity_mb,
                "fixed_kv_ttl_ms": 1000,
                "encoder_ttl_ms": 5000,
                "max_time_ms": 20000,
                "workflows": [
                    {
                        "workflow_id": "w0",
                        "start_ms": 0,
                        "model_segments_ms": [100, 100],
                        "tool_waits_ms": [500],
                        "expected_tool_waits_ms": [500],
                        "kv_size_mb": 4000,
                        "encoder_size_mb": 400,
                        "prefill_ms": 200,
                        "encoder_ms": 100,
                    }
                ],
            }
        )

    def test_fixed_kv_lease_avoids_second_turn_recompute(self) -> None:
        spec = self._spec()
        no_cache = LeaseSimulator(spec, LeasePolicy.NO_CACHE).run()
        fixed = LeaseSimulator(spec, LeasePolicy.FIXED_KV_LEASE).run()

        self.assertEqual(no_cache.total_recompute_ms, 600)
        self.assertEqual(fixed.total_recompute_ms, 300)
        self.assertEqual(fixed.kv_hits, 1)
        self.assertLess(fixed.average_jct_ms, no_cache.average_jct_ms)

    def test_joint_lease_demotes_when_kv_does_not_fit(self) -> None:
        spec = self._spec(capacity_mb=1000)
        joint = LeaseSimulator(spec, LeasePolicy.JOINT_LEASE).run()

        self.assertEqual(joint.demotions, 1)
        self.assertEqual(joint.encoder_hits, 1)
        self.assertEqual(joint.kv_hits, 0)
        self.assertEqual(joint.peak_retained_mb, 400)

    def test_long_pause_expires_fixed_kv(self) -> None:
        data = {
            "name": "expiry-test",
            "tick_ms": 10,
            "retention_capacity_mb": 5000,
            "fixed_kv_ttl_ms": 1000,
            "encoder_ttl_ms": 5000,
            "max_time_ms": 20000,
            "workflows": [
                {
                    "workflow_id": "w0",
                    "start_ms": 0,
                    "model_segments_ms": [100, 100],
                    "tool_waits_ms": [2000],
                    "expected_tool_waits_ms": [500],
                    "kv_size_mb": 4000,
                    "encoder_size_mb": 400,
                    "prefill_ms": 200,
                    "encoder_ms": 100,
                }
            ],
        }
        fixed = LeaseSimulator(
            LeaseExperimentSpec.from_dict(data), LeasePolicy.FIXED_KV_LEASE
        ).run()

        self.assertEqual(fixed.lease_expirations, 1)
        self.assertEqual(fixed.cache_misses, 1)

    def test_metrics_record_effective_experiment_parameters(self) -> None:
        metrics = LeaseSimulator(
            self._spec(capacity_mb=1234), LeasePolicy.JOINT_LEASE
        ).run()

        self.assertEqual(metrics.retention_capacity_mb, 1234)
        self.assertEqual(metrics.fixed_kv_ttl_ms, 1000)
        self.assertEqual(metrics.encoder_ttl_ms, 5000)


if __name__ == "__main__":
    unittest.main()
