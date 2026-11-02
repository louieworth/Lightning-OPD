"""GPU utilization monitor: parse DCGM samples and flag under-used GPUs.

A GPU counts as busy by max(SM active, DRAM active): decoding is memory-bound, so a well-used
GPU during generation can show modest SM activity but high DRAM activity.
"""

import importlib.util
from pathlib import Path
import sys
import unittest

# Load by path: in the venv, "tools" resolves to Megatron-LM's tools package.
SPEC = importlib.util.spec_from_file_location("gpu_monitor", Path(__file__).resolve().parents[1] / "tools/gpu_monitor.py")
monitor = importlib.util.module_from_spec(SPEC)
sys.modules["gpu_monitor"] = monitor  # dataclasses resolve their module while the file executes
SPEC.loader.exec_module(monitor)

DMON = """#Entity   SMACT        SMOCC        TENSO        DRAMA        GPUTL        FBUSD
ID
GPU 1     0.912        0.400        0.300        0.200        100          60000
GPU 0     N/A          N/A          N/A          N/A          0            900
"""


def window(busy_by_gpu, seconds=300, start=0.0):
    """One sample per 10 s for each (node, gpu) -> busy value (as both SM and DRAM activity)."""
    return [monitor.Sample(start + t, node, gpu, dict(sm_active=busy, dram_active=busy))
            for t in range(0, seconds, 10) for (node, gpu), busy in busy_by_gpu.items()]


class ParseTests(unittest.TestCase):
    def test_dmon_rows_become_samples_and_missing_values_are_none(self):
        samples = monitor.parse_dmon(DMON, time=5.0, node="head")
        self.assertEqual([(s.gpu, s.metrics["sm_active"], s.metrics["gpu_util"]) for s in samples],
                         [(1, 0.912, 100.0), (0, None, 0.0)])
        self.assertEqual(samples[0].metrics["fb_used_mib"], 60000.0)
        self.assertTrue(all(s.time == 5.0 and s.node == "head" for s in samples))


class AssessTests(unittest.TestCase):
    def test_busy_cluster_raises_no_alert(self):
        report = monitor.assess(window({("h", g): 0.9 for g in range(8)}), now=300, window_s=300)
        self.assertEqual(report.alerts, [])
        self.assertAlmostEqual(report.mean_busy, 0.9)

    def test_low_mean_utilization_is_flagged(self):
        report = monitor.assess(window({("h", g): 0.3 for g in range(8)}), now=300, window_s=300)
        self.assertEqual([alert.kind for alert in report.alerts], ["LOW_UTILIZATION"])

    def test_idle_gpus_beside_busy_ones_are_named(self):
        busy = {("h", g): 0.9 for g in range(6)} | {("w", 0): 0.0, ("w", 1): 0.01}
        report = monitor.assess(window(busy), now=300, window_s=300)
        idle = [alert for alert in report.alerts if alert.kind == "IDLE_GPUS"]
        self.assertEqual(len(idle), 1)
        self.assertEqual(idle[0].gpus, [("w", 0), ("w", 1)])

    def test_whole_cluster_stall_is_flagged_once(self):
        report = monitor.assess(window({("h", g): 0.0 for g in range(8)}), now=300, window_s=300)
        self.assertEqual([alert.kind for alert in report.alerts], ["STALLED"])

    def test_no_alert_until_a_full_window_is_observed(self):
        """Right after (re)starting the monitor, 30 idle seconds are not a 5-minute stall."""
        report = monitor.assess(window({("h", g): 0.0 for g in range(8)}, seconds=30), now=30, window_s=300)
        self.assertEqual(report.alerts, [])

    def test_only_the_recent_window_counts(self):
        old_idle = window({("h", 0): 0.0}, seconds=600, start=0)
        recent_busy = window({("h", 0): 0.95}, seconds=300, start=600)
        report = monitor.assess(old_idle + recent_busy, now=900, window_s=300)
        self.assertEqual(report.alerts, [])

    def test_activity_on_a_gpu_without_our_memory_is_not_ours(self):
        """A host process outside the container (~0.8 GB on every GPU) runs kernels on GPUs we leave idle."""
        foreign = [monitor.Sample(t, "h", 0, dict(sm_active=0.9, dram_active=0.2, fb_used_mib=821.0))
                   for t in range(0, 300, 10)]
        ours = [monitor.Sample(t, "h", 1, dict(sm_active=0.9, dram_active=0.2, fb_used_mib=40000.0))
                for t in range(0, 300, 10)]
        report = monitor.assess(foreign + ours, now=300, window_s=300)
        self.assertEqual(report.per_gpu, {("h", 0): 0.0, ("h", 1): 0.9})
        self.assertEqual([alert.gpus for alert in report.alerts if alert.kind == "IDLE_GPUS"], [[("h", 0)]])

    def test_missing_profiling_values_fall_back_to_gpu_utilization(self):
        samples = [monitor.Sample(t, "h", 0, dict(sm_active=None, dram_active=None, gpu_util=95.0))
                   for t in range(0, 300, 10)]
        report = monitor.assess(samples, now=300, window_s=300)
        self.assertAlmostEqual(report.mean_busy, 0.95)


if __name__ == "__main__":
    unittest.main()
