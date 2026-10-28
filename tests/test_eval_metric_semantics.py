"""Avg@4 averages four attempts; pass@4 accepts any correct attempt."""

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("eval_metric_semantics", ROOT / "slime/eval/eval_metrics.py")
METRICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METRICS)


class EvalMetricSemanticsTests(unittest.TestCase):
    def test_four_attempt_mean_differs_from_any_correct_attempt(self):
        metrics = METRICS.compute_eval_metrics(
            ["amc23"] * 8, ["a"] * 4 + ["b"] * 4,
            [1, 0, 0, 0, 1, 1, 1, 1], k=4)
        self.assertEqual(metrics, {"eval/amc23/avg@4": 0.625, "eval/amc23/pass@4": 1.0})


if __name__ == "__main__":
    unittest.main()
