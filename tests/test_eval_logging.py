"""CPU checks for evaluation metric names and their original evaluation steps."""

import ast
import logging
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]


def load_function(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


class EvalLoggingTests(unittest.TestCase):
    def setUp(self):
        self.tracking = Mock()
        self.log_eval = load_function(
            "slime/ray/rollout.py",
            "_log_eval_rollout_data",
            {
                "__builtins__": __builtins__,
                "Any": object,
                "logger": logging.getLogger(__name__),
                "tracking_utils": self.tracking,
                "compute_rollout_step": lambda args, rollout_id: rollout_id * 2,
                "compute_metrics_from_samples": lambda args, samples: {"response_len/mean": 12.0},
                "dict_add_prefix": lambda values, prefix: {prefix + key: value for key, value in values.items()},
                "compute_pass_rate": lambda **kwargs: {"pass@2": 0.75},
            },
        )
        self.args = SimpleNamespace(
            custom_eval_rollout_log_function_path=None,
            log_passrate=True,
            n_samples_per_eval_prompt=2,
        )

    def test_exact_avg_moves_without_mutating_evaluator_metrics(self):
        # The per-question average can differ from a mean of all samples.
        metrics = {"eval/math500/avg@1": 0.6, "eval/math500/pass@1": 0.6,
                   "eval/math500/num_questions": 2, "eval/math500": 0.5}
        original = dict(metrics)
        data = {"math500": {"rewards": [1.0, 0.0, 0.0], "samples": [object()], "truncated": [0, 1, 0]}}
        result = self.log_eval(24, self.args, data, metrics)
        self.assertEqual(metrics, original)
        self.assertEqual(result["eval_metric/math500-avg@1"], 0.6)
        self.assertNotIn("eval/math500/avg@1", result)
        self.assertNotIn("eval/math500", result)
        self.assertEqual(result["eval/math500/pass@1"], 0.6)
        self.assertEqual(result["eval/math500/num_questions"], 2)
        self.assertEqual(result["eval/math500/response_len/mean"], 12.0)
        self.assertEqual(result["eval/math500-truncated_ratio"], 1 / 3)
        self.assertEqual(result["eval/math500-pass@2"], 0.75)
        self.assertEqual(result["eval/step"], 50)
        self.assertEqual(result["eval_step"], 50)
        self.tracking.log.assert_called_once_with(self.args, result, step_key="eval/step")

    def test_fallback_and_zero_avg_each_produce_one_metric_per_dataset(self):
        data = {"amc23": {"rewards": [1.0, 0.0]}, "hmmt26": {"rewards": [1.0]}}
        result = self.log_eval(0, self.args, data, {"eval/hmmt26/avg@1": 0.0})
        averages = {key: value for key, value in result.items() if key.startswith("eval_metric/")}
        self.assertEqual(averages, {"eval_metric/amc23-avg@1": 0.5, "eval_metric/hmmt26-avg@1": 0.0})
        self.assertFalse(any(key.startswith("eval/") and key.endswith("/avg@1") for key in result))
        self.assertNotIn("eval/amc23", result)
        self.assertNotIn("eval/hmmt26", result)

    def test_wandb_avg_group_uses_existing_eval_axis(self):
        wandb = Mock()
        initialize = load_function("slime/utils/wandb_utils.py", "_init_wandb_common",
                                   {"wandb": wandb, "EVAL_STEP_METRIC": "eval_step"})
        initialize()
        wandb.define_metric.assert_any_call("eval_step", hidden=True)
        wandb.define_metric.assert_any_call("eval_metric/*", step_metric="eval_step", step_sync=False)

    def test_initial_periodic_and_final_steps_use_completed_updates(self):
        self.args.log_passrate = False
        data = {"math500": {"rewards": [0.0]}}
        for index, completed in [(0, 0), (24, 25), (49, 50), (149, 150), (0, 1)]:
            result = self.log_eval(index, self.args, data, completed_steps=completed)
            self.assertEqual(result["eval_step"], completed * 2)


if __name__ == "__main__":
    unittest.main()
