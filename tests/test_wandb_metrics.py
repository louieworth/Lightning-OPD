"""Check that pruning W&B metrics preserves real evaluation steps and zero scores."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from test_eval_logging import load_function

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("wandb_metrics", ROOT / "slime/utils/wandb_metrics.py")
metrics_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics_module)
filter_metrics = metrics_module.filter_wandb_metrics


class WandbMetricTests(unittest.TestCase):
    def test_prunes_only_requested_groups_and_preserves_zero_avg(self):
        source = {"train/loss": 0.5, "train/step": 24, "rollout/rewards": 0.2,
                  "eval/step": 24, "eval_metric/math500-avg@1": 0.0,
                  "eval/math500/pass@1": 0.0, "perf/time": 5, "lightning/block": 2,
                  "system/cpu": 20}
        retained = filter_metrics(source)
        self.assertEqual(retained, {"train/loss": 0.5, "train/step": 24, "rollout/rewards": 0.2,
                                    "eval_metric/math500-avg@1": 0.0,
                                    "eval_metric/math500-pass@1": 0.0, "eval_step": 25})
        self.assertIn("eval/step", source)

    def test_logging_filters_wandb_and_preserves_tensorboard_diagnostics(self):
        wandb, adapter = Mock(), Mock()
        function = load_function("slime/utils/tracking_utils.py", "log",
                                 {"wandb": wandb, "filter_wandb_metrics": filter_metrics,
                                  "_TensorboardAdapter": lambda args: adapter})
        args = SimpleNamespace(use_wandb=True, use_tensorboard=True)
        function(args, {"eval/step": 49, "eval_metric/amc23-avg@1": 0.0,
                        "eval/amc23/pass@1": 0.0}, "eval/step")
        wandb.log.assert_called_once_with({"eval_step": 50, "eval_metric/amc23-avg@1": 0.0,
                                          "eval_metric/amc23-pass@1": 0.0})
        adapter.log.assert_called_once_with(
            data={"eval_metric/amc23-avg@1": 0.0, "eval/amc23/pass@1": 0.0}, step=49)

    def test_dropped_performance_log_does_not_add_a_wandb_step(self):
        wandb = Mock()
        function = load_function("slime/utils/tracking_utils.py", "log",
                                 {"wandb": wandb, "filter_wandb_metrics": filter_metrics})
        function(SimpleNamespace(use_wandb=True, use_tensorboard=False),
                 {"rollout/step": 10, "perf/actor_train_time": 12}, "rollout/step")
        wandb.log.assert_not_called()

    def test_explicit_completed_axis_is_not_converted_twice(self):
        self.assertEqual(filter_metrics({"eval/step": 25, "eval_step": 25,
                                         "eval_metric/math500-avg@1": 0.0}),
                         {"eval_step": 25, "eval_metric/math500-avg@1": 0.0})

    def test_keeps_each_dataset_mean_completion_length(self):
        source = {"eval_step": 20, "eval/amc23/response_len/mean": 123.5,
                  "eval/aime25/response_len/mean": 0.0,
                  "eval/amc23/response_len/median": 99.0,
                  "eval/amc23/pass@4": 0.5}
        self.assertEqual(filter_metrics(source), {
            "eval_step": 20, "eval_metric/amc23-response-length": 123.5,
            "eval_metric/aime25-response-length": 0.0, "eval_metric/amc23-pass@4": 0.5})
        self.assertIn("eval/amc23/response_len/mean", source)

    def test_length_and_accuracy_share_the_completed_eval_axis(self):
        records = [{"eval/step": 19, "eval/amc23/response_len/mean": 123.5,
                    "eval_metric/amc23-avg@1": 0.25},
                   {"eval_step": 20, "eval_metric/response-length/amc23": 123.5}]
        rows = metrics_module.canonical_wandb_history(records)
        self.assertEqual(rows, [{"_step": 20, "eval_step": 20,
                                "eval_metric/amc23-avg@1": 0.25,
                                "eval_metric/amc23-response-length": 123.5}])

    def test_legacy_score_labels_follow_known_dataset_sample_counts(self):
        source = {"eval_step": 20, "eval_metric/amc23-avg@1": 0.0,
                  "eval_metric/aime25-avg@1": 0.5,
                  "eval_metric/response-length/amc23": 123.5}
        self.assertEqual(filter_metrics(source, {"amc23": 4, "aime25": 2}), {
            "eval_step": 20, "eval_metric/amc23-avg@4": 0.0,
            "eval_metric/aime25-avg@2": 0.5,
            "eval_metric/amc23-response-length": 123.5})
        self.assertIn("eval_metric/amc23-avg@1", source)

    def test_standard_opd_eval_keeps_completed_step_and_all_three_metrics(self):
        source = {"eval/step": 113, "eval_step": 113,
                  "eval/amc23/avg@4": 0.25, "eval/amc23/pass@4": 0.525,
                  "eval/amc23/response-length": 995.5375}
        self.assertEqual(filter_metrics(source, {"amc23": 4}), {
            "eval_step": 113, "eval_metric/amc23-avg@4": 0.25,
            "eval_metric/amc23-pass@4": 0.525,
            "eval_metric/amc23-response-length": 995.5375})

    def test_pass_at_one_is_not_relabelled_as_pass_at_four(self):
        result = filter_metrics({"eval_step": 20, "eval/amc23/pass@1": 0.25,
                                 "eval_metric/amc23-avg@1": 0.25}, {"amc23": 4})
        self.assertEqual(result["eval_metric/amc23-pass@1"], 0.25)
        self.assertNotIn("eval_metric/amc23-pass@4", result)
        self.assertEqual(result["eval_metric/amc23-avg@4"], 0.25)

    def test_canonical_history_has_one_eval_point_at_each_completed_step(self):
        records = [{"train/step": 24, "train/loss": 0.5},
                   {"rollout/step": 24, "rollout/rewards": 0.2},
                   {"eval/step": 0, "eval_metric/math500-avg@1": 0.0},
                   {"eval/step": 24, "eval_metric/math500-avg@1": 0.3},
                   {"eval_step": 25, "eval_metric/math500-avg@1": 0.3},
                   {"eval/step": 149, "eval_metric/math500-avg@1": 0.4}]
        rows = metrics_module.canonical_wandb_history(records)
        self.assertEqual([row["_step"] for row in rows], [0, 25, 150])
        self.assertEqual([row["eval_step"] for row in rows], [0, 25, 150])
        self.assertEqual(rows[1]["train/step"], 24)
        self.assertEqual(rows[1]["train/loss"], 0.5)
        self.assertEqual(rows[1]["rollout/rewards"], 0.2)
        with self.assertRaisesRegex(ValueError, "Conflicting metric"):
            metrics_module.canonical_wandb_history(records + [
                {"eval_step": 25, "eval_metric/math500-avg@1": 0.4}])

    def test_canonical_history_preserves_distinct_rollout_values(self):
        records = [{"rollout/step": 0, "rollout/response_len/mean": 100},
                   {"rollout/step": 0, "rollout/response_len/mean": 200},
                   {"eval_step": 25, "eval_metric/math500-avg@1": 0.5}]
        rows = metrics_module.canonical_wandb_history(records, preserve_conflicts=True)
        self.assertEqual({row["rollout/response_len/mean"] for row in rows
                          if "rollout/response_len/mean" in row}, {100, 200})
        points = [row for row in rows if "eval_metric/math500-avg@1" in row]
        self.assertEqual([(row["_step"], row["eval_step"]) for row in points], [(25, 25)])

    def test_mixed_history_record_keeps_both_axes(self):
        rows = metrics_module.canonical_wandb_history([
            {"train/step": 24, "train/loss": 0.5, "eval_step": 25,
             "eval_metric/math500-avg@1": 0.4}])
        self.assertEqual(rows[0]["train/step"], 24)
        self.assertEqual(rows[0]["eval_step"], 25)


if __name__ == "__main__":
    unittest.main()
