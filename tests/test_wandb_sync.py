"""Regression checks for Avg@1 axes after distributed offline synchronization."""

import ast
from copy import deepcopy
import json
from pathlib import Path
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "configs/environment/sync_wandb.py"
namespace = {"deepcopy": deepcopy, "json": json, "EVAL_STEP_METRIC": "eval_step",
             "EXCLUDED_WANDB_PREFIXES": ("system/", "perf/", "lightning/", "eval/")}
tree = ast.parse(SOURCE.read_text())
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in {"eval_metric_axis_definitions", "eval_metric_names",
                               "hide_excluded_metric_definitions"}]
exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), namespace)
repair = namespace["eval_metric_axis_definitions"]
names = namespace["eval_metric_names"]
hide = namespace["hide_excluded_metric_definitions"]
FIELDS = {"name": "1", "glob_name": "2", "step_metric": "4", "step_metric_index": "5"}


class WandbSyncTests(unittest.TestCase):
    def test_restores_lost_axis_without_changing_training_indices(self):
        previous = [{"1": "train/step"}, {"1": "eval_step"},
                    {"1": "train/loss", "5": 1, "6": [1]}]
        original = deepcopy(previous)
        result = repair(previous, {"eval_metric/math500-avg@1", "train/loss"}, FIELDS)
        self.assertEqual(previous, original)
        self.assertEqual(result[:3], original)
        self.assertEqual(result[3], {"1": "eval_metric/math500-avg@1", "5": 2})
        self.assertEqual(repair(result, {"eval_metric/math500-avg@1"}, FIELDS), result)

    def test_repairs_explicit_and_wildcard_axes_preserving_summary_options(self):
        previous = [{"1": "eval_step"}, {"2": "eval_metric/*", "5": 2},
                    {"1": "eval_metric/math500-avg@1", "4": "_step", "7": [1]},
                    {"1": "eval/math500/pass@1", "4": "existing_axis"}]
        result = repair(previous, {"eval_metric/math500-avg@1"}, FIELDS)
        self.assertEqual(result[1], {"2": "eval_metric/*", "5": 1})
        self.assertEqual(result[2], {"1": "eval_metric/math500-avg@1", "5": 1, "7": [1]})
        self.assertEqual(result[3], previous[3])

    def test_adds_missing_axis_and_does_not_touch_runs_without_avg_metrics(self):
        self.assertEqual(repair([], set(), FIELDS), [])
        self.assertEqual(repair([], {"eval_metric/amc23-avg@1"}, FIELDS),
                         [{"1": "eval_step"}, {"1": "eval_metric/amc23-avg@1", "5": 1}])

    def test_reads_history_key_catalogue_and_keeps_zero_value_metrics(self):
        catalogue = {"keys": {"eval_metric/math500-avg@1": {"count": 3},
                               "eval/math500/pass@1": {"count": 3}}}
        self.assertEqual(names(json.dumps(catalogue)), {"eval_metric/math500-avg@1"})

    def test_hides_existing_diagnostics_without_renumbering_other_axes(self):
        previous = [{"1": "train/step"}, {"1": "train/loss", "5": 1},
                    {"1": "perf/actor_train_time", "5": 1, "6": [1]}]
        fields = dict(FIELDS, options="6", hidden_option=2)
        result = hide(previous, {"eval/math500/pass@1", "train/loss"}, fields)
        self.assertEqual(result[:2], previous[:2])
        self.assertEqual(result[2], {"1": "perf/actor_train_time", "5": 1, "6": [1, 2]})
        self.assertEqual(result[3], {"1": "eval/math500/pass@1", "6": [2]})
        self.assertEqual(hide(result, {"eval/math500/pass@1"}, fields), result)


if __name__ == "__main__":
    unittest.main()
