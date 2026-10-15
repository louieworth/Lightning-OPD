"""Exercise offline filtering against real W&B protobuf and datastore files."""

import ast
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "configs/environment/sync_wandb.py"
namespace = {"json": json, "shutil": shutil, "EVAL_STEP_METRIC": "eval_step",
             "EXCLUDED_WANDB_PREFIXES": ("system/", "perf/", "lightning/", "eval/")}
metrics = ROOT / "slime/utils/wandb_metrics.py"
exec(compile(metrics.read_text(), str(metrics), "exec"), namespace)
tree = ast.parse(source.read_text())
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in {"filtered_run_file", "deduplicate_eval_metrics"}]
exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)


class OfflineWandbTests(unittest.TestCase):
    def test_filters_a_real_log_and_preserves_source_and_indexed_avg_axis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "offline-run-original" / "run-test.wandb"
            original.parent.mkdir()
            writer = DataStore()
            writer.open_for_write(str(original))
            axis = wandb_internal_pb2.Record()
            axis.metric.name = "eval/step"
            writer.write(axis)
            average = wandb_internal_pb2.Record()
            average.metric.name = "eval_metric/math500-avg@1"
            average.metric.step_metric_index = 1
            writer.write(average)
            stats = wandb_internal_pb2.Record()
            stats.stats.SetInParent()
            writer.write(stats)
            history = wandb_internal_pb2.Record()
            for key, value in {"eval/step": 49, "eval_metric/math500-avg@1": 0.0,
                               "eval/math500/pass@1": 0.0, "perf/time": 12,
                               "lightning/block": 3}.items():
                item = history.history.item.add()
                item.key, item.value_json = key, json.dumps(value)
            writer.write(history)
            writer.close()
            original_bytes = original.read_bytes()
            target = namespace["filtered_run_file"](original, root / "filtered")
            self.assertEqual(original.read_bytes(), original_bytes)
            reader = DataStore()
            reader.open_for_scan(str(target))
            records = []
            while (data := reader.scan_data()) is not None:
                record = wandb_internal_pb2.Record()
                record.ParseFromString(data)
                records.append(record)
            reader.close()
            self.assertFalse(any(record.WhichOneof("record_type") == "stats" for record in records))
            self.assertEqual(records[0].metric.name, "eval_step")
            self.assertTrue(records[0].metric.options.hidden)
            self.assertEqual(records[1].metric.step_metric, "eval_step")
            row = {item.key: json.loads(item.value_json) for item in records[-1].history.item}
            self.assertEqual(row, {"eval_step": 50, "eval_metric/math500-avg@1": 0.0})
            replay = namespace["filtered_run_file"](
                original, root / "replayed", {(50, "eval_metric/math500-avg@1"): 0.0})
            reader = DataStore()
            reader.open_for_scan(str(replay))
            histories = 0
            while (data := reader.scan_data()) is not None:
                record = wandb_internal_pb2.Record()
                record.ParseFromString(data)
                histories += record.WhichOneof("record_type") == "history"
            reader.close()
            self.assertEqual(histories, 0)

    def test_repeated_sync_removes_existing_eval_and_preserves_other_metrics(self):
        seen = {(25, "eval_metric/math500-avg@1"): 0.0}
        row = {"eval_step": 25, "eval_metric/math500-avg@1": 0.0,
               "train/step": 24, "train/loss": 0.5}
        retained = namespace["deduplicate_eval_metrics"](row, seen)
        self.assertEqual(retained, {"train/step": 24, "train/loss": 0.5})
        with self.assertRaisesRegex(ValueError, "Conflicting evaluation"):
            namespace["deduplicate_eval_metrics"](dict(row, **{"eval_metric/math500-avg@1": 0.1}), seen)


if __name__ == "__main__":
    unittest.main()
