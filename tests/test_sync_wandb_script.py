"""Check discovery, real-step alignment and retry-safe uploads without a network."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sync_by_sbatch", ROOT / "scripts/sync_wandb.py")
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)
FIELDS = {"name": "1", "options": "6", "step_metric": "4", "step_metric_index": "5", "hidden_option": 2}


def evaluation(step=20):
    return {"eval_step": step, "eval_metric/math-avg@4": 0.25,
            "eval_metric/math-pass@4": 0.5, "eval_metric/math-response-length": 123.5}


class FakeRun:
    def __init__(self):
        self.state = "finished"
        self.config, self.rawconfig, self.summary = {}, {}, {}
        self.system_metrics, self.history = {}, []
        self.url = "https://wandb.ai/entity/project/test"

    def scan_history(self, **kwargs):
        return iter(deepcopy(self.history))


class FakeApi:
    def __init__(self):
        self.remote = None

    def run(self, path):
        if self.remote is None:
            raise FakeWandb.errors.CommError("Could not find run")
        return self.remote

    def flush(self):
        pass


class FakeWandb:
    errors = SimpleNamespace(CommError=type("CommError", (Exception,), {}))
    Settings = SimpleNamespace

    def __init__(self, api):
        self.api, self.inits = api, []

    def init(self, **kwargs):
        self.inits.append(kwargs)
        if self.api.remote is None:
            self.api.remote = FakeRun()
        return FakeTarget(self.api.remote)


class FakeTarget:
    def __init__(self, run):
        self.run = run
        self.config = SimpleNamespace(update=lambda values, **kwargs: run.config.update(values))
        self.summary = run.summary

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def define_metric(self, name, **kwargs):
        # Wildcards are not needed to inspect exact evaluation definitions.
        metric = {"1": name, "6": [2] if kwargs.get("hidden", False) else []}
        if "step_metric" in kwargs:
            metric["4"] = kwargs["step_metric"]
        self.run.rawconfig.setdefault("_wandb", {}).setdefault("m", []).append(metric)

    def log(self, row):
        self.run.history.append(dict(row, _step=len(self.run.history)))
        self.summary.update(row)


class DiscoveryTests(unittest.TestCase):
    def test_resolves_filename_job_name_and_experiment_without_executing_shell(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            config = repo / "configs/config/recipe.yaml"
            config.parent.mkdir(parents=True)
            config.write_text("experiment_name: actual_name\n")
            # Agent YAML can contain a list, rather than an experiment mapping.
            (repo / "configs/agent.yaml").write_text("- tool: example\n")
            sbatch = repo / "configs/sbatch/wrapper.sbatch"
            sbatch.parent.mkdir()
            sbatch.write_text("#SBATCH --job-name=job-name\nTASK_CONFIG=configs/config/recipe.yaml\nexit 99\n")
            for name in ("wrapper", "wrapper.sbatch", str(sbatch), "job-name", "actual_name"):
                self.assertEqual(sync.find_recipe(name, [repo]), (repo, config))

    def test_shared_job_name_reports_ambiguity(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "configs/sbatch").mkdir(parents=True)
            for name in ("one", "two"):
                (repo / f"configs/{name}.yaml").write_text(f"experiment_name: {name}\n")
                (repo / f"configs/sbatch/{name}.sbatch").write_text(f"#SBATCH --job-name=same\npython configs/{name}.yaml\n")
            with self.assertRaisesRegex(ValueError, "Cannot resolve one recipe"):
                sync.find_recipe("same", [repo])

    def test_standard_opd_uses_marker_and_saved_experiment_name(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            result = repo / "results/timestamp-models"
            result.mkdir(parents=True)
            (result / "config.yaml").write_text("experiment_name: baseline\nval_n: 4\n")
            (result / "eval_metrics").mkdir()
            (result / "eval_metrics/20.json").write_text("{}")
            recipe = repo / "recipe.yaml"
            recipe.write_text(yaml.safe_dump({"method": "opd", "experiment_name": "baseline", "output_dir": str(repo / "results")}))
            marker = repo / "wandb/.run_roots/slurm-123.txt"
            marker.parent.mkdir(parents=True)
            marker.write_text(str(result))
            preflight = repo / "results/preflight-only"
            preflight.mkdir()
            (preflight / "config.yaml").write_text("experiment_name: baseline\n")
            self.assertEqual(sync.find_run_dirs(repo, recipe, job_id="123"), [result])
            self.assertEqual(sync.find_run_dirs(repo, recipe), [result])
            with self.assertRaisesRegex(ValueError, "No saved results"):
                sync.find_run_dirs(repo, recipe, job_id="456")


class AlignmentTests(unittest.TestCase):
    def test_lightning_legacy_indices_and_multistream_deduplication(self):
        records = [{"train/step": 119, "train/loss": 0.4}, {"rollout/step": 119, "rollout/reward": 0.5},
                   {"eval/step": 119, "eval_metric/math-avg@1": 0.25,
                    "eval/math/pass@4": 0.5, "eval/math/response_len/mean": 123.5}, evaluation(120)]
        rows = sync.merge_rows(records * 2, "lightning_opd", {"math": 4})
        self.assertEqual(len(rows), 3)
        self.assertIn(evaluation(120), rows)

    def test_native_opd_completed_step_is_not_incremented(self):
        raw = {"eval/step": 113, "eval/math/avg@4": 0.25,
               "eval/math/pass@4": 0.5, "eval/math/response-length": 123.5,
               "train/step": 112, "train/loss": 0.4, "actor/loss": 0.4}
        rows = sync.merge_rows([raw], "opd", {"math": 4})
        self.assertIn(evaluation(113), rows)
        self.assertIn({"train/step": 112, "train/loss": 0.4, "actor/loss": 0.4}, rows)

    def test_source_conflict_and_unbound_axes_fail(self):
        with self.assertRaisesRegex(ValueError, "Conflicting source"):
            sync.merge_rows([evaluation(), dict(evaluation(), **{"eval_metric/math-pass@4": 0.6})], "opd", {"math": 4})
        with self.assertRaisesRegex(ValueError, "lacks its real axis"):
            sync.merge_rows([{"train/loss": 0.4}], "opd", {"math": 4})

    def test_standard_console_restores_training_tail_and_excludes_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "slurm.log"
            path.write_text("\x1b[36m(Worker pid=1)\x1b[0m step:113 - actor/loss:0.4 - train/step:112 - train/loss:0.4 - perf/time:9\n")
            rows = sync.merge_rows(list(sync.console_rows(path, "opd")), "opd", {"math": 4})
            self.assertEqual(rows, [{"train/step": 112, "actor/loss": 0.4, "train/loss": 0.4}])

    def test_cloud_conflict_and_duplicate_evaluations_fail(self):
        source = {"method": "opd", "rows": [evaluation()]}
        self.assertEqual(sync.missing_rows(source, [dict(evaluation(), _step=1)]), [])
        with self.assertRaisesRegex(ValueError, "Cloud/source conflict"):
            sync.missing_rows(source, [dict(evaluation(), **{"eval_metric/math-pass@4": 0.7})])
        with self.assertRaisesRegex(ValueError, "Duplicate cloud evaluation"):
            sync.missing_rows(source, [evaluation(), evaluation()])
        with self.assertRaisesRegex(ValueError, "extends beyond"):
            sync.missing_rows(source, [evaluation(40)])

    def test_file_changes_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.json"
            path.write_text("{}")
            source = {"files": {str(path): sync.signature(path)}}
            sync.check_stable(source)
            path.write_text("{\"changed\": true}")
            with self.assertRaisesRegex(ValueError, "Source changed"):
                sync.check_stable(source)


class SourceTests(unittest.TestCase):
    def test_repo_wandb_logs_are_discovered_beside_external_results(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            folder = repo / "external-results/run"
            metrics = folder / "eval_metrics/20.json"
            metrics.parent.mkdir(parents=True)
            cfg = {"method": "lightning_opd", "experiment_name": "run", "model_path": "Qwen/Student",
                   "teacher_model_path": "Qwen/Teacher", "val_n": 4, "val_files": ["/data/math.parquet"]}
            (folder / "config.yaml").write_text(yaml.safe_dump(cfg))
            metrics.write_text(json.dumps(dict(evaluation(), _eval_completed_steps=20)))
            stream = repo / "wandb/run/wandb/offline-run-abc/run-abc.wandb"
            stream.parent.mkdir(parents=True)
            stream.touch()
            original = {"run_id": "abc", "job_ids": set(), "rows": [], "signature": sync.signature(stream)}
            with patch.object(sync, "read_stream", return_value=original):
                source = sync.build_source(repo, folder)
            self.assertEqual(source["rows"], [evaluation()])
            self.assertIn(str(stream), source["files"])
            self.assertFalse((folder / "wandb").exists())

    def test_saved_recipe_json_and_console_combine_without_inventing_pass_or_length(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            folder = repo / "results/run"
            folder.mkdir(parents=True)
            cfg = {"method": "lightning_opd", "experiment_name": "run", "model_path": "Qwen/Student-Base",
                   "teacher_model_path": "Qwen/Teacher", "val_n": 4, "val_files": ["/data/math.parquet"],
                   "wandb_api_key": "must-never-upload"}
            (folder / "config.yaml").write_text(yaml.safe_dump(cfg))
            stream = folder / "wandb/offline-run-abc/run-abc.wandb"
            stream.parent.mkdir(parents=True)
            stream.touch()
            metrics = folder / "eval_metrics/119.json"
            metrics.parent.mkdir()
            metrics.write_text(json.dumps({"_eval_completed_steps": 120, "eval/math/avg@4": 0.25, "eval/math/pass@4": 0.5}))
            log = repo / "tmp/slurm-old-name-123.log"
            log.parent.mkdir()
            log.write_text("rollout.py:123 - eval 119: {'eval/step': 119, 'eval/math/response_len/mean': 123.5}\n")
            original = {"run_id": "abc", "job_ids": {"123"}, "rows": [], "signature": sync.signature(stream)}
            with patch.object(sync, "read_stream", return_value=original):
                source = sync.build_source(repo, folder)
                self.assertEqual(source["rows"], [evaluation(120)])
                self.assertEqual(source["metadata"]["val_n"], 4)
                self.assertEqual(source["metadata"]["student_model"], "Qwen/Student-Base")
                self.assertNotIn("wandb_api_key", source["metadata"])
                log.write_text("")
                with self.assertRaisesRegex(ValueError, "Incomplete evaluation"):
                    sync.build_source(repo, folder)

    def test_reads_real_protobuf_history_and_job_identity(self):
        from wandb.proto import wandb_internal_pb2 as pb
        from wandb.sdk.internal.datastore import DataStore
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run-abc.wandb"
            writer = DataStore()
            writer.open_for_write(str(path))
            record = pb.Record()
            item = record.run.config.update.add()
            item.key, item.value_json = "env_vars", '{"SLURM_JOB_ID": "123"}'
            writer.write(record)
            record = pb.Record()
            for key, value in evaluation().items():
                item = record.history.item.add()
                item.key, item.value_json = key, json.dumps(value)
            writer.write(record)
            writer.close()
            before = path.read_bytes()
            source = sync.read_stream(path)
            self.assertEqual(source["job_ids"], {"123"})
            self.assertEqual(source["rows"], [evaluation()])
            self.assertEqual(path.read_bytes(), before)


class UploadTests(unittest.TestCase):
    def source(self):
        return {"name": "run", "method": "opd", "files": {}, "job_ids": [],
                "metadata": {"student_model": "Qwen/Student", "teacher_model": "Qwen/Teacher", "val_n": 4,
                             "model_path": "Qwen/Student", "teacher_model_path": "Qwen/Teacher", "eval_samples_per_dataset": {"math": 4}},
                "rows": [{"train/step": 19, "train/loss": 0.4, "actor/loss": 0.4}, evaluation()]}

    def test_upload_retry_and_verification_preserve_points_and_avoid_extra_init(self):
        api = FakeApi()
        wandb = FakeWandb(api)
        with tempfile.TemporaryDirectory() as directory, patch.object(sync, "metric_fields", return_value=FIELDS):
            audit = Path(directory)
            for verify_only in (False, False, True):
                sync.sync_source(self.source(), api, wandb, "entity", "project", "test", audit, verify_only)
            self.assertEqual(len(wandb.inits), 1)
            self.assertEqual(len(api.remote.history), 2)
            self.assertTrue((audit / "verification.json").is_file())
            self.assertEqual(api.remote.config["student_model"], "Qwen/Student")
            self.assertIn({"1": "actor/loss", "4": "train/step", "6": []}, api.remote.rawconfig["_wandb"]["m"])

    def test_existing_history_and_old_names_are_preserved_but_hidden(self):
        api = FakeApi()
        api.remote = FakeRun()
        api.remote.history = [{"_step": 0, "eval/step": 20, "eval/math/pass@4": 0.5}]
        api.remote.summary = dict(api.remote.history[0])
        original = deepcopy(api.remote.history)
        wandb = FakeWandb(api)
        with tempfile.TemporaryDirectory() as directory, patch.object(sync, "metric_fields", return_value=FIELDS):
            sync.sync_source(self.source(), api, wandb, "entity", "project", "test", Path(directory))
            self.assertEqual(api.remote.history[:1], original)
            self.assertNotIn("eval/math/pass@4", api.remote.summary)
            self.assertIn("eval_metric/math-pass@4", api.remote.summary)
            definitions = api.remote.rawconfig["_wandb"]["m"]
            self.assertIn({"1": "eval/math/pass@4", "6": [2]}, definitions)

    def test_verify_only_and_conflict_do_not_mutate_cloud(self):
        api = FakeApi()
        api.remote = FakeRun()
        wandb = FakeWandb(api)
        with tempfile.TemporaryDirectory() as directory, patch.object(sync, "metric_fields", return_value=FIELDS):
            with self.assertRaisesRegex(ValueError, "does not yet meet"):
                sync.sync_source(self.source(), api, wandb, "entity", "project", "test", Path(directory), True)
            api.remote.history = [{"_step": 0, "train/step": 19, "train/loss": 9}]
            with self.assertRaisesRegex(ValueError, "Cloud/source conflict"):
                sync.sync_source(self.source(), api, wandb, "entity", "project", "test", Path(directory))
            self.assertEqual(wandb.inits, [])


if __name__ == "__main__":
    unittest.main()
