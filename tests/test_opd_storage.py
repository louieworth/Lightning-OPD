"""CPU regressions for the quota exhaustion that interrupted jobs 637107-637109."""

import contextlib
import errno
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from slime.utils import storage


ROOT = Path(__file__).resolve().parents[1]


def launcher():
    spec = importlib.util.spec_from_file_location("storage_launcher", ROOT / "configs/opd/qwen3-4b-opd.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StorageTests(unittest.TestCase):
    def test_index_counts_weights_once_and_reserves_optimizer_space(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "metadata": {"total_size": 3 * storage.GIB},
                "weight_map": {"a": "shard.safetensors", "b": "shard.safetensors"},
            }))
            self.assertEqual(storage.model_weight_bytes(root), 3 * storage.GIB)
            self.assertEqual(storage.checkpoint_reserve_bytes(root / "checkpoints", root), 24 * storage.GIB)
            with patch.object(storage.shutil, "disk_usage", return_value=types.SimpleNamespace(free=28 * storage.GIB)):
                with self.assertRaisesRegex(RuntimeError, "need at least 29.0 GiB"):
                    storage.check_checkpoint_space(root / "checkpoints", root)

    def test_actual_checkpoint_size_is_used_when_weights_are_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "latest_checkpointed_iteration.txt").write_text("39")
            checkpoint = root / "iter_0000039"
            checkpoint.mkdir()
            (checkpoint / "weights").write_bytes(b"previous completed checkpoint")
            required = storage.checkpoint_reserve_bytes(root, None)
            self.assertEqual(required, len(b"previous completed checkpoint"))
            with patch.object(storage.shutil, "disk_usage", return_value=types.SimpleNamespace(free=0)):
                with self.assertRaisesRegex(RuntimeError, "project quota"):
                    storage.check_checkpoint_space(root, None)
            self.assertEqual((root / "latest_checkpointed_iteration.txt").read_text(), "39")
            self.assertEqual((checkpoint / "weights").read_bytes(), b"previous completed checkpoint")

    def test_missing_output_uses_existing_parent_without_creating_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "new-experiment/checkpoints"
            with patch.object(storage.shutil, "disk_usage", return_value=types.SimpleNamespace(free=0)) as usage:
                with self.assertRaisesRegex(RuntimeError, "Insufficient output disk space"):
                    storage.require_free_space(target, 5 * storage.GIB)
            usage.assert_called_once_with(Path(directory).resolve())
            self.assertFalse(target.parent.exists())

    def test_preflight_accounts_for_initialization_and_next_checkpoint(self):
        module = launcher()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 3 * storage.GIB}}))
            cfg = dict(output_dir=str(root), experiment_name="test", save_interval=20)
            initial = root / "initial"
            with patch.object(module, "checkpoint_path", return_value=initial), \
                    patch.object(module, "require_free_space", return_value=100 * storage.GIB) as check, \
                    contextlib.redirect_stdout(io.StringIO()):
                module.check_training_storage(cfg, str(model))
                check.assert_called_once_with(root / "test", 32 * storage.GIB)

    def test_full_filesystem_stops_before_starting_cluster_services(self):
        module = launcher()
        cfg = dict(output_dir="/unused", experiment_name="test")
        with patch.object(module, "cluster_nodes", return_value=(["node"], 0)), \
                patch.object(module, "check_training_storage", side_effect=RuntimeError("Insufficient output disk space")), \
                patch.object(module, "background") as background:
            with self.assertRaisesRegex(RuntimeError, "Insufficient output disk space"):
                module.train_cluster(cfg, "student", "teacher", {})
        background.assert_not_called()

    def test_quota_failure_marker_does_not_mask_the_original_training_error(self):
        module = launcher()
        with tempfile.TemporaryDirectory() as directory:
            cfg = dict(output_dir=directory, experiment_name="test", method="lightning_opd",
                       n_gpus_per_node=4)
            write = Path.write_text

            def fail(path, *args, **kwargs):
                if path.name == "initialization.json":
                    raise RuntimeError("original training failure")
                if path.suffix == ".done":
                    raise OSError(errno.EDQUOT, "Disk quota exceeded")
                return write(path, *args, **kwargs)

            errors = io.StringIO()
            with patch.object(module, "cluster_nodes", return_value=(["node"], 0)), \
                    patch.object(module, "check_training_storage"), \
                    patch.object(module, "check_inputs"), patch.object(module, "check_run_state"), \
                    patch.object(module.socket, "gethostbyname", return_value="127.0.0.1"), \
                    patch.object(Path, "write_text", fail), \
                    patch.dict(module.os.environ, {"LIGHTNING_OPD_JOB_ID": "123", "CUDA_VISIBLE_DEVICES": "0,1,2,3"}), \
                    contextlib.redirect_stderr(errors):
                with self.assertRaisesRegex(RuntimeError, "original training failure"):
                    module.train_cluster(cfg, "student", "teacher", {})
            self.assertIn("Disk quota exceeded", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
