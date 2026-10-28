"""CPU checks for saving, evaluating and retaining one complete checkpoint."""

import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import yaml


ROOT = Path(__file__).resolve().parents[1]


class CheckpointPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("checkpoint_policy_launcher", ROOT / "configs/opd/qwen3-4b-opd.py")
        cls.launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.launcher)
        cls.train_function = next(node for node in ast.parse((ROOT / "train.py").read_text()).body
                                  if isinstance(node, ast.FunctionDef) and node.name == "train")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="opd-eval-checkpoint-test-")
        self.addCleanup(temporary.cleanup)
        self.run = Path(temporary.name)
        self.checkpoints = self.run / "checkpoints"
        self.events = []

    def write_checkpoint(self, index):
        weights = self.checkpoints / f"iter_{index:07d}"
        weights.mkdir(parents=True, exist_ok=True)
        (weights / "model-optimizer-rng.pt").write_bytes(b"complete training state")
        (self.checkpoints / "latest_checkpointed_iteration.txt").write_text(str(index))

    def write_cursor(self, index):
        directory = self.checkpoints / "rollout"
        directory.mkdir(exist_ok=True)
        (directory / f"global_dataset_state_dict_{index}.pt").write_bytes(b"data cursor")

    def drive(self, *, start=0, total=40, interval=20, fail_save=None, fail_eval=None):
        cfg = dict(method="opd", num_rollout=total, save_interval=interval, save_total_limit=1)
        (self.run / "config.yaml").write_text(yaml.safe_dump(cfg))
        tracker = self.checkpoints / "latest_checkpointed_iteration.txt"
        if start and not tracker.exists():
            self.write_checkpoint(start - 1)
            self.write_cursor(start - 1)
            metrics = self.run / "eval_metrics"
            metrics.mkdir()
            (metrics / f"{start - 1}.json").write_text("{}")
        args = types.SimpleNamespace(
            num_rollout=total, start_rollout_id=start, eval_interval=interval,
            eval_on_resume=bool(start), val_before_train=False,
            offload_train=False, offload_rollout=False, enable_weights_backuper=False,
            use_critic=False, rollout_global_dataset=True, update_weights_interval=1,
            save_interval=interval, save=str(self.checkpoints),
            eval_output_dir=str(self.run / "eval_metrics"), eval_datasets=[],
        )
        remote = lambda function: types.SimpleNamespace(remote=function)

        def save(index, force_sync=False):
            self.assertTrue(force_sync)
            self.events.append(("save", index + 1))
            if index + 1 == fail_save:
                partial = self.checkpoints / f"iter_{index:07d}"
                partial.mkdir()
                raise RuntimeError("checkpoint save failed")
            self.write_checkpoint(index)

        def save_cursor(index):
            self.events.append(("cursor", index + 1))
            self.write_cursor(index)

        def evaluate(index, completed_steps):
            self.events.append(("eval", completed_steps))
            if completed_steps == 40:
                # The previous evaluated checkpoint must survive until the new
                # evaluation has actually succeeded, not just until model save.
                self.assertTrue((self.checkpoints / "iter_0000019").is_dir())
                self.assertTrue((self.checkpoints / "iter_0000039").is_dir())
            if completed_steps == fail_eval:
                raise RuntimeError("evaluation failed")
            args.eval_completed_steps = completed_steps
            self.launcher.evaluate(args, index, None)

        actor = types.SimpleNamespace(
            async_train=lambda index, data: self.events.append(("train", index + 1)),
            save_model=save, update_weights=lambda: None, clear_memory=lambda: None,
            dispose=lambda: None,
        )
        manager = types.SimpleNamespace(
            generate=remote(lambda index: None), eval=remote(evaluate),
            save=remote(save_cursor), dispose=remote(lambda: None),
        )
        namespace = dict(
            ray=types.SimpleNamespace(get=lambda value: value),
            GPU_MEMORY_TYPE_WEIGHTS="weights", GPU_MEMORY_TYPE_KV_CACHE="kv", GPU_MEMORY_TYPE_CUDA_GRAPH=None,
            configure_logger=lambda: None, init_tracking=lambda args: None,
            create_placement_groups=lambda args: {"rollout": None},
            create_rollout_manager=lambda *args: (manager, None),
            create_training_models=lambda *args: (actor, None),
        )
        rollout = types.SimpleNamespace(
            GenerateState=lambda args: None,
            generate_rollout=lambda *args, **kwargs: types.SimpleNamespace(metrics={}, data={}),
        )
        with patch.dict(sys.modules, {"slime.rollout.sglang_rollout": rollout}):
            exec(compile(ast.Module(body=[self.train_function], type_ignores=[]), "train.py", "exec"), namespace)
            namespace["train"](args)

    def test_step_40_eval_removes_step_20_checkpoint_and_cursor_but_keeps_metrics(self):
        self.drive()
        self.assertEqual([value for name, value in self.events if name == "save"], [20, 40])
        self.assertEqual([value for name, value in self.events if name == "eval"], [20, 40])
        for step in (20, 40):
            self.assertLess(self.events.index(("save", step)), self.events.index(("cursor", step)))
            self.assertLess(self.events.index(("cursor", step)), self.events.index(("eval", step)))
        self.assertEqual([path.name for path in self.checkpoints.glob("iter_*")], ["iter_0000039"])
        self.assertEqual([path.name for path in (self.checkpoints / "rollout").glob("*.pt")],
                         ["global_dataset_state_dict_39.pt"])
        self.assertEqual(sorted(path.name for path in (self.run / "eval_metrics").glob("*.json")),
                         ["19.json", "39.json"])

    def test_failed_save_retains_last_evaluated_checkpoint(self):
        with self.assertRaisesRegex(RuntimeError, "checkpoint save failed"):
            self.drive(start=20, fail_save=40)
        self.assertTrue((self.checkpoints / "iter_0000019").is_dir())
        self.assertTrue((self.checkpoints / "rollout/global_dataset_state_dict_19.pt").is_file())
        self.assertEqual((self.checkpoints / "latest_checkpointed_iteration.txt").read_text(), "19")
        self.assertNotIn(("eval", 40), self.events)

    def test_failed_eval_keeps_previous_checkpoint_until_resume_finishes_eval(self):
        with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
            self.drive(start=20, fail_eval=40)
        self.assertTrue((self.checkpoints / "iter_0000019").is_dir())
        self.assertTrue((self.checkpoints / "iter_0000039").is_dir())
        self.assertFalse((self.run / "eval_metrics/39.json").exists())
        self.assertFalse((self.run / "training_status.json").exists())
        self.events.clear()
        self.drive(start=40)
        self.assertEqual(self.events, [("eval", 40)])
        self.assertFalse((self.checkpoints / "iter_0000019").exists())
        self.assertTrue((self.checkpoints / "iter_0000039").is_dir())
        self.assertTrue((self.run / "eval_metrics/39.json").is_file())
        self.assertEqual(json.loads((self.run / "training_status.json").read_text()),
                         dict(status="complete", global_step=40, target_steps=40))

    def test_final_update_is_saved_and_evaluated_between_regular_intervals(self):
        self.drive(total=45)
        self.assertEqual([value for name, value in self.events if name == "save"], [20, 40, 45])
        self.assertEqual([value for name, value in self.events if name == "eval"], [20, 40, 45])
        self.assertEqual([path.name for path in self.checkpoints.glob("iter_*")], ["iter_0000044"])
        self.assertTrue((self.run / "eval_metrics/44.json").is_file())

    def test_missing_new_cursor_prevents_old_checkpoint_cleanup(self):
        self.write_checkpoint(19)
        self.write_cursor(19)
        self.write_checkpoint(39)
        with self.assertRaisesRegex(RuntimeError, "rollout cursor"):
            self.launcher.prune_training_checkpoints(self.checkpoints, 1, 39, require_rollout_state=True)
        self.assertTrue((self.checkpoints / "iter_0000019").is_dir())
        self.assertTrue((self.checkpoints / "iter_0000039").is_dir())

    def test_all_current_opd_recipes_save_at_each_eval_and_keep_one(self):
        checked = 0
        for path in sorted((ROOT / "configs/config").glob("*.yaml")):
            cfg = yaml.safe_load(path.read_text())
            if cfg.get("method") not in {"opd", "lightning_opd"}:
                continue
            with self.subTest(config=path.name):
                cfg = self.launcher.read_config(path, cfg["method"])
                self.assertEqual(cfg["save_interval"], cfg["eval_steps"])
                self.assertEqual(cfg["save_total_limit"], 1)
                self.assertFalse(cfg.get("async_save"))
                flags = self.launcher.training_args(cfg, "student", None, "/tmp/eval.yaml")
                self.assertNotIn("--save-final-only", flags)
                checked += 1
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()
