"""Regression for SFT workers launched without the repo on PYTHONPATH."""

import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class SFTRuntimeTests(unittest.TestCase):
    def test_resume_allows_node_repacking_but_preserves_optimizer_world_size(self):
        tree = ast.parse((ROOT / "scripts/qwen3_sft.py").read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "check_resume_recipe")
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "qwen3_sft.py", "exec"), namespace)
        previous = yaml.safe_load((ROOT / "configs/config/qwen3_4b_sft_downloaded.yaml").read_text())
        current = yaml.safe_load((ROOT / "configs/config/qwen3_4b_sft_downloaded_h200.yaml").read_text())
        self.assertEqual((previous["nnodes"], previous["n_gpus_per_node"]), (2, 4))
        self.assertEqual((current["nnodes"], current["n_gpus_per_node"]), (1, 8))
        check = namespace["check_resume_recipe"]
        check(previous, current)
        check(current, previous)
        for field, value in (("n_gpus_per_node", 16), ("deepspeed_stage", 0),
                             ("gradient_accumulation_steps", 16), ("per_device_train_batch_size", 8)):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    check(previous, dict(current, **{field: value}))
        batch = current["nnodes"] * current["n_gpus_per_node"] * current["per_device_train_batch_size"]
        self.assertEqual(batch * current["gradient_accumulation_steps"], current["global_batch_size"])

    def test_checkpoint_schedule_can_change_without_changing_training_recipe(self):
        tree = ast.parse((ROOT / "scripts/qwen3_sft.py").read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "check_resume_recipe")
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "qwen3_sft.py", "exec"), namespace)
        previous = {"max_steps": 3000, "learning_rate": 8e-5, "save_steps": 100, "save_total_limit": 1}
        current = dict(previous, save_strategy="no", walltime_checkpoint_margin_seconds=1200)
        current.pop("save_steps")
        namespace["check_resume_recipe"](previous, current)
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            namespace["check_resume_recipe"](previous, dict(current, learning_rate=1e-4))
        old_stage = dict(current, deepspeed_stage=0)
        new_stage = dict(current, deepspeed_stage=2)
        namespace["check_resume_recipe"](old_stage, new_stage, allow_untrained=True)
        with self.assertRaisesRegex(ValueError, "deepspeed_stage"):
            namespace["check_resume_recipe"](old_stage, new_stage)
        for filename in ("qwen3_4b_sft.yaml", "qwen3_4b_sft_downloaded.yaml"):
            config = yaml.safe_load((ROOT / "configs/config" / filename).read_text())
            self.assertEqual(config["save_strategy"], "no")
            self.assertFalse(config["save_only_model"])

    def test_full_checkpoint_at_walltime_and_training_target_without_periodic_saves(self):
        tree = ast.parse((ROOT / "scripts/qwen3_sft.py").read_text())
        callback = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.ClassDef) and node.name == "WalltimeCheckpoint")
        reached = [False]
        torch = types.SimpleNamespace(
            tensor=lambda value, **kwargs: types.SimpleNamespace(item=lambda: value),
            cuda=types.SimpleNamespace(current_device=lambda: 0),
            distributed=types.SimpleNamespace(is_initialized=lambda: False),
        )
        namespace = {"TrainerCallback": object, "torch": torch, "cfg": {"max_steps": 3000},
                     "deadline_reached": lambda: reached[0]}
        exec(compile(ast.Module(body=[callback], type_ignores=[]), "qwen3_sft.py", "exec"), namespace)
        checkpoint = namespace["WalltimeCheckpoint"]()
        for step, deadline, save, stop in ((100, False, False, False), (100, True, True, True),
                                           (3000, False, True, True), (3000, True, True, True)):
            with self.subTest(step=step, deadline=deadline):
                reached[0] = deadline
                # DefaultFlowCallback stops at max_steps, even with save_strategy: no.
                control = types.SimpleNamespace(should_save=False, should_training_stop=step >= 3000)
                checkpoint.on_step_end(None, types.SimpleNamespace(global_step=step), control)
                self.assertEqual((control.should_save, control.should_training_stop), (save, stop))

    def test_standalone_worker_can_load_shared_environment_from_another_directory(self):
        with tempfile.TemporaryDirectory(prefix="sft-worker-path-test-") as directory:
            temporary = Path(directory)
            overlay = temporary / "packages"
            overlay.mkdir()
            # Match a worker's script entry path: no cwd/repo entry, and only
            # package directories on PYTHONPATH. setup_env must import slime.
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join((str(overlay), str(Path(yaml.__file__).parents[1])))
            env["PYTHONNOUSERSITE"] = "1"
            code = f"""
import pathlib
import runpy
import sys
root = pathlib.Path({str(ROOT)!r})
assert str(root) not in sys.path
module = runpy.run_path(str(root / 'scripts/qwen3_sft.py'))
module['setup_env']({{
    'env_file': 'absent-test.env', 'hf_home': 'test-cache',
    'wandb_mode': 'disabled', 'project_name': 'test', 'wandb_entity': 'test',
    'output_dir': '.', 'experiment_name': 'test', 'local_models_only': True,
}})
import slime.rollout.lightning_opd_blocks as blocks
assert pathlib.Path(blocks.__file__).resolve() == root / 'slime/rollout/lightning_opd_blocks.py'
print('standalone SFT worker environment loaded')
"""
            result = subprocess.run([sys.executable, "-S", "-B", "-c", code],
                                    cwd=temporary, env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("standalone SFT worker environment loaded", result.stdout)


if __name__ == "__main__":
    unittest.main()
