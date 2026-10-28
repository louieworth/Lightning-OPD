"""W&B exposes launcher metadata and effective per-dataset evaluation counts."""

from copy import deepcopy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_eval_logging import load_function


class WandbConfigLoggingTests(unittest.TestCase):
    def setUp(self):
        self.compute = load_function("slime/utils/wandb_utils.py", "_compute_config_for_logging",
                                     {"deepcopy": deepcopy, "os": os})

    def test_records_launcher_models_and_effective_four_sample_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.yaml"
            args = SimpleNamespace(
                lightning_block_config=str(config), n_samples_per_eval_prompt=1,
                n_samples_per_prompt=1,
                eval_datasets=[SimpleNamespace(name=name, n_samples_per_eval_prompt=4)
                               for name in ("amc23", "aime25")])
            for student in ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-1.7B-Base"):
                with self.subTest(student=student):
                    config.write_text(f"model_path: {student}\n"
                                      "teacher_model_path: Qwen/Qwen3-4B-Instruct-2507\n"
                                      "policy_rollout_k: 40\nval_n: 4\n")
                    with patch.dict(os.environ, {"SLURM_JOB_ID": "626791", "WANDB_API_KEY": "secret"}):
                        result = self.compute(args)
                    self.assertEqual(result["student_model"], student)
                    self.assertEqual(result["teacher_model"], "Qwen/Qwen3-4B-Instruct-2507")
                    self.assertEqual(result["policy_rollout_k"], 40)
                    self.assertEqual(result["val_n"], 4)
                    self.assertEqual(result["n_samples_per_eval_prompt"], 4)
                    self.assertEqual(result["eval_samples_per_dataset"], {"amc23": 4, "aime25": 4})
                    self.assertEqual(result["n_samples_per_prompt"], 1)
                    self.assertEqual(args.n_samples_per_eval_prompt, 1)
                    self.assertEqual(result["env_vars"], {"SLURM_JOB_ID": "626791"})

    def test_preserves_different_dataset_counts_without_inventing_common_value(self):
        args = SimpleNamespace(n_samples_per_eval_prompt=1,
                               eval_datasets=[SimpleNamespace(name="a", n_samples_per_eval_prompt=2),
                                              SimpleNamespace(name="b", n_samples_per_eval_prompt=4)])
        result = self.compute(args)
        self.assertEqual(result["eval_samples_per_dataset"], {"a": 2, "b": 4})
        self.assertEqual(result["n_samples_per_eval_prompt"], 1)
        self.assertNotIn("val_n", result)

    def test_epoch_and_update_budgets_are_fixed_config_for_lightning_and_online_opd(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.yaml"
            config.write_text("epoch: 2\nnum_rollout_per_epoch: 120\nnum_rollout: 240\n")
            for path_argument in (dict(lightning_block_config=str(config)),
                                  dict(eval_config=str(Path(directory) / "eval.yaml"))):
                with self.subTest(path_argument=path_argument):
                    args = SimpleNamespace(num_rollout=240, **path_argument)
                    result = self.compute(args)
                    self.assertEqual(result["epoch"], 2)
                    self.assertEqual(result["num_rollout_per_epoch"], 120)
                    self.assertEqual(result["num_rollout"], 240)
                    self.assertFalse(hasattr(args, "epoch"))
                    self.assertFalse(hasattr(args, "num_rollout_per_epoch"))

    def test_standard_opd_reads_model_ids_next_to_eval_config(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.yaml"
            config.write_text("model_path: Qwen/Qwen3-1.7B-Base\n"
                              "teacher_model_path: Qwen/Qwen3-8B\nval_n: 4\n")
            result = self.compute(SimpleNamespace(eval_config=str(Path(directory) / "eval.yaml")))
        self.assertEqual(result["student_model"], "Qwen/Qwen3-1.7B-Base")
        self.assertEqual(result["teacher_model"], "Qwen/Qwen3-8B")
        self.assertEqual(result["val_n"], 4)


if __name__ == "__main__":
    unittest.main()
