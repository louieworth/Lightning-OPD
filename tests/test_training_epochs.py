"""Epochs scale the existing training budget; K stays an explicit total count."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from slime.rollout import lightning_opd_blocks as blocks
from slime.utils.training_epochs import normalize_epochs


ROOT = Path(__file__).resolve().parents[1]


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def encode(self, text, **kwargs):
        return list(text.encode())


class TrainingEpochTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("epoch_launcher", ROOT / "configs/opd/qwen3-4b-opd.py")
        cls.launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.launcher)

    def test_default_is_one_epoch_and_k_is_not_multiplied(self):
        source = dict(num_rollout=120, policy_rollout_k=40)
        self.assertEqual(normalize_epochs(source)["num_rollout"], 120)
        cfg = normalize_epochs(dict(source, epoch=2))
        self.assertEqual((cfg["num_rollout"], cfg["policy_rollout_k"]), (240, 40))
        self.assertNotIn("epoch", source)
        for invalid in (0, -1, True, 1.5, "2", None):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "epoch"):
                normalize_epochs(dict(source, epoch=invalid))

    def test_four_base_configs_keep_defaults_and_support_two_epochs(self):
        paths = [ROOT / "configs/config" / name for name in (
            "qwen3_1_7b_base_nt_teacher_4b_instruct_2507_k1.yaml",
            "qwen3_1_7b_base_nt_teacher_4b_instruct_2507_k2.yaml",
            "qwen3_1_7b_base_nt_teacher_4b_instruct_2507_k40.yaml",
            "qwen3_1_7b_base_opd_nt_teacher_4b_instruct_2507.yaml",
        )]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            for original in paths:
                source = yaml.safe_load(original.read_text())
                source["output_dir"] = directory
                path.write_text(yaml.safe_dump(source))
                cfg = self.launcher.read_config(path, source["method"])
                self.assertEqual((cfg["epoch"], cfg["num_rollout"]), (1, 120))
                source["epoch"] = 2
                if "policy_rollout_k" in source:
                    source["policy_rollout_k"] *= 2
                path.write_text(yaml.safe_dump(source))
                cfg = self.launcher.read_config(path, source["method"])
                self.assertEqual((cfg["num_rollout_per_epoch"], cfg["num_rollout"]), (120, 240))
                flags = self.launcher.training_args(cfg, "student", None)
                self.assertEqual(flags[flags.index("--num-rollout") + 1], "240")
                self.assertNotIn("--epoch", flags)
                self.assertNotIn("--num-rollout-per-epoch", flags)
                path.write_text(yaml.safe_dump(cfg))
                reloaded = self.launcher.read_config(path, source["method"])
                self.assertEqual(reloaded["num_rollout"], 240)

    def test_epoch_two_presets_keep_original_k_and_other_hyperparameters(self):
        for k in (1, 2, 40):
            stem = f"qwen3_1_7b_base_nt_teacher_4b_instruct_2507_k{k}"
            original = ROOT / "configs/config" / f"{stem}.yaml"
            preset = original.with_name(f"{stem}_epoch2.yaml")
            with self.subTest(config=preset.name):
                before = yaml.safe_load(original.read_text())
                after = yaml.safe_load(preset.read_text())
                self.assertEqual(after, dict(before, epoch=2, experiment_name=f"{stem}_epoch2"))
                cfg = self.launcher.read_config(preset, "lightning_opd")
                self.assertEqual((cfg["epoch"], cfg["num_rollout_per_epoch"], cfg["num_rollout"]),
                                 (2, 120, 240))
                self.assertEqual(cfg["policy_rollout_k"], k)
                plan = blocks.schedule(cfg)
                self.assertEqual(len(plan), k)
                self.assertTrue(all(block["end_step"] - block["start_step"] == 240 // k
                                    for block in plan))
                self.assertEqual(plan[-1]["end_sample"], 30720)
                flags = self.launcher.training_args(cfg, "student", None)
                self.assertEqual(flags[flags.index("--num-rollout") + 1], "240")
                self.assertEqual(flags[flags.index("--wandb-group") + 1], f"{stem}_epoch2")
                self.assertNotIn("--epoch", flags)
                self.assertNotIn("--num-rollout-per-epoch", flags)

    def test_epoch_two_k80_retains_three_updates_per_policy_block(self):
        cfg = normalize_epochs(dict(num_rollout=120, epoch=2, policy_rollout_k=80, rollout_batch_size=128))
        plan = blocks.schedule(cfg)
        self.assertEqual(len(plan), 80)
        self.assertTrue(all(block["end_step"] - block["start_step"] == 3 for block in plan))
        self.assertEqual(plan[-1]["end_sample"], 30720)

    def test_real_fixed_prompt_data_doubles_without_changing_first_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.parquet"
            pq.write_table(pa.Table.from_pylist([
                dict(prompt="first", label="1", metadata={"uid": "a"}),
                dict(prompt="second", label="2", metadata={"uid": "b"}),
                dict(prompt="third", label="3", metadata={"uid": "c"}),
            ]), source)
            base = yaml.safe_load((ROOT / "configs/config/qwen3_1_7b_base_nt_teacher_4b_instruct_2507_k1.yaml").read_text())
            base.update(output_dir=directory, num_rollout=6, rollout_batch_size=2,
                        policy_rollout_k=2, train_files=str(source))
            first = normalize_epochs(dict(base, epoch=1, experiment_name="one"))
            second = normalize_epochs(dict(base, epoch=2, policy_rollout_k=4, experiment_name="two"))
            first_path = blocks.prepare_fixed_prompts(first, "student", Tokenizer())
            second_path = blocks.prepare_fixed_prompts(second, "student", Tokenizer())
            one = pq.read_table(first_path).to_pylist()
            two = pq.read_table(second_path).to_pylist()
            self.assertEqual(len(two), 2 * len(one))
            self.assertEqual(two[:len(one)], one)
            self.assertEqual([row["metadata"]["lightning_sample_index"] for row in two], list(range(24)))
            blocks.prepare_fixed_prompts(second, "student", Tokenizer())
            self.assertEqual(pq.read_table(second_path).to_pylist(), two)

    def test_changing_epoch_cannot_resume_an_existing_one_epoch_run(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = yaml.safe_load((ROOT / "configs/config/qwen3_1_7b_base_opd_nt_teacher_4b_instruct_2507.yaml").read_text())
            cfg.update(output_dir=directory, experiment_name="old", resume=True)
            run = Path(directory) / "old"
            (run / "checkpoints").mkdir(parents=True)
            (run / "config.yaml").write_text(yaml.safe_dump(normalize_epochs(cfg)))
            (run / "checkpoints/latest_checkpointed_iteration.txt").write_text("119")
            with self.assertRaisesRegex(ValueError, "recipe changed"):
                self.launcher.check_run_state(normalize_epochs(dict(cfg, epoch=2)), "student", "teacher")
