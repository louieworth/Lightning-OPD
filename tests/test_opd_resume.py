"""CPU checks for complete evaluations and continuation of the actual data cursor."""

import ast
import asyncio
import copy
import importlib.util
import json
import logging
import os
from pathlib import Path
import pickle
import random
import tempfile
import types
import unittest
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def definitions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), namespace)


class ResumeTests(unittest.TestCase):
    def test_saved_cursor_continues_the_uninterrupted_shuffled_prompt_sequence(self):
        with tempfile.TemporaryDirectory(prefix="opd-cursor-test-") as directory:
            def save(value, path):
                Path(path).write_bytes(pickle.dumps(value))
            namespace = dict(DataSource=object, Sample=types.SimpleNamespace, torch=types.SimpleNamespace(
                save=save, load=lambda path: pickle.loads(Path(path).read_bytes())),
                os=os, copy=copy, logger=logging.getLogger(__name__))
            definitions("slime/rollout/data_source.py", {"RolloutDataSource"}, namespace)
            source_cls = namespace["RolloutDataSource"]

            class Dataset:
                def __init__(self):
                    self.original = [types.SimpleNamespace(prompt=str(index)) for index in range(5)]
                    self.shuffle(0)

                def shuffle(self, epoch):
                    order = list(range(len(self.original)))
                    random.Random(42 + epoch).shuffle(order)
                    self.samples = [self.original[index] for index in order]

                def __len__(self):
                    return len(self.samples)

            def source():
                value = source_cls.__new__(source_cls)
                value.args = types.SimpleNamespace(save=directory, load=directory, rollout_global_dataset=True,
                                                   rollout_shuffle=True, n_samples_per_prompt=1)
                value.dataset = Dataset()
                value.epoch_id = value.sample_offset = value.sample_index = value.sample_group_index = 0
                value.metadata = {}
                return value

            def next_batch(value):
                return [(sample.prompt, sample.index, sample.group_index)
                        for group in value.get_samples(2) for sample in group]

            uninterrupted = source()
            all_batches = [next_batch(uninterrupted) for _ in range(14)]
            interrupted = source()
            self.assertEqual([next_batch(interrupted) for _ in range(6)], all_batches[:6])
            interrupted.save(5)
            restored = source()
            restored.load(5)
            self.assertEqual((restored.epoch_id, restored.sample_offset, restored.sample_index), (2, 2, 12))
            self.assertEqual([next_batch(restored) for _ in range(8)], all_batches[6:])

    def test_standard_resume_rejects_changed_training_recipe_and_missing_cursor(self):
        spec = importlib.util.spec_from_file_location("resume_launcher", ROOT / "configs/opd/qwen3-4b-opd.py")
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        with tempfile.TemporaryDirectory(prefix="opd-recipe-test-") as directory:
            cfg = yaml.safe_load((ROOT / "configs/config/qwen3_4b_base_opd_nt_teacher_8b.yaml").read_text())
            cfg.update(output_dir=directory, experiment_name="test", resume=True)
            run = launcher.run_dir(cfg)
            weights = run / "checkpoints/iter_0000002"
            weights.mkdir(parents=True)
            (weights / "weights.pt").write_bytes(b"weights")
            # Existing final-only checkpoints remain resumable after changing
            # the save schedule to one checkpoint per evaluation.
            previous = dict(cfg, save_model="final", save_interval=cfg["num_rollout"])
            (run / "config.yaml").write_text(yaml.safe_dump(previous))
            (weights.parent / "latest_checkpointed_iteration.txt").write_text("2")
            cursor = weights.parent / "rollout/global_dataset_state_dict_2.pt"
            cursor.parent.mkdir()
            cursor.write_bytes(b"cursor")
            launcher.check_run_state(cfg, "student", "teacher")
            launcher.check_run_state(dict(cfg, val_files=[], save_total_limit=2), "student", "teacher")
            for change in (dict(train_files="other.parquet"), dict(rollout_seed=43),
                           dict(rollout_shuffle=not cfg["rollout_shuffle"]), dict(lr=1e-4),
                           dict(rollout_batch_size=32), dict(n_samples_per_prompt=2),
                           dict(opd_loss_token_mode="top_k_teacher")):
                with self.subTest(change=change), self.assertRaisesRegex(ValueError, "recipe changed"):
                    launcher.check_run_state(dict(cfg, **change), "student", "teacher")
            legacy = {key: value for key, value in cfg.items() if not key.startswith("opd_")}
            (run / "config.yaml").write_text(yaml.safe_dump(legacy))
            launcher.check_run_state(cfg, "student", "teacher")
            launcher.check_run_state(dict(cfg, opd_token_top_k=64), "student", "teacher")
            topk = dict(cfg, opd_loss_token_mode="top_k_student")
            (run / "config.yaml").write_text(yaml.safe_dump(topk))
            with self.assertRaisesRegex(ValueError, "opd_token_top_k"):
                launcher.check_run_state(dict(topk, opd_token_top_k=64), "student", "teacher")
            (run / "config.yaml").write_text(yaml.safe_dump(cfg))
            cursor.unlink()
            with self.assertRaisesRegex(ValueError, "rollout cursor"):
                launcher.check_run_state(cfg, "student", "teacher")

    def test_evaluation_waits_for_all_dataset_requests(self):
        completed = []

        async def dataset_eval(args, index, dataset):
            await asyncio.sleep(0.01 if dataset.name == "a" else 0.02)
            completed.append(dataset.name)
            return {dataset.name: {"samples": [dataset.name]}}

        namespace = dict(asyncio=asyncio, Namespace=types.SimpleNamespace, Any=Any,
                         Sample=types.SimpleNamespace, RolloutFnEvalOutput=types.SimpleNamespace,
                         eval_rollout_single_dataset=dataset_eval)
        definitions("slime/rollout/sglang_rollout.py", {"eval_rollout"}, namespace)
        args = types.SimpleNamespace(group_rm=False, eval_datasets=[
            types.SimpleNamespace(name="a"), types.SimpleNamespace(name="b")])
        output, auxiliary = asyncio.run(namespace["eval_rollout"](args, 19))
        self.assertEqual(set(completed), {"a", "b"})
        self.assertEqual(set(output.data), {"a", "b"})
        self.assertEqual(auxiliary, [])


if __name__ == "__main__":
    unittest.main()
