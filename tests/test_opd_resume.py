"""CPU checks for evaluation deadlines and continuation of the actual data cursor."""

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
import time
import types
import unittest
from typing import Any

import yaml

from slime.utils.walltime import CheckpointDeadlineReached, evaluation_before_deadline


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
            cfg = yaml.safe_load((ROOT / "configs/config/qwen3_4b_opd.yaml").read_text())
            cfg.update(output_dir=directory, experiment_name="test", resume=True)
            run = launcher.run_dir(cfg)
            weights = run / "checkpoints/iter_0000002"
            weights.mkdir(parents=True)
            (weights / "weights.pt").write_bytes(b"weights")
            (run / "config.yaml").write_text(yaml.safe_dump(cfg))
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

    def test_long_evaluation_cancels_every_request_without_partial_results(self):
        cancelled, events = [], []

        async def generate(args, sample, **kwargs):
            try:
                await asyncio.sleep(60)
            finally:
                cancelled.append(sample.index)

        async def abort(args, index):
            events.append("abort")

        class Progress:
            def __init__(self, **kwargs):
                pass
            def update(self, count):
                pass
            def close(self):
                events.append("closed")

        dataset = types.SimpleNamespace(cache_key=("test",), temperature=0.7, top_p=0.8, top_k=20,
                                        max_response_len=16384, n_samples_per_eval_prompt=1,
                                        inject_metadata=lambda metadata: metadata)
        args = types.SimpleNamespace(group_rm=False, eval_datasets=[dataset], hf_checkpoint="student",
                                     apply_chat_template=True, rollout_stop=None, rollout_stop_token_ids=None,
                                     rollout_skip_special_tokens=True, checkpoint_deadline=None)
        namespace = dict(asyncio=asyncio, copy=copy, Namespace=types.SimpleNamespace,
                         EvalDatasetConfig=types.SimpleNamespace, Sample=types.SimpleNamespace, Any=Any,
                         RolloutFnEvalOutput=types.SimpleNamespace, tqdm=Progress, abort=abort,
                         generate_and_rm=generate, logger=logging.getLogger(__name__),
                         evaluation_before_deadline=evaluation_before_deadline,
                         CheckpointDeadlineReached=CheckpointDeadlineReached,
                         EVAL_PROMPT_DATASET={("test", "student", True): types.SimpleNamespace(
                             samples=[types.SimpleNamespace(metadata={}) for _ in range(3)])})
        definitions("slime/rollout/sglang_rollout.py", {"eval_rollout", "eval_rollout_single_dataset"}, namespace)
        args.checkpoint_deadline = time.time() + 0.05
        with self.assertRaises(CheckpointDeadlineReached):
            asyncio.run(namespace["eval_rollout"](args, 24))
        self.assertEqual(sorted(cancelled), [0, 1, 2])
        self.assertEqual(events, ["closed", "abort"])

    def test_evaluation_pause_saves_last_update_after_restoring_training_workers(self):
        for start, stop in ((0, 0), (0, 2), (0, 6), (2, 2), (3, 4)):
            with self.subTest(start=start, stop=stop), tempfile.TemporaryDirectory(prefix="opd-eval-stop-") as directory:
                events, state = [], {"step": start}
                remote = lambda function: types.SimpleNamespace(remote=function)
                def event(name, value=None):
                    events.append((name, value))
                def update(index, data):
                    state["step"] = index + 1
                    event("train", index)
                def evaluate(index, completed_steps=None):
                    self.assertEqual(completed_steps, state["step"])
                    event("eval", completed_steps)
                    if state["step"] == stop:
                        raise CheckpointDeadlineReached("deadline")
                def save(index, force_sync=False):
                    self.assertTrue(force_sync)
                    event("save", index)
                actor = types.SimpleNamespace(async_train=update, save_model=save,
                    update_weights=lambda: None, clear_memory=lambda: None,
                    onload=lambda: event("train_onload"), offload=lambda: event("train_offload"))
                manager = types.SimpleNamespace(generate=remote(lambda index: None), eval=remote(evaluate),
                    offload=remote(lambda: event("rollout_offload")), onload=remote(lambda **kw: None),
                    save=remote(lambda index: event("cursor", index)), dispose=remote(lambda: None))
                args = types.SimpleNamespace(num_rollout=6, start_rollout_id=start, eval_interval=2,
                    eval_on_resume=bool(start), offload_train=True, offload_rollout=True,
                    enable_weights_backuper=False, use_critic=False, rollout_global_dataset=True,
                    save_interval=6, save_final_only=True, update_weights_interval=1,
                    checkpoint_deadline=None, save=str(Path(directory) / "checkpoints"))
                namespace = dict(ray=types.SimpleNamespace(get=lambda value: value),
                    GPU_MEMORY_TYPE_WEIGHTS="weights", GPU_MEMORY_TYPE_KV_CACHE="kv", GPU_MEMORY_TYPE_CUDA_GRAPH=None,
                    configure_logger=lambda: None, init_tracking=lambda args: None,
                    create_placement_groups=lambda args: {"rollout": None},
                    create_rollout_manager=lambda *args: (manager, None), create_training_models=lambda *args: (actor, None))
                definitions("train.py", {"train"}, namespace)
                namespace["train"](args)
                expected = [stop - 1] if stop > start else []
                self.assertEqual([value for name, value in events if name == "save"], expected)
                self.assertEqual([value for name, value in events if name == "cursor"], expected)
                self.assertEqual([value for name, value in events if name == "train"], list(range(start, stop)))
                self.assertTrue(all(value % args.eval_interval == 0
                                    for name, value in events if name == "eval"))
                if stop == 2 and not start:
                    position = events.index(("save", 1))
                    self.assertEqual(events[position - 2:position], [("rollout_offload", None), ("train_onload", None)])
                status = json.loads((Path(directory) / "training_status.json").read_text())
                self.assertEqual(status, dict(status="paused", global_step=stop, target_steps=6))


if __name__ == "__main__":
    unittest.main()
