"""CPU regression checks for fixed data, policy refreshes and global training positions."""

import ast
import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from slime.rollout import lightning_opd_blocks as blocks

ROOT = Path(__file__).resolve().parents[1]


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def encode(self, text, **kwargs):
        return [ord(char) for char in text]

    def decode(self, tokens, **kwargs):
        return str(tokens)

    def get_vocab(self):
        return {"shared": 1}


class Remote:
    def __init__(self, function):
        self.remote = function


class BlockTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lightning-block-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cfg = yaml.safe_load((ROOT / "configs/config/qwen3_4b_lightning_opd.yaml").read_text())
        self.cfg.update(output_dir=str(self.root), experiment_name="test", num_rollout=6,
                        rollout_batch_size=2, global_batch_size=2, policy_rollout_k=2,
                        eval_steps=2, rollout_max_prompt_len=8, rollout_max_response_len=4,
                        precompute_concurrency=2,
                        train_files=str(self.root / "dapo.parquet"))
        pq.write_table(pa.Table.from_pylist([
            dict(prompt="one", label="1", metadata={"sample_uid": "a"}),
            dict(prompt="two", label="2", metadata={"sample_uid": "b"}),
            dict(prompt="three", label="3", metadata={"sample_uid": "c"}),
            dict(prompt="too long for filtering", label="4", metadata={"sample_uid": "d"}),
        ]), self.cfg["train_files"])
        self.models = dict(student="student", teacher="teacher")
        self.run = blocks.run_directory(self.cfg)
        self.run.mkdir()
        blocks.atomic_json(self.run / "initialization.json", self.models)

    def snapshot(self):
        self.cfg["save_interval"] = blocks.schedule(self.cfg)[0]["end_step"]
        path = self.run / "config.yaml"
        path.write_text(yaml.safe_dump(self.cfg))
        return types.SimpleNamespace(lightning_block_config=str(path), hf_checkpoint="student",
                                     sglang_router_ip="localhost", sglang_router_port=1)

    def test_balanced_batch_aligned_schedule(self):
        cfg = dict(self.cfg, num_rollout=150, rollout_batch_size=256)
        for k in (1, 2, 3, 4, 5, 150):
            plan = blocks.schedule(dict(cfg, policy_rollout_k=k))
            sizes = [item["end_step"] - item["start_step"] for item in plan]
            self.assertEqual(len(plan), k)
            self.assertEqual(sum(sizes), 150)
            self.assertLessEqual(max(sizes) - min(sizes), 1)
            self.assertEqual(plan[-1]["end_sample"], 38400)
            for previous, following in zip(plan, plan[1:]):
                self.assertEqual(previous["end_sample"], following["start_sample"])
        for invalid in (0, -1, 151, True, 2.0, "2", None):
            with self.assertRaises(ValueError):
                blocks.schedule(dict(cfg, policy_rollout_k=invalid))

    def test_fixed_sequence_is_repeated_deterministic_and_k_independent(self):
        path = blocks.prepare_fixed_prompts(self.cfg, "student", Tokenizer())
        first = pq.read_table(path).to_pylist()
        self.assertEqual(len(first), 12)
        self.assertEqual([row["metadata"]["lightning_sample_index"] for row in first], list(range(12)))
        self.assertEqual(len({row["prompt"] for row in first}), 3)
        other = dict(self.cfg, experiment_name="k3", policy_rollout_k=3)
        second_path = blocks.prepare_fixed_prompts(other, "student", Tokenizer())
        self.assertEqual(first, pq.read_table(second_path).to_pylist())
        with self.assertRaisesRegex(ValueError, "block plan changed"):
            blocks.prepare_fixed_prompts(dict(self.cfg, policy_rollout_k=3), "student", Tokenizer())

    def test_real_parquet_generation_and_teacher_alignment(self):
        blocks.prepare_fixed_prompts(self.cfg, "student", Tokenizer())
        args = self.snapshot()
        manager = types.SimpleNamespace(args=args)
        policy = {"step": 0, "calls": 0, "bad_teacher": False}

        def request(url, payload):
            if payload["sampling_params"]["max_new_tokens"]:
                policy["calls"] += 1
                token = 100 + policy["step"]
                return dict(meta_info=dict(output_token_logprobs=[[-0.5, token, ""]], finish_reason={"type": "stop"}))
            ids = payload["input_ids"]
            items = [[None, ids[0], ""]] + [[-0.2, token, ""] for token in ids[1:]]
            if policy["bad_teacher"]:
                items[-1][1] += 1
            return dict(meta_info=dict(input_token_logprobs=items))

        @contextlib.contextmanager
        def background(*args):
            yield object()

        launcher = types.SimpleNamespace(request=request, background=background,
                                         server_command=lambda *args: [], wait_ready=lambda *args: None)
        transformers = types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer()))
        with patch.dict(sys.modules, {"configs.opd.qwen3-4b-opd": launcher, "transformers": transformers}):
            blocks.prepare_student_block(manager, 0, "initial")
            blocks.score_teacher_block(args, 0, "initial")
            first = pq.read_table(blocks.block_paths(self.cfg, 0)[1]).to_pylist()
            self.assertEqual([row["metadata"]["lightning_sample_index"] for row in first], list(range(6)))
            self.assertTrue(all(row["metadata"]["response_tokens"] == [100] for row in first))
            self.assertTrue(all(row["metadata"]["teacher_log_probs"] == [-0.2] for row in first))
            before = policy["calls"]
            blocks.prepare_student_block(manager, 0, "initial")
            self.assertEqual(policy["calls"], before)
            policy["step"] = 3
            blocks.prepare_student_block(manager, 1, "updated")
            blocks.score_teacher_block(args, 1, "updated")
            second = pq.read_table(blocks.block_paths(self.cfg, 1)[1]).to_pylist()
            self.assertEqual([row["metadata"]["lightning_sample_index"] for row in second], list(range(6, 12)))
            self.assertTrue(all(row["metadata"]["response_tokens"] == [103] for row in second))
            self.assertTrue(all(row["metadata"]["policy_update_step"] == 3 for row in second))
            args.__dict__.update(rollout_max_prompt_len=8, input_key="prompt", label_key="label",
                                 metadata_key="metadata", tool_key=None, multimodal_keys=None, rollout_seed=42)
            manager.data_source = types.SimpleNamespace(metadata={})

            def dataset(path, **kwargs):
                rows = pq.read_table(path).to_pylist()
                return Dataset(rows)

            class Dataset:
                def __init__(self, rows):
                    self.samples = [types.SimpleNamespace(metadata=row["metadata"]) for row in rows]

                def __len__(self):
                    return len(self.samples)

            data_module = types.SimpleNamespace(create_dataset=dataset)
            processing = types.SimpleNamespace(load_tokenizer=lambda *a, **kw: Tokenizer(),
                                              load_processor=lambda *a, **kw: None)
            with patch.dict(sys.modules, {"slime.utils.data": data_module, "slime.utils.processing_utils": processing}):
                blocks.load_training_block(manager, 1, 3, "updated")
                self.assertEqual(manager.data_source.sample_offset, 0)
                self.assertEqual(manager.data_source.sample_index, 6)
                self.assertEqual(manager.data_source.metadata["lightning_global_sample_offset"], 6)
                # A walltime checkpoint can stop inside a later block. Resume
                # its original cached policy, rather than generating new data.
                args.__dict__.update(offload_train=True, offload_rollout=True, update_weights_interval=1)
                manager.data_source.metadata["lightning_global_sample_offset"] = 8
                manager.lightning_resume_metadata = Remote(lambda: dict(manager.data_source.metadata))
                manager.prepare_lightning_block = Remote(lambda *a: self.fail("resumed block regenerated"))
                manager.load_lightning_block = Remote(lambda i, r, p: blocks.load_training_block(manager, i, r, p))
                manager.offload = Remote(lambda: None)
                manager.onload = Remote(lambda **kw: None)
                constants = types.SimpleNamespace(GPU_MEMORY_TYPE_WEIGHTS="weights", GPU_MEMORY_TYPE_KV_CACHE="kv")
                with patch.dict(sys.modules, {"ray": types.SimpleNamespace(get=lambda value: value),
                                              "sglang.srt.constants": constants}), \
                        patch.object(blocks, "score_teacher_block", side_effect=AssertionError("teacher rescored")):
                    coordinator = blocks.BlockCoordinator(args, manager)
                    coordinator.ensure_block(4)
                    self.assertEqual(manager.data_source.sample_offset, 2)
                    self.assertEqual(manager.data_source.sample_index, 8)
                    self.assertEqual(manager.data_source.metadata["lightning_policy_id"], "updated")
                    coordinator.loaded = None
                    manager.data_source.metadata["lightning_global_sample_offset"] = 0
                    with self.assertRaisesRegex(ValueError, "resume cursor"):
                        coordinator.ensure_block(4)
                data_module.create_dataset = lambda path, **kw: Dataset(list(reversed(second)))
                with self.assertRaisesRegex(ValueError, "reordered"):
                    blocks.load_training_block(manager, 1, 3, "updated")
            identity = blocks.block_identity(self.cfg, self.models, 1, "another-run")
            self.assertFalse(blocks.cached(blocks.block_paths(self.cfg, 1)[1], identity, 6))
            blocks.prepare_student_block(manager, 1, "bad")
            policy["bad_teacher"] = True
            with self.assertRaisesRegex(ValueError, "misaligned"):
                blocks.score_teacher_block(args, 1, "bad")
            self.assertTrue(blocks.cached(blocks.block_paths(self.cfg, 1)[1],
                                          blocks.block_identity(self.cfg, self.models, 1, "updated"), 6))

    def test_teacher_feature_upgrade_preserves_saved_rollout_policy_and_resume_cursor(self):
        blocks.prepare_fixed_prompts(self.cfg, "student", Tokenizer())
        args = self.snapshot()
        manager = types.SimpleNamespace(args=args)
        calls = {"generation": 0, "features": 0}
        feature_reference = dict(path=str(self.root / "features.pt"), head_path=str(self.root / "head.pt"),
                                 vocab_size=128, response_length=1)

        def request(url, payload):
            if payload["sampling_params"]["max_new_tokens"]:
                calls["generation"] += 1
                return dict(meta_info=dict(output_token_logprobs=[[-.5, 100, ""]], finish_reason={"type": "stop"}))
            ids = payload["input_ids"]
            metadata = dict(input_token_logprobs=[[None, ids[0], ""]] + [[-.2, token, ""] for token in ids[1:]])
            if "opd_response_length" in payload:
                self.assertEqual(payload["opd_response_length"], 1)
                calls["features"] += 1
                metadata["opd_teacher_features"] = feature_reference
            return dict(meta_info=metadata)

        @contextlib.contextmanager
        def background(*args):
            yield object()

        launcher = types.SimpleNamespace(request=request, background=background, server_command=lambda *a: [],
                                         wait_ready=lambda *a: None)
        transformers = types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()))
        with patch.dict(sys.modules, {"configs.opd.qwen3-4b-opd": launcher, "transformers": transformers}):
            blocks.prepare_student_block(manager, 0, "initial")
            blocks.score_teacher_block(args, 0, "initial")
            rollouts, scored = blocks.block_paths(self.cfg, 0)
            original = pq.read_table(scored).to_pylist()
            rollouts.unlink()
            rollouts.with_suffix(".manifest.json").unlink()
            self.cfg["opd_loss_token_mode"] = "full_vocab"
            args = self.snapshot()
            blocks.score_teacher_block(args, 0, "initial")
            upgraded = pq.read_table(scored).to_pylist()
            self.assertEqual(calls, {"generation": 6, "features": 6})
            for old, new in zip(original, upgraded, strict=True):
                self.assertEqual(old["prompt"], new["prompt"])
                self.assertEqual(old["metadata"]["response_tokens"], new["metadata"]["response_tokens"])
                self.assertEqual(old["metadata"]["teacher_log_probs"], new["metadata"]["teacher_log_probs"])
                self.assertEqual(new["metadata"]["teacher_features"], feature_reference)
            identity = blocks.block_identity(self.cfg, self.models, 0, "initial")
            self.assertTrue(blocks.cached(scored, blocks.teacher_identity(self.cfg, identity), 6))
            self.assertFalse(blocks.cached(scored, identity, 6))
            args.__dict__.update(offload_train=True, offload_rollout=True, update_weights_interval=1)
            loaded = []
            manager.args = args
            manager.lightning_resume_metadata = Remote(lambda: dict(
                lightning_block_index=0, lightning_policy_id="initial", lightning_global_sample_offset=2))
            manager.prepare_lightning_block = Remote(lambda *a: self.fail("resume regenerated student responses"))
            manager.offload, manager.onload = Remote(lambda: None), Remote(lambda **k: None)
            manager.load_lightning_block = Remote(lambda *a: loaded.append(a))
            constants = types.SimpleNamespace(GPU_MEMORY_TYPE_WEIGHTS="weights", GPU_MEMORY_TYPE_KV_CACHE="kv")
            with patch.dict(sys.modules, {"ray": types.SimpleNamespace(get=lambda value: value),
                                          "sglang.srt.constants": constants}):
                blocks.BlockCoordinator(args, manager).ensure_block(1)
            self.assertEqual(loaded, [(0, 1, "initial")])
            self.assertEqual(calls, {"generation": 6, "features": 6})

    def test_driver_refreshes_policy_but_keeps_global_eval_and_saves_once(self):
        function = next(item for item in ast.parse((ROOT / "train.py").read_text()).body
                        if isinstance(item, ast.FunctionDef) and item.name == "train")
        constants = types.SimpleNamespace(GPU_MEMORY_TYPE_WEIGHTS="weights", GPU_MEMORY_TYPE_KV_CACHE="kv",
                                         GPU_MEMORY_TYPE_CUDA_GRAPH="graph")
        self.cfg.update(num_rollout=150, rollout_batch_size=256, global_batch_size=256, eval_steps=25)
        for k in (1, 2, 4, 150):
            with self.subTest(k=k):
                self.cfg["policy_rollout_k"] = k
                args = self.snapshot()
                args.__dict__.update(num_rollout=150, start_rollout_id=0, eval_interval=25, eval_on_resume=False,
                                     offload_train=True, offload_rollout=True, enable_weights_backuper=False,
                                     use_critic=False, rollout_global_dataset=True, rollout_batch_size=256,
                                     save_interval=self.cfg["save_interval"], update_weights_interval=1, save=None)
                events, state = [], {"step": 0, "train_loaded": False, "inference_loaded": True}

                def stage(name, value=None):
                    events.append((name, value))
                    if name == "offload_inference":
                        state["inference_loaded"] = False
                    elif name == "onload_inference":
                        state["inference_loaded"] = True
                    elif name == "prepare":
                        self.assertFalse(state["train_loaded"])
                        self.assertTrue(state["inference_loaded"])
                        self.assertEqual(state["step"], blocks.schedule(self.cfg)[value]["start_step"])

                manager = types.SimpleNamespace(
                    prepare_lightning_block=Remote(lambda index, policy: stage("prepare", index)),
                    load_lightning_block=Remote(lambda index, rollout, policy: stage("load", index)),
                    offload=Remote(lambda: stage("offload_inference")),
                    onload=Remote(lambda **kw: stage("onload_inference")),
                    generate=Remote(lambda index: stage("generate", index)),
                    eval=Remote(lambda index=None, **kw: stage("eval", index if index is not None else kw["rollout_id"])),
                    save=Remote(lambda index: stage("cursor", index)), dispose=Remote(lambda: None),
                )

                def train(index, data):
                    state["train_loaded"] = True
                    state["step"] += 1
                    stage("train", index)

                def offload():
                    state["train_loaded"] = False

                actor = types.SimpleNamespace(async_train=train, save_model=lambda index, **kw: stage("save", index),
                                              update_weights=lambda: None, clear_memory=lambda: None,
                                              onload=lambda: state.update(train_loaded=True), offload=offload)

                def score(args, index, policy):
                    self.assertFalse(state["train_loaded"])
                    self.assertFalse(state["inference_loaded"])
                    stage("teacher", index)

                ray = types.SimpleNamespace(get=lambda value: value)
                namespace = dict(ray=ray, GPU_MEMORY_TYPE_WEIGHTS="weights", GPU_MEMORY_TYPE_KV_CACHE="kv",
                                 GPU_MEMORY_TYPE_CUDA_GRAPH="graph", configure_logger=lambda: None,
                                 init_tracking=lambda args: None, create_placement_groups=lambda args: {"rollout": None},
                                 create_rollout_manager=lambda *args: (manager, None),
                                 create_training_models=lambda *args: (actor, None))
                with patch.dict(sys.modules, {"ray": ray, "sglang.srt.constants": constants}), \
                        patch.object(blocks, "score_teacher_block", score):
                    exec(compile(ast.Module(body=[function], type_ignores=[]), "train.py", "exec"), namespace)
                    namespace["train"](args)
                self.assertEqual([value for name, value in events if name == "prepare"], list(range(k)))
                self.assertEqual([value for name, value in events if name == "teacher"], list(range(k)))
                self.assertEqual([value for name, value in events if name == "eval"], [0, 24, 49, 74, 99, 124, 149])
                self.assertEqual([value for name, value in events if name == "save"], [self.cfg["save_interval"] - 1])
                self.assertEqual([value for name, value in events if name == "train"], list(range(150)))
                self.assertEqual(state["step"], 150)
                # Resume from the only saved checkpoint; global steps do not reset.
                events.clear()
                args.start_rollout_id = self.cfg["save_interval"]
                state.update(step=args.start_rollout_id, train_loaded=False)
                with patch.dict(sys.modules, {"ray": ray, "sglang.srt.constants": constants}), \
                        patch.object(blocks, "score_teacher_block", score):
                    namespace["train"](args)
                self.assertEqual([value for name, value in events if name == "prepare"], list(range(1, k)))
                self.assertEqual([value for name, value in events if name == "save"], [])
                self.assertEqual([value for name, value in events if name == "train"], list(range(args.start_rollout_id, 150)))
                self.assertEqual([value for name, value in events if name == "eval"],
                                 [i for i in (24, 49, 74, 99, 124, 149) if i >= args.start_rollout_id])

    def test_launcher_keeps_standard_opd_save_policy(self):
        spec = importlib.util.spec_from_file_location("block_launcher_test", ROOT / "configs/opd/qwen3-4b-opd.py")
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        cfg = launcher.read_config(ROOT / "configs/config/qwen3_4b_lightning_opd.yaml", "lightning_opd")
        self.assertEqual(cfg["save_interval"], 150)
        flags = launcher.training_args(cfg, "student", None, "/tmp/eval.yaml")
        self.assertIn("--lightning-block-config", flags)
        self.assertNotIn("--policy-rollout-k", flags)
        self.assertEqual(flags[flags.index("--rollout-function-path") + 1],
                         "slime.rollout.lightning_opd_blocks.generate_cached_rollout")
        base = yaml.safe_load((ROOT / "configs/config/qwen3_4b_lightning_opd.yaml").read_text())
        temporary = self.root / "launcher.yaml"
        temporary.write_text(yaml.safe_dump(dict(base, policy_rollout_k=4)))
        self.assertEqual(launcher.read_config(temporary, "lightning_opd")["save_interval"], 150)
        for change in (dict(policy_rollout_k=True), dict(policy_rollout_k=0), dict(policy_rollout_k=151),
                       dict(rollout_shuffle=True), dict(save_model="active"), dict(save_eval_multiplier=2)):
            temporary.write_text(yaml.safe_dump(dict(base, **change)))
            with self.assertRaises(ValueError):
                launcher.read_config(temporary, "lightning_opd")
        opd = launcher.read_config(ROOT / "configs/config/qwen3_4b_opd.yaml", "opd")
        self.assertEqual((opd["save_model"], opd["save_interval"]), ("final", 150))
        self.assertIn("--save-final-only", launcher.training_args(opd, "student", None, "/tmp/eval.yaml"))
        self.assertIn("--save-final-only", flags)
        legacy = dict(base, policy_rollout_k=4)
        legacy.pop("save_model")
        temporary.write_text(yaml.safe_dump(legacy))
        self.assertEqual(launcher.read_config(temporary, "lightning_opd")["save_interval"], 38)
        standard = yaml.safe_load((ROOT / "configs/config/qwen3_4b_opd.yaml").read_text())
        temporary.write_text(yaml.safe_dump(dict(standard, save_model="active", save_eval_multiplier=2)))
        self.assertEqual(launcher.read_config(temporary, "opd")["save_interval"], 50)

    def test_final_and_walltime_saves_are_synchronous_and_preserve_update_cursor(self):
        function = next(item for item in ast.parse((ROOT / "train.py").read_text()).body
                        if isinstance(item, ast.FunctionDef) and item.name == "train")
        for lightning in (False, True):
            for stop_at in (None, 2, 4, 6):
                with self.subTest(lightning=lightning, stop_at=stop_at):
                    events, state = [], {"step": 0}
                    args = types.SimpleNamespace(
                        num_rollout=6, start_rollout_id=0, eval_interval=2, eval_on_resume=False,
                        offload_train=False, offload_rollout=False, enable_weights_backuper=False,
                        use_critic=False, rollout_global_dataset=True, rollout_batch_size=2,
                        save_interval=6, save_final_only=True, update_weights_interval=1,
                        checkpoint_deadline=10 if stop_at else None, save=str(self.run / "checkpoints"),
                        lightning_block_config="unused" if lightning else None,
                    )
                    def event(name, value=None):
                        events.append((name, value))
                    def update(index, data):
                        state["step"] = index + 1
                        event("train", index)
                    def save(index, force_sync=False):
                        self.assertTrue(force_sync)
                        event("save", index)
                    actor = types.SimpleNamespace(async_train=update, save_model=save,
                                                  update_weights=lambda: None, clear_memory=lambda: None)
                    manager = types.SimpleNamespace(
                        generate=Remote(lambda index: None),
                        eval=Remote(lambda index=None, **kw: event("eval", kw.get("rollout_id", index))),
                        save=Remote(lambda index: event("cursor", index)),
                        dispose=Remote(lambda: event("dispose")),
                    )
                    coordinator = types.SimpleNamespace(ensure_block=lambda index: None)
                    ray = types.SimpleNamespace(get=lambda value: value)
                    namespace = dict(ray=ray, GPU_MEMORY_TYPE_WEIGHTS="weights", GPU_MEMORY_TYPE_KV_CACHE="kv",
                                     GPU_MEMORY_TYPE_CUDA_GRAPH=None, configure_logger=lambda: None,
                                     init_tracking=lambda args: None, create_placement_groups=lambda args: {"rollout": None},
                                     create_rollout_manager=lambda *args: (manager, 2),
                                     create_training_models=lambda *args: (actor, None))
                    with patch.object(blocks, "BlockCoordinator", return_value=coordinator), \
                            patch("time.time", side_effect=lambda: 20 if stop_at and state["step"] >= stop_at else 0):
                        exec(compile(ast.Module(body=[function], type_ignores=[]), "train.py", "exec"), namespace)
                        namespace["train"](args)
                    completed = stop_at or 6
                    self.assertEqual([v for n, v in events if n == "train"], list(range(completed)))
                    self.assertEqual([v for n, v in events if n == "save"], [completed - 1])
                    self.assertEqual([v for n, v in events if n == "cursor"], [completed - 1])
                    self.assertEqual([v for n, v in events if n == "eval"],
                                     [0] + [i for i in (1, 3, 5) if i < completed - bool(stop_at)])
                    self.assertEqual(events[-1], ("dispose", None))
                    status = json.loads((self.run / "training_status.json").read_text())
                    self.assertEqual(status, dict(status="paused" if stop_at else "complete",
                                                  global_step=completed, target_steps=6))

    def test_fixed_sampler_checks_absolute_batch_positions(self):
        rows = [types.SimpleNamespace(metadata=dict(lightning_sample_index=i, policy_block_index=1)) for i in (6, 7)]
        data_source = types.SimpleNamespace(get_samples=lambda size: [[row] for row in rows])
        args = types.SimpleNamespace(rollout_batch_size=2)
        fake_rollout = types.SimpleNamespace(GenerateState=lambda args: None,
                                            _handle_lightning_opd_sample=lambda sample, state: sample)
        fake_output = types.SimpleNamespace(RolloutFnTrainOutput=lambda **kwargs: types.SimpleNamespace(**kwargs))
        with patch.dict(sys.modules, {"slime.rollout.sglang_rollout": fake_rollout,
                                      "slime.rollout.base_types": fake_output}):
            output = blocks.generate_cached_rollout(args, 3, data_source)
            self.assertEqual(output.metrics["lightning/data_position"], 8)
            self.assertTrue(all(row.reward == {"lightning_opd": True} for row in rows))
            rows.reverse()
            with self.assertRaisesRegex(ValueError, "absolute"):
                blocks.generate_cached_rollout(args, 3, data_source)

    def test_eval_cleanup_and_resume_use_absolute_steps(self):
        spec = importlib.util.spec_from_file_location("eval_block_launcher", ROOT / "configs/opd/qwen3-4b-opd.py")
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        self.cfg.update(num_rollout=150, rollout_batch_size=256, global_batch_size=256, eval_steps=25)
        self.cfg["resume"] = True
        for k in (2, 4):
            with self.subTest(k=k):
                self.cfg.update(policy_rollout_k=k, experiment_name=f"eval-k{k}")
                run = blocks.run_directory(self.cfg)
                run.mkdir()
                self.cfg["save_interval"] = blocks.schedule(self.cfg)[0]["end_step"]
                (run / "config.yaml").write_text(yaml.safe_dump(self.cfg))
                latest = self.cfg["save_interval"] - 1
                checkpoint = run / "checkpoints"
                weights = checkpoint / f"iter_{latest:07d}"
                weights.mkdir(parents=True)
                (weights / "weights.pt").write_bytes(b"checkpoint")
                (checkpoint / "latest_checkpointed_iteration.txt").write_text(str(latest))
                (checkpoint / "rollout").mkdir()
                (checkpoint / "rollout" / f"global_dataset_state_dict_{latest}.pt").write_bytes(b"cursor")
                args = types.SimpleNamespace(save=str(checkpoint), eval_output_dir=str(run / "eval_metrics"),
                                             eval_datasets=[], rollout_global_dataset=True)
                fake_rollout = types.SimpleNamespace(GenerateState=lambda args: None,
                                                     generate_rollout=lambda *a, **kw: types.SimpleNamespace(metrics={}, data={}))
                with patch.dict(sys.modules, {"slime.rollout.sglang_rollout": fake_rollout}):
                    for index in (0, 24, 49, 74, 99, 124, 149):
                        launcher.evaluate(args, index, None)
                self.assertTrue(weights.is_dir())
                flags = launcher.training_args(self.cfg, "student", None, "/tmp/eval.yaml")
                self.assertNotIn("--eval-on-resume", flags)
                if k == 2:
                    (run / "eval_metrics/74.json").unlink()
                    flags = launcher.training_args(self.cfg, "student", None, "/tmp/eval.yaml")
                    self.assertIn("--eval-on-resume", flags)
                launcher.check_run_state(self.cfg, "student", "teacher")
                with self.assertRaisesRegex(ValueError, "recipe changed"):
                    launcher.check_run_state(dict(self.cfg, policy_rollout_k=k + 1), "student", "teacher")


if __name__ == "__main__":
    unittest.main()
