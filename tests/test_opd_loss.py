"""Numerical checks for OPD supports, gradients and tensor parallelism."""

import ast
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from typing import Iterator
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import yaml

from slime.utils.opd_config import needs_teacher_features, validate_opd_options
from slime.utils.opd_loss import (
    gather_vocabulary, reverse_kl_loss, sampled_advantage,
)

ROOT = Path(__file__).resolve().parents[1]


def tp_worker(rank, rendezvous, output_dir):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2)
    try:
        for mode in ("top_k_student", "top_k_teacher", "full_vocab"):
            full_student = torch.tensor([[0.2, 2.0, 1.0, -1.0, 0.5, 3.0]])
            full_teacher = torch.tensor([[1.0, 0.1, 2.0, 0.0, 2.5, -1.0]])
            shard = full_student.chunk(2, -1)[rank].clone().requires_grad_()
            teacher_shard = full_teacher.chunk(2, -1)[rank].clone()
            gathered = gather_vocabulary(shard)
            teacher = gather_vocabulary(teacher_shard)
            loss = reverse_kl_loss(gathered, teacher, mode, top_k=3, maximum=None).sum()
            loss.backward()
            torch.save({"loss": loss.detach(), "grad": shard.grad},
                       Path(output_dir) / f"{mode}_{rank}.pt")
    finally:
        dist.destroy_process_group()


class LossTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.student = torch.tensor([[2.0, 0.5, -1.0, 1.0, 0.0], [-0.5, 0.0, 2.0, 1.0, 0.5]])
        self.teacher = torch.tensor([[-1.0, 1.0, 0.5, 2.0, 0.0], [2.0, 0.5, -0.5, 0.0, 1.0]])

    def test_reverse_kl_matches_independent_distribution_reference_and_gradient(self):
        for mode in ("full_vocab", "top_k_teacher", "top_k_student"):
            with self.subTest(mode=mode):
                student = self.student.clone().requires_grad_()
                teacher = self.teacher.clone().requires_grad_()
                actual = reverse_kl_loss(student, teacher, mode, top_k=3, maximum=None)
                reference_student = self.student.clone().requires_grad_()
                if mode == "full_vocab":
                    s, t = reference_student, self.teacher
                else:
                    source = self.teacher if mode == "top_k_teacher" else self.student
                    ids = source.topk(3, -1).indices
                    s, t = reference_student.gather(-1, ids), self.teacher.gather(-1, ids)
                expected = F.kl_div(t.log_softmax(-1), s.log_softmax(-1), reduction="none", log_target=True).sum(-1)
                torch.testing.assert_close(actual, expected)
                actual.sum().backward()
                expected.sum().backward()
                torch.testing.assert_close(student.grad, reference_student.grad)
                self.assertIsNone(teacher.grad)

    def test_topk_renormalization_and_offset_invariance(self):
        for mode in ("top_k_teacher", "top_k_student"):
            loss = reverse_kl_loss(self.student, self.teacher, mode, 3, None)
            shifted = reverse_kl_loss(self.student + 200, self.teacher - 200, mode, 3, None)
            torch.testing.assert_close(loss, shifted)
            # K=V recovers full-vocabulary KL, proving the normalization covers the selected support.
            torch.testing.assert_close(reverse_kl_loss(self.student, self.teacher, mode, 5, None),
                                       reverse_kl_loss(self.student, self.teacher, "full_vocab", 5, None))

    def test_sampled_signal_and_stability_clips(self):
        teacher, student = torch.tensor([-2., -100., -1.]), torch.tensor([-1., -1., -100.])
        torch.testing.assert_close(sampled_advantage(teacher, student, None, None), teacher - student)
        torch.testing.assert_close(sampled_advantage(teacher, student, -10., 3.), torch.tensor([-1., -3., 3.]))
        s, t = torch.tensor([[100., -100.]], requires_grad=True), torch.tensor([[-100., 100.]])
        clipped = reverse_kl_loss(s, t, "full_vocab", 2, 10.)
        torch.testing.assert_close(clipped, torch.tensor([10.]))
        clipped.sum().backward()
        self.assertTrue(torch.isfinite(s.grad).all())

    def test_equal_models_zero_kl(self):
        equal = torch.zeros(3, 4)
        for mode in ("top_k_student", "top_k_teacher", "full_vocab"):
            torch.testing.assert_close(reverse_kl_loss(equal, equal, mode, 2), torch.zeros(3))

    def test_distributed_vocabulary_gradients_are_not_duplicated(self):
        with tempfile.TemporaryDirectory(prefix="opd-tp-test-") as directory:
            mp.spawn(tp_worker, args=(str(Path(directory) / "rendezvous"), directory), nprocs=2, join=True)
            for mode in ("top_k_student", "top_k_teacher", "full_vocab"):
                student = torch.tensor([[0.2, 2.0, 1.0, -1.0, 0.5, 3.0]], requires_grad=True)
                teacher = torch.tensor([[1.0, 0.1, 2.0, 0.0, 2.5, -1.0]])
                loss = reverse_kl_loss(student, teacher, mode, 3, None).sum()
                loss.backward()
                for rank in range(2):
                    actual = torch.load(Path(directory) / f"{mode}_{rank}.pt", weights_only=True)
                    torch.testing.assert_close(actual["loss"], loss.detach())
                    torch.testing.assert_close(actual["grad"], student.grad.chunk(2, -1)[rank])


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_response_logits_are_causally_aligned_without_a_full_size_temperature_copy(self):
        tree = ast.parse((ROOT / "slime/backends/megatron_utils/loss.py").read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_responses")
        namespace = dict(torch=torch, Namespace=types.SimpleNamespace, Iterator=Iterator,
                         mpu=types.SimpleNamespace(get_context_parallel_world_size=lambda: 1))
        exec(compile(ast.Module(body=[function], type_ignores=[]), "loss.py", "exec"), namespace)
        logits = torch.arange(45, dtype=torch.float32).reshape(1, 9, 5)
        tokens = [torch.tensor([4, 4, 1, 2, 0]), torch.tensor([4, 4, 3, 1])]
        responses = list(namespace["get_responses"](
            logits, args=types.SimpleNamespace(rollout_temperature=1.), unconcat_tokens=tokens,
            total_lengths=[5, 4], response_lengths=[3, 2]))
        torch.testing.assert_close(responses[0][0], logits[0, 1:4])
        torch.testing.assert_close(responses[1][0], logits[0, 6:8])
        torch.testing.assert_close(responses[0][1], tokens[0][-3:])
        for chunk, _ in responses:
            self.assertEqual(chunk.untyped_storage().data_ptr(), logits.untyped_storage().data_ptr())

    def test_managed_feature_teacher_uses_causal_next_token_alignment_and_reuses_cache(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from slime.rollout.opd_teacher import TeacherScorer, atomic_save

        torch.manual_seed(3)
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=16, hidden_size=16, intermediate_size=32,
                                           num_hidden_layers=1, num_attention_heads=2,
                                           num_key_value_heads=1, head_dim=8)).eval()
        with tempfile.TemporaryDirectory(prefix="opd-teacher-test-") as directory:
            scorer = TeacherScorer.__new__(TeacherScorer)
            scorer.model, scorer.head, scorer.root = model, model.get_output_embeddings(), Path(directory)
            scorer.chunk_size, scorer.context_length = 2, 32
            scorer.head_path = Path(directory) / "output_head.pt"
            atomic_save({"weight": scorer.head.weight.detach().cpu(), "bias": None}, scorer.head_path)
            ids = [1, 3, 4, 5, 2]
            output = scorer.score({"input_ids": ids, "opd_response_length": 3})
            actual = output["meta_info"]["input_token_logprobs"][-3:]
            with torch.no_grad():
                logits = model(torch.tensor([ids]), use_cache=False).logits[0, 1:4]
                expected = logits.log_softmax(-1).gather(-1, torch.tensor(ids[-3:]).unsqueeze(-1)).squeeze(-1)
            torch.testing.assert_close(torch.tensor([item[0] for item in actual]), expected)
            features = output["meta_info"]["opd_teacher_features"]
            artifact = torch.load(features["path"], weights_only=True, mmap=True)
            reconstructed = F.linear(artifact["hidden_states"], scorer.head.weight)
            torch.testing.assert_close(reconstructed, logits)
            with patch.object(model.base_model, "forward", side_effect=AssertionError("cached features recomputed")):
                self.assertEqual(scorer.score({"input_ids": ids, "opd_response_length": 3}), output)

    def test_chunked_training_reads_features_excludes_padding_and_recomputes_correct_gradients(self):
        mpu = types.SimpleNamespace(get_tensor_model_parallel_rank=lambda: 0,
                                    get_tensor_model_parallel_world_size=lambda: 1,
                                    get_tensor_model_parallel_group=lambda: None)
        with patch.dict(sys.modules, {"megatron": types.ModuleType("megatron"),
                                      "megatron.core": types.SimpleNamespace(mpu=mpu)}):
            spec = importlib.util.spec_from_file_location("opd_backend_test", ROOT / "slime/backends/megatron_utils/opd.py")
            backend = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(backend)
        with tempfile.TemporaryDirectory(prefix="opd-feature-test-") as directory:
            head_path = Path(directory) / "head.pt"
            weight = torch.tensor([[1., 0.], [0., 2.], [1., 1.], [-1., 0.]])
            torch.save({"weight": weight, "bias": None}, head_path)
            hidden_list = [torch.tensor([[1., .5], [-1., 1.], [.5, .5]]), torch.tensor([[2., -1.], [0., .5]])]
            tokens_list = [torch.tensor([1, 2, 0]), torch.tensor([3, 1])]
            references = []
            for index, (hidden, tokens) in enumerate(zip(hidden_list, tokens_list, strict=True)):
                path = Path(directory) / f"features{index}.pt"
                torch.save({"hidden_states": hidden, "response_tokens": tokens}, path)
                references.append(dict(path=str(path), head_path=str(head_path), vocab_size=4))
            for mode in ("top_k_student", "top_k_teacher", "full_vocab"):
                backend.clear_teacher_head_cache()
                args = types.SimpleNamespace(opd_loss_token_mode=mode, opd_token_top_k=2,
                                             opd_distribution_chunk_size=2, opd_loss_max_clip=None, rollout_temperature=.8)
                shards = [torch.tensor([[1., .2, .5, -.5, 999.]] * len(tokens), requires_grad=True) for tokens in tokens_list]
                responses = iter(zip(shards, tokens_list, strict=True))
                losses = backend.compute_distribution_losses(args, dict(teacher_features=references), shards[0], responses)
                teachers = torch.cat([F.linear(hidden, weight) for hidden in hidden_list])
                students = torch.cat(shards)[..., :4] / .8
                reference = reverse_kl_loss(students, teachers, mode, 2, None)
                torch.testing.assert_close(losses, reference)
                losses.sum().backward()
                # The padded high logit must have zero gradient and never affect selection/normalization.
                for shard in shards:
                    torch.testing.assert_close(shard.grad[..., -1], torch.zeros(len(shard)))
                gradients = [shard.grad.clone() for shard in shards]
                for shard in shards:
                    shard.grad = None
                reference.sum().backward()
                for actual, shard in zip(gradients, shards, strict=True):
                    torch.testing.assert_close(actual, shard.grad)
            bad = dict(references[0], response_length=3)
            path = Path(directory) / "bad.pt"
            torch.save({"hidden_states": hidden_list[0], "response_tokens": torch.tensor([0, 0, 0])}, path)
            bad["path"] = str(path)
            with self.assertRaisesRegex(ValueError, "misaligned"):
                backend.compute_distribution_losses(args, dict(teacher_features=[bad]), shards[0], iter([(shards[0], tokens_list[0])]))

    def test_online_and_offline_feature_metadata_reaches_reward_postprocessor(self):
        # Import without the multimodal/tokenizer stack; the real postprocessor is executed.
        processing = types.SimpleNamespace(encode_image_for_rollout_engine=lambda value: value)
        with patch.dict(sys.modules, {"slime.utils.processing_utils": processing}):
            spec = importlib.util.spec_from_file_location("opd_rewards_test", ROOT / "slime/rollout/on_policy_distillation.py")
            rewards = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(rewards)
        args = types.SimpleNamespace(opd_loss_token_mode="top_k_teacher")
        features = dict(path="/shared/features.pt", head_path="/shared/head.pt", vocab_size=8)
        offline = types.SimpleNamespace(metadata=dict(teacher_log_probs=[-.1, -.2], teacher_features=features),
                                        response_length=2, get_reward_value=lambda args: {"lightning_opd": True})
        online = types.SimpleNamespace(metadata={}, response_length=2, get_reward_value=lambda args: {
            "meta_info": dict(input_token_logprobs=[[None, 1, ""], [None, 2, ""], [-.1, 3, ""], [-.2, 4, ""]],
                              opd_teacher_features=features)})
        rewards.post_process_rewards(args, [offline, online])
        for sample in (offline, online):
            self.assertEqual(sample.teacher_features, features)
            torch.testing.assert_close(sample.teacher_log_probs, torch.tensor([-.1, -.2]))
        offline.metadata.pop("teacher_features")
        with self.assertRaisesRegex(ValueError, "missing"):
            rewards.post_process_rewards(args, [offline])

    def test_default_configs_cli_and_invalid_options(self):
        spec = importlib.util.spec_from_file_location("opd_launcher_test", ROOT / "configs/opd/qwen3-4b-opd.py")
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        expected_options = dict(opd_loss_token_mode="sample", opd_token_top_k=32,
                                opd_distribution_chunk_size=32,
                                opd_loss_max_clip=10., opd_log_prob_min_clip=-10.)
        for path in (ROOT / "configs/config").glob("*opd*.yaml"):
            cfg = launcher.read_config(path, yaml.safe_load(path.read_text())["method"])
            self.assertEqual({key: cfg[key] for key in expected_options}, expected_options)
            # Both launchers share the same loss defaults.
            with tempfile.TemporaryDirectory(prefix="opd-defaults-test-") as directory:
                missing_options = {key: value for key, value in cfg.items() if key not in expected_options}
                missing_path = Path(directory) / "config.yaml"
                missing_path.write_text(yaml.safe_dump(missing_options))
                defaults = launcher.read_config(missing_path, cfg["method"])
                self.assertEqual({key: defaults[key] for key in expected_options}, expected_options)
            flags = launcher.training_args(cfg, "student", None)
            self.assertEqual(flags[flags.index("--opd-token-top-k") + 1], "32")
            self.assertEqual(flags[flags.index("--opd-loss-token-mode") + 1], "sample")
            teacher = launcher.server_command(cfg, "teacher", 1, 2, .6, True)
            student = launcher.server_command(cfg, "student", 1, 2, .4)
            self.assertIn("sglang.launch_server", teacher)
            self.assertIn("sglang.launch_server", student)
            self.assertFalse(needs_teacher_features(cfg))
        self.assertFalse(needs_teacher_features({}))
        self.assertTrue(needs_teacher_features({"opd_loss_token_mode": "top_k_teacher"}))
        for invalid in ({"opd_token_top_k": 0}, {"opd_loss_token_mode": "topk"}, {"opd_token_top_k": True},
                        {"opd_loss_max_clip": -1}, {"opd_log_prob_min_clip": 1},
                        {"opd_loss_token_mode": "top_k_teacher", "context_parallel_size": 2},
                        {"opd_loss_token_mode": "full_vocab", "train_backend": "fsdp"},
                        {"opd_loss_max_clip": True}, {"opd_log_prob_min_clip": "-10"}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_opd_options(invalid)


if __name__ == "__main__":
    unittest.main()
