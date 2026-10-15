"""Chunked OPD distribution loss on the training student's current logits."""

from functools import lru_cache

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from megatron.core import mpu

from slime.utils.opd_config import option
from slime.utils.opd_loss import gather_vocabulary, reverse_kl_loss


_GPU_HEADS = {}


def clear_teacher_head_cache():
    # Actor offload must also release the frozen teacher output-head shard.
    _GPU_HEADS.clear()


@lru_cache(maxsize=2)
def _cpu_head(path):
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def _head_shard(reference, width, device):
    rank = mpu.get_tensor_model_parallel_rank()
    start, end = rank * width, (rank + 1) * width
    key = (reference["head_path"], start, end, str(device))
    if key not in _GPU_HEADS:
        head = _cpu_head(reference["head_path"])
        if head["weight"].shape[0] != reference["vocab_size"]:
            raise ValueError("Teacher output-head vocabulary does not match the feature metadata.")
        weight = head["weight"][start:end].to(device)
        bias = None if head["bias"] is None else head["bias"][start:end].to(device)
        _GPU_HEADS[key] = weight, bias
    return _GPU_HEADS[key]


def compute_distribution_losses(args, batch, logits, response_iterator):
    """Return response-aligned losses; existing reducers apply masks/weights.

    Teacher hidden states and head are frozen. TP vocabulary gathers are bounded by
    opd_distribution_chunk_size; checkpointing prevents retaining dense probabilities
    for every token until backward. Padded Megatron vocabulary entries are excluded.
    """
    references = batch.get("teacher_features")
    if not references or any(reference is None for reference in references):
        raise ValueError("OPD distribution loss requires teacher feature caches; regenerate teacher scoring.")
    mode = option(args, "opd_loss_token_mode", "sample")
    top_k = option(args, "opd_token_top_k", 32)
    chunk_size = option(args, "opd_distribution_chunk_size", 32)
    maximum = option(args, "opd_loss_max_clip", 10.0)
    group = mpu.get_tensor_model_parallel_group()
    losses = []
    for (student_chunk, response_tokens), reference in zip(response_iterator, references, strict=True):
        vocab_size = reference["vocab_size"]
        if vocab_size > student_chunk.size(-1) * mpu.get_tensor_model_parallel_world_size():
            raise ValueError("Teacher vocabulary is larger than the student vocabulary.")
        if top_k > vocab_size:
            raise ValueError("opd_token_top_k exceeds the teacher/student vocabulary size.")
        artifact = torch.load(reference["path"], map_location="cpu", weights_only=True, mmap=True)
        hidden = artifact["hidden_states"]
        if hidden.size(0) != len(response_tokens) or not torch.equal(artifact["response_tokens"], response_tokens.cpu()):
            raise ValueError("Teacher features are misaligned with the training response tokens.")
        weight, bias = _head_shard(reference, student_chunk.size(-1), logits.device)

        # Bind every loop value: backward recomputation must not use the last sample's head/reference.
        def run_chunk(student, features, weight=weight, bias=bias, vocab_size=vocab_size):
            teacher = F.linear(features.to(weight.dtype), weight, bias).float()
            teacher = F.pad(teacher, (0, student.size(-1) - teacher.size(-1)), value=float("-inf"))
            teacher = gather_vocabulary(teacher, group)[..., :vocab_size]
            student = gather_vocabulary(student, group)[..., :vocab_size]
            return reverse_kl_loss(student, teacher, mode, top_k, maximum)

        for start in range(0, len(response_tokens), chunk_size):
            features = hidden[start : start + chunk_size].to(logits.device)
            student = student_chunk[start : start + chunk_size] / args.rollout_temperature
            losses.append(checkpoint(run_chunk, student, features, use_reentrant=False))
    # Empty CP-local responses are not allowed here (CP=1); masked responses are still represented.
    return torch.cat(losses)
