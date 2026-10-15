"""Reverse-KL objectives, independent of Megatron."""

import torch
import torch.distributed as dist


def sampled_advantage(teacher_log_probs, student_log_probs, minimum=-10.0, maximum=10.0):
    """Negative k1 reverse-KL estimate with optional stability clipping."""
    teacher, student = teacher_log_probs.float(), student_log_probs.float()
    if minimum is not None:
        teacher, student = teacher.clamp_min(minimum), student.clamp_min(minimum)
    advantage = teacher - student
    return advantage if maximum is None else advantage.clamp(-maximum, maximum)


def reverse_kl_loss(student_logits, teacher_logits, mode, top_k=32, maximum=10.0):
    """Per-position reverse KL. Both top-k modes renormalize on the SAME selected support.

    Teacher logits are fixed targets. Full-vocab uses every real vocabulary entry.
    Float32 log-softmax avoids probability underflow; no logprob floor distorts these distributions.
    """
    student, teacher = student_logits.float(), teacher_logits.detach().float()
    if student.shape != teacher.shape:
        raise ValueError("Teacher and student vocabulary logits must have the same shape.")
    if not 1 <= top_k <= student.size(-1):
        raise ValueError("opd_token_top_k must not exceed the real vocabulary size.")
    if mode in ("top_k_student", "top_k_teacher"):
        selector = student.detach() if mode == "top_k_student" else teacher
        ids = selector.topk(top_k, dim=-1).indices
        student, teacher = student.gather(-1, ids), teacher.gather(-1, ids)
    elif mode != "full_vocab":
        raise ValueError(f"Distribution reverse KL does not support mode {mode!r}.")
    log_student, log_teacher = student.log_softmax(-1), teacher.log_softmax(-1)
    losses = (log_student.exp() * (log_student - log_teacher)).sum(-1).clamp_min(0.0)
    return losses if maximum is None else losses.clamp_max(maximum)


class _GatherVocabulary(torch.autograd.Function):
    """Replicated TP loss: gather vocabulary in forward, split its gradient in backward.

    Each TP rank computes the same loss, so backward must NOT sum duplicate gradients.
    """

    @staticmethod
    def forward(ctx, logits, group):
        ctx.group = group
        ctx.size = dist.get_world_size(group) if dist.is_initialized() else 1
        ctx.rank = dist.get_rank(group) if ctx.size > 1 else 0
        if ctx.size == 1:
            return logits
        shards = [torch.empty_like(logits) for _ in range(ctx.size)]
        dist.all_gather(shards, logits.contiguous(), group=group)
        return torch.cat(shards, dim=-1)

    @staticmethod
    def backward(ctx, gradient):
        return gradient.chunk(ctx.size, dim=-1)[ctx.rank].contiguous(), None


def gather_vocabulary(logits, group=None):
    return _GatherVocabulary.apply(logits, group)
