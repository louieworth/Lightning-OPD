"""Initial student-teacher top-K overlap, as defined in "When Do We Need On-Policy Distillation?"
(arXiv 2610.11291, Sec. 2.2): on the initial student's own rollouts, at every response position t,
Overlap@K(t) = |TopK(pi_student(.|s_t)) ∩ TopK(pi_teacher(.|s_t))| / K, aggregated as the token-weighted
mean over all response tokens. Temperature does not change top-K sets, so raw logits are used.

    python3 tools/student_teacher_overlap.py --student /opt/tiger/models/Qwen3-1.7B-Base \
        --teacher /opt/tiger/models/Qwen3-4B-Instruct-2507 \
        --rollouts outputs/<exp>/data/blocks/block_0000/student_rollouts.parquet [--n 64] [--k 64]

Runs on CPU in fp32 by default (--device cuda --dtype bfloat16 for a GPU); prints one JSON line with the
overlap and per-rollout spread.
"""

import argparse
import json
import math
import time

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def top_k(model, ids, start, k):
    """Top-k token ids of the next-token distribution at every position predicting ids[start:]."""
    with torch.inference_mode():
        logits = model(torch.tensor([ids], device=model.device)).logits[0, start - 1:len(ids) - 1]
    return torch.topk(logits.float(), k, dim=-1).indices


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--student", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--rollouts", required=True, help="initial-student rollouts parquet (prompt, metadata.response_tokens)")
    parser.add_argument("--n", type=int, default=64, help="rollouts to use (the first n, fixed order)")
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--max-response", type=int, default=None, help="optional cap on response tokens per rollout")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--label", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    tokenizer = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    load = dict(torch_dtype=getattr(torch, args.dtype), trust_remote_code=True, attn_implementation="sdpa")
    student = AutoModelForCausalLM.from_pretrained(args.student, **load).to(args.device).eval()
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, **load).to(args.device).eval()

    rows = pq.read_table(args.rollouts, columns=["prompt", "metadata"]).slice(0, args.n).to_pylist()
    hits = tokens = 0
    per_rollout = []
    started = time.time()
    for row in rows:
        metadata = row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"])
        response = list(metadata["response_tokens"])[:args.max_response]
        prompt = tokenizer.encode(row["prompt"], add_special_tokens=False)
        ids = prompt + response
        s, t = top_k(student, ids, len(prompt), args.k), top_k(teacher, ids, len(prompt), args.k)
        # |S ∩ T| per position: compare each student id against the teacher's k ids.
        shared = (s.unsqueeze(-1) == t.unsqueeze(-2)).any(-1).sum(-1).cpu()
        hits += int(shared.sum())
        tokens += len(response)
        per_rollout.append(float(shared.sum()) / (args.k * len(response)))
    overlap = hits / (args.k * tokens)
    mean = sum(per_rollout) / len(per_rollout)
    spread = math.sqrt(sum((x - mean) ** 2 for x in per_rollout) / max(1, len(per_rollout) - 1))
    print(json.dumps(dict(label=args.label, student=args.student, teacher=args.teacher, rollouts=len(rows),
                          response_tokens=tokens, k=args.k, overlap_pct=round(100 * overlap, 2),
                          per_rollout_mean_pct=round(100 * mean, 2), per_rollout_std_pct=round(100 * spread, 2),
                          dtype=args.dtype, seconds=round(time.time() - started))))


if __name__ == "__main__":
    main()
