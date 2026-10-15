# OPD loss token selection

The YAML launchers support the following options. All checked-in OPD YAML recipes
default to `sample`; `policy_rollout_k` retains its existing meaning (number of
policy rollout blocks).
Standard OPD and Lightning OPD share the loss defaults and implementation.

```yaml
opd_loss_token_mode: sample  # sample | top_k_student | top_k_teacher | full_vocab
opd_token_top_k: 32          # Vocabulary candidates, independent of policy_rollout_k.
opd_distribution_chunk_size: 32
opd_loss_max_clip: 10.0      # null disables clipping.
opd_log_prob_min_clip: -10.0 # sample only; null disables the logprob floor.
```

## Objectives

All modes use the student's generated prefix, and all distillation directions are
reverse KL (student || teacher). Student probabilities use `rollout_temperature`;
teacher scoring uses its untempered distribution (temperature 1).

* `sample`: retain the sampled-token PPO loss. At each response position,
  `advantage = log p_teacher(y_t) - log p_student(y_t)`. If configured, floor both
  logprobs before subtraction, then clip the advantage symmetrically. The existing
  PPO ratio clipping (`eps_clip`, default 0.2) and gradient clipping remain active.
* `top_k_student`: select the K highest-probability **student** tokens at the prefix.
* `top_k_teacher`: select the K highest-probability **teacher** tokens at the prefix.
* `full_vocab`: use every real vocabulary entry, excluding Megatron padding.

Both top-k modes gather teacher **and** student logits at the same selected token
IDs, independently apply log-softmax **within that set**, and directly optimize
`sum(p_student_in_set * (log p_student_in_set - log p_teacher_in_set))`.
This is the support renormalization from
[Revisiting On-Policy Distillation, equations (6)–(8)](https://arxiv.org/html/2603.25562v1#S4.SS1).
The only difference between the two top-k modes is who selects the set.
`full_vocab` applies the same formula to the complete vocabulary.

Distribution modes use direct KL gradients, rather than a PPO surrogate. Their
per-position KL is capped by `opd_loss_max_clip`; a capped position has zero
distillation gradient. The sampled-token logprob floor is deliberately **not**
applied to these probability distributions: float32 log-softmax keeps their
normalization and reverse-KL definition intact.

The settings are stability controls inspired by the local verl baseline; that
baseline's `forward_kl_topk` is a different loss direction. Selecting a top-k mode
does not automatically change rollout top-p, learning rate, or special-token masks.

## Teacher data and compatibility

Sample-only training needs just scalar teacher logprobs and can reuse the original
caches. Any distribution loss requires more teacher information. Managed teacher
scoring automatically uses the feature scorer; student generation/evaluation still
use SGLang.

The scorer caches response-aligned final hidden states plus one frozen output head
under the experiment's `data/teacher_features/` directory. The head is memory mapped
on CPU and vocabulary-sharded on training GPUs. Logits/probabilities are reconstructed
in bounded chunks; activation checkpointing avoids retaining a full L × vocabulary
matrix through backward. Actor offload also clears the GPU head cache.

The feature scorer uses Hugging Face SDPA with `device_map="auto"` across the reserved
teacher GPUs, instead of SGLang tensor parallelism. Teacher and student must have
identical token IDs/vocabulary sizes. This implementation currently supports Megatron, text
and `context_parallel_size: 1`; student tensor/data/pipeline parallelism remains supported.
An external teacher URL must implement the feature scorer's `opd_response_length`
request and `meta_info.opd_teacher_features` response protocol, with cache paths on
the filesystem shared by every training node.

BF16 feature storage is approximately `2 * response_tokens * teacher_hidden_size`
bytes, plus the shared head. This is much smaller than dense vocabulary caches but
larger than scalar-only caches. Distribution losses also add head reconstruction
and, for top-k modes, token selection work.

Scored block manifests distinguish scalar-only and feature caches. A resumed block
can be upgraded by scoring its **saved responses**, preserving its original rollout
policy and data cursor. No new student rollout is needed for that upgrade. Legacy
non-block offline data without features needs re-scoring into a new prepared dataset.
Changing the loss token mode, or K for a distribution loss, requires a new experiment
name.
Do not remove feature files while their scored dataset/checkpoint is still needed.

The raw CLI defaults preserve sample-only operation without clipping.
Both YAML launchers default to the loss settings and stability bounds shown above.
With `sample`, standard OPD and Lightning OPD use scalar-only teacher scoring
without feature caching. No running jobs are restarted or re-submitted by these
options.
