"""Write the 1102-cluster presets (configs/config/byte16_*.yaml) from the Slurm presets.

    python3 tools/make_byte16_presets.py

Four teacher-student pairs from "When Do We Need On-Policy Distillation?" (arXiv 2610.11291,
Table 3 rows a, d, i, l), each as K=1 (Lightning OPD / Semi-OPD), K=40 (Lightning OPD with 40 current-policy
rollout blocks of 3 steps; user request 11/03) and K=120 (standard OPD).
Every training setting comes from the Qwen3-1.7B / Qwen3-4B-Instruct-2507 Slurm presets; only paths,
models, the 16-GPU layout and the user-approved evaluation and throughput settings change.
tests/test_byte16_presets.py enforces exactly that.
"""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs/config"
MODELS = Path("/opt/tiger/models")
DATA = ROOT / "data/lightning_opd"
SOURCES = {"k1": "qwen3_1_7b_nt_teacher_4b_instruct_2507_k1.yaml", "k40": "qwen3_1_7b_nt_teacher_4b_instruct_2507_k1.yaml",
           "k120": "qwen3_1_7b_opd_nt_teacher_4b_instruct_2507.yaml"}
PAIRS = {  # name -> (student, teacher, model_type, chat template kwargs)
    "qwen3_1_7b_nt_teacher_4b_instruct_2507": ("Qwen3-1.7B", "Qwen3-4B-Instruct-2507", "qwen3-1.7B", {"enable_thinking": False}),
    "qwen3_1_7b_nt_teacher_4b": ("Qwen3-1.7B", "Qwen3-4B", "qwen3-1.7B", {"enable_thinking": False}),
    # User request 11/04: the base student with A's teacher; the base Slurm presets differ from the
    # instruct ones only in model_path, so the same sources apply.
    "qwen3_1_7b_base_nt_teacher_4b_instruct_2507": (
        "Qwen3-1.7B-Base", "Qwen3-4B-Instruct-2507", "qwen3-1.7B", {"enable_thinking": False}),
    # User request 11/04: the 4B base student with the same teacher; qwen3_4b_base_nt_teacher_8b.yaml has
    # the same training settings as the 1.7B base presets, so the same sources apply.
    "qwen3_4b_base_nt_teacher_4b_instruct_2507": (
        "Qwen3-4B-Base", "Qwen3-4B-Instruct-2507", "qwen3-4B", {"enable_thinking": False}),
    # DeepSeek-R1-Distill templates always open <think>: thinking mode, no switch.
    "r1_distill_qwen_1_5b_teacher_justrl_1_5b": (
        "DeepSeek-R1-Distill-Qwen-1.5B", "JustRL-DeepSeek-1.5B", "deepseek-r1-distill-qwen-1.5B", {}),
    "r1_distill_qwen_7b_teacher_light_r1_14b": (
        "DeepSeek-R1-Distill-Qwen-7B", "Light-R1-14B-DS", "deepseek-r1-distill-qwen-7B", {}),
}
STEP_ZERO = {"qwen3_1_7b_nt_teacher_4b_instruct_2507", "r1_distill_qwen_1_5b_teacher_justrl_1_5b",
             "r1_distill_qwen_7b_teacher_light_r1_14b", "qwen3_1_7b_base_nt_teacher_4b_instruct_2507",
             "qwen3_4b_base_nt_teacher_4b_instruct_2507"}  # Once per student; pair B shares A's student.
COMMON = dict(
    hf_home=str(ROOT / ".runtime/hf"), output_dir=str(ROOT / "outputs"), ray_cluster="existing",
    nnodes=2, n_gpus_per_node=8, source_data_dir=str(ROOT / "data/source"),
    train_files=str(DATA / "train/dapo_math_17k_en.parquet"),
    val_files=[str(DATA / f"eval/{name}.parquet") for name in ("aime24", "aime25", "aime26", "hmmt26")],
    val_n=8, val_max_completion_length=32768,
    # User-approved 2026-11-02: generation throughput only (requests per SGLang engine, client threads).
    sglang_max_running_requests=64, sglang_server_concurrency=64, precompute_concurrency=1024,
)
# The source OPD preset's 2 : 4 : 2 split doubled, with its SGLang memory share 0.4 (user decision
# 11/02: rollout 10 / teacher TP2 / memory 0.85-0.7 sped up no training step of A k120).
DEDICATED = dict(actor_num_nodes=1, actor_num_gpus_per_node=4, colocate=False, rollout_num_gpus=8,
                 teacher_num_gpus=4, teacher_tp_size=4)
LAYOUT = {
    "k1": dict(actor_num_nodes=2, actor_num_gpus_per_node=8, colocate=True, rollout_num_gpus=16),
    # User decision 11/03: Lightning with K > 10 keeps every model resident on the K=120 split. The
    # colocated K=40 offloaded, onloaded and restarted models every 3 steps (16-GPU SM mean 6.3%) and
    # the cluster reclaimed it twice.
    "k40": dict(DEDICATED, policy_rollout_k=40),
    "k120": DEDICATED,
}
# User-approved 11/02: D k120's 7B actor (4 GPUs, TP2) hit a fragmentation OOM at step 3 (4.64 GiB of logits
# requested, 10.28 GiB reserved but unallocated). Allocator setting for its training processes only.
# D k40 trains the same 7B actor on the same 4 GPUs.
# The 4B base actor also trains on those 4 GPUs (allocator setting only, as for D).
MEMORY = {(pair, k): dict(train_env_vars={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
          for pair in ("r1_distill_qwen_7b_teacher_light_r1_14b", "qwen3_4b_base_nt_teacher_4b_instruct_2507")
          for k in ("k40", "k120")}
# User-approved 11/04: Qwen3-1.7B-Base's config allows 32768 positions, fewer than a 4096-token prompt plus the
# 32768-token evaluation (its step-0 evaluation failed). Its engines use Qwen3-1.7B's 40960 (same RoPE base).
CONTEXT = {pair: dict(sglang_context_length=40960) for pair in
           ("qwen3_1_7b_base_nt_teacher_4b_instruct_2507", "qwen3_4b_base_nt_teacher_4b_instruct_2507")}
# User-approved 11/04 for the 4B base pair (and the resumed 1.7B base K=120): engines decoded mostly <=16 sequences and hit the 64-request
# cap with full KV cache during evaluations. Up to 256 running requests per engine; dedicated rollout GPUs
# (K=40/K=120) also get SGLang memory 0.7 (0.85 ran out of activation memory on 11/02). Colocated K=1
# shares its GPUs with the actor and keeps 0.4.
THROUGHPUT = {("qwen3_4b_base_nt_teacher_4b_instruct_2507", k): dict(
    sglang_max_running_requests=256, sglang_server_concurrency=256, precompute_concurrency=4096,
    **({} if k == "k1" else dict(sglang_mem_fraction_static=0.7))) for k in ("k1", "k40", "k120")}
THROUGHPUT[("qwen3_1_7b_base_nt_teacher_4b_instruct_2507", "k120")] = THROUGHPUT[("qwen3_4b_base_nt_teacher_4b_instruct_2507", "k120")]
HEADER = ("# 1102 cluster (2 pods x 8 H100, pre-started Ray, no shared filesystem); generated by\n"
          "# tools/make_byte16_presets.py from {source}. K={k} ({method}). Changed: paths, models,\n"
          "# 16-GPU layout, eval = AIME24/25/26 + HMMT Feb26, avg@8, 32k, step-0 eval once per student\n"
          "# (user-approved items 7-10); generation throughput 64 requests/engine, 1024 client threads\n"
          "# (user-approved 11/02); K=40/K=120: the OPD GPU split doubled (actor 4, rollout 8, teacher TP4).\n"
          "# All training settings are the source preset's.\n")


def scalar(value):
    return yaml.safe_dump(value, default_flow_style=True).strip().removesuffix("\n...")


def override(text, values):
    """Replace top-level `key: value` lines (and list blocks), keeping every other line and comment."""
    out, skip, done = [], False, set()
    for line in text.splitlines():
        if skip:
            if line.startswith("  - "):
                continue
            skip = False
        match = re.match(r"^([a-z_0-9]+):", line)
        key = match.group(1) if match else None
        if key in values:
            done.add(key)
            if isinstance(values[key], list):
                out.append(f"{key}:")
                out.extend(f"  - {item}" for item in values[key])
                skip = True
            else:
                out.append(f"{key}: {scalar(values[key])}")
            continue
        out.append(line)
    missing = [key for key in values if key not in done]
    if missing:
        out.append("# Keys added for the 1102 cluster.")
        out.extend(f"{key}: {scalar(values[key])}" for key in missing)
    return "\n".join(out) + "\n"


def main():
    written = []
    for pair, (student, teacher, model_type, chat) in PAIRS.items():
        for k, source in SOURCES.items():
            text = (CONFIGS / source).read_text()
            name = f"{pair}_{k}_byte16"
            values = dict(COMMON, **LAYOUT[k], model_path=str(MODELS / student), teacher_model_path=str(MODELS / teacher),
                          model_type=model_type, apply_chat_template_kwargs=chat, experiment_name=name,
                          val_before_train=k == "k1" and pair in STEP_ZERO, **MEMORY.get((pair, k), {}),
                          **CONTEXT.get(pair, {}))
            values.update(THROUGHPUT.get((pair, k), {}))
            if "wandb_dir" in yaml.safe_load(text):
                values["wandb_dir"] = f"wandb/{name}"
            target = CONFIGS / f"byte16_{pair}_{k}.yaml"
            method = {"k1": "Lightning OPD / Semi-OPD", "k40": "Lightning OPD, 40 rollout blocks"}.get(k, "standard OPD")
            target.write_text(HEADER.format(source=source, k=k[1:], method=method) + override(text, values))
            cfg = yaml.safe_load(target.read_text())
            assert all(cfg[key] == value for key, value in values.items()), target
            written.append(target.name)
    for stale in sorted(set(path.name for path in CONFIGS.glob("byte16_*.yaml")) - set(written)):
        (CONFIGS / stale).unlink()
        print("removed", stale)
    print("\n".join(written))


if __name__ == "__main__":
    main()
