# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-data Lightning OPD with K policy rollouts and one persistent trainer.

Keep planning/preparation imports CPU-only. The driver controls GPU ownership:
student inference -> offload inference -> teacher scoring -> stop teacher -> train.
"""

import concurrent.futures
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import uuid

from slime.utils.opd_config import needs_teacher_features


def enabled(cfg):
    return cfg.get("method") == "lightning_opd" and "policy_rollout_k" in cfg


def schedule(cfg):
    steps, batch, k = cfg["num_rollout"], cfg["rollout_batch_size"], cfg["policy_rollout_k"]
    if type(k) is not int or not 1 <= k <= steps:
        raise ValueError("policy_rollout_k must be an integer between 1 and num_rollout.")
    quotient, remainder = divmod(steps, k)
    blocks, start = [], 0
    for index in range(k):
        end = start + quotient + (index < remainder)
        blocks.append(dict(index=index, start_step=start, end_step=end,
                           start_sample=start * batch, end_sample=end * batch))
        start = end
    return blocks


def fixed_data(cfg):
    path = cfg.get("lightning_fixed_data")
    if path:
        result = Path(path)
    else:
        result = Path(cfg["output_dir"]) / cfg["experiment_name"] / "data/fixed_prompts.parquet"
    if result.suffix != ".parquet":
        raise ValueError("lightning_fixed_data must be a .parquet path or null.")
    return result


def run_directory(cfg):
    return Path(cfg["output_dir"]) / cfg["experiment_name"]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sequence_identity(cfg, student):
    source = Path(cfg["train_files"]).resolve()
    stat = source.stat()
    # K is deliberately excluded: all K variants use the same prompt order.
    return dict(version=1, student=student, train_files=str(source),
                size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                max_train_samples=cfg["max_train_samples"],
                max_prompt_len=cfg["rollout_max_prompt_len"],
                chat_template_kwargs=cfg["apply_chat_template_kwargs"],
                rollout_seed=cfg["rollout_seed"],
                total_samples=cfg["num_rollout"] * cfg["rollout_batch_size"])


def prepare_fixed_prompts(cfg, student, tokenizer=None):
    """Repeat deterministically shuffled DAPO decks to exactly the training budget."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    target = fixed_data(cfg)
    identity = sequence_identity(cfg, student)
    manifest = target.with_suffix(".manifest.json")
    if target.exists():
        if not manifest.is_file() or json.loads(manifest.read_text()) != identity:
            raise ValueError(f"{target}: fixed prompt recipe changed; choose a new experiment/data path.")
        if pq.ParquetFile(target).metadata.num_rows != identity["total_samples"]:
            raise ValueError(f"{target}: fixed prompt row count differs from the training budget.")
    else:
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(student, trust_remote_code=True)
        table = pq.read_table(cfg["train_files"])
        if cfg["max_train_samples"] is not None:
            table = table.slice(0, cfg["max_train_samples"])
        eligible = []
        for source_index, row in enumerate(table.to_pylist()):
            messages = row["prompt"] if isinstance(row["prompt"], list) else [{"role": "user", "content": row["prompt"]}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **cfg["apply_chat_template_kwargs"]
            )
            if len(tokenizer.encode(prompt, add_special_tokens=False)) <= cfg["rollout_max_prompt_len"]:
                eligible.append(dict(prompt=prompt, label=row["label"], metadata=row.get("metadata") or {},
                                     source_index=source_index))
        if not eligible:
            raise ValueError("No DAPO prompts remain after prompt-length filtering.")
        rows, cycle = [], 0
        while len(rows) < identity["total_samples"]:
            order = list(range(len(eligible)))
            random.Random(cfg["rollout_seed"] + cycle).shuffle(order)
            for index in order[:identity["total_samples"] - len(rows)]:
                row = eligible[index]
                metadata = dict(row["metadata"], lightning_sample_index=len(rows),
                                lightning_source_index=row["source_index"], lightning_dataset_cycle=cycle)
                rows.append(dict(prompt=row["prompt"], label=row["label"], metadata=metadata))
            cycle += 1
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp.parquet")
        pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd", row_group_size=1024)
        temporary.replace(target)
        atomic_json(manifest, identity)
        print(f"Built {len(rows):,} fixed prompts from {len(eligible):,} eligible DAPO rows: {target}", flush=True)
    plan = dict(version=1, sequence=digest(identity), blocks=schedule(cfg))
    plan_path = run_directory(cfg) / "data/block_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("The Lightning block plan changed; choose a new experiment_name.")
    atomic_json(plan_path, plan)
    return target


def recipe(args):
    import yaml

    cfg = yaml.safe_load(Path(args.lightning_block_config).read_text())
    models = json.loads((run_directory(cfg) / "initialization.json").read_text())
    return cfg, models


def block_paths(cfg, index):
    directory = run_directory(cfg) / "data/blocks" / f"block_{index:04d}"
    return directory / "student_rollouts.parquet", directory / "lightning_opd.parquet"


def block_identity(cfg, models, index, policy_id):
    manifest = fixed_data(cfg).with_suffix(".manifest.json")
    return dict(version=1, sequence=digest(json.loads(manifest.read_text())),
                blocks=schedule(cfg), block_index=index, policy_id=policy_id,
                policy_step=schedule(cfg)[index]["start_step"], student=models["student"], teacher=models["teacher"],
                seed=cfg["seed"], max_response_len=cfg["rollout_max_response_len"],
                temperature=cfg["rollout_temperature"], top_p=cfg["rollout_top_p"], top_k=cfg["rollout_top_k"])


def cached(path, identity, expected_rows):
    import pyarrow.parquet as pq

    manifest = path.with_suffix(".manifest.json")
    return (path.is_file() and manifest.is_file()
            and json.loads(manifest.read_text()) == identity
            and pq.ParquetFile(path).metadata.num_rows == expected_rows)


def teacher_identity(cfg, identity):
    # Scalar caches remain compatible; feature caches never masquerade as scalar-only scoring.
    return dict(identity, teacher_features_version=1) if needs_teacher_features(cfg) else identity


def write_batches(path, batches, identity, expected_rows):
    """Commit bounded row groups; a failed request never publishes partial data."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    writer, count = None, 0
    try:
        for rows in batches:
            table = pa.Table.from_pylist(rows)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
            count += len(rows)
            print(f"{path.parent.name}/{path.name}: {count:,}/{expected_rows:,}", flush=True)
        if count != expected_rows:
            raise ValueError(f"{path}: expected {expected_rows} rows, received {count}.")
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(path)
    atomic_json(path.with_suffix(".manifest.json"), identity)


def prepare_student_block(manager, index, policy_id):
    """Generate the whole block from the current, already synchronized student."""
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    args = manager.args
    cfg, models = recipe(args)
    block = schedule(cfg)[index]
    count = block["end_sample"] - block["start_sample"]
    identity = block_identity(cfg, models, index, policy_id)
    rollouts, scored = block_paths(cfg, index)
    if cached(scored, teacher_identity(cfg, identity), count) or cached(rollouts, identity, count):
        return
    if not hasattr(manager, "_lightning_fixed_table"):
        manager._lightning_fixed_table = pq.read_table(fixed_data(cfg))
    table = manager._lightning_fixed_table.slice(block["start_sample"], count)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    launcher = importlib.import_module("configs.opd.qwen3-4b-opd")
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    def generate(row):
        position = row["metadata"]["lightning_sample_index"]
        output = launcher.request(url, dict(
            input_ids=tokenizer.encode(row["prompt"], add_special_tokens=False), return_logprob=True,
            sampling_params=dict(max_new_tokens=cfg["rollout_max_response_len"],
                                 temperature=cfg["rollout_temperature"], top_p=cfg["rollout_top_p"],
                                 top_k=cfg["rollout_top_k"], sampling_seed=cfg["seed"] + position,
                                 skip_special_tokens=False, no_stop_trim=False)))
        if output["meta_info"].get("finish_reason", {}).get("type") == "abort":
            raise RuntimeError(f"Student generation aborted at fixed sample {position}.")
        tokens = [int(item[1]) for item in output["meta_info"].get("output_token_logprobs", [])]
        if not tokens or len(tokens) > cfg["rollout_max_response_len"]:
            raise ValueError(f"Invalid student response tokens at fixed sample {position}.")
        metadata = dict(row["metadata"], is_lightning_opd=True, policy_block_index=index,
                        policy_update_step=block["start_step"], response_tokens=tokens,
                        loss_mask=[1] * len(tokens), response=tokenizer.decode(tokens, skip_special_tokens=False))
        return dict(prompt=row["prompt"], label=row["label"], metadata=metadata)

    def batches():
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg["precompute_concurrency"]) as pool:
            for batch in table.to_batches(max_chunksize=1024):
                yield list(pool.map(generate, batch.to_pylist()))

    print(f"Policy rollout {index + 1}/{cfg['policy_rollout_k']}: policy after step {block['start_step']}, {count:,} samples.", flush=True)
    write_batches(rollouts, batches(), identity, count)


def score_teacher_block(args, index, policy_id):
    """The driver owns Slurm's head-node GPUs while all Ray models are offloaded."""
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    cfg, models = recipe(args)
    block = schedule(cfg)[index]
    count = block["end_sample"] - block["start_sample"]
    identity = block_identity(cfg, models, index, policy_id)
    rollouts, scored = block_paths(cfg, index)
    if cached(scored, teacher_identity(cfg, identity), count):
        return
    if not cached(rollouts, identity, count):
        # Upgrading a scalar-only scored block must preserve its saved rollout policy.
        if needs_teacher_features(cfg) and cached(scored, identity, count):
            rollouts = scored
        else:
            raise ValueError("Student block cache is incomplete or belongs to another policy.")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(models["teacher"], trust_remote_code=True)
    if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError("Student and teacher tokenizers must share token IDs for OPD.")
    launcher = importlib.import_module("configs.opd.qwen3-4b-opd")
    env = os.environ.copy()
    devices = env.get("CUDA_VISIBLE_DEVICES", ",".join(map(str, range(cfg["n_gpus_per_node"])))).split(",")
    if len(devices) < cfg["teacher_tp_size"]:
        raise ValueError("Teacher TP exceeds the head node's visible GPUs.")
    env["CUDA_VISIBLE_DEVICES"] = ",".join(devices[:cfg["teacher_tp_size"]])
    url = f"http://127.0.0.1:{cfg['precompute_port']}"

    def score(row):
        tokens = row["metadata"]["response_tokens"]
        output = launcher.request(url + "/generate", dict(
            input_ids=tokenizer.encode(row["prompt"], add_special_tokens=False) + tokens,
            sampling_params=dict(temperature=0, max_new_tokens=0, skip_special_tokens=False),
            return_logprob=True, logprob_start_len=0,
            **({"opd_response_length": len(tokens)} if needs_teacher_features(cfg) else {})))
        items = output["meta_info"]["input_token_logprobs"][1:][-len(tokens):]
        if len(items) != len(tokens) or [int(item[1]) for item in items] != tokens:
            raise ValueError("Teacher returned misaligned response-token logprobs.")
        values = [float(item[0]) for item in items]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Teacher returned non-finite logprobs.")
        row["metadata"]["teacher_log_probs"] = values
        if needs_teacher_features(cfg):
            row["metadata"]["teacher_features"] = output["meta_info"]["opd_teacher_features"]
        return row

    def batches():
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg["precompute_concurrency"]) as pool:
            for batch in pq.ParquetFile(rollouts).iter_batches(batch_size=1024):
                yield list(pool.map(score, batch.to_pylist()))

    with launcher.background(
        launcher.server_command(cfg, models["teacher"], cfg["precompute_port"],
                                cfg["teacher_tp_size"], cfg["teacher_mem_fraction_static"], True),
        env, scored.parent / "teacher.log",
    ) as process:
        launcher.wait_ready(url + "/health_generate", cfg["startup_timeout"], process)
        write_batches(scored, batches(), teacher_identity(cfg, identity), count)


def load_training_block(manager, index, rollout_id, policy_id):
    from slime.utils.data import create_dataset
    from slime.utils.processing_utils import load_processor, load_tokenizer

    args = manager.args
    cfg, models = recipe(args)
    block = schedule(cfg)[index]
    count = block["end_sample"] - block["start_sample"]
    path = block_paths(cfg, index)[1]
    if not cached(path, teacher_identity(cfg, block_identity(cfg, models, index, policy_id)), count):
        raise ValueError("Cannot train on an incomplete or stale-policy Lightning block.")
    source = manager.data_source
    source.dataset = create_dataset(
        str(path), tokenizer=load_tokenizer(args.hf_checkpoint, trust_remote_code=True),
        processor=load_processor(args.hf_checkpoint, trust_remote_code=True),
        max_length=args.rollout_max_prompt_len, prompt_key=args.input_key,
        label_key=args.label_key, metadata_key=args.metadata_key, tool_key=args.tool_key,
        multimodal_keys=args.multimodal_keys, apply_chat_template=False, seed=args.rollout_seed,
    )
    if len(source.dataset) != count:
        raise ValueError("Block loading filtered samples; the fixed global data positions would change.")
    for offset, sample in enumerate(source.dataset.samples):
        metadata = sample.metadata or {}
        tokens, logprobs = metadata.get("response_tokens"), metadata.get("teacher_log_probs")
        if (
            metadata.get("lightning_sample_index") != block["start_sample"] + offset
            or tokens is None or not len(tokens) or logprobs is None or len(tokens) != len(logprobs)
        ):
            raise ValueError("Lightning block has reordered rows or misaligned response/teacher logprobs.")
    source.sample_offset = (rollout_id - block["start_step"]) * cfg["rollout_batch_size"]
    source.sample_group_index = rollout_id * cfg["rollout_batch_size"]
    source.sample_index = source.sample_group_index
    source.epoch_id = 0
    source.metadata.update(lightning_block_index=index, lightning_policy_id=policy_id,
                           lightning_global_sample_offset=rollout_id * cfg["rollout_batch_size"])


def generate_cached_rollout(args, rollout_id, data_source, evaluation=False):
    """Consume exactly one fixed batch, with no resampling, filters or network RM."""
    from slime.rollout.base_types import RolloutFnTrainOutput
    from slime.rollout.sglang_rollout import GenerateState, _handle_lightning_opd_sample

    if evaluation:
        raise ValueError("Use the separate evaluation function for Lightning block evaluation.")
    groups = data_source.get_samples(args.rollout_batch_size)
    if len(groups) != args.rollout_batch_size:
        raise ValueError("The fixed Lightning batch is incomplete.")
    state = GenerateState(args)
    for offset, group in enumerate(groups):
        if len(group) != 1:
            raise ValueError("Block Lightning requires one response per fixed data position.")
        sample = group[0]
        if (sample.metadata or {}).get("lightning_sample_index") != rollout_id * args.rollout_batch_size + offset:
            raise ValueError("Training batch differs from its absolute Lightning data positions.")
        _handle_lightning_opd_sample(sample, state)
        sample.reward = {"lightning_opd": True}
    return RolloutFnTrainOutput(samples=groups, metrics={
        "lightning/policy_block": groups[0][0].metadata["policy_block_index"],
        "lightning/data_position": (rollout_id + 1) * args.rollout_batch_size,
    })


class BlockCoordinator:
    def __init__(self, args, rollout_manager):
        import ray

        self.ray, self.args, self.manager = ray, args, rollout_manager
        self.cfg, _ = recipe(args)
        if not args.offload_train or not args.offload_rollout or args.update_weights_interval != 1:
            raise ValueError("Block Lightning requires offloaded training/inference and weight sync every step.")
        self.blocks = schedule(self.cfg)
        self.loaded = None
        # Never reuse later blocks generated by a previous, unsaved policy path.
        # Block 0 is safe to reuse because its policy is the fixed initialization.
        self.lineage = uuid.uuid4().hex

    def ensure_block(self, rollout_id):
        from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

        block = next(item for item in self.blocks if item["start_step"] <= rollout_id < item["end_step"])
        index = block["index"]
        if self.loaded == index:
            return
        resuming_block = rollout_id != block["start_step"]
        if resuming_block:
            metadata = self.ray.get(self.manager.lightning_resume_metadata.remote())
            policy_id = metadata.get("lightning_policy_id")
            if (metadata.get("lightning_block_index") != index or not policy_id
                    or metadata.get("lightning_global_sample_offset") != rollout_id * self.cfg["rollout_batch_size"]):
                raise ValueError("Block Lightning resume cursor does not match its cached policy block.")
        else:
            policy_id = "initial" if index == 0 else self.lineage
            self.ray.get(self.manager.prepare_lightning_block.remote(index, policy_id))
        self.ray.get(self.manager.offload.remote())
        try:
            if not resuming_block or needs_teacher_features(self.cfg):
                score_teacher_block(self.args, index, policy_id)
            self.ray.get(self.manager.load_lightning_block.remote(index, rollout_id, policy_id))
        finally:
            tags = [GPU_MEMORY_TYPE_WEIGHTS]
            try:
                from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH

                tags.append(GPU_MEMORY_TYPE_CUDA_GRAPH)
            except ImportError:
                pass
            tags.append(GPU_MEMORY_TYPE_KV_CACHE)
            for tag in tags:
                self.ray.get(self.manager.onload.remote(tags=[tag]))
        self.loaded = index
