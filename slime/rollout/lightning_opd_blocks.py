# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-data Lightning OPD with K policy rollouts and one persistent trainer.

Keep planning/preparation imports CPU-only. The driver controls GPU ownership:
student inference -> offload inference -> teacher scoring -> stop teacher -> train.
"""

import concurrent.futures
import contextlib
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import uuid

from slime.utils import ray_job
from slime.utils.opd_config import needs_teacher_features


ROW_GROUP = 1024  # Parquet row group size and progress-report granularity.


def ordered_groups(function, rows, concurrency, size=ROW_GROUP):
    """Apply `function` to every row with up to `concurrency` calls in flight; yield results in row order,
    `size` (default ROW_GROUP) at a time. A finished call frees its slot for the next row at once, so no row group waits
    for its slowest row (block generation is long-tailed). Queued calls are cancelled on failure.
    """
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)
    try:
        group = []
        for result in pool.map(function, rows):
            group.append(result)
            if len(group) == size:
                yield group
                group = []
        if group:
            yield group
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def enabled(cfg):
    return cfg.get("method") == "lightning_opd" and "policy_rollout_k" in cfg


def resident(cfg):
    """Block Lightning on dedicated actor / student-engine / teacher GPUs (colocate: false).

    Everything stays loaded for the whole job: blocks never offload, onload or restart a model. The
    launcher starts one live teacher on its own GPUs and records its URL as lightning_teacher_url.
    """
    return enabled(cfg) and not cfg.get("colocate", True)


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


def engines_needed_after(blocks, rollout_id, evaluating):
    """Whether the inference engines need the weights trained in step `rollout_id`.

    Within a block they serve nothing but an evaluation or the next block's generation, so pushing
    weights to them (and swapping their memory back in) after every update is pure overhead.
    """
    return evaluating or any(block["start_step"] == rollout_id + 1 for block in blocks)


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
    # Version 2: source-file order on every pass (version 1 shuffled each pass).
    return dict(version=2, student=student, train_files=str(source),
                size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                max_train_samples=cfg["max_train_samples"],
                max_prompt_len=cfg["rollout_max_prompt_len"],
                chat_template_kwargs=cfg["apply_chat_template_kwargs"],
                rollout_seed=cfg["rollout_seed"],
                total_samples=cfg["num_rollout"] * cfg["rollout_batch_size"])


def prepare_fixed_prompts(cfg, student, tokenizer=None):
    """Repeat the eligible DAPO rows in source-file order to exactly the training budget."""
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
            for row in eligible[:identity["total_samples"] - len(rows)]:
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


def checkpoint_identity(args, completed_steps):
    """Identify the saved boundary policy without reading multi-GB tensor shards."""
    if not getattr(args, "save", None) or completed_steps <= 0:
        return None
    root = Path(args.save)
    marker = root / "latest_checkpointed_iteration.txt"
    if not marker.is_file() or marker.read_text().strip() != str(completed_steps - 1):
        return None
    checkpoint = root / f"iter_{completed_steps - 1:07d}"
    files = sorted(checkpoint.glob("*.distcp"))
    metadata = [checkpoint / "common.pt", checkpoint / ".metadata"]
    if not files or not all(path.is_file() for path in metadata):
        return None
    return dict(path=str(checkpoint.resolve()), completed_steps=completed_steps,
                metadata={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in metadata},
                shards=[dict(name=path.name, size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
                        for path in files])


def boundary_policy_id(args, cfg, models, index, rollout_id, new_policy_id):
    """Reuse a boundary cache only when it is bound to the checkpoint just loaded."""
    checkpoint = checkpoint_identity(args, rollout_id)
    if checkpoint is None:
        return new_policy_id
    rollouts, scored = block_paths(cfg, index)
    provenance = rollouts.parent / "policy_checkpoint.json"
    resuming = (getattr(args, "start_rollout_id", None) == rollout_id
                and getattr(args, "load", None)
                and Path(args.load).resolve() == Path(args.save).resolve())
    if resuming and provenance.is_file():
        saved = json.loads(provenance.read_text())
        policy_id = saved.get("policy_id")
        if saved.get("checkpoint") == checkpoint and saved.get("policy_step") == rollout_id and policy_id:
            identity = block_identity(cfg, models, index, policy_id)
            block = schedule(cfg)[index]
            count = block["end_sample"] - block["start_sample"]
            if cached(scored, teacher_identity(cfg, identity), count) or cached(rollouts, identity, count):
                print(f"Reusing policy block {index + 1} from checkpoint after step {rollout_id}.", flush=True)
                return policy_id
    atomic_json(provenance, dict(version=1, checkpoint=checkpoint, policy_step=rollout_id,
                                 policy_id=new_policy_id))
    return new_policy_id


def identity_metadata(identity):
    """Parquet schema metadata naming the block's student, teacher, policy and sampling recipe."""
    return {b"lightning_identity": json.dumps(identity, sort_keys=True).encode()}


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
                writer = pq.ParquetWriter(temporary, table.schema.with_metadata(identity_metadata(identity)),
                                          compression="zstd")
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


def shared_rollouts(cfg, identity):
    """Initial-policy student rollouts depend on the student, prompts and sampling, not on the teacher:
    experiments that differ only in their teacher share one copy under <output_dir>/rollout_cache."""
    student_identity = {key: value for key, value in identity.items() if key != "teacher"}
    path = Path(cfg["output_dir"]) / "rollout_cache" / digest(student_identity)[:16] / "student_rollouts.parquet"
    return path, student_identity


def copy_block(source, target, identity):
    """Copy a block file, recording `identity` (its new owner's student/teacher) in data and manifest."""
    import pyarrow.parquet as pq

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.parquet")
    table = pq.read_table(source)
    pq.write_table(table.replace_schema_metadata(identity_metadata(identity)), temporary, compression="zstd",
                   row_group_size=ROW_GROUP)
    temporary.replace(target)
    atomic_json(target.with_suffix(".manifest.json"), identity)


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
    shared, shared_identity = shared_rollouts(cfg, identity) if policy_id == "initial" else (None, None)
    if shared is not None and cached(shared, shared_identity, count):
        copy_block(shared, rollouts, identity)
        print(f"Policy rollout {index + 1}: reusing the initial student's responses from {shared.parent}.", flush=True)
        return
    table = fixed_block_rows(manager, cfg, block)
    generate = student_generator(args, cfg, index, block)
    print(f"Policy rollout {index + 1}/{cfg['policy_rollout_k']}: policy after step {block['start_step']}, {count:,} samples.", flush=True)
    write_batches(rollouts, ordered_groups(generate, table, cfg["precompute_concurrency"]), identity, count)
    if shared is not None:
        copy_block(rollouts, shared, shared_identity)


def fixed_block_rows(manager, cfg, block):
    """The block's fixed prompt rows, in absolute position order."""
    import pyarrow.parquet as pq

    if not hasattr(manager, "_lightning_fixed_table"):
        manager._lightning_fixed_table = pq.read_table(fixed_data(cfg))
    return manager._lightning_fixed_table.slice(block["start_sample"], block["end_sample"] - block["start_sample"]).to_pylist()


def student_generator(args, cfg, index, block):
    """row -> the row with the current student's response, sampled with the row's fixed position seed."""
    from transformers import AutoTokenizer

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
                                 stop_token_ids=cfg.get("rollout_stop_token_ids"),
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

    return generate


def teacher_response_logprobs(output, tokens, sglang_vocab_size=None):
    """Validate response alignment, including SGLang's last-vocabulary-ID label bug.

    SGLang clips input logprob *labels* with ``id < vocab_size - 1``, even
    though it computes the logprobs from the original input IDs. Only the last
    valid ID may therefore be labelled 0. Never change the response or scores,
    accept out-of-vocabulary tokens, or relax alignment for the feature teacher.
    """
    items = output["meta_info"]["input_token_logprobs"][1:][-len(tokens):]
    if not tokens or len(items) != len(tokens):
        raise ValueError("Teacher returned misaligned response-token logprobs (length mismatch).")
    for position, (item, token) in enumerate(zip(items, tokens, strict=True)):
        returned = int(item[1])
        if sglang_vocab_size is not None and not 0 <= token < sglang_vocab_size:
            raise ValueError(f"Response token {token} at position {position} is outside the teacher vocabulary.")
        clipped_last_id = sglang_vocab_size is not None and token == sglang_vocab_size - 1 and returned == 0
        if returned != token and not clipped_last_id:
            raise ValueError(f"Teacher returned misaligned response-token logprobs at position {position}: "
                             f"expected token {token}, got {returned}.")
    values = [float(item[0]) for item in items]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Teacher returned non-finite logprobs.")
    return values


def scoring_servers(cfg, nodes, visible=None):
    """(node, CUDA devices, port) of each teacher scoring server.

    Scoring runs while every Ray model is offloaded, so the servers fill every GPU of every node,
    teacher_tp_size GPUs each. Logical GPU i of a node is the i-th entry of `visible` (this node's
    CUDA_VISIBLE_DEVICES; Slurm may assign a subset). Feature scoring caches teacher features on its
    node's disk, which other nodes cannot read, so it keeps one server on the first node (this one).
    """
    tp, per_node = cfg["teacher_tp_size"], cfg["n_gpus_per_node"]
    devices = visible.split(",") if visible else [str(gpu) for gpu in range(per_node)]
    if per_node % tp or len(devices) < per_node:
        raise ValueError(f"Teacher scoring splits {per_node} GPUs per node into groups of teacher_tp_size={tp}; "
                         f"visible devices: {','.join(devices)}.")
    if needs_teacher_features(cfg):
        nodes, per_node = nodes[:1], tp
    return [(node, ",".join(devices[first:first + tp]), cfg["precompute_port"] + first // tp)
            for node in nodes for first in range(0, per_node, tp)]


class TeacherServers:
    """The teacher scoring servers of a job (see scoring_servers), started on first use.

    Persistent (the default for log-prob scoring): the servers start once with SGLang's memory saver
    and a CPU copy of the weights, and between blocks only release their GPU memory to the student
    and trainer, so each later block resumes them in seconds instead of starting K new servers (loading
    the teacher and capturing CUDA graphs on every GPU while the GPUs otherwise sit idle).
    Otherwise, or for feature scoring (slime.rollout.opd_teacher has no memory saver), the servers are
    stopped after every block, as before.
    """

    def __init__(self, cfg, models, log_dir):
        self.cfg, self.models, self.log_dir = cfg, models, Path(log_dir)
        self.persistent = bool(cfg.get("teacher_persistent", True)) and not needs_teacher_features(cfg)
        # Resident jobs: the launcher owns one live teacher on dedicated GPUs; never start, release or stop it.
        self.external = cfg.get("lightning_teacher_url")
        self.stack, self.urls, self.processes, self.released = None, [], [], False

    @property
    def launcher(self):
        return importlib.import_module("configs.opd.qwen3-4b-opd")

    def acquire(self):
        """Return the ready servers' base URLs, starting or resuming them as needed."""
        if self.external:
            self.launcher.wait_ready(self.external + "/health_generate", self.cfg["startup_timeout"])
            return [self.external]
        if self.stack is None:
            self._start()
        elif self.released:
            for url in self.urls:
                self.launcher.request(url + "/resume_memory_occupation", {})
            self.released = False
        for url, process in zip(self.urls, self.processes, strict=True):
            self.launcher.wait_ready(url + "/health_generate", self.cfg["startup_timeout"], process)
        return self.urls

    def release(self):
        """Hand every GPU back after a block's scoring."""
        if self.stack is None or self.external:
            return
        if not self.persistent:
            self.close()
            return
        for url in self.urls:
            self.launcher.request(url + "/release_memory_occupation", {})
        self.released = True

    def close(self):
        stack, self.stack, self.urls, self.processes, self.released = self.stack, None, [], [], False
        if stack is not None:
            stack.close()

    def _start(self):
        cfg = self.cfg
        servers = scoring_servers(cfg, ray_job.gpu_nodes(cfg["nnodes"]), os.environ.get("CUDA_VISIBLE_DEVICES"))
        stack = contextlib.ExitStack()
        try:
            # Start every server before waiting for any, so they load the teacher at the same time.
            for node, devices, port in servers:
                command = self.launcher.server_command(cfg, self.models["teacher"], port, cfg["teacher_tp_size"],
                                                       cfg["teacher_mem_fraction_static"], True, node.strip("[]"))
                if self.persistent:
                    command += ["--enable-memory-saver", "--enable-weights-cpu-backup"]
                self.processes.append(stack.enter_context(ray_job.gpu_process(
                    command, None, 0, {f"node:{node}": 0.001}, self.log_dir / f"teacher-{port}.log", devices)))
                self.urls.append(f"http://{node}:{port}")
        except BaseException:
            stack.close()
            self.urls, self.processes = [], []
            raise
        self.stack = stack


def score_teacher_block(args, index, policy_id, servers=None):
    """Score a block's cached student rollouts with the teacher on every GPU of the job (see scoring_servers).

    `servers` (a TeacherServers) keeps the teacher alive across blocks; None starts and stops it here.
    """
    import pyarrow.parquet as pq
    from transformers import AutoConfig, AutoTokenizer

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
    owned = servers is None
    if owned:
        servers = TeacherServers(dict(cfg, teacher_persistent=False), models, scored.parent)
    urls = []
    score = teacher_scorer(args, cfg, models, index, urls)

    def batches():
        rows = [row for batch in pq.ParquetFile(rollouts).iter_batches(batch_size=ROW_GROUP) for row in batch.to_pylist()]
        yield from ordered_groups(score, list(enumerate(rows)), cfg["precompute_concurrency"])

    try:
        urls[:] = servers.acquire()
        write_batches(scored, batches(), teacher_identity(cfg, identity), count)
    except BaseException:
        servers.close()  # A failed block never leaves servers holding GPUs for the next attempt.
        raise
    finally:
        if owned:
            servers.close()
        else:
            servers.release()


def teacher_scorer(args, cfg, models, index, urls):
    """(position, row) -> the row with its teacher log-probs; position picks a server from `urls` (read per call)."""
    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(models["teacher"], trust_remote_code=True)
    if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError("Student and teacher tokenizers must share token IDs for OPD.")
    # Tokenizer size can exclude padded but valid model IDs; use the teacher's
    # actual output vocabulary. The feature teacher returns exact labels.
    sglang_vocab_size = None if needs_teacher_features(cfg) else AutoConfig.from_pretrained(
        models["teacher"], trust_remote_code=True).vocab_size
    launcher = importlib.import_module("configs.opd.qwen3-4b-opd")

    def score(item):
        position, row = item
        tokens = row["metadata"]["response_tokens"]
        output = launcher.request(urls[position % len(urls)] + "/generate", dict(
            input_ids=tokenizer.encode(row["prompt"], add_special_tokens=False) + tokens,
            sampling_params=dict(temperature=0, max_new_tokens=0, skip_special_tokens=False),
            return_logprob=True, logprob_start_len=0,
            **({"opd_response_length": len(tokens)} if needs_teacher_features(cfg) else {})))
        try:
            row["metadata"]["teacher_log_probs"] = teacher_response_logprobs(output, tokens, sglang_vocab_size)
        except ValueError as error:
            raise ValueError(f"{error} Block {index}, fixed sample {row['metadata']['lightning_sample_index']}.") from error
        if needs_teacher_features(cfg):
            row["metadata"]["teacher_features"] = output["meta_info"]["opd_teacher_features"]
        return row

    return score


def load_training_block(manager, index, rollout_id, policy_id):
    args = manager.args
    cfg, models = recipe(args)
    block = schedule(cfg)[index]
    count = block["end_sample"] - block["start_sample"]
    path = block_paths(cfg, index)[1]
    if not cached(path, teacher_identity(cfg, block_identity(cfg, models, index, policy_id)), count):
        raise ValueError("Cannot train on an incomplete or stale-policy Lightning block.")
    load_rows(manager, cfg, path, index, rollout_id, policy_id, block["start_sample"], count,
              (rollout_id - block["start_step"]) * cfg["rollout_batch_size"])


def load_rows(manager, cfg, path, index, rollout_id, policy_id, first_sample, count, offset):
    """Make the parquet rows at absolute positions [first_sample, first_sample + count) the training data,
    with the cursor `offset` rows in (the next rows are those of `rollout_id`)."""
    from slime.utils.data import create_dataset
    from slime.utils.processing_utils import load_processor, load_tokenizer

    args = manager.args
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
    for position, sample in enumerate(source.dataset.samples):
        metadata = sample.metadata or {}
        tokens, logprobs = metadata.get("response_tokens"), metadata.get("teacher_log_probs")
        if (
            metadata.get("lightning_sample_index") != first_sample + position
            or tokens is None or not len(tokens) or logprobs is None or len(tokens) != len(logprobs)
        ):
            raise ValueError("Lightning block has reordered rows or misaligned response/teacher logprobs.")
    source.sample_offset = offset
    source.sample_group_index = rollout_id * cfg["rollout_batch_size"]
    source.sample_index = source.sample_group_index
    source.epoch_id = 0
    source.metadata.update(lightning_block_index=index, lightning_policy_id=policy_id,
                           lightning_global_sample_offset=rollout_id * cfg["rollout_batch_size"])


class BlockStream:
    """Resident Block Lightning: generate, score and hand out one block batch by batch.

    A background thread (in the rollout manager) sends every fixed position of the block to the student
    engines, which hold the block-start policy until the stream is finished, and each finished response
    straight to the resident teacher. Batch k, the block's k-th rollout_batch_size fixed positions, is
    ready once all of its rows are scored, so training on it overlaps the generation of later batches.
    Data, order and policy are exactly those of the unstreamed block (a batch never takes whichever rows
    finish first: that would train short responses first). The block files are written when all rows are
    done, so resuming and caching work as before.
    """

    def __init__(self, manager, index, policy_id):
        import threading

        self.manager, self.index, self.policy_id = manager, index, policy_id
        self.cfg, self.models = recipe(manager.args)
        self.block = schedule(self.cfg)[index]
        self.batch = self.cfg["rollout_batch_size"]
        self.count = self.block["end_sample"] - self.block["start_sample"]
        self.identity = block_identity(self.cfg, self.models, index, policy_id)
        self.rollouts, self.scored = block_paths(self.cfg, index)
        self.paths = [None] * (self.count // self.batch)
        self.ready = [threading.Event() for _ in self.paths]
        self.error = None
        self.thread = threading.Thread(target=self._run, name=f"lightning-block-{index}", daemon=True)

    def start(self):
        self.thread.start()
        return self

    def wait(self, k):
        """Path of batch k's scored rows, once they are all ready."""
        self.ready[k].wait()
        if self.paths[k] is None:
            raise RuntimeError(f"Lightning block {self.index} stopped before batch {k} was ready.") from self.error
        return self.paths[k]

    def finish(self):
        """Wait until every row is generated, scored and the block files are written."""
        self.thread.join()
        if self.error is not None:
            raise RuntimeError(f"Lightning block {self.index} failed.") from self.error

    def _publish(self, k, rows):
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = self.scored.parent / "stream" / f"batch_{k:03d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.parquet")
        table = pa.Table.from_pylist(rows)
        pq.write_table(table.replace_schema_metadata(identity_metadata(self.identity)), temporary, compression="zstd")
        temporary.replace(path)
        self.paths[k] = path
        self.ready[k].set()

    def _run(self):
        import pyarrow.parquet as pq

        try:
            cfg, identity, count, args = self.cfg, self.identity, self.count, self.manager.args
            scored_identity = teacher_identity(cfg, identity)
            if cached(self.scored, scored_identity, count):
                rows = pq.read_table(self.scored).to_pylist()
                for k in range(len(self.paths)):
                    self._publish(k, rows[k * self.batch:(k + 1) * self.batch])
                return
            shared, shared_identity = (shared_rollouts(cfg, identity) if self.policy_id == "initial"
                                       else (None, None))
            generate = None
            if cached(self.rollouts, identity, count):
                rows = pq.read_table(self.rollouts).to_pylist()
            elif shared is not None and cached(shared, shared_identity, count):
                copy_block(shared, self.rollouts, identity)
                rows = pq.read_table(self.rollouts).to_pylist()
                print(f"Policy rollout {self.index + 1}: reusing the initial student's responses "
                      f"from {shared.parent}.", flush=True)
            else:
                rows = fixed_block_rows(self.manager, cfg, self.block)
                generate = student_generator(args, cfg, self.index, self.block)
                print(f"Policy rollout {self.index + 1}/{cfg['policy_rollout_k']}: policy after step "
                      f"{self.block['start_step']}, {count:,} samples, streamed to the teacher.", flush=True)
            score = teacher_scorer(args, cfg, self.models, self.index, [cfg["lightning_teacher_url"]])

            def work(item):
                position, row = item
                if generate is not None:
                    row = generate(row)
                student = dict(row, metadata=dict(row["metadata"]))  # scoring adds teacher_log_probs in place
                return student, score((position, row))

            students, scored = [], []
            groups = ordered_groups(work, list(enumerate(rows)), cfg["precompute_concurrency"], self.batch)
            for k, group in enumerate(groups):
                students += [student for student, _ in group]
                scored += [row for _, row in group]
                self._publish(k, [row for _, row in group])
            if generate is not None:
                write_batches(self.rollouts, [students], identity, count)
                if shared is not None:
                    copy_block(self.rollouts, shared, shared_identity)
            write_batches(self.scored, [scored], scored_identity, count)
        except BaseException as error:
            self.error = error
        finally:
            for event in self.ready:
                event.set()


def load_training_batch(manager, index, rollout_id, policy_id):
    """Train step `rollout_id` on its batch of the streaming block, waiting until the batch is scored."""
    stream = getattr(manager, "_lightning_stream", None)
    if stream is None or (stream.index, stream.policy_id) != (index, policy_id):
        raise ValueError(f"No streaming Lightning block {index} for policy {policy_id}.")
    cfg, block = stream.cfg, stream.block
    k = rollout_id - block["start_step"]
    path = stream.wait(k)
    load_rows(manager, cfg, path, index, rollout_id, policy_id, block["start_sample"] + k * stream.batch,
              stream.batch, 0)


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
        self.cfg, self.models = recipe(args)
        self.resident = resident(self.cfg)
        if args.update_weights_interval != 1:
            raise ValueError("Block Lightning requires weight sync every step.")
        if self.resident:
            if args.offload_train or args.offload_rollout or not self.cfg.get("lightning_teacher_url"):
                raise ValueError("Resident Block Lightning keeps every model loaded and needs the launcher's teacher.")
        elif not args.offload_train or not args.offload_rollout:
            raise ValueError("Colocated Block Lightning requires offloaded training/inference.")
        self.blocks = schedule(self.cfg)
        self.loaded, self.streaming = None, None
        self.teacher = TeacherServers(self.cfg, self.models, run_directory(self.cfg) / "data/teacher_servers")
        # Unsaved policy paths need a fresh lineage. Saved block-boundary policies
        # can reuse a complete cache tied to the exact checkpoint being restored.
        self.lineage = uuid.uuid4().hex

    def resume_policy(self, index, rollout_id):
        metadata = self.ray.get(self.manager.lightning_resume_metadata.remote())
        policy_id = metadata.get("lightning_policy_id")
        if (metadata.get("lightning_block_index") != index or not policy_id
                or metadata.get("lightning_global_sample_offset") != rollout_id * self.cfg["rollout_batch_size"]):
            raise ValueError("Block Lightning resume cursor does not match its cached policy block.")
        return policy_id

    def boundary_policy(self, index, rollout_id):
        if index == 0:
            return "initial"
        return boundary_policy_id(self.args, self.cfg, self.models, index, rollout_id, self.lineage)

    def ensure_block(self, rollout_id):
        """Make step `rollout_id`'s batch the rollout manager's training data."""
        block = next(item for item in self.blocks if item["start_step"] <= rollout_id < item["end_step"])
        if self.resident:
            return self.ensure_resident_batch(block, rollout_id)
        from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

        index = block["index"]
        if self.loaded == index:
            return
        resuming_block = rollout_id != block["start_step"]
        if resuming_block:
            policy_id = self.resume_policy(index, rollout_id)
        else:
            policy_id = self.boundary_policy(index, rollout_id)
            self.ray.get(self.manager.prepare_lightning_block.remote(index, policy_id))
        self.ray.get(self.manager.offload.remote())
        try:
            if not resuming_block or needs_teacher_features(self.cfg):
                score_teacher_block(self.args, index, policy_id, self.teacher)
            if index == len(self.blocks) - 1:
                self.teacher.close()  # No later block needs it; free its CPU weight copy and GPU context.
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

    def ensure_resident_batch(self, block, rollout_id):
        """Resident: stream the block from its first step; each step waits only for its own batch."""
        index = block["index"]
        if self.loaded != index:
            if rollout_id != block["start_step"]:
                # Resuming inside a block: settle() completed its files before the checkpoint was saved.
                policy_id = self.resume_policy(index, rollout_id)
                self.ray.get(self.manager.load_lightning_block.remote(index, rollout_id, policy_id))
                self.loaded, self.streaming = index, None
                return
            policy_id = self.boundary_policy(index, rollout_id)
            self.ray.get(self.manager.start_lightning_stream.remote(index, policy_id))
            self.loaded, self.streaming = index, policy_id
        if self.streaming is not None:
            self.ray.get(self.manager.load_lightning_batch.remote(index, rollout_id, self.streaming))

    def settle(self):
        """Before the engines get new weights or a checkpoint is saved: finish the streaming block.

        Its remaining rows must come from the block-start policy, and a resumed job reads the block files.
        """
        if self.resident and self.streaming is not None:
            self.ray.get(self.manager.finish_lightning_stream.remote())

    def close(self):
        """Stop the persistent teacher servers."""
        self.teacher.close()
