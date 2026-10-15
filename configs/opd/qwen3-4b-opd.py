# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""YAML launcher shared by the Qwen3-4B OPD and Lightning OPD entry points."""

import argparse
import importlib
import concurrent.futures
import contextlib
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import sysconfig
import time
import urllib.error
import urllib.request

import yaml
from slime.rollout import lightning_opd_blocks as lightning_blocks
from slime.utils.opd_config import needs_teacher_features, validate_opd_options

ROOT = Path(__file__).resolve().parents[2]
MODULE = "configs.opd.qwen3-4b-opd"
# These configure the launcher; all other YAML keys map directly to slime CLI flags.
LAUNCHER_KEYS = {
    "method",
    "model_path",
    "teacher_model_path",
    "auto_prepare_model",
    "local_models_only",
    "hf_home",
    "env_file",
    "output_dir",
    "experiment_name",
    "resume",
    "python_executable",
    "megatron_path",
    "cuda_home",
    "model_type",
    "conversion_num_gpus",
    "nnodes",
    "n_gpus_per_node",
    "startup_timeout",
    "teacher_num_gpus",
    "teacher_tp_size",
    "teacher_port",
    "teacher_mem_fraction_static",
    "teacher_context_length",
    "teacher_chunked_prefill_size",
    "teacher_url",
    "train_files",
    "source_data_dir",
    "lightning_opd_data",
    "lightning_fixed_data",
    "policy_rollout_k",
    "max_train_samples",
    "precompute_concurrency",
    "precompute_port",
    "precompute_tp_size",
    "use_wandb",
    "wandb_mode",
    "wandb_entity",
    "project_name",
    "eval_steps",
    "val_files",
    "val_max_prompt_length",
    "val_max_completion_length",
    "val_n",
    "val_temperature",
    "val_top_p",
    "val_top_k",
    "save_eval_generations",
    "save_model",
    "save_eval_multiplier",
    "save_total_limit",
    "walltime_checkpoint_margin_seconds",
}


def read_config(path, method, resume=False):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict) or cfg.get("method") != method:
        raise ValueError(f"{path}: expected method: {method}")
    if resume:
        cfg["resume"] = True
    if cfg.get("resume") == "auto":
        cfg["resume"] = (run_dir(cfg) / "checkpoints/latest_checkpointed_iteration.txt").is_file()
    cfg.setdefault("save_total_limit", None)
    cfg.setdefault("opd_loss_token_mode", "sample")
    cfg.setdefault("opd_token_top_k", 32)
    cfg.setdefault("opd_distribution_chunk_size", 32)
    cfg.setdefault("opd_loss_max_clip", 10.0)
    cfg.setdefault("opd_log_prob_min_clip", -10.0)
    validate_opd_options(cfg)
    if Path(cfg["experiment_name"]).name != cfg["experiment_name"]:
        raise ValueError("experiment_name must be a single directory name.")
    if "wandb_key" in cfg or "wandb_api_key" in cfg:
        raise ValueError("Keep WANDB_API_KEY in env_file, never in YAML.")
    positive = (
        "nnodes",
        "n_gpus_per_node",
        "actor_num_nodes",
        "actor_num_gpus_per_node",
        "conversion_num_gpus",
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
        "global_batch_size",
        "rollout_batch_size",
        "n_samples_per_prompt",
        "rollout_num_gpus_per_engine",
        "teacher_tp_size",
        "precompute_tp_size",
        "precompute_concurrency",
        "startup_timeout",
        "num_rollout",
    )
    if any(not isinstance(cfg[key], int) or cfg[key] < 1 for key in positive):
        raise ValueError("Node/GPU counts, batch sizes, parallel sizes, steps and timeouts must be positive integers.")
    if lightning_blocks.enabled(cfg):
        blocks = lightning_blocks.schedule(cfg)
        if cfg["n_samples_per_prompt"] != 1 or cfg["rollout_batch_size"] != cfg["global_batch_size"]:
            raise ValueError("Block Lightning requires one response per prompt and rollout_batch_size == global_batch_size.")
        if cfg["rollout_shuffle"] or not cfg["colocate"]:
            raise ValueError("Block Lightning requires rollout_shuffle: false and colocate: true.")
        if cfg.get("async_save") or cfg.get("update_weights_interval", 1) != 1 or cfg.get("use_critic"):
            raise ValueError("Block Lightning requires synchronous saves, update_weights_interval: 1 and no critic.")
        if cfg.get("save_model") == "final" and "save_eval_multiplier" not in cfg:
            cfg["save_interval"] = cfg["num_rollout"]
        elif "save_model" in cfg or "save_eval_multiplier" in cfg:
            raise ValueError("Block Lightning supports save_model: final or its legacy first-block save policy.")
        else:
            cfg["save_interval"] = blocks[0]["end_step"]
    elif "save_model" in cfg or "save_eval_multiplier" in cfg:
        cfg.setdefault("save_model", "active")
        cfg.setdefault("save_eval_multiplier", 1)
        multiplier = cfg["save_eval_multiplier"]
        if type(multiplier) is not int or multiplier < 1:
            raise ValueError("save_eval_multiplier must be an integer >= 1.")
        mode = cfg["save_model"]
        if mode is None:
            cfg["save_interval"] = None
        elif mode == "final":
            cfg["save_interval"] = cfg["num_rollout"]
        elif mode == "active":
            if type(cfg.get("eval_steps")) is not int or cfg["eval_steps"] < 1:
                raise ValueError("save_model: active requires positive eval_steps.")
            cfg["save_interval"] = cfg["eval_steps"] * multiplier
        else:
            raise ValueError("save_model must be null (no saves), final, or active.")
    if method == "opd" and not cfg.get("teacher_url") and cfg["teacher_num_gpus"] < 1:
        raise ValueError("A managed OPD teacher requires teacher_num_gpus > 0.")
    if cfg["max_train_samples"] is not None and cfg["max_train_samples"] < 1:
        raise ValueError("max_train_samples must be positive or null.")
    if cfg["wandb_mode"] not in {"offline", "online", "disabled"}:
        raise ValueError("wandb_mode must be offline, online, or disabled.")
    actor = cfg["actor_num_nodes"] * cfg["actor_num_gpus_per_node"]
    teacher = cfg["teacher_num_gpus"] if method == "opd" and not cfg.get("teacher_url") else 0
    total = cfg["nnodes"] * cfg["n_gpus_per_node"]
    rollout = actor if cfg["colocate"] else cfg["rollout_num_gpus"]
    if actor + (0 if cfg["colocate"] else rollout) + teacher > total:
        raise ValueError("Actor + rollout + teacher GPU allocations exceed nnodes * n_gpus_per_node.")
    if cfg["nnodes"] > 1 and actor + (0 if cfg["colocate"] else rollout) + teacher != total:
        raise ValueError(
            "Multi-node GPU allocations must cover every configured GPU for slime's contiguous GPU ordering. "
            "With an external teacher, add the freed teacher GPUs to rollout_num_gpus."
        )
    conversion_available = cfg["n_gpus_per_node"] - (teacher if cfg["nnodes"] == 1 else 0)
    if not 1 <= cfg["conversion_num_gpus"] <= conversion_available:
        raise ValueError("conversion_num_gpus exceeds available head-node GPUs.")
    if not 1 <= cfg["precompute_tp_size"] <= cfg["n_gpus_per_node"]:
        raise ValueError("precompute_tp_size must fit on one node.")
    if teacher >= cfg["n_gpus_per_node"]:
        raise ValueError("The last node must retain at least one GPU for Ray; reduce teacher_num_gpus.")
    if teacher and teacher != cfg["teacher_tp_size"]:
        raise ValueError("The managed teacher requires teacher_num_gpus == teacher_tp_size.")
    parallel = cfg["tensor_model_parallel_size"] * cfg["pipeline_model_parallel_size"] * cfg["context_parallel_size"]
    if actor % parallel or cfg["global_batch_size"] % (actor // parallel):
        raise ValueError(
            "Model parallelism must divide actor GPUs; global_batch_size must be divisible by data parallelism."
        )
    batch = cfg["rollout_batch_size"] * cfg["n_samples_per_prompt"]
    if batch < cfg["global_batch_size"] or batch % cfg["global_batch_size"]:
        raise ValueError("rollout_batch_size * n_samples_per_prompt must be a multiple of global_batch_size.")
    if cfg["rollout_num_gpus_per_engine"] > cfg["n_gpus_per_node"] or rollout % cfg["rollout_num_gpus_per_engine"]:
        raise ValueError("rollout_num_gpus_per_engine must divide rollout GPUs and fit on one node.")
    if cfg.get("eval_steps") and not rollout:
        raise ValueError("Evaluation requires student rollout GPUs (use colocate for Lightning OPD).")
    if cfg.get("eval_steps") and not cfg.get("val_files"):
        raise ValueError("eval_steps requires val_files.")
    if method == "lightning_opd" and cfg["apply_chat_template"]:
        raise ValueError("Lightning OPD parquet already contains formatted prompts; set apply_chat_template: false.")
    limit = cfg["save_total_limit"]
    if limit is not None:
        if type(limit) is not int or limit < 1:
            raise ValueError("save_total_limit must be a positive integer or null (keep all checkpoints).")
        if cfg.get("async_save"):
            raise ValueError("save_total_limit requires synchronous checkpoint saving; set async_save: false.")
        if cfg.get("save_retain_interval") is not None:
            raise ValueError("Set save_retain_interval: null when using save_total_limit for post-eval cleanup.")
        if not lightning_blocks.enabled(cfg) and cfg.get("save_model") != "final" and cfg.get("eval_steps") and cfg.get("save_interval") is not None:
            interval = cfg["save_interval"]
            if type(interval) is not int or interval < 1 or interval % cfg["eval_steps"]:
                raise ValueError("With save_total_limit, save_interval must be a positive multiple of eval_steps.")
    return cfg


def load_env(path):
    """Read simple dotenv assignments without executing shell code or printing secrets."""
    if not path:
        return
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        key, sep, value = line.partition("=")
        if not sep or not key.strip().isidentifier():
            raise ValueError(f"Invalid dotenv assignment at {path}:{number}")
        parts = shlex.split(value, comments=True)
        os.environ.setdefault(key.strip(), " ".join(parts))


def runtime_env(cfg):
    env = os.environ.copy()
    env.update(
        {
            "HF_HOME": cfg["hf_home"],
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "RAY_USAGE_STATS_ENABLED": "0",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_CUMEM_ENABLE": "0",
            "WANDB_MODE": cfg["wandb_mode"],
            "WANDB_ENTITY": cfg["wandb_entity"] or "",
            "WANDB_DISABLE_CODE": "true",
            "WANDB_DISABLE_GIT": "true",
        }
    )
    env["PYTHONPATH"] = os.pathsep.join(
        str(x) for x in (ROOT, cfg.get("megatron_path"), env.get("PYTHONPATH")) if x
    )
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    # Preloaded TorchMemorySaver workers start before torch loads its CUDA libs.
    # Use the CUDA runtime bundled with this interpreter's PyTorch wheel.
    cuda_runtime = Path(sysconfig.get_paths()["purelib"]) / "nvidia/cuda_runtime/lib"
    if cuda_runtime.is_dir():
        env["LD_LIBRARY_PATH"] = str(cuda_runtime) + (
            os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
        )
    if cfg.get("cuda_home"):
        env["CUDA_HOME"] = env["CUDA_PATH"] = str(Path(cfg["cuda_home"]).expanduser())
        env["PATH"] = str(Path(env["CUDA_HOME"]) / "bin") + os.pathsep + env["PATH"]
    if cfg["local_models_only"]:
        env.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    if env.get("WANDB_KEY") and not env.get("WANDB_API_KEY"):
        env["WANDB_API_KEY"] = env["WANDB_KEY"]
    env.pop("WANDB_KEY", None)
    return env


def run_dir(cfg):
    return Path(cfg["output_dir"]).expanduser().resolve() / cfg["experiment_name"]


def model_path(value, cfg, download=False):
    path = Path(value).expanduser()
    if path.is_dir():
        return str(path.resolve())
    if path.is_absolute() or value.startswith("."):
        raise FileNotFoundError(f"Model directory does not exist: {value}")
    cache = Path(cfg["hf_home"]) / "hub" / ("models--" + value.replace("/", "--"))
    ref = cache / "refs/main"
    snapshots = sorted((cache / "snapshots").glob("*"))
    snapshot = cache / "snapshots" / ref.read_text().strip() if ref.exists() else None
    if snapshot is None and len(snapshots) == 1:
        snapshot = snapshots[0]
    if snapshot is not None and (snapshot / "config.json").is_file():
        return str(snapshot.resolve())
    if download and not cfg["local_models_only"]:
        from huggingface_hub import snapshot_download

        return snapshot_download(value, cache_dir=str(Path(cfg["hf_home"]) / "hub"))
    raise FileNotFoundError(
        f"{value} is not cached under {cache}. Download it before submitting, or set local_models_only: false."
    )


def selected_models(cfg, download=False):
    student = model_path(cfg["model_path"], cfg, download)
    teacher = model_path(cfg["teacher_model_path"], cfg, download)
    for model in (student, teacher):
        directory = Path(model)
        if not (directory / "config.json").is_file() or not (directory / "tokenizer_config.json").is_file():
            raise FileNotFoundError(f"Model config/tokenizer files missing: {model}")
        indexes = list(directory.glob("*.index.json"))
        if indexes:
            for index in indexes:
                shards = set(json.loads(index.read_text()).get("weight_map", {}).values())
                if any(not (directory / shard).is_file() for shard in shards):
                    raise FileNotFoundError(f"Model weight shards missing: {model}")
        elif not list(directory.glob("*.safetensors")) and not list(directory.glob("pytorch_model*.bin")):
            raise FileNotFoundError(f"Model weights missing: {model}")
    if needs_teacher_features(cfg):
        from transformers import AutoTokenizer

        student_tokenizer = AutoTokenizer.from_pretrained(student, trust_remote_code=True, local_files_only=True)
        teacher_tokenizer = AutoTokenizer.from_pretrained(teacher, trust_remote_code=True, local_files_only=True)
        student_vocab_size = json.loads((Path(student) / "config.json").read_text())["vocab_size"]
        teacher_vocab_size = json.loads((Path(teacher) / "config.json").read_text())["vocab_size"]
        if student_tokenizer.get_vocab() != teacher_tokenizer.get_vocab() or student_vocab_size != teacher_vocab_size:
            raise ValueError("OPD distribution loss requires identical teacher/student token IDs and vocabulary sizes.")
    return student, teacher


def lightning_data(cfg):
    if lightning_blocks.enabled(cfg):
        return lightning_blocks.fixed_data(cfg)
    return (
        Path(cfg["lightning_opd_data"])
        if cfg.get("lightning_opd_data")
        else run_dir(cfg) / "data/lightning_opd.parquet"
    )


def require_parquet(path):
    path = Path(path)
    if path.suffix != ".parquet":
        raise ValueError(f"Only parquet data is supported by these launchers: {path}")
    return path


def convert_data(source, target, limit=None):
    """Add slime's flat label and benchmark metadata, retaining verl columns."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    source, target = require_parquet(source), require_parquet(target)
    stat = source.stat()
    identity = {
        "source": str(source.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "limit": limit,
        "adapter_version": 2,
    }
    manifest = target.with_suffix(".source.json")
    if target.exists() and manifest.exists() and json.loads(manifest.read_text()) == identity:
        return str(target)
    table = pq.read_table(source)
    if limit is not None:
        table = table.slice(0, limit)
    rows = table.to_pylist()
    if not rows:
        raise ValueError(f"Empty dataset: {source}")
    for index, row in enumerate(rows):
        truth = row.get("label")
        if truth is None:
            truth = (row.get("reward_model") or {}).get("ground_truth")
        if truth is None:
            raise ValueError(f"{source}: missing label / reward_model.ground_truth at row {index}")
        extra = row.get("extra_info") or {}
        metadata = dict(row.get("metadata") or {})
        metadata.update(
            data_source=str(row.get("data_source") or source.stem),
            sample_uid=str(extra.get("index", index)),
            extra_info=extra,
        )
        row.update(label=str(truth), metadata=metadata)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.parquet")
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
    temporary.replace(target)
    manifest.write_text(json.dumps(identity, indent=2) + "\n")
    return str(target)


def prepare_prompts(cfg):
    """CPU-only preparation of DAPO-Math-17k and all requested eval parquet."""
    import pyarrow.parquet as pq

    records = []
    for split, targets in (("train", [cfg["train_files"]]), ("eval", cfg["val_files"])):
        for target in targets:
            target = require_parquet(target)
            source = Path(cfg["source_data_dir"]) / split / target.name
            if source.resolve() == target.resolve():
                raise ValueError("source_data_dir must differ from the prepared train/eval destinations.")
            convert_data(source, target)
            rows = pq.ParquetFile(target).metadata.num_rows
            record = {
                "dataset": target.stem,
                "split": split,
                "rows": rows,
                "source": str(source.resolve()),
                "path": str(target.resolve()),
            }
            records.append(record)
            print(f"Prepared {target.stem}: {rows:,} rows -> {target}", flush=True)
    index = Path(cfg["train_files"]).parent.parent / "prepared_inputs.json"
    index.write_text(json.dumps(records, indent=2) + "\n")
    return records


def prepare_eval(cfg):
    if not cfg.get("eval_steps"):
        return None
    datasets = [{"name": Path(source).stem, "path": str(require_parquet(source))} for source in cfg["val_files"]]
    result = {
        "eval": {
            "defaults": {
                "input_key": "prompt",
                "label_key": "label",
                "metadata_key": "metadata",
                "temperature": cfg["val_temperature"],
                "top_p": cfg["val_top_p"],
                "top_k": cfg["val_top_k"],
                "max_response_len": cfg["val_max_completion_length"],
                "n_samples_per_eval_prompt": cfg["val_n"],
            },
            "datasets": datasets,
        }
    }
    path = run_dir(cfg) / "eval.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(result, sort_keys=False))
    return str(path)


def baseline_module(name):
    return importlib.import_module("slime.eval." + name)


async def baseline_eval_reward(args, sample, **kwargs):
    metadata = sample.metadata or {}
    source = metadata.get("data_source")
    if not source:
        raise ValueError("Evaluation metadata must contain data_source; run --prepare-prompts.")
    # Keep score/acc/formatted consistent with slime.eval.baseline_reward, including MC handling.
    return await baseline_module("reward_async").compute_score(
        source, sample.response, sample.label, metadata.get("extra_info") or {}
    )


def prune_training_checkpoints(save_dir, save_total_limit, rollout_id=None, require_rollout_state=False):
    """Remove old training checkpoints only after the latest save/eval completed.

    The Megatron tracker is written after all ranks finish a synchronous save.
    Keep that checkpoint and its rollout cursor so the next allocation can resume.
    Never follow checkpoint symlinks or touch initialization/data/eval/W&B files.
    """
    if save_total_limit is None:
        return []
    if type(save_total_limit) is not int or save_total_limit < 1:
        raise ValueError("save_total_limit must be a positive integer or null.")
    directory = Path(save_dir)
    tracker = directory / "latest_checkpointed_iteration.txt"
    if tracker.is_symlink():
        raise RuntimeError(f"Checkpoint tracker is a symlink: {tracker}; no cleanup performed.")
    if not tracker.is_file():
        if rollout_id in (None, 0):  # Initial evaluation has no training checkpoint.
            return []
        raise FileNotFoundError(f"No completed checkpoint tracker: {tracker}")
    latest = int(tracker.read_text().strip())
    if rollout_id is not None and latest != rollout_id:
        raise RuntimeError(f"Evaluated rollout {rollout_id} differs from saved checkpoint {latest}; no cleanup performed.")
    current = directory / f"iter_{latest:07d}"
    if current.is_symlink() or not current.is_dir() or not any(current.iterdir()):
        raise RuntimeError(f"Latest checkpoint is missing, empty or a symlink: {current}; no cleanup performed.")
    rollout_directory = directory / "rollout"
    if rollout_directory.is_symlink():
        raise RuntimeError(f"Rollout state directory is a symlink: {rollout_directory}; no cleanup performed.")
    state = rollout_directory / f"global_dataset_state_dict_{latest}.pt"
    if require_rollout_state and (state.is_symlink() or not state.is_file() or state.stat().st_size == 0):
        raise RuntimeError(f"Latest rollout cursor is missing or empty: {state}; no cleanup performed.")
    candidates = sorted(
        (
            path for path in directory.iterdir()
            if re.fullmatch(r"iter_\d{7}", path.name)
            and not path.is_symlink() and path.is_dir()
            and int(path.name[5:]) <= latest
        ),
        key=lambda path: int(path.name[5:]), reverse=True,
    )
    kept = {int(path.name[5:]) for path in candidates[:save_total_limit]}
    removed = []
    for path in candidates[save_total_limit:]:
        shutil.rmtree(path)
        removed.append(path.name)
    for path in rollout_directory.glob("global_dataset_state_dict_*.pt"):
        match = re.fullmatch(r"global_dataset_state_dict_(\d+)\.pt", path.name)
        if match and not path.is_symlink() and path.is_file():
            index = int(match.group(1))
            if index < latest and index not in kept:
                path.unlink()
    if removed:
        print(f"Checkpoint cleanup: kept latest {save_total_limit}; removed {', '.join(removed)}", flush=True)
    return removed


def evaluate(args, rollout_id, data_source, evaluation=True):
    """Grade answers with slime.eval scorers and report avg@k/pass@k."""
    from slime.rollout.sglang_rollout import GenerateState, generate_rollout

    # This singleton retains its first args for later training tasks. Initialize
    # it with training args before passing the evaluation-only scorer override.
    GenerateState(args)
    eval_args = copy.copy(args)
    eval_args.custom_rm_path = MODULE + ".baseline_eval_reward"
    eval_args.eval_reward_key = "acc"
    eval_args.apply_chat_template = True
    output = generate_rollout(eval_args, rollout_id, data_source, evaluation=True)
    metrics = dict(output.metrics or {})
    datasets = {dataset.name: dataset for dataset in args.eval_datasets}
    for name, data in output.data.items():
        samples = data["samples"]
        metadata = [sample.metadata for sample in samples]
        metrics.update(
            baseline_module("eval_metrics").compute_eval_metrics(
                [item["data_source"] for item in metadata],
                [item["sample_uid"] for item in metadata],
                [sample.reward["acc"] for sample in samples],
                k=datasets[name].n_samples_per_eval_prompt,
            )
        )
    output.metrics = metrics
    # Evaluation still writes metrics when training checkpoint saving is disabled.
    target = (
        Path(args.eval_output_dir)
        if getattr(args, "eval_output_dir", None)
        else Path(args.save).parent / "eval_metrics"
    )
    target.mkdir(parents=True, exist_ok=True)
    destination = target / f"{rollout_id}.json"
    temporary = destination.with_suffix(".tmp.json")
    saved_metrics = dict(metrics, _eval_completed_steps=getattr(
        args, "eval_completed_steps", 0 if rollout_id == 0 else rollout_id + 1))
    temporary.write_text(json.dumps(saved_metrics, indent=2) + "\n")
    temporary.replace(destination)
    config = target.parent / "config.yaml"
    if config.is_file():
        cfg = yaml.safe_load(config.read_text())
        interval = cfg.get("save_interval")
        saved = (
            rollout_id + 1 == interval
            if lightning_blocks.enabled(cfg)
            else interval is not None and (rollout_id + 1) % interval == 0
        )
        if saved:
            prune_training_checkpoints(
                args.save, cfg.get("save_total_limit"), rollout_id,
                require_rollout_state=args.rollout_global_dataset,
            )
    return output


def model_args(cfg, env):
    # Reading the existing architecture script keeps Qwen3 dimensions in one place.
    script = ROOT / "configs/models" / (cfg["model_type"] + ".sh")
    command = ["bash", "-c", 'source "$1"; printf "%s\\0" "${MODEL_ARGS[@]}"', "bash", str(script)]
    result = subprocess.run(command, check=True, capture_output=True, env=env)
    return result.stdout.decode().rstrip("\0").split("\0")


def checkpoint_path(cfg, student):
    key = hashlib.sha256(student.encode()).hexdigest()[:12]
    return run_dir(cfg) / "models" / key / "torch_dist"


def training_args(cfg, student, teacher_url, eval_path=None):
    args = {key: value for key, value in cfg.items() if key not in LAUNCHER_KEYS}
    checkpoint = str(checkpoint_path(cfg, student))
    args.update(
        hf_checkpoint=student,
        ref_load=checkpoint,
        # A missing training tracker lets slime select ref_load, finetune mode
        # and start_rollout_id=0. A present tracker resumes optimizer/RNG.
        load=str(run_dir(cfg) / "checkpoints"),
        save=str(run_dir(cfg) / "checkpoints") if cfg.get("save_interval") is not None else None,
        num_gpus_per_node=cfg["n_gpus_per_node"],
        prompt_data=str(lightning_data(cfg))
        if cfg["method"] == "lightning_opd"
        else str(require_parquet(cfg["train_files"]))
        + (f"@[:{cfg['max_train_samples']}]" if cfg["max_train_samples"] else ""),
    )
    if teacher_url:
        args["rm_url"] = teacher_url.rstrip("/") + ("" if teacher_url.endswith("/generate") else "/generate")
    if cfg.get("save_model") == "final":
        args["save_final_only"] = True
    if os.environ.get("OPD_CHECKPOINT_DEADLINE") and cfg.get("save_interval") is not None:
        args["checkpoint_deadline"] = float(os.environ["OPD_CHECKPOINT_DEADLINE"])
    if lightning_blocks.enabled(cfg):
        args["lightning_block_config"] = str(run_dir(cfg) / "config.yaml")
        args["rollout_function_path"] = "slime.rollout.lightning_opd_blocks.generate_cached_rollout"
    if eval_path:
        args.update(
            eval_config=eval_path,
            eval_interval=cfg["eval_steps"],
            eval_output_dir=str(run_dir(cfg) / "eval_metrics"),
            eval_max_prompt_len=cfg["val_max_prompt_length"],
            eval_function_path=MODULE + ".evaluate",
        )
        if cfg["save_eval_generations"]:
            args["save_debug_rollout_data"] = str(run_dir(cfg) / "generations/{rollout_id}.pt")
        tracker = run_dir(cfg) / "checkpoints/latest_checkpointed_iteration.txt"
        if cfg.get("eval_on_resume") and cfg["resume"] and tracker.is_file():
            completed = int(tracker.read_text().strip())
            if (
                (run_dir(cfg) / "eval_metrics" / f"{completed}.json").is_file()
                or (completed + 1) % cfg["eval_steps"]
            ):
                args["eval_on_resume"] = False  # Keep the existing completed evaluation.
    if cfg["use_wandb"]:
        args.update(
            use_wandb=True,
            wandb_project=cfg["project_name"],
            wandb_group=cfg["experiment_name"],
            wandb_team=cfg["wandb_entity"],
            wandb_mode=cfg["wandb_mode"],
            wandb_dir=str(run_dir(cfg) / "wandb"),
            disable_wandb_random_suffix=True,
        )
    result = []
    for key, value in args.items():
        if value is None or value is False:
            continue
        result.append("--" + key.replace("_", "-"))
        if value is not True:
            if isinstance(value, list):
                result.extend(str(x) for x in value)
            else:
                result.append(json.dumps(value) if isinstance(value, dict) else str(value))
    return result


def check_inputs(cfg, preparation=False):
    sources = [cfg["train_files"]] + (cfg["val_files"] if cfg.get("eval_steps") else [])
    for source in sources:
        require_parquet(source)
    missing = [str(p) for p in sources if not Path(p).is_file()]
    if cfg["method"] == "lightning_opd":
        require_parquet(lightning_data(cfg))
    if cfg["method"] == "lightning_opd" and not preparation and not lightning_data(cfg).is_file():
        missing.append(str(lightning_data(cfg)) + " (run --prepare-data first)")
    if missing:
        raise FileNotFoundError("Missing data:\n  " + "\n  ".join(missing))
    import pyarrow.parquet as pq

    for source in sources:
        if Path(source).suffix == ".parquet":
            fields = pq.read_schema(source).names
            if not {"prompt", "label", "metadata"}.issubset(fields):
                raise ValueError(f"{source}: expected prepared prompt/label/metadata columns; run --prepare-prompts.")
    if lightning_blocks.enabled(cfg) and not preparation:
        if pq.ParquetFile(lightning_data(cfg)).metadata.num_rows != cfg["num_rollout"] * cfg["rollout_batch_size"]:
            raise ValueError("Fixed Lightning prompt count differs from num_rollout * rollout_batch_size.")
        return
    if cfg["method"] == "lightning_opd" and not preparation:
        parquet = pq.ParquetFile(lightning_data(cfg))
        for batch in parquet.iter_batches(batch_size=1024, columns=["metadata"]):
            for row in batch.to_pylist():
                metadata = row["metadata"] or {}
                tokens, logprobs = metadata.get("response_tokens"), metadata.get("teacher_log_probs")
                if (
                    not (metadata.get("is_lightning_opd") or metadata.get("is_offline_opd"))
                    or not tokens
                    or logprobs is None
                    or len(tokens) != len(logprobs)
                ):
                    raise ValueError(
                        "Lightning OPD data must have is_lightning_opd, response_tokens and aligned teacher_log_probs."
                    )


def check_dependencies(cfg, preparation=False):
    if cfg.get("cuda_home") and not (Path(cfg["cuda_home"]).expanduser() / "bin/nvcc").is_file():
        raise FileNotFoundError(f"CUDA compiler missing: {cfg['cuda_home']}/bin/nvcc")
    names = ["torch", "transformers", "pyarrow", "sglang"]
    if not preparation:
        names += ["ray", "sglang_router", "megatron.core", "megatron.training", "mbridge"]
        if cfg.get("eval_steps"):
            names.append("math_verify")
            names += ["slime.eval.baseline_reward", "slime.eval.reward_async", "slime.eval.eval_metrics"]
        if cfg["colocate"]:
            names.append("torch_memory_saver")
    missing = []
    for name in names:
        try:
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        except (ImportError, ModuleNotFoundError):
            missing.append(name)
    if missing:
        raise RuntimeError(
            f"{sys.executable}: missing dependencies: {', '.join(missing)}. Use the README training environment; set python_executable / megatron_path in YAML."
        )


def request(url, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=600) as response:
        return json.loads(response.read() or b"{}")


def wait_ready(url, timeout, process=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"Server exited with status {process.returncode}; inspect its log.")
        try:
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url, timeout=5):
                return
        except (OSError, urllib.error.URLError):
            time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {url}")


@contextlib.contextmanager
def background(command, env, log_path):
    """Only stop processes launched by this job; Ray --block owns its children."""
    with Path(log_path).open("w") as log:
        process = subprocess.Popen(
            command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            yield process
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def server_command(cfg, model, port, tp, memory, teacher_scoring=False):
    if teacher_scoring and needs_teacher_features(cfg):
        return [sys.executable, "-m", "slime.rollout.opd_teacher", "--model-path", model,
                "--port", str(port), "--cache-dir", str(run_dir(cfg) / "data/teacher_features"),
                "--context-length", str(cfg["teacher_context_length"]),
                "--chunk-size", str(cfg.get("opd_distribution_chunk_size", 32)),
                "--memory-fraction", str(memory)]
    return [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--tp",
        str(tp),
        "--chunked-prefill-size",
        str(cfg["teacher_chunked_prefill_size"]),
        "--mem-fraction-static",
        str(memory),
        "--context-length",
        str(cfg["teacher_context_length"]),
    ]


def provenance(cfg, student, teacher):
    source = Path(cfg["train_files"]).resolve()
    stat = source.stat()
    return {
        "student": student,
        "teacher": teacher,
        "train_files": str(source),
        "train_file_size": stat.st_size,
        "train_file_mtime_ns": stat.st_mtime_ns,
        "max_train_samples": cfg["max_train_samples"],
        "max_response_len": cfg["rollout_max_response_len"],
        "temperature": cfg["rollout_temperature"],
        "top_p": cfg["rollout_top_p"],
        "top_k": cfg["rollout_top_k"],
        "chat_template_kwargs": cfg["apply_chat_template_kwargs"],
        "seed": cfg["seed"],
        "max_prompt_len": cfg["rollout_max_prompt_len"],
    }


def scored_provenance(cfg, student, teacher):
    return lightning_blocks.teacher_identity(cfg, provenance(cfg, student, teacher))


def prepare_lightning_data(cfg, student, teacher, env):
    """Collect student responses, then score them offline with the same teacher."""
    if lightning_blocks.enabled(cfg):
        # Responses/logprobs must be generated inside training, using each block's policy.
        return lightning_blocks.prepare_fixed_prompts(cfg, student)
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    target = lightning_data(cfg)
    manifest = target.with_suffix(".manifest.json")
    identity = provenance(cfg, student, teacher)
    if target.exists():
        if not manifest.exists() or json.loads(manifest.read_text()) != scored_provenance(cfg, student, teacher):
            raise ValueError(
                f"{target} already exists with different/unknown provenance. Choose a new lightning_opd_data path."
            )
        print(f"Reusing verified precomputed data: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(student, trust_remote_code=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher, trust_remote_code=True)
    if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError("Student and teacher tokenizers must share token IDs for OPD.")
    table = pq.read_table(require_parquet(cfg["train_files"]))
    if cfg["max_train_samples"] is not None:
        table = table.slice(0, cfg["max_train_samples"])
    rows = table.to_pylist()
    for row in rows:
        messages = row["prompt"] if isinstance(row["prompt"], list) else [{"role": "user", "content": row["prompt"]}]
        row["prompt"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **cfg["apply_chat_template_kwargs"]
        )
    rows = [
        row
        for row in rows
        if len(tokenizer.encode(row["prompt"], add_special_tokens=False)) <= cfg["rollout_max_prompt_len"]
    ]
    if not rows:
        raise ValueError("No training prompts remain after rollout_max_prompt_len filtering.")
    print(f"Preparing {len(rows)} student responses and teacher logprobs.", flush=True)
    url = f"http://127.0.0.1:{cfg['precompute_port']}"
    rollout_path = run_dir(cfg) / "data/student_rollouts.parquet"
    if rollout_path.exists():
        rollout_manifest = rollout_path.with_suffix(".manifest.json")
        if not rollout_manifest.exists() or json.loads(rollout_manifest.read_text()) != identity:
            raise ValueError("Existing student rollouts have different provenance; use a new experiment_name.")
        rows = pq.read_table(rollout_path).to_pylist()
        print(f"Reusing verified student responses: {rollout_path}", flush=True)
    else:
        print("Starting student server to collect responses.", flush=True)

        def generate(pair):
            index, row = pair
            output = request(
                url + "/generate",
                {
                    "text": row["prompt"],
                    "sampling_params": {
                        "max_new_tokens": cfg["rollout_max_response_len"],
                        "temperature": cfg["rollout_temperature"],
                        "top_p": cfg["rollout_top_p"],
                        "top_k": cfg["rollout_top_k"],
                        "sampling_seed": cfg["seed"] + index,
                        "skip_special_tokens": False,
                    },
                },
            )
            response_tokens = tokenizer.encode(output["text"], add_special_tokens=False)[
                : cfg["rollout_max_response_len"]
            ]
            if not response_tokens:
                raise ValueError(f"Student returned an empty response for row {index}.")
            return {
                "prompt": row["prompt"],
                "label": row["label"],
                "metadata": {
                    "is_lightning_opd": True,
                    "response_tokens": response_tokens,
                    "loss_mask": [1] * len(response_tokens),
                    "response": tokenizer.decode(response_tokens, skip_special_tokens=False),
                },
            }

        with background(
            server_command(
                cfg, student, cfg["precompute_port"], cfg["precompute_tp_size"], cfg["sglang_mem_fraction_static"]
            ),
            env,
            run_dir(cfg) / "student-server.log",
        ) as process:
            wait_ready(url + "/health_generate", cfg["startup_timeout"], process)
            with concurrent.futures.ThreadPoolExecutor(max_workers=cfg["precompute_concurrency"]) as pool:
                rows = list(pool.map(generate, enumerate(rows)))
        temporary = rollout_path.with_suffix(".tmp.parquet")
        pq.write_table(pa.Table.from_pylist(rows), temporary)
        temporary.replace(rollout_path)
        rollout_path.with_suffix(".manifest.json").write_text(json.dumps(identity, indent=2))
        print(f"Student server stopped; saved {len(rows)} responses: {rollout_path}", flush=True)

    def score(row):
        tokens = row["metadata"]["response_tokens"]
        output = request(
            url + "/generate",
            {
                "input_ids": tokenizer.encode(row["prompt"], add_special_tokens=False) + tokens,
                "sampling_params": {"temperature": 0, "max_new_tokens": 0, "skip_special_tokens": False},
                "return_logprob": True,
                "logprob_start_len": 0,
                **({"opd_response_length": len(tokens)} if needs_teacher_features(cfg) else {}),
            },
        )
        logprobs = [float(item[0]) for item in output["meta_info"]["input_token_logprobs"][-len(tokens):]]
        if len(logprobs) != len(tokens):
            raise ValueError("Teacher returned misaligned token logprobs.")
        row["metadata"]["teacher_log_probs"] = logprobs
        if needs_teacher_features(cfg):
            row["metadata"]["teacher_features"] = output["meta_info"]["opd_teacher_features"]
        return row

    print("Starting teacher server to precompute response-token logprobs.", flush=True)
    with background(
        server_command(
            cfg, teacher, cfg["precompute_port"], cfg["teacher_tp_size"], cfg["teacher_mem_fraction_static"], True
        ),
        env,
        run_dir(cfg) / "precompute-teacher.log",
    ) as process:
        wait_ready(url + "/health_generate", cfg["startup_timeout"], process)
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg["precompute_concurrency"]) as pool:
            rows = list(pool.map(score, rows))
    temporary = target.with_suffix(".tmp.parquet")
    pq.write_table(pa.Table.from_pylist(rows), temporary)
    temporary.replace(target)
    manifest.write_text(json.dumps(scored_provenance(cfg, student, teacher), indent=2))
    print(f"Teacher server stopped; prepared {len(rows)} examples: {target}", flush=True)


def cluster_nodes(cfg):
    if os.environ.get("SLURM_JOB_NODELIST"):
        hosts = subprocess.check_output(
            ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]], text=True
        ).split()
        # Compute nodes map their own short hostname to 127.0.1.1. Resolve
        # qualified names so every node sees the same routable IP ordering.
        hostname = socket.gethostname()
        domain = hostname.partition(".")[2]
        if domain:
            hosts = [host if "." in host else f"{host}.{domain}" for host in hosts]
        # slime orders bundles by numeric IP. Reserve the teacher on the final node
        # so the remaining GPUs form a contiguous prefix of that ordering.
        hosts = sorted(hosts, key=lambda host: tuple(map(int, socket.gethostbyname(host).split("."))))
        if len(hosts) != cfg["nnodes"]:
            raise ValueError(f"Slurm allocated {len(hosts)} nodes, YAML requires nnodes: {cfg['nnodes']}.")
        current_node = os.environ.get("SLURMD_NODENAME", hostname).split(".")[0]
        node_index = [host.split(".")[0] for host in hosts].index(current_node)
    else:
        if cfg["nnodes"] != 1:
            raise ValueError("Multi-node jobs must be launched through the provided sbatch/srun script.")
        hosts, node_index = ["127.0.0.1"], 0
    return hosts, node_index


def check_run_state(cfg, student, teacher):
    """Reject conflicting experiment state before starting expensive GPU work."""
    directory = run_dir(cfg)
    initialization = directory / "initialization.json"
    if initialization.exists() and json.loads(initialization.read_text()) != {"student": student, "teacher": teacher}:
        raise ValueError("This experiment used a different student/teacher. Choose a new experiment_name.")
    saved = directory / "checkpoints/latest_checkpointed_iteration.txt"
    if saved.exists() and not cfg["resume"]:
        raise ValueError("Existing training checkpoints: pass --resume or choose a new experiment_name.")
    if cfg["resume"] and not saved.exists():
        raise FileNotFoundError("--resume requires an existing experiment checkpoint.")
    if saved.exists():
        previous = directory / "config.yaml"
        if not previous.is_file():
            raise ValueError("OPD resume requires its saved configuration.")
        old = yaml.safe_load(previous.read_text())
        keys = ("method", "num_rollout", "rollout_batch_size", "global_batch_size", "train_files",
                "n_samples_per_prompt", "rollout_shuffle", "rollout_seed", "seed", "max_train_samples", "rollout_max_prompt_len",
                "rollout_max_response_len", "rollout_temperature", "rollout_top_p", "rollout_top_k",
                "input_key", "label_key", "metadata_key", "apply_chat_template", "apply_chat_template_kwargs",
                "optimizer", "lr", "lr_decay_style", "weight_decay", "adam_beta1", "adam_beta2",
                "advantage_estimator", "kl_loss_coef", "kl_loss_type", "entropy_coef")
        if lightning_blocks.enabled(cfg):
            keys += ("policy_rollout_k", "lightning_fixed_data")
        changed = [key for key in keys if old.get(key) != cfg.get(key)]
        # Missing mode in older snapshots means sample.
        old_mode, new_mode = old.get("opd_loss_token_mode", "sample"), cfg.get("opd_loss_token_mode", "sample")
        if old_mode != new_mode:
            changed.append("opd_loss_token_mode")
        elif new_mode != "sample" and old.get("opd_token_top_k", 32) != cfg.get("opd_token_top_k", 32):
            changed.append("opd_token_top_k")
        if changed:
            raise ValueError(f"OPD resume recipe changed ({', '.join(changed)}); use a new experiment_name.")
        completed = int(saved.read_text().strip())
        cursor = saved.parent / "rollout" / f"global_dataset_state_dict_{completed}.pt"
        weights = saved.parent / f"iter_{completed:07d}"
        if (not 0 <= completed < cfg["num_rollout"] or not cursor.is_file() or cursor.stat().st_size == 0
                or not weights.is_dir() or not any(weights.iterdir())):
            raise ValueError("OPD resume requires a valid training checkpoint and its saved rollout cursor.")


def preflight(cfg, student, teacher):
    hosts, node_index = cluster_nodes(cfg)
    devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if devices is not None and len(devices.split(",")) != cfg["n_gpus_per_node"]:
        raise ValueError("CUDA_VISIBLE_DEVICES must expose n_gpus_per_node GPUs to each srun task.")
    check_run_state(cfg, student, teacher)
    if cfg.get("eval_steps") and cfg["num_rollout"] % cfg["eval_steps"]:
        raise ValueError("For end-to-end final evaluation, num_rollout must be divisible by eval_steps.")
    if not lightning_blocks.enabled(cfg) and cfg.get("eval_steps") and cfg.get("save_interval") and cfg["num_rollout"] % cfg["save_interval"]:
        raise ValueError(
            "For an end-to-end final checkpoint, num_rollout must be divisible by the derived save_interval. "
            "Choose another save_eval_multiplier or set save_model: final."
        )
    if cfg["method"] == "lightning_opd" and not lightning_blocks.enabled(cfg):
        for path in (lightning_data(cfg), run_dir(cfg) / "data/student_rollouts.parquet"):
            manifest = path.with_suffix(".manifest.json")
            expected = scored_provenance(cfg, student, teacher) if path == lightning_data(cfg) else provenance(cfg, student, teacher)
            if path.exists() and (
                not manifest.exists() or json.loads(manifest.read_text()) != expected
            ):
                raise ValueError(f"{path}: cached data has different/unknown provenance; use a new experiment_name.")
    print(f"Preflight passed on node {node_index + 1}/{len(hosts)} ({hosts[node_index]}).", flush=True)


def train_cluster(cfg, student, teacher, env):
    hosts, node_index = cluster_nodes(cfg)
    directory = run_dir(cfg)
    directory.mkdir(parents=True, exist_ok=True)
    job = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    done = directory / f"cluster-{job}.done"
    head_ip = socket.gethostbyname(hosts[0])
    local_ip = socket.gethostbyname(hosts[node_index])
    port = 20000 + int(job) % 20000
    env.update(
        RAY_ADDRESS=f"{head_ip}:{port}",
        MASTER_ADDR=head_ip,
        no_proxy=f"127.0.0.1,localhost,{','.join(socket.gethostbyname(h) for h in hosts)}",
    )
    env["NO_PROXY"] = env["no_proxy"]
    env["WANDB_DIR"] = str(directory / "wandb")
    managed_teacher = cfg["method"] == "opd" and not cfg.get("teacher_url")
    reserved = cfg["teacher_num_gpus"] if managed_teacher and node_index == len(hosts) - 1 else 0
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", ",".join(map(str, range(cfg["n_gpus_per_node"])))).split(",")
    if len(devices) != cfg["n_gpus_per_node"]:
        raise ValueError("CUDA_VISIBLE_DEVICES must expose n_gpus_per_node GPUs to each srun task.")
    ray_env = env.copy()
    ray_env["CUDA_VISIBLE_DEVICES"] = ",".join(devices[:-reserved] if reserved else devices)
    teacher_url = cfg.get("teacher_url")
    if managed_teacher:
        teacher_url = f"http://{socket.gethostbyname(hosts[-1])}:{cfg['teacher_port']}/generate"
    try:
        with contextlib.ExitStack() as stack:
            if reserved:
                teacher_env = env.copy()
                teacher_env["CUDA_VISIBLE_DEVICES"] = ",".join(devices[-reserved:])
                teacher_process = stack.enter_context(
                    background(
                        server_command(
                            cfg,
                            teacher,
                            cfg["teacher_port"],
                            cfg["teacher_tp_size"],
                            cfg["teacher_mem_fraction_static"],
                            True,
                        ),
                        teacher_env,
                        directory / f"teacher-node{node_index}.log",
                    )
                )
                wait_ready(teacher_url.rsplit("/", 1)[0] + "/health_generate", cfg["startup_timeout"], teacher_process)
            ray_command = [
                sys.executable,
                "-m",
                "ray.scripts.scripts",
                "start",
                "--block",
                "--disable-usage-stats",
                "--node-ip-address",
                local_ip,
                "--num-gpus",
                str(len(devices) - reserved),
                "--num-cpus",
                os.environ.get("SLURM_CPUS_PER_TASK", "48"),
            ]
            if node_index == 0:
                done.unlink(missing_ok=True)
                check_inputs(cfg)
                check_run_state(cfg, student, teacher)
                initialization = directory / "initialization.json"
                identity = {"student": student, "teacher": teacher}
                initialization.write_text(json.dumps(identity, indent=2))
                checkpoint = checkpoint_path(cfg, student)
                if not (checkpoint / "latest_checkpointed_iteration.txt").is_file():
                    if not cfg["auto_prepare_model"]:
                        raise FileNotFoundError(
                            f"Converted checkpoint missing: {checkpoint}; enable auto_prepare_model."
                        )
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "torch.distributed.run",
                            "--standalone",
                            "--nproc-per-node",
                            str(cfg["conversion_num_gpus"]),
                            "tools/convert_hf_to_torch_dist.py",
                            *model_args(cfg, env),
                            "--attention-backend",
                            cfg["attention_backend"],
                            "--hf-checkpoint",
                            student,
                            "--save",
                            str(checkpoint),
                        ],
                        check=True,
                        cwd=ROOT,
                        env=ray_env,
                    )
                ray_command += [
                    "--head",
                    "--port",
                    str(port),
                    "--dashboard-host",
                    head_ip,
                    "--dashboard-port",
                    "8265",
                    "--temp-dir",
                    f"/tmp/lightning-opd-ray-{job}",
                ]
            else:
                wait_ready(f"http://{head_ip}:8265/api/version", cfg["startup_timeout"])
                ray_command += ["--address", env["RAY_ADDRESS"]]
            ray_process = stack.enter_context(
                background(ray_command, ray_env, directory / f"ray-node{node_index}.log")
            )
            if node_index:
                while not done.exists():
                    if ray_process.poll() is not None:
                        raise RuntimeError("Ray worker exited; inspect its log.")
                    if reserved and teacher_process.poll() is not None:
                        raise RuntimeError("Teacher exited; inspect its log.")
                    time.sleep(2)
                if done.read_text().strip() != "success":
                    raise RuntimeError("Training failed on the head node; inspect the Slurm log.")
                return
            wait_ready(f"http://{head_ip}:8265/api/version", cfg["startup_timeout"], ray_process)
            import ray

            ray.init(address=env["RAY_ADDRESS"])
            deadline = time.monotonic() + cfg["startup_timeout"]
            expected = cfg["nnodes"] * cfg["n_gpus_per_node"] - (cfg["teacher_num_gpus"] if managed_teacher else 0)
            while ray.cluster_resources().get("GPU", 0) < expected:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Ray did not register {expected} GPUs.")
                time.sleep(2)
            ray.shutdown()
            if teacher_url:
                wait_ready(teacher_url.rsplit("/", 1)[0] + "/health_generate", cfg["startup_timeout"])
            if lightning_blocks.enabled(cfg):
                lightning_blocks.prepare_fixed_prompts(cfg, student)
            elif cfg["method"] == "lightning_opd" and lightning_data(cfg).with_suffix(".manifest.json").exists():
                actual = json.loads(lightning_data(cfg).with_suffix(".manifest.json").read_text())
                if actual != scored_provenance(cfg, student, teacher):
                    raise ValueError(
                        "Precomputed data provenance differs from this student/teacher/sampling configuration."
                    )
            (directory / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
            command = [
                sys.executable,
                "train.py",
                *model_args(cfg, env),
                *training_args(cfg, student, teacher_url, prepare_eval(cfg)),
            ]
            print("Training command:", shlex.join(command), flush=True)
            subprocess.run(command, check=True, cwd=ROOT, env=env)
            status_path = directory / "training_status.json"
            status = json.loads(status_path.read_text()) if cfg.get("save_interval") is not None and status_path.is_file() else {}
            paused = status.get("status") == "paused"
            if lightning_blocks.enabled(cfg) and (not paused or status["global_step"] > 0):
                tracker = directory / "checkpoints/latest_checkpointed_iteration.txt"
                expected = (
                    status["global_step"] - 1 if paused
                    else cfg["num_rollout"] - 1 if cfg.get("save_model") == "final"
                    else lightning_blocks.schedule(cfg)[0]["end_step"] - 1
                )
                if not tracker.is_file() or int(tracker.read_text().strip()) != expected:
                    raise RuntimeError("Training exited without its required completed checkpoint.")
            if paused:
                continuation = ("resubmit to resume the saved checkpoint." if status["global_step"]
                                else "no updates completed; resubmit to start from initialization.")
                print(f"Training paused for wall time: {status['global_step']}/{cfg['num_rollout']} steps; "
                      + continuation, flush=True)
            if not paused and cfg.get("eval_steps") and cfg["num_rollout"] % cfg["eval_steps"] == 0:
                final_metrics = directory / "eval_metrics" / f"{cfg['num_rollout'] - 1}.json"
                if not final_metrics.is_file():
                    raise FileNotFoundError(f"Training exited without final evaluation metrics: {final_metrics}")
                print(f"Training and final evaluation completed: {final_metrics}", flush=True)
            if cfg.get("save_interval") is not None:
                prune_training_checkpoints(
                    directory / "checkpoints", cfg.get("save_total_limit"),
                    require_rollout_state=True,
                )
        if node_index == 0:
            done.write_text("success")
    except BaseException:
        if node_index == 0:
            done.write_text("failed")
        raise


def main(method="opd"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/config" / f"qwen3_4b_{method}.yaml")
    parser.add_argument("--env-file", type=Path, help="Override env_file from YAML")
    parser.add_argument("--resume", action="store_true", help="Resume this experiment's Megatron checkpoint")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--dry-run", action="store_true", help="Print configuration and CLI flags without writing or starting services"
    )
    modes.add_argument(
        "--check", action="store_true", help="Check local models, input schemas and runtime dependencies"
    )
    modes.add_argument(
        "--preflight",
        action="store_true",
        help="Check the allocated cluster and full runtime before generating offline data; no GPU services start",
    )
    modes.add_argument(
        "--prepare-data",
        action="store_true",
        help="Collect student rollouts and precompute teacher logprobs on one node",
    )
    modes.add_argument(
        "--prepare-prompts",
        action="store_true",
        help="CPU-only: prepare DAPO-Math-17k and all eval datasets as parquet",
    )
    cli = parser.parse_args()
    cfg = read_config(cli.config, method, resume=cli.resume)
    if cli.prepare_prompts:
        prepare_prompts(cfg)
        return
    if cli.prepare_data and method != "lightning_opd":
        parser.error("--prepare-data is only available for Lightning OPD.")
    load_env(cli.env_file or cfg.get("env_file"))
    env = runtime_env(cfg)
    sys.path[:0] = [str(ROOT)] + ([cfg["megatron_path"]] if cfg.get("megatron_path") else [])
    student, teacher = selected_models(cfg, download=not (cli.dry_run or cli.check or cli.preflight))
    if cli.dry_run:
        print(
            yaml.safe_dump(
                {
                    "method": method,
                    "student": student,
                    "teacher": teacher,
                    "results": str(run_dir(cfg)),
                    "lightning_opd_data": str(lightning_data(cfg)) if method == "lightning_opd" else None,
                    "policy_blocks": lightning_blocks.schedule(cfg) if lightning_blocks.enabled(cfg) else None,
                    "nodes": cfg["nnodes"],
                    "gpus_per_node": cfg["n_gpus_per_node"],
                    "teacher_gpus": cfg["teacher_num_gpus"] if method == "opd" else 0,
                },
                sort_keys=False,
            )
        )
        print(
            "Training flags:",
            shlex.join(
                training_args(
                    cfg,
                    student,
                    cfg.get("teacher_url")
                    or (f"http://<last-node>:{cfg['teacher_port']}/generate" if method == "opd" else None),
                    str(run_dir(cfg) / "eval.yaml") if cfg.get("eval_steps") else None,
                )
            ),
        )
        return
    check_dependencies(cfg, preparation=cli.prepare_data)
    check_inputs(cfg, preparation=cli.prepare_data or cli.preflight)
    if cli.preflight:
        preflight(cfg, student, teacher)
        return
    if cli.check:
        print("Local models, datasets and runtime dependencies are ready.")
        return
    os.environ.update(env)
    if cli.prepare_data:
        run_dir(cfg).mkdir(parents=True, exist_ok=True)
        prepare_lightning_data(cfg, student, teacher, env)
    else:
        train_cluster(cfg, student, teacher, env)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError, TimeoutError) as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        sys.exit(1)
