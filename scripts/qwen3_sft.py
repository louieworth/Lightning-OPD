"""Cluster SFT stages; all experiment settings come from one flat YAML."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
# torchrun starts workers with scripts/ on sys.path. The shared environment
# loader imports repo modules, even when PYTHONPATH contains only SFT packages.
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
PAUSED = 75


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run_dir(cfg):
    return Path(cfg["output_dir"]) / cfg["experiment_name"]


def local_model(name, cfg):
    path = Path(name)
    if not path.is_dir():
        cache = Path(cfg["hf_home"]) / "hub" / ("models--" + name.replace("/", "--"))
        revision = (cache / "refs/main").read_text().strip()
        path = cache / "snapshots" / revision
    index = path / "model.safetensors.index.json"
    weights = set(json.loads(index.read_text())["weight_map"].values()) if index.exists() else {"model.safetensors"}
    for filename in ["config.json", "tokenizer_config.json", *weights]:
        if not (path / filename).is_file():
            raise ValueError(f"Missing cached model file: {path / filename}")
    return path.resolve()


def setup_env(cfg):
    # Reuse the existing non-executing .env parser; never print credentials.
    spec = importlib.util.spec_from_file_location("opd_config", ROOT / "configs/opd/qwen3-4b-opd.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    launcher.load_env(cfg.get("env_file", ".env"))
    os.environ.update(HF_HOME=cfg["hf_home"], WANDB_MODE=cfg["wandb_mode"],
                      WANDB_PROJECT=cfg["project_name"], WANDB_ENTITY=cfg["wandb_entity"],
                      WANDB_DIR=str(run_dir(cfg)), TOKENIZERS_PARALLELISM="false")
    if cfg.get("local_models_only"):
        os.environ["HF_HUB_OFFLINE"] = "1"
    # One offline run per allocation; separate segments can be synced later.
    os.environ.pop("WANDB_RUN_ID", None)
    os.environ.pop("WANDB_RESUME", None)


def generation_identity(cfg):
    prompts_manifest = json.loads(Path(cfg["prompts_file"]).with_suffix(".manifest.json").read_text())
    return {"teacher": str(local_model(cfg["teacher_model_path"], cfg)),
            "prompt_sha256": prompts_manifest["prompts_sha256"],
            "world_size": cfg["nnodes"] * cfg["n_gpus_per_node"] // cfg["teacher_tensor_parallel_size"],
            **{key: cfg[key] for key in ("seed", "num_sft_prompts", "generation_batch_size",
                "generation_shard_size",
                "generation_temperature", "generation_top_p", "generation_max_tokens",
                "teacher_tensor_parallel_size", "teacher_max_model_len")}}


def sft_dataset_identity(cfg):
    if cfg.get("sft_data_source", "generated") == "generated":
        return generation_identity(cfg)
    if cfg["sft_data_source"] != "huggingface":
        raise ValueError("Unsupported sft_data_source")
    path = Path(cfg["train_files"])
    manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    expected = {"source": "huggingface", "dataset_repo": cfg["sft_dataset_repo"],
                "dataset_revision": cfg["sft_dataset_revision"],
                "filename": cfg["sft_dataset_filename"], "sha256": cfg["sft_dataset_sha256"],
                "size_bytes": cfg["sft_dataset_size_bytes"],
                "num_rows": cfg["num_sft_prompts"],
                "teacher_model_path": cfg["teacher_model_path"]}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Downloaded SFT dataset provenance differs from configuration")
    if path.stat().st_size != manifest["size_bytes"]:
        raise ValueError("Downloaded SFT dataset size changed")
    if pq.ParquetFile(path).metadata.num_rows != cfg["num_sft_prompts"]:
        raise ValueError("Downloaded SFT dataset is incomplete")
    if not manifest.get("audit", {}).get("all_rows_valid"):
        raise ValueError("Downloaded SFT dataset has not passed validation")
    return manifest


def check_resume_recipe(previous, cfg, allow_untrained=False):
    # Logging and checkpoint schedules do not change the trained model/data.
    mutable = {"project_name", "wandb_entity", "wandb_mode", "use_wandb", "env_file",
               "save_strategy", "save_steps", "save_total_limit", "walltime_checkpoint_margin_seconds"}
    topology_keys = {"nnodes", "n_gpus_per_node"}
    if topology_keys <= previous.keys() and topology_keys <= cfg.keys():
        # ZeRO optimizer and per-rank RNG state depend on the global GPU count,
        # not whether those ranks run on one node or several nodes.
        previous_world = previous["nnodes"] * previous["n_gpus_per_node"]
        current_world = cfg["nnodes"] * cfg["n_gpus_per_node"]
        if previous_world == current_world:
            mutable.update(topology_keys)
    if allow_untrained:
        # An initialization failure has no optimizer state to migrate. Its
        # completed tokenized data remain valid with a different ZeRO stage.
        mutable.add("deepspeed_stage")
    changed = sorted(key for key in previous.keys() | cfg.keys()
                     if key not in mutable and previous.get(key) != cfg.get(key))
    if changed:
        raise ValueError("Training/data recipe changed: " + ", ".join(changed) + "; use a new experiment_name")


def configure(cfg, config_path):
    if cfg["nnodes"] * cfg["n_gpus_per_node"] * cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"] != cfg["global_batch_size"]:
        raise ValueError("GPU count x micro batch x gradient accumulation must equal global_batch_size")
    if cfg["n_gpus_per_node"] % cfg["teacher_tensor_parallel_size"]:
        raise ValueError("teacher_tensor_parallel_size must divide n_gpus_per_node")
    if cfg["generation_shard_size"] % cfg["generation_batch_size"]:
        raise ValueError("generation_shard_size must be a multiple of generation_batch_size")
    output = run_dir(cfg)
    pipeline = output / ".pipeline"
    pipeline.mkdir(parents=True, exist_ok=True)
    snapshot = pipeline / "config.yaml"
    if snapshot.exists():
        previous = yaml.safe_load(snapshot.read_text())
        has_training_state = any(output.glob("checkpoint-*")) or (output / "trainer_state.json").exists()
        check_resume_recipe(previous, cfg, allow_untrained=not has_training_state)
    snapshot.write_text(config_path.read_text())
    identity = sft_dataset_identity(cfg)
    if cfg.get("sft_data_source", "generated") == "generated":
        generation = Path(cfg["generation_dir"])
        generation.mkdir(parents=True, exist_ok=True)
        manifest = generation / "manifest.json"
    else:
        manifest = pipeline / "dataset_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("SFT dataset provenance changed; choose a new data/experiment directory")
    if not manifest.exists():
        write_json(manifest, identity)
    dataset_info = {"openthoughts3_qwen3_8b": {
        "file_name": cfg["train_files"], "formatting": "sharegpt", "columns": {"messages": "messages"},
        "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user",
                 "assistant_tag": "assistant", "system_tag": "system"}}}
    write_json(pipeline / "dataset_info.json", dataset_info)
    deepspeed = {"train_batch_size": "auto", "train_micro_batch_size_per_gpu": "auto",
                "gradient_accumulation_steps": "auto", "gradient_clipping": "auto",
                "zero_allow_untested_optimizer": True, "bf16": {"enabled": "auto"},
                "fp16": {"enabled": "auto", "loss_scale": 0, "loss_scale_window": 1000,
                         "initial_scale_power": 16, "hysteresis": 2, "min_loss_scale": 1},
                "zero_optimization": {"stage": cfg["deepspeed_stage"], "allgather_partitions": True,
                    "allgather_bucket_size": 500000000, "overlap_comm": False, "reduce_scatter": True,
                    "reduce_bucket_size": 500000000, "contiguous_gradients": True,
                    "round_robin_gradients": True}}
    write_json(pipeline / "deepspeed.json", deepspeed)
    train_keys = ("stage", "do_train", "finetuning_type", "enable_liger_kernel", "packing", "template",
                  "cutoff_len", "learning_rate", "max_steps", "lr_scheduler_type", "warmup_ratio", "bf16",
                  "flash_attn", "gradient_checkpointing", "optim", "adam_beta1", "adam_beta2", "adam_epsilon",
                  "weight_decay", "max_grad_norm", "train_on_prompt", "preprocessing_num_workers",
                  "dataloader_persistent_workers", "dataloader_pin_memory", "dataloader_num_workers",
                  "logging_steps", "save_total_limit", "save_only_model", "plot_loss",
                  "overwrite_cache", "overwrite_output_dir", "ddp_timeout", "per_device_train_batch_size",
                  "gradient_accumulation_steps", "seed")
    train = {key: cfg[key] for key in train_keys}
    train.update(save_strategy=cfg.get("save_strategy", "steps"), save_steps=cfg.get("save_steps", 500))
    train.update(model_name_or_path=str(local_model(cfg["model_path"], cfg)),
                 dataset="openthoughts3_qwen3_8b", dataset_dir=str(pipeline),
                 output_dir=str(output), run_name=cfg["experiment_name"],
                 deepspeed=str(pipeline / "deepspeed.json"),
                 report_to="wandb" if cfg["use_wandb"] else "none", do_eval=False,
                 data_shared_file_system=True, tokenized_path=str(pipeline / "tokenized"),
                 cache_dir=str(pipeline / "hf_cache"))
    (pipeline / "llamafactory.yaml").write_text(yaml.safe_dump(train, sort_keys=False))
    teacher = identity.get("teacher", identity.get("teacher_model_path"))
    print(f"SFT configured: {output}; dataset source: {cfg.get('sft_data_source', 'generated')}; "
          f"teacher: {teacher}", flush=True)


def deadline_reached():
    return time.time() >= float(os.environ.get("SFT_CHECKPOINT_DEADLINE", "inf"))


def compact_generation_shard(directory, start, end, cfg):
    paths = [directory / f"batch-{index // cfg['generation_batch_size']:06d}.parquet"
             for index in range(start, end, cfg["generation_batch_size"])]
    if not all(path.exists() for path in paths):
        return None
    output = directory / f"shard-{start // cfg['generation_shard_size']:06d}.parquet"
    temporary = output.with_name(output.name + ".tmp")
    with pq.ParquetWriter(temporary, pq.read_schema(paths[0]), compression="zstd") as writer:
        for path in paths:
            writer.write_table(pq.read_table(path))
    temporary.replace(output)
    # Only redundant generated batch copies are removed, after the larger
    # Parquet is atomically published. Every response remains on shared storage.
    for path in paths:
        path.unlink()
    return output


def generate_worker(cfg, rank):
    identity = generation_identity(cfg)
    world = identity["world_size"]
    prompts = pq.read_table(cfg["prompts_file"]).to_pylist()[rank::world]
    directory = Path(cfg["generation_dir"]) / f"rank{rank:05d}"
    directory.mkdir(parents=True, exist_ok=True)
    batch_size = cfg["generation_batch_size"]
    ready = {}
    shard_size = cfg["generation_shard_size"]
    for shard_start in range(0, len(prompts), shard_size):
        shard_end = min(shard_start + shard_size, len(prompts))
        path = directory / f"shard-{shard_start // shard_size:06d}.parquet"
        if path.exists():
            ids = pq.read_table(path, columns=["sample_id"]).column(0).to_pylist()
            if ids != [item["sample_id"] for item in prompts[shard_start:shard_end]]:
                raise ValueError(f"Invalid compacted generation shard: {path}")
            for batch_start in range(shard_start, shard_end, batch_size):
                ready[batch_start] = path
    for batch_start in range(0, len(prompts), batch_size):
        batch_index = batch_start // batch_size
        path = directory / f"batch-{batch_index:06d}.parquet"
        if path.exists():
            ids = pq.read_table(path, columns=["sample_id"]).column(0).to_pylist()
            expected = [item["sample_id"] for item in prompts[batch_start:batch_start + batch_size]]
            if ids != expected:
                raise ValueError(f"Invalid generation shard: {path}")
            if batch_start in ready:
                # A prior interruption may have occurred between publishing a
                # compacted shard and deleting its redundant small batch copies.
                path.unlink()
            else:
                ready[batch_start] = path
    for shard_start in range(0, len(prompts), shard_size):
        shard_end = min(shard_start + shard_size, len(prompts))
        if all(index in ready for index in range(shard_start, shard_end, batch_size)):
            compact_generation_shard(directory, shard_start, shard_end, cfg)
    if len(ready) == (len(prompts) + batch_size - 1) // batch_size:
        print(f"Rank {rank}: all {len(prompts)} responses already ready", flush=True)
        return 0
    if deadline_reached():
        return PAUSED
    os.environ["VLLM_ATTENTION_BACKEND"] = cfg["generation_attention_backend"]
    from vllm import LLM, SamplingParams

    llm = LLM(model=identity["teacher"], tensor_parallel_size=cfg["teacher_tensor_parallel_size"],
              trust_remote_code=True, seed=cfg["seed"], max_model_len=cfg["teacher_max_model_len"],
              gpu_memory_utilization=cfg["teacher_gpu_memory_utilization"])
    sampling = SamplingParams(temperature=cfg["generation_temperature"], top_p=cfg["generation_top_p"],
                              max_tokens=cfg["generation_max_tokens"], n=1)
    for start in range(0, len(prompts), batch_size):
        if start in ready:
            continue
        if deadline_reached():
            print(f"Rank {rank}: wall-time checkpoint; resubmit to continue", flush=True)
            return PAUSED
        batch = prompts[start:start + batch_size]
        outputs = llm.chat([item["prompt"] for item in batch], sampling, use_tqdm=False)
        if len(outputs) != len(batch):
            raise ValueError("Teacher returned an incomplete batch")
        rows = []
        for item, output in zip(batch, outputs):
            completion = output.outputs[0]
            text = completion.text
            # Identical thinking-tag repair to data_curation/pipeline.py.
            if "</think>" in text and not text.strip().startswith("<think>"):
                text = "<think>\n" + text
            rows.append({"sample_id": item["sample_id"], "messages": item["prompt"] + [
                {"role": "assistant", "content": text}], "tokens": len(completion.token_ids)})
        path = directory / f"batch-{start // batch_size:06d}.parquet"
        temporary = path.with_name(path.name + ".tmp")
        pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
        temporary.replace(path)
        end = start + len(batch)
        if end % shard_size == 0 or end == len(prompts):
            compact_generation_shard(directory, start // shard_size * shard_size, end, cfg)
        print(f"Rank {rank}: saved samples {start}:{start + len(batch)} / {len(prompts)}", flush=True)
    return 0


def generate_node(cfg, config_path):
    node = int(os.environ["SLURM_PROCID"])
    tp = cfg["teacher_tensor_parallel_size"]
    workers_per_node = cfg["n_gpus_per_node"] // tp
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(devices) != cfg["n_gpus_per_node"] or not all(devices):
        raise ValueError(f"Expected {cfg['n_gpus_per_node']} Slurm-assigned GPUs, got {devices}")
    processes = []
    paused = False
    for local in range(workers_per_node):
        rank = node * workers_per_node + local
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(devices[local * tp:(local + 1) * tp])
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            env.pop(key, None)
        processes.append(subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
            "--config", str(config_path), "--stage", "generate-worker", "--rank", str(rank)],
            cwd=ROOT, env=env))
    # Poll all workers: an early failure must not wait for healthy workers to
    # generate their entire 37.5K shard before the allocation is released.
    try:
        while processes:
            for process in list(processes):
                status = process.poll()
                if status is None:
                    continue
                processes.remove(process)
                if status == PAUSED:
                    paused = True
                elif status != 0:
                    return status
            time.sleep(1)
        if paused:
            print(f"Node {node}: generation saved at wall-time deadline", flush=True)
        return 0
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            process.wait()


def merge(cfg):
    output = Path(cfg["train_files"])
    manifest = output.with_suffix(".manifest.json")
    identity = generation_identity(cfg)
    if output.exists() and manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError("SFT dataset teacher provenance mismatch")
        if pq.ParquetFile(output).metadata.num_rows != cfg["num_sft_prompts"]:
            raise ValueError("SFT dataset is incomplete")
        print(f"Reusing completed SFT Parquet: {output}", flush=True)
        return
    expected = set(pq.read_table(cfg["prompts_file"], columns=["sample_id"]).column(0).to_pylist())
    files = sorted(Path(cfg["generation_dir"]).glob("rank*/*.parquet"))
    total = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    if total != cfg["num_sft_prompts"]:
        raise ValueError(f"Teacher generation incomplete: {total}/{cfg['num_sft_prompts']}")
    temporary = output.with_name(output.name + ".tmp")
    seen = set()
    with pq.ParquetWriter(temporary, pq.read_schema(files[0]), compression="zstd") as writer:
        for path in files:
            table = pq.read_table(path)
            ids = table.column("sample_id").to_pylist()
            if len(set(ids)) != len(ids) or seen.intersection(ids) or not set(ids).issubset(expected):
                raise ValueError(f"Duplicate or unexpected sample IDs: {path}")
            seen.update(ids)
            writer.write_table(table)
    if seen != expected:
        raise ValueError("Teacher responses do not cover the selected prompts")
    temporary.replace(output)
    write_json(manifest, identity)
    print(f"Merged {len(seen)} SAME-teacher responses: {output}", flush=True)


def train(cfg):
    if cfg.get("sft_data_source", "generated") == "huggingface":
        sft_dataset_identity(cfg)
    import torch
    from transformers import TrainerCallback
    from llamafactory.train.tuner import run_exp

    class WalltimeCheckpoint(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            should_stop = torch.tensor(int(deadline_reached()), device=torch.cuda.current_device())
            if torch.distributed.is_initialized():
                torch.distributed.broadcast(should_stop, src=0)
            if should_stop.item() and state.global_step < cfg["max_steps"]:
                control.should_save = True
                control.should_training_stop = True
            elif state.global_step >= cfg["max_steps"]:
                # save_strategy: no also disables the Trainer's final checkpoint.
                # Keep a complete optimizer/RNG checkpoint at the training target.
                control.should_save = True
            return control

        def on_save(self, args, state, control, **kwargs):
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            if state.is_world_process_zero:
                checkpoint = run_dir(cfg) / f"checkpoint-{state.global_step}"
                write_json(checkpoint / "sft_checkpoint_complete.json", {"global_step": state.global_step})
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            return control

    args = yaml.safe_load((run_dir(cfg) / ".pipeline/llamafactory.yaml").read_text())
    checkpoints = sorted((path for path in run_dir(cfg).glob("checkpoint-*")
                          if (path / "trainer_state.json").exists()
                          and (path / "sft_checkpoint_complete.json").exists()),
                         key=lambda path: int(path.name.split("-")[-1]))
    if cfg["resume"] and checkpoints:
        latest = checkpoints[-1]
        if json.loads((latest / "trainer_state.json").read_text())["global_step"] >= cfg["max_steps"]:
            # The final optimizer checkpoint may be complete even if Slurm
            # interrupted the following HF export. Restore the final export.
            if int(os.environ.get("RANK", "0")) == 0:
                for source in latest.iterdir():
                    if source.is_file() and source.name not in {
                        "training_args.bin", "sft_checkpoint_complete.json"} and (
                        source.suffix in {".json", ".safetensors", ".model", ".txt"}):
                        target = run_dir(cfg) / source.name
                        temporary = target.with_name(target.name + ".tmp")
                        shutil.copyfile(source, temporary)
                        temporary.replace(target)
            print("SFT has already reached max_steps", flush=True)
            return 0
        args["resume_from_checkpoint"] = str(latest)
        print(f"Resuming complete training state from {latest}", flush=True)
    elif any(run_dir(cfg).glob("checkpoint-*")) or (run_dir(cfg) / "trainer_state.json").exists():
        raise ValueError("Existing output without a complete resumable checkpoint; inspect before restarting")
    run_exp(args, callbacks=[WalltimeCheckpoint()])
    return 0


def finish(cfg):
    state = json.loads((run_dir(cfg) / "trainer_state.json").read_text())
    complete = state["global_step"] >= cfg["max_steps"]
    write_json(run_dir(cfg) / ".pipeline/status.json", {
        "status": "complete" if complete else "paused", "global_step": state["global_step"],
        "target_steps": cfg["max_steps"], "teacher_model_path": cfg["teacher_model_path"],
        "sft_data_source": cfg.get("sft_data_source", "generated"),
        "sft_dataset_repo": cfg.get("sft_dataset_repo"),
        "model_path": str(run_dir(cfg)), "slurm_job_id": os.environ.get("SLURM_JOB_ID")})
    print(f"SFT {'complete' if complete else 'paused for wall time'}: "
          f"{state['global_step']}/{cfg['max_steps']} steps; {run_dir(cfg)}", flush=True)


def pause_generation(cfg):
    write_json(run_dir(cfg) / ".pipeline/status.json", {
        "status": "paused", "phase": "teacher_generation", "global_step": 0,
        "target_steps": cfg["max_steps"], "target_responses": cfg["num_sft_prompts"],
        "teacher_model_path": cfg["teacher_model_path"],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/config/qwen3_4b_sft.yaml")
    parser.add_argument("--stage", required=True,
                        choices=["configure", "generate-node", "generate-worker", "merge", "train", "finish", "pause-generation"])
    parser.add_argument("--rank", type=int)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if cfg.get("sft_data_source", "generated") != "generated" and args.stage in {
        "generate-node", "generate-worker", "merge", "pause-generation"}:
        raise ValueError("Downloaded data requires the direct-SFT sbatch; generation stages are disabled")
    if args.stage == "configure":
        configure(cfg, args.config)
    elif args.stage == "generate-node":
        return generate_node(cfg, args.config)
    elif args.stage == "generate-worker":
        return generate_worker(cfg, args.rank)
    elif args.stage == "merge":
        merge(cfg)
    elif args.stage == "train":
        setup_env(cfg)
        return train(cfg)
    elif args.stage == "finish":
        finish(cfg)
    elif args.stage == "pause-generation":
        pause_generation(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
