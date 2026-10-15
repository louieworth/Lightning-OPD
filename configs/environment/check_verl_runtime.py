"""Validate the configured runtime on a GPU node with real kernels and models."""

import argparse
import importlib
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/config/qwen3_4b_lightning_opd.yaml")
    parser.add_argument("--kernels-only", action="store_true")
    parser.add_argument("--train-smoke", action="store_true", help="Also run one training step and two-row evaluation")
    cli = parser.parse_args()
    if cli.kernels_only and cli.train_smoke:
        parser.error("--train-smoke requires the model checks")
    launcher = importlib.import_module("configs.opd.qwen3-4b-opd")
    cfg = launcher.read_config(cli.config, "lightning_opd")
    launcher.load_env(cfg.get("env_file"))
    env = launcher.runtime_env(cfg)
    # Exercise only the backend selected by the training YAML.
    backend = cfg["attention_backend"]
    env.update(
        NVTE_FLASH_ATTN=str(int(backend in ("auto", "flash"))),
        NVTE_FUSED_ATTN=str(int(backend in ("auto", "fused"))),
        NVTE_UNFUSED_ATTN=str(int(backend in ("auto", "unfused"))),
    )
    os.environ.update(env)
    sys.path.insert(0, cfg["megatron_path"])
    launcher.check_dependencies(cfg)

    for name in (
        "sglang.srt.entrypoints.http_server",
        "sglang_router.launch_router",
        "megatron.training.training",
        "mbridge",
        "slime.backends.megatron_utils.actor",
        "train",
    ):
        importlib.import_module(name)
        print(f"Runtime import passed: {name}", flush=True)

    import torch
    from flash_attn import flash_attn_func
    import sgl_kernel
    import transformer_engine.pytorch as te
    from torch_memory_saver import torch_memory_saver

    if not torch.cuda.is_available():
        raise RuntimeError("Run this check inside a GPU allocation.")
    devices = torch.cuda.device_count()
    for index in range(devices):
        with torch.cuda.device(index):
            tensor = torch.randn(4, 64, dtype=torch.float16, device="cuda")
            norm = sgl_kernel.rmsnorm(tensor, torch.ones(64, dtype=torch.float16, device="cuda"), 1e-6)
            assert torch.isfinite(norm).all()
            query = torch.randn(1, 16, 2, 64, dtype=torch.float16, device="cuda")
            attention = flash_attn_func(query, query, query, causal=True)
            assert torch.isfinite(attention).all()
            layer = te.Linear(64, 32, params_dtype=torch.bfloat16, device="cuda")
            inputs = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            output = layer(inputs)
            if isinstance(output, tuple):
                output = output[0]
            output.float().sum().backward()
            assert torch.isfinite(inputs.grad).all()
            core_attention = te.DotProductAttention(
                32, 128, num_gqa_groups=8, qkv_format="thd", attn_mask_type="padding_causal"
            )
            packed_query = torch.randn(16, 32, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            packed_key = torch.randn(16, 8, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            packed_value = torch.randn_like(packed_key, requires_grad=True)
            boundaries = torch.tensor([0, 8, 16], device="cuda", dtype=torch.int32)
            core_output = core_attention(
                packed_query, packed_key, packed_value, cu_seqlens_q=boundaries, cu_seqlens_kv=boundaries,
                max_seqlen_q=8, max_seqlen_kv=8,
            )
            core_output.float().sum().backward()
            assert torch.isfinite(packed_query.grad).all()
            torch.cuda.synchronize()
            print(f"CUDA kernels passed: GPU {index} ({torch.cuda.get_device_name(index)})", flush=True)
    torch.cuda.set_device(0)
    with torch_memory_saver.region(tag="runtime-check", enable_cpu_backup=True):
        saved_tensor = torch.ones(4096, device="cuda")
    torch_memory_saver.pause(tag="runtime-check")
    torch_memory_saver.resume(tag="runtime-check")
    assert saved_tensor.sum().item() == 4096
    del saved_tensor, tensor, norm, query, attention, layer, inputs, output
    del core_attention, packed_query, packed_key, packed_value, boundaries, core_output
    torch.cuda.empty_cache()
    print("Memory offload/resume passed.", flush=True)

    directory = ROOT / ".runtime/verl/checks" / os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    directory.mkdir(parents=True, exist_ok=True)
    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "sglang", "sglang-router", "sgl-kernel", "torch-memory-saver", "megatron-core", "mbridge")
    }
    report = {
        "python": sys.executable, "versions": versions, "cuda": torch.version.cuda, "gpus": devices,
        "attention_backend": cfg["attention_backend"],
        "scope": "kernels" if cli.kernels_only else ("training_and_eval" if cli.train_smoke else "models"),
        "megatron_gpt_sha256": hashlib.sha256(
            (Path(cfg["megatron_path"]) / "megatron/core/models/gpt/gpt_model.py").read_bytes()
        ).hexdigest(),
    }
    if not cli.kernels_only:
        # Keep this existing two-sample hardware check independent of the full
        # fixed-data K-block experiment. It does not validate the K-block GPU path.
        cfg.pop("policy_rollout_k", None)
        cfg.pop("lightning_fixed_data", None)
        if devices < max(cfg["precompute_tp_size"], cfg["teacher_tp_size"]):
            raise RuntimeError("The model check requires at least two visible GPUs.")
        cfg.update(output_dir=str(directory), experiment_name="model-check", max_train_samples=2,
                   rollout_max_response_len=32, precompute_concurrency=2, conversion_num_gpus=min(4, devices))
        launcher.run_dir(cfg).mkdir(parents=True, exist_ok=True)
        student, teacher = launcher.selected_models(cfg)
        launcher.prepare_lightning_data(cfg, student, teacher, env)
        launcher.check_inputs(cfg)
        checkpoint = launcher.checkpoint_path(cfg, student)
        if not (checkpoint / "latest_checkpointed_iteration.txt").is_file():
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([
                sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node",
                str(cfg["conversion_num_gpus"]), "tools/convert_hf_to_torch_dist.py",
                *launcher.model_args(cfg, env), "--attention-backend", cfg["attention_backend"],
                "--hf-checkpoint", student, "--save", str(checkpoint),
            ], cwd=ROOT, env=env, check=True)
        if not (checkpoint / "latest_checkpointed_iteration.txt").is_file():
            raise RuntimeError("Model conversion did not produce a checkpoint tracker.")
        report.update(offline_data=str(launcher.lightning_data(cfg)), converted_checkpoint=str(checkpoint))
        if cli.train_smoke:
            import pyarrow.parquet as pq

            evaluation = launcher.run_dir(cfg) / "data/eval_smoke.parquet"
            pq.write_table(pq.read_table(cfg["val_files"][0]).slice(0, 2), evaluation)
            cfg.update(
                nnodes=1, n_gpus_per_node=devices, actor_num_nodes=1, actor_num_gpus_per_node=devices,
                rollout_num_gpus=devices, num_rollout=1, rollout_batch_size=2, global_batch_size=2,
                max_tokens_per_gpu=1024, eval_steps=1, save_interval=1, save_retain_interval=1,
                val_files=[str(evaluation)], val_n=1, val_max_completion_length=32, use_wandb=False,
            )
            launcher.preflight(cfg, student, teacher)
            launcher.train_cluster(cfg, student, teacher, env)
            tracker = launcher.run_dir(cfg) / "checkpoints/latest_checkpointed_iteration.txt"
            # Slime checkpoints use rollout IDs, so the first updated model is 0.
            if not tracker.is_file() or int(tracker.read_text().strip()) != 0:
                raise RuntimeError("Training smoke check did not save an updated checkpoint.")
            report["training_and_eval"] = str(launcher.run_dir(cfg) / "eval_metrics/0.json")
            report["training_checkpoint"] = str(tracker.parent)
    (directory / "passed.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Runtime validation passed: {directory / 'passed.json'}", flush=True)


if __name__ == "__main__":
    main()
