"""Validate the packaged backend and the actual eight-H100 communication path."""

import argparse
from datetime import timedelta
import importlib
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys


def imports():
    marker = Path('/opt/lightning-container.json')
    if not marker.is_file():
        raise RuntimeError('This is not the prepared OPD image. Build it with bash run_docker.sh build.')
    manifest = json.loads(marker.read_text())
    if manifest.get('schema') != 1 or not all(manifest.get(key) for key in ('base', 'source_sha256', 'versions')):
        raise RuntimeError('The OPD image build manifest is incomplete; rebuild the image.')
    sources = {}
    for name in ('torch', 'ray', 'sglang', 'sglang_router', 'mbridge', 'math_verify',
                 'pyarrow', 'transformer_engine.pytorch', 'flash_attn', 'sgl_kernel',
                 'torch_memory_saver', 'megatron.core', 'megatron.training.training',
                 'slime.backends.megatron_utils.actor', 'slime.rollout.sglang_rollout', 'train'):
        module = importlib.import_module(name)
        source = getattr(module, '__file__', None)
        if source and ('/.runtime/' in source or '/envs/verl/' in source):
            raise RuntimeError(f'{name} was loaded from a host environment: {source}')
        sources[name] = source
    if not sources['megatron.core'].startswith('/opt/Megatron-LM/'):
        raise RuntimeError('Megatron must come from the full source inside the image.')
    repository = Path(os.environ.get('LIGHTNING_OPD_REPO', '/opt/Lightning-OPD')).resolve()
    for name in ('slime.backends.megatron_utils.actor', 'slime.rollout.sglang_rollout', 'train'):
        if not Path(sources[name]).resolve().is_relative_to(repository):
            raise RuntimeError(f'{name} must come from the current repository: {sources[name]}')
    patch = manifest.get('tms_native_patch', {})
    hook = Path(sources['torch_memory_saver']).parent.parent / 'torch_memory_saver_hook_mode_preload.abi3.so'
    if (patch.get('fix') != 'cuda-free-owned-pointer-outside-region'
            or patch.get('library') != str(hook.resolve())
            or hashlib.sha256(hook.read_bytes()).hexdigest() != patch.get('library_sha256')):
        raise RuntimeError('The image is missing its verified TorchMemorySaver cudaFree fix; rebuild it.')
    freeze = Path('/opt/lightning-pip-freeze.txt')
    if not freeze.is_file():
        raise RuntimeError('The image is missing its complete package inventory; rebuild it.')
    return {'python': sys.executable, 'sources': sources, **manifest,
            'pip_freeze_sha256': hashlib.sha256(freeze.read_bytes()).hexdigest()}


def preload_memory_hook():
    if os.environ.get('LIGHTNING_OPD_CHECK_PRELOADED'):
        return
    package = importlib.util.find_spec('torch_memory_saver')
    if package is None or not package.origin:
        raise RuntimeError('TorchMemorySaver is missing from the image.')
    hook = Path(package.origin).parent.parent / 'torch_memory_saver_hook_mode_preload.abi3.so'
    if not hook.is_file():
        raise RuntimeError(f'TorchMemorySaver CUDA hook missing: {hook}')
    env = os.environ.copy()
    env['LD_PRELOAD'] = str(hook)
    env['LIGHTNING_OPD_CHECK_PRELOADED'] = '1'
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


def distributed():
    import torch
    import torch.distributed as dist
    from flash_attn import flash_attn_func
    import sgl_kernel
    import transformer_engine.pytorch as te
    from torch_memory_saver import torch_memory_saver

    local_rank = int(os.environ['LOCAL_RANK'])
    if torch.cuda.device_count() != 4:
        raise RuntimeError(f'Expected four visible GPUs per node, got {torch.cuda.device_count()}')
    torch.cuda.set_device(local_rank)
    gpu = torch.cuda.get_device_name(local_rank)
    if 'H100' not in gpu:
        raise RuntimeError(f'Expected H100, got {gpu}')
    dist.init_process_group('nccl', timeout=timedelta(minutes=5))
    try:
        if dist.get_world_size() != 8:
            raise RuntimeError('The distributed check requires eight GPU ranks on two nodes.')
        value = torch.tensor([dist.get_rank() + 1.0], device='cuda')
        dist.all_reduce(value)
        if value.item() != 36:
            raise RuntimeError(f'NCCL all-reduce returned {value.item()}, expected 36.')
        group = dist.new_group(backend='gloo', timeout=timedelta(minutes=5))
        control = torch.ones(1)
        dist.all_reduce(control, group=group)
        if control.item() != 8:
            raise RuntimeError('Cross-node Gloo all-reduce failed.')

        query = torch.randn(1, 16, 2, 64, dtype=torch.float16, device='cuda')
        attention = flash_attn_func(query, query, query, causal=True)
        norm = sgl_kernel.rmsnorm(query.reshape(-1, 64), torch.ones(64, device='cuda', dtype=query.dtype), 1e-6)
        if not torch.isfinite(attention).all() or not torch.isfinite(norm).all():
            raise RuntimeError('FlashAttention/SGLang kernel returned non-finite values.')
        layer = te.Linear(64, 32, params_dtype=torch.bfloat16, device='cuda')
        inputs = torch.randn(4, 64, dtype=torch.bfloat16, device='cuda', requires_grad=True)
        output = layer(inputs)
        (output[0] if isinstance(output, tuple) else output).float().sum().backward()
        if not torch.isfinite(inputs.grad).all():
            raise RuntimeError('Transformer Engine backward failed.')
        packed_attention = te.DotProductAttention(32, 128, num_gqa_groups=8, qkv_format='thd',
                                                 attn_mask_type='padding_causal')
        q = torch.randn(16, 32, 128, dtype=torch.bfloat16, device='cuda', requires_grad=True)
        k = torch.randn(16, 8, 128, dtype=torch.bfloat16, device='cuda', requires_grad=True)
        v = torch.randn_like(k, requires_grad=True)
        boundaries = torch.tensor([0, 8, 16], dtype=torch.int32, device='cuda')
        packed_attention(q, k, v, cu_seqlens_q=boundaries, cu_seqlens_kv=boundaries,
                         max_seqlen_q=8, max_seqlen_kv=8).float().sum().backward()
        if not torch.isfinite(q.grad).all():
            raise RuntimeError('Packed GQA attention backward failed.')
        with torch_memory_saver.region(tag='container-check', enable_cpu_backup=True):
            saved = torch.ones(4096, device='cuda')
        torch_memory_saver.pause(tag='container-check')
        torch_memory_saver.resume(tag='container-check')
        if saved.sum().item() != 4096:
            raise RuntimeError('GPU memory offload/resume failed.')
        torch.cuda.synchronize()
        # Exercise the failing cudaFree path before writing a passed report.
        # The region has ended, so its VMM addresses must still be recognized.
        if torch_memory_saver._impl._binary_wrapper.cdll.tms_get_interesting_region():
            raise RuntimeError('The memory-release check must run outside a TMS region.')
        del saved
        pool = torch_memory_saver._impl._mem_pools.pop(('container-check', True))
        del pool
        torch.cuda.synchronize()
        dist.barrier()
        return {'rank': dist.get_rank(), 'world_size': 8, 'gpu': gpu,
                'checks': ['NCCL', 'Gloo', 'FlashAttention', 'SGLang RMSNorm',
                           'Transformer Engine backward', 'packed GQA backward', 'memory offload/resume',
                           'GPU memory release outside a TMS region']}
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--imports', action='store_true')
    mode.add_argument('--distributed', action='store_true')
    parser.add_argument('--report-dir', type=Path, required=True)
    cli = parser.parse_args()
    # Match the selected H100 recipes' fused Transformer Engine backend;
    # FlashAttention is also exercised explicitly below.
    os.environ['NVTE_FLASH_ATTN'] = '0'
    os.environ['NVTE_FUSED_ATTN'] = '1'
    os.environ['NVTE_UNFUSED_ATTN'] = '0'
    if cli.distributed:
        preload_memory_hook()
    report = imports()
    rank = os.environ.get('LIGHTNING_OPD_NODE_RANK', '0')
    name = f'imports-node-{rank}.json'
    if cli.distributed:
        report.update(distributed())
        name = f'gpu-rank-{report["rank"]}.json'
    cli.report_dir.mkdir(parents=True, exist_ok=True)
    (cli.report_dir / name).write_text(json.dumps(report, indent=2) + '\n')
    print(f'Container check passed: {name}', flush=True)


if __name__ == '__main__':
    main()
