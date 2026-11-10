"""Keep every GPU of this node above a utilization floor while training leaves it idle.

    python3 tools/gpu_keepalive.py [--target 0.35] [--max-duty 0.35] [--log-dir tmp/gpu_keepalive]

The cluster reclaims machines whose GPUs stay under-used for 30 minutes. Training with dedicated
teacher/actor/rollout GPUs leaves some of them idle for long stretches (e.g. the teacher GPUs while the
student generates), and the platform's own keep-alive backs off whenever a job holds GPU memory.

One process per GPU runs bf16 matmul bursts only to fill the gap: every --period seconds it reads the
GPU's NVML utilization averaged over --window seconds, subtracts the share of time its own kernels ran,
and runs bursts for duty = clamp(target - others, 0, max_duty) of the next period. When training (or
the platform keep-alive) keeps the GPU busier than --target, it runs nothing. Memory: ~1 GB per GPU
(CUDA context plus two 8192 x 8192 bf16 matrices).
"""

import argparse
import collections
import multiprocessing
import os
from pathlib import Path
import signal
import time


def run_gpu(index, args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
    import pynvml
    import torch

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    torch.cuda.set_device(0)
    a = torch.randn(args.size, args.size, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(args.size, args.size, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):  # warm up and time one matmul
        torch.matmul(a, b)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    matmul_s = (time.perf_counter() - start) / 10

    samples = collections.deque()  # (time, nvml utilization 0-1, own busy seconds in that period)
    log = Path(args.log_dir) / f"gpu{index}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    last_log, own_busy = 0.0, 0.0
    with log.open("a") as out:
        out.write(f"{time.strftime('%F %T')} start gpu={index} matmul={matmul_s * 1e3:.2f}ms target={args.target} "
                  f"max_duty={args.max_duty}\n")
        out.flush()
        while True:
            now = time.time()
            samples.append((now, pynvml.nvmlDeviceGetUtilizationRates(handle).gpu / 100, own_busy))
            while samples and samples[0][0] < now - args.window:
                samples.popleft()
            utilization = sum(s[1] for s in samples) / len(samples)
            own = sum(s[2] for s in samples) / (len(samples) * args.period)
            others = max(0.0, utilization - own)
            duty = min(max(args.target - others, 0.0), args.max_duty)
            began = time.perf_counter()
            bursts = int(duty * args.period / matmul_s)
            if bursts:
                for _ in range(bursts):
                    torch.matmul(a, b)
                torch.cuda.synchronize()
            own_busy = time.perf_counter() - began if bursts else 0.0
            time.sleep(max(0.0, args.period - (time.perf_counter() - began)))
            if now - last_log >= args.log_every:
                out.write(f"{time.strftime('%F %T')} util={utilization:.2f} own={own:.2f} others={others:.2f} "
                          f"duty={duty:.2f}\n")
                out.flush()
                last_log = now


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpus", default=None, help="comma-separated GPU indices (default: all)")
    parser.add_argument("--target", type=float, default=0.35, help="utilization floor to fill up to (0-1)")
    parser.add_argument("--max-duty", type=float, default=0.35, help="largest share of time the filler may run")
    parser.add_argument("--period", type=float, default=0.2, help="seconds per control step")
    parser.add_argument("--window", type=float, default=5.0, help="seconds of utilization history to average")
    parser.add_argument("--size", type=int, default=8192, help="square matmul size")
    parser.add_argument("--log-dir", default=str(Path(__file__).resolve().parents[1] / "tmp/gpu_keepalive"))
    parser.add_argument("--log-every", type=float, default=60.0)
    args = parser.parse_args()
    import pynvml

    pynvml.nvmlInit()
    gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else list(range(pynvml.nvmlDeviceGetCount()))
    context = multiprocessing.get_context("spawn")
    workers = [context.Process(target=run_gpu, args=(gpu, args), daemon=True) for gpu in gpus]
    for worker in workers:
        worker.start()

    def stop(*_):
        for worker in workers:
            worker.terminate()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while True:  # restart a GPU's filler if it dies (e.g. a transient CUDA error)
        for i, worker in enumerate(workers):
            if not worker.is_alive():
                workers[i] = context.Process(target=run_gpu, args=(gpus[i], args), daemon=True)
                workers[i].start()
        time.sleep(10)


if __name__ == "__main__":
    main()
