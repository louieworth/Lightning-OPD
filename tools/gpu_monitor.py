"""Watch how busy every GPU of the Ray cluster is and flag under-use.

    python3 tools/gpu_monitor.py --out tmp/gpu_monitor/<run> [--log <training log>]

Each node is sampled once per second with DCGM (`dcgmi dmon`) through a CPU-only Ray actor;
nothing runs on the GPUs. A GPU counts as busy by max(SM active, DRAM active), because decoding
is memory-bound: a well-used GPU during generation can show modest SM activity but high DRAM
activity. Without profiling values the GPU utilization counter is used instead. A GPU where this
job holds no memory counts as idle: a host process outside the container runs kernels on idle GPUs.

Outputs in --out: samples.csv (every GPU, every second), status.log (one line per interval with
the latest training-log line as a phase hint) and alerts.log (deduplicated alerts).
"""

import argparse
import csv
import dataclasses
import os
import statistics
import subprocess
import time
from pathlib import Path

FIELDS = {1002: "sm_active", 1003: "sm_occupancy", 1004: "tensor_active", 1005: "dram_active",
          203: "gpu_util", 252: "fb_used_mib"}


@dataclasses.dataclass(frozen=True)
class Sample:
    time: float
    node: str
    gpu: int
    metrics: dict


@dataclasses.dataclass(frozen=True)
class Alert:
    kind: str
    message: str
    gpus: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class Report:
    mean_busy: float | None
    per_gpu: dict
    alerts: list


def parse_dmon(text, time, node, fields=tuple(FIELDS.values())):
    samples = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2 + len(fields) or parts[0] != "GPU" or not parts[1].isdigit():
            continue
        values = [None if value == "N/A" else float(value) for value in parts[2:2 + len(fields)]]
        samples.append(Sample(time, node, int(parts[1]), dict(zip(fields, values, strict=True))))
    return samples


# A host process outside the container holds ~0.8 GB on every GPU and runs kernels on GPUs this job
# leaves idle. Our engines and trainers always allocate several GB on the GPUs they use, so activity
# on a GPU holding less than this is not ours.
OURS_MIN_MIB = 2048


def busy(sample):
    used = sample.metrics.get("fb_used_mib")
    if used is not None and used < OURS_MIN_MIB:
        return 0.0
    activity = [sample.metrics.get(key) for key in ("sm_active", "dram_active")]
    activity = [value for value in activity if value is not None]
    if activity:
        return max(activity)
    util = sample.metrics.get("gpu_util")
    return None if util is None else util / 100


def assess(samples, now, window_s, low=0.5, idle=0.05):
    """Alerts for the last `window_s` seconds: whole-cluster stall, idle GPUs beside busy ones, low mean."""
    values = {}
    for sample in samples:
        value = busy(sample)
        if now - window_s < sample.time <= now and value is not None:
            values.setdefault((sample.node, sample.gpu), []).append(value)
    per_gpu = {gpu: statistics.fmean(found) for gpu, found in sorted(values.items())}
    if not per_gpu:
        return Report(None, per_gpu, [])
    times = [sample.time for sample in samples if now - window_s < sample.time <= now]
    if max(times) - min(times) < 0.9 * window_s:  # Alert only on a fully observed window.
        return Report(statistics.fmean(per_gpu.values()), per_gpu, [])
    mean = statistics.fmean(per_gpu.values())
    minutes = window_s / 60
    if max(per_gpu.values()) < idle:
        return Report(mean, per_gpu, [Alert("STALLED", f"every GPU idle for {minutes:g} min", list(per_gpu))])
    alerts = []
    idle_gpus = [gpu for gpu, value in per_gpu.items() if value < idle]
    if idle_gpus and max(per_gpu.values()) >= low:
        alerts.append(Alert("IDLE_GPUS", f"{len(idle_gpus)} GPUs idle for {minutes:g} min while others work",
                            idle_gpus))
    if mean < low:
        alerts.append(Alert("LOW_UTILIZATION", f"mean busy {mean:.0%} < {low:.0%} over {minutes:g} min"))
    return Report(mean, per_gpu, alerts)


def phase_hint(log_path, size=65536):
    """The latest informative training-log line, to attribute utilization to a phase."""
    if not log_path or not Path(log_path).is_file():
        return ""
    with open(log_path, "rb") as stream:
        stream.seek(max(0, os.path.getsize(log_path) - size))
        lines = stream.read().decode(errors="ignore").splitlines()
    keys = ("Policy rollout", "student_rollouts", "lightning_opd.parquet", "eval", "rollout", "step", "Training")
    for line in reversed(lines):
        if "NCCL" not in line and any(key in line for key in keys):
            return " ".join(line.split())[-160:]
    return ""


def samplers():
    """One CPU-only DCGM sampler actor per alive node (shipped by value; no repository code needed)."""
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    class Sampler:
        def read(self, seconds, field_ids):
            import subprocess
            import time

            start = time.time()
            result = subprocess.run(["dcgmi", "dmon", "-e", ",".join(map(str, field_ids)), "-d", "1000",
                                     "-c", str(seconds)], capture_output=True, text=True, timeout=seconds + 30)
            return start, result.stdout

    remote = ray.remote(num_cpus=0)(Sampler)
    return {node["NodeManagerAddress"]: remote.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node["NodeID"], soft=False)).remote()
            for node in ray.nodes() if node["Alive"] and node["Resources"].get("GPU")}


def timed(text, start, node):
    """dmon prints one row per GPU per second; give each block of rows its second."""
    rows = parse_dmon(text, start, node)
    gpus = len({sample.gpu for sample in rows}) or 1
    return [dataclasses.replace(sample, time=start + 1 + index // gpus) for index, sample in enumerate(rows)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log", help="training log to take phase hints from")
    parser.add_argument("--interval", type=int, default=30, help="seconds per sampling round")
    parser.add_argument("--window", type=int, default=300, help="seconds an alert condition must last")
    parser.add_argument("--low", type=float, default=0.5, help="mean busy fraction below which to alert")
    parser.add_argument("--repeat", type=int, default=900, help="seconds before the same alert repeats")
    args = parser.parse_args()
    import ray

    ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), logging_level="ERROR", log_to_driver=False)
    args.out.mkdir(parents=True, exist_ok=True)
    actors, history, last_alert = samplers(), [], {}
    names = list(FIELDS.values())
    with open(args.out / "samples.csv", "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if csv_file.tell() == 0:
            writer.writerow(["time", "node", "gpu", *names])
        while True:
            rounds = ray.get([actor.read.remote(args.interval, list(FIELDS)) for actor in actors.values()])
            fresh = [s for (start, text), node in zip(rounds, actors, strict=True) for s in timed(text, start, node)]
            writer.writerows([[f"{s.time:.0f}", s.node, s.gpu, *[s.metrics[n] for n in names]] for s in fresh])
            csv_file.flush()
            now = max((s.time for s in fresh), default=time.time())
            history = [s for s in history + fresh if s.time > now - args.window]
            report = assess(history, now, args.window, low=args.low)
            nodes = {node: statistics.fmean(v for (n, _), v in report.per_gpu.items() if n == node)
                     for node in {n for n, _ in report.per_gpu}}
            mean = lambda key: statistics.fmean([s.metrics[key] for s in fresh if s.metrics.get(key) is not None] or [0])
            stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(now))
            with open(args.out / "status.log", "a") as status:
                status.write(f"{stamp} busy {report.mean_busy if report.mean_busy is not None else float('nan'):.2f} "
                             f"sm {mean('sm_active'):.2f} dram {mean('dram_active'):.2f} tensor {mean('tensor_active'):.2f} "
                             f"nodes {' '.join(f'{n}={v:.2f}' for n, v in sorted(nodes.items()))} "
                             f"| {phase_hint(args.log)}\n")
            for alert in report.alerts:
                if now - last_alert.get(alert.kind, -1e18) >= args.repeat:
                    last_alert[alert.kind] = now
                    with open(args.out / "alerts.log", "a") as alerts:
                        gpus = f" gpus={alert.gpus}" if alert.gpus and alert.kind != "STALLED" else ""
                        alerts.write(f"{stamp} {alert.kind}: {alert.message}{gpus} | {phase_hint(args.log)}\n")


if __name__ == "__main__":
    main()
