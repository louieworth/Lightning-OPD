"""Keep an inference-only HuggingFace copy of every evaluated policy.

    python3 tools/export_eval_hf.py [--outputs outputs] [--watch 60]

Training saves the full Megatron state before each evaluation and keeps only the newest one
(save_total_limit: 1), so older evaluation points cannot be re-evaluated later. This converts every
<output>/<experiment>/checkpoints/iter_<N> to <output>/<experiment>/hf_eval/step_<N+1>/ (bf16
safetensors plus the student's config and tokenizer, via tools/convert_torch_dist_to_hf.py) before
it is pruned. Optimizer state, RNG and data cursors are not kept: these copies are for evaluation
and inference, not for resuming.

A finished copy has export.json; a half-written one is never left at the final path. Experiments
still training come first, because their older checkpoint is the next to be pruned.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def pending(outputs):
    jobs = []
    for checkpoints in Path(outputs).glob("*/checkpoints"):
        run = checkpoints.parent
        init = run / "initialization.json"
        if not init.is_file():
            continue
        student = json.loads(init.read_text())["student"]
        training = not (run / "training_status.json").is_file()
        for iteration in checkpoints.glob("iter_*"):
            if not (iteration / ".metadata").is_file():
                continue  # still being written
            step = int(iteration.name.split("_")[1]) + 1
            target = run / "hf_eval" / f"step_{step:04d}"
            if not (target / "export.json").is_file():
                jobs.append((not training, iteration.stat().st_mtime, iteration, target, student))
    return [job[2:] for job in sorted(jobs, key=lambda job: job[:2])]


def export(iteration, target, student):
    temporary = target.with_name(target.name + ".tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    started = time.time()
    result = subprocess.run([sys.executable, str(ROOT / "tools/convert_torch_dist_to_hf.py"),
                             "--input-dir", str(iteration), "--output-dir", str(temporary),
                             "--origin-hf-dir", student], cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0 or not any(temporary.glob("*.safetensors")):
        shutil.rmtree(temporary, ignore_errors=True)
        raise RuntimeError(f"{iteration}: conversion failed\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
    (temporary / "export.json").write_text(json.dumps(dict(
        source=str(iteration), step=int(target.name.split("_")[1]), student=student,
        seconds=round(time.time() - started)), indent=2) + "\n")
    shutil.rmtree(target, ignore_errors=True)
    temporary.rename(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs", default=str(ROOT / "outputs"))
    parser.add_argument("--watch", type=float, default=0, help="seconds between scans; 0 exports once and exits")
    args = parser.parse_args()
    while True:
        for iteration, target, student in pending(args.outputs):
            if not (iteration / ".metadata").is_file():
                continue  # pruned since the scan
            print(time.strftime("%F %T"), "export", iteration, "->", target, flush=True)
            try:
                export(iteration, target, student)
                print(time.strftime("%F %T"), "done", target, flush=True)
            except Exception as error:  # a pruned or partial checkpoint must not stop the watcher
                print(time.strftime("%F %T"), "FAILED", error, flush=True)
        if not args.watch:
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
