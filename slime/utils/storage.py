"""Check output filesystem headroom before allocating GPUs or saving weights."""

import json
import os
from pathlib import Path
import re
import shutil


GIB = 1024 ** 3


def model_weight_bytes(model):
    if model is None:
        return 0
    directory = Path(model)
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = directory / name
        if index.is_file():
            data = json.loads(index.read_text())
            size = data.get("metadata", {}).get("total_size")
            if size is not None:
                return int(size)
            return sum((directory / name).stat().st_size for name in set(data["weight_map"].values()))
    files = list(directory.glob("*.safetensors")) or list(directory.glob("pytorch_model*.bin"))
    return sum(path.stat().st_size for path in files if path.is_file())


def checkpoint_reserve_bytes(save_dir, model):
    # BF16 weights plus FP32 master weights and Adam's two FP32 moments need
    # about seven times the HF weight size. Allow another copy of the weights
    # for serialization overhead; the old checkpoint stays until eval succeeds.
    required = 8 * model_weight_bytes(model)
    directory = Path(save_dir)
    tracker = directory / "latest_checkpointed_iteration.txt"
    if tracker.is_file():
        latest = tracker.read_text().strip()
        if re.fullmatch(r"\d+", latest):
            checkpoint = directory / f"iter_{int(latest):07d}"
            actual = 0
            for root, dirs, files in os.walk(checkpoint, followlinks=False):
                dirs[:] = [name for name in dirs if not (Path(root) / name).is_symlink()]
                actual += sum(path.stat().st_size for name in files
                              if not (path := Path(root) / name).is_symlink())
            required = max(required, actual)
    return required


def require_free_space(destination, required_bytes):
    directory = Path(destination).resolve()
    while not directory.exists():
        directory = directory.parent
    free = shutil.disk_usage(directory).free
    if free < required_bytes:
        raise RuntimeError(
            f"Insufficient output disk space at {directory}: {free / GIB:.1f} GiB free, "
            f"need at least {required_bytes / GIB:.1f} GiB for the next checkpoint and working space. "
            "Check filesystem/project quota and release space before retrying; "
            "the previous completed checkpoint has been retained."
        )
    return free


def check_checkpoint_space(save_dir, model):
    return require_free_space(save_dir, checkpoint_reserve_bytes(save_dir, model) + 5 * GIB)
