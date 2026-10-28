"""Resolve SFT batch settings using the active Python environment."""

import os
from pathlib import Path
import shutil
import sys

import yaml


def interpreter(command):
    resolved = shutil.which(str(command or sys.executable))
    if resolved is None:
        raise ValueError(f"Python not found: {command}")
    return resolved


def settings(cfg):
    training = interpreter(os.environ.get("LIGHTNING_OPD_PYTHON") or cfg.get("python_executable"))
    generation = interpreter(cfg.get("generation_python") or training)
    return [
        training,
        generation,
        cfg.get("cuda_home") or os.environ.get("CUDA_HOME", ""),
        cfg["nnodes"],
        cfg["n_gpus_per_node"],
        cfg["master_port"],
        cfg["walltime_checkpoint_margin_seconds"],
        os.environ.get("SFT_DEPENDENCY_ARCHIVE") or cfg.get("sft_dependency_archive") or "",
        cfg["hf_home"],
    ]


if __name__ == "__main__":
    try:
        cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
        print("\n".join(str(value) for value in settings(cfg)))
    except (OSError, ValueError, KeyError) as error:
        sys.exit(f"Invalid SFT environment settings: {error}")
