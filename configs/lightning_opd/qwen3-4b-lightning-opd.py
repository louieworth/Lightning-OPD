# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch Qwen3-4B Lightning OPD using configs/config/qwen3_4b_lightning_opd.yaml."""

import importlib.util
from pathlib import Path
import sys


if __name__ == "__main__":
    path = Path(__file__).resolve().parents[1] / "opd/qwen3-4b-opd.py"
    spec = importlib.util.spec_from_file_location("qwen3_4b_opd_launcher", path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    try:
        launcher.main(method="lightning_opd")
    except (ValueError, FileNotFoundError, RuntimeError, TimeoutError) as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        sys.exit(1)
