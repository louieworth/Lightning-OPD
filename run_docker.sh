#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Host orchestration uses only system Python's standard library.
set -euo pipefail
TASK_CONTAINER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$TASK_CONTAINER_ROOT"
if [[ $# == 0 || "$1" == --* && "$1" != --help && "$1" != -h ]]; then
    set -- submit "$@"
fi
exec "${LIGHTNING_OPD_HOST_PYTHON:-/usr/bin/python3}" -I -S -B \
    "$TASK_CONTAINER_ROOT/containers/launch.py" "$@"
