# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import threading

import wandb
from slime.utils.tensorboard_utils import _TensorboardAdapter

from . import wandb_utils
from .wandb_metrics import filter_wandb_metrics


def init_tracking(args, primary: bool = True, **kwargs):
    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
    else:
        wandb_utils.init_wandb_secondary(args, **kwargs)


logger = logging.getLogger(__name__)
# wandb.finish() waits for every record to reach the server and has no timeout. On 11/05 an upload stalled
# behind the proxy and held a finished job (results and GPUs already released) for 40+ minutes.
FINISH_TIMEOUT_S = 300


def finish_tracking(timeout=FINISH_TIMEOUT_S):
    """Upload this process's remaining W&B records before Ray stops it; secondaries leave the run state alone.

    Waits at most `timeout` seconds. The records are already in the local run directory, so a stalled
    upload never blocks shutdown: `wandb sync <run dir>` uploads them later. Returns whether finish completed.
    """
    run = wandb.run
    if run is None:
        return True
    done = threading.Event()

    def finish():
        try:
            wandb.finish()
        finally:
            done.set()

    threading.Thread(target=finish, name="wandb-finish", daemon=True).start()
    if done.wait(timeout):
        return True
    logger.warning("W&B finish did not complete within %s s; continuing shutdown. Upload later with "
                   "`wandb sync %s`.", timeout, getattr(run, "dir", "<run dir>"))
    return False


# TODO further refactor, e.g. put TensorBoard init to the "init" part
def log(args, metrics, step_key: str):
    if args.use_wandb:
        retained = filter_wandb_metrics(metrics)
        if any(key not in {"train/step", "rollout/step", "eval_step"} for key in retained):
            wandb.log(retained)

    if args.use_tensorboard:
        metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
        _TensorboardAdapter(args).log(data=metrics_except_step, step=metrics[step_key])
