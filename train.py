# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ray
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

try:
    from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
except ImportError:
    GPU_MEMORY_TYPE_CUDA_GRAPH = None

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger
from slime.utils.tracking_utils import init_tracking


def train(args):
    import json
    from pathlib import Path
    import time

    from slime.utils.walltime import CheckpointDeadlineReached

    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout:
        ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]))

    if args.offload_train and not args.enable_weights_backuper:
        actor_model.onload()
    # always update weight first so that sglang has the loaded weights from training.
    actor_model.update_weights()
    if args.offload_train and not args.enable_weights_backuper:
        actor_model.offload()

    if args.offload_rollout:
        if GPU_MEMORY_TYPE_CUDA_GRAPH is not None:
            ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_CUDA_GRAPH]))
        ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_KV_CACHE]))

    def offload_train():
        if args.offload_train:
            if args.use_critic:
                critic_model.offload()
                if rollout_id >= args.num_critic_only_steps:
                    actor_model.offload()
            else:
                actor_model.offload()
        else:
            actor_model.clear_memory()

    def onload_rollout():
        if args.offload_rollout:
            ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]))

    completed_steps = args.start_rollout_id
    last_saved_step = completed_steps if completed_steps else None
    paused = False

    def checkpoint_deadline_reached():
        deadline = getattr(args, "checkpoint_deadline", None)
        return deadline is not None and time.time() >= deadline

    def save_checkpoint(index, force_sync):
        nonlocal last_saved_step
        if (not args.use_critic) or (index >= args.num_critic_only_steps):
            actor_model.save_model(index, force_sync=force_sync)
        if args.use_critic:
            critic_model.save_model(index, force_sync=force_sync)
        if args.rollout_global_dataset:
            ray.get(rollout_manager.save.remote(index))
        last_saved_step = index + 1

    def pause_between_updates():
        nonlocal paused
        if completed_steps and last_saved_step != completed_steps:
            # Evaluation uses the shared inference GPUs with training offloaded.
            # Release those GPUs before restoring optimizer/model state to save.
            if args.offload_rollout:
                ray.get(rollout_manager.offload.remote())
            if args.offload_train:
                actor_model.onload()
                if args.use_critic:
                    critic_model.onload()
            save_checkpoint(completed_steps - 1, force_sync=True)
        paused = True

    def evaluate(index, eval_completed_steps):
        try:
            ray.get(rollout_manager.eval.remote(index, completed_steps=eval_completed_steps))
        except CheckpointDeadlineReached:
            pause_between_updates()
            return False
        return True

    # Eval-only and an interrupted initial evaluation have no completed updates
    # to checkpoint; the next allocation starts again from initialization.
    if args.num_rollout == 0 and args.eval_interval is not None:
        evaluate(0, 0)

    # A job can stop after saving its last checkpoint but before evaluation.
    # Evaluate the loaded weights even when the requested training budget is done.
    if (args.eval_on_resume and args.eval_interval is not None and args.start_rollout_id > 0
            and (args.start_rollout_id % args.eval_interval == 0
                 or args.start_rollout_id == args.num_rollout)):
        evaluate(args.start_rollout_id - 1, args.start_rollout_id)

    block_coordinator = None
    if getattr(args, "lightning_block_config", None):
        from slime.rollout.lightning_opd_blocks import BlockCoordinator

        block_coordinator = BlockCoordinator(args, rollout_manager)

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if paused:
            break
        if checkpoint_deadline_reached():
            pause_between_updates()
            break
        # TODO extract the duplicated eval logic
        if args.eval_interval is not None and rollout_id == 0:
            if not evaluate(rollout_id, 0):
                break

        if block_coordinator is not None:
            block_coordinator.ensure_block(rollout_id)

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        if args.use_critic:
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_ref)
            if rollout_id >= args.num_critic_only_steps:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
            ray.get(critic_train_handle)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        completed_steps = rollout_id + 1
        stop_for_walltime = checkpoint_deadline_reached()
        if getattr(args, "save_final_only", False):
            should_save = completed_steps == args.num_rollout
        elif block_coordinator is not None:
            should_save = rollout_id + 1 == block_coordinator.blocks[0]["end_step"]
        else:
            should_save = args.save_interval is not None and (
                (rollout_id + 1) % args.save_interval == 0
                or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
            )
        if should_save or stop_for_walltime:
            force_sync = stop_for_walltime or getattr(args, "save_final_only", False)
            save_checkpoint(rollout_id, force_sync=force_sync)

        if stop_for_walltime:
            # Leave any pending evaluation to the next allocation as well.
            paused = True
            break

        # Policy lag: only sync rollout engine weights every update_weights_interval steps.
        # When lag > 1, the rollout engine runs with stale weights; IS correction
        # (--use-rollout-logprobs + --use-tis) handles the resulting distribution shift.
        should_sync = (rollout_id - args.start_rollout_id + 1) % args.update_weights_interval == 0

        if args.enable_weights_backuper:
            offload_train()
            onload_rollout()
            if should_sync:
                actor_model.update_weights()
        else:
            actor_model.clear_memory()
            onload_rollout()
            if should_sync:
                actor_model.update_weights()
            offload_train()

        if args.offload_rollout:
            if GPU_MEMORY_TYPE_CUDA_GRAPH is not None:
                ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_CUDA_GRAPH]))
            ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_KV_CACHE]))

        if args.eval_interval is not None and (
            (rollout_id + 1) % args.eval_interval == 0
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        ):
            if not evaluate(rollout_id, rollout_id + 1):
                break

    ray.get(rollout_manager.dispose.remote())
    if args.save:
        status = Path(args.save).parent / "training_status.json"
        temporary = status.with_suffix(".tmp.json")
        temporary.write_text(json.dumps({"status": "paused" if paused else "complete",
                                         "global_step": completed_steps, "target_steps": args.num_rollout}) + "\n")
        temporary.replace(status)


if __name__ == "__main__":
    args = parse_args()
    train(args)
