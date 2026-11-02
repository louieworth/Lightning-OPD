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

    configure_logger()
    from slime.utils import ray_job

    ray_job.connect()  # Pre-started cluster: start this job with the launcher's runtime env.
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

    def save_checkpoint(index):
        from slime.utils.storage import check_checkpoint_space

        if args.save:
            check_checkpoint_space(args.save, getattr(args, "hf_checkpoint", None))
        if (not args.use_critic) or (index >= args.num_critic_only_steps):
            actor_model.save_model(index, force_sync=True)
        if args.use_critic:
            critic_model.save_model(index, force_sync=True)
        if args.rollout_global_dataset:
            ray.get(rollout_manager.save.remote(index))

    def evaluate(index, eval_completed_steps):
        ray.get(rollout_manager.eval.remote(index, completed_steps=eval_completed_steps))

    # An initial evaluation has no training checkpoint to save or prune.
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
        from slime.rollout import lightning_opd_blocks

        block_coordinator = lightning_opd_blocks.BlockCoordinator(args, rollout_manager)

    # Block Lightning keeps the engines offloaded between updates that no evaluation or generation needs.
    engines_offloaded = False

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # TODO extract the duplicated eval logic
        if args.eval_interval is not None and rollout_id == 0 and getattr(args, 'val_before_train', False):
            evaluate(rollout_id, 0)

        if block_coordinator is not None:
            block_coordinator.ensure_block(rollout_id)

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.offload_rollout and not engines_offloaded:
            ray.get(rollout_manager.offload.remote())

        if args.use_critic:
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_ref)
            if rollout_id >= args.num_critic_only_steps:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
            ray.get(critic_train_handle)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        completed_steps = rollout_id + 1
        epoch_end = num_rollout_per_epoch is not None and completed_steps % num_rollout_per_epoch == 0
        final_step = completed_steps == args.num_rollout
        if getattr(args, "save_final_only", False):
            should_save = final_step
        else:
            should_save = args.save_interval is not None and (
                completed_steps % args.save_interval == 0 or epoch_end or final_step
            )
        if should_save:
            # Save the full model and data cursor before evaluation. The
            # evaluator prunes older checkpoints only after writing its results.
            save_checkpoint(rollout_id)

        # Policy lag: only sync rollout engine weights every update_weights_interval steps.
        # When lag > 1, the rollout engine runs with stale weights; IS correction
        # (--use-rollout-logprobs + --use-tis) handles the resulting distribution shift.
        should_sync = (rollout_id - args.start_rollout_id + 1) % args.update_weights_interval == 0
        evaluating = args.eval_interval is not None and (
            completed_steps % args.eval_interval == 0 or epoch_end or final_step
        )
        engines_needed = block_coordinator is None or lightning_opd_blocks.engines_needed_after(
            block_coordinator.blocks, rollout_id, evaluating
        )

        if not engines_needed:
            pass  # Block Lightning: the actor stays on the GPUs (and keeps its NCCL groups); engines stay offloaded.
        elif args.enable_weights_backuper:
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

        if args.offload_rollout and engines_needed:
            if GPU_MEMORY_TYPE_CUDA_GRAPH is not None:
                ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_CUDA_GRAPH]))
            ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_KV_CACHE]))
        engines_offloaded = args.offload_rollout and not engines_needed

        if evaluating:
            evaluate(rollout_id, completed_steps)

    actor_model.dispose()
    if critic_model is not None:
        critic_model.dispose()
    ray.get(rollout_manager.dispose.remote())
    if args.save:
        status = Path(args.save).parent / "training_status.json"
        temporary = status.with_suffix(".tmp.json")
        temporary.write_text(json.dumps({"status": "complete",
                                         "global_step": completed_steps, "target_steps": args.num_rollout}) + "\n")
        temporary.replace(status)


if __name__ == "__main__":
    args = parse_args()
    try:
        train(args)
    finally:
        ray.shutdown()
