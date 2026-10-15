# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import slime.utils.external_utils.command_utils as U

# Lightning OPD: response tokens + teacher logprobs are pre-computed in parquet.
# No teacher server needed at training time -> all 8 GPUs go to the actor.
#
# Required env vars:
#   SFT_CHECKPOINT      - path to the SFT checkpoint (HF format)
#   LIGHTNING_OPD_DATA   - path to the precomputed parquet

MODEL_NAME = "Qwen3-8B-Base-Open-Thoughts-Qwen3-32B-sft-3k"
MODEL_TYPE = "qwen3-8B"
NUM_GPUS = 8
SFT_CHECKPOINT = os.environ["SFT_CHECKPOINT"]


def prepare():
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=NUM_GPUS,
        hf_checkpoint=SFT_CHECKPOINT,
    )


def execute(rerun=True):
    load_save_path = f"/root/models/{MODEL_NAME}_ckpt__{Path(__file__).stem}/"

    ckpt_args = (
        f"--hf-checkpoint {SFT_CHECKPOINT} "
        f"--ref-load /root/models/{MODEL_NAME}_torch_dist "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        "--save-interval 10 "
        "--save-retain-interval 100 "
    )

    rollout_args = (
        f"--prompt-data {os.environ['LIGHTNING_OPD_DATA']} "
        "--input-key prompt "
        "--label-key label "
        "--rollout-shuffle "
        "--num-rollout 3000 "
        "--rollout-batch-size 256 "
        "--n-samples-per-prompt 1 "
        "--rollout-max-response-len 4096 "
        "--global-batch-size 256 "
    )

    rm_args = (
        "--custom-rm-path slime.rollout.on_policy_distillation.reward_func "
        "--custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards "
        "--include-verifiable-reward "
    )

    perf_args = (
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 16384 "
    )

    grpo_args = (
        "--advantage-estimator on_policy_distillation "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 2e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    wandb_args = ""
    if os.environ.get("WANDB_KEY"):
        wandb_args = (
            "--use-wandb "
            "--wandb-project onestep-opd "
            f"--wandb-group {Path(__file__).stem} "
            f"--wandb-key {os.environ['WANDB_KEY']} "
        )

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.4 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 8 "
        "--rollout-num-gpus 0 "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{rm_args} "
        f"{grpo_args} "
        f"{optimizer_args} "
        f"{wandb_args} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{misc_args} "
    )

    U.execute_train(
        rerun=rerun,
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
    )


if __name__ == "__main__":
    prepare()
    execute(rerun=False)
