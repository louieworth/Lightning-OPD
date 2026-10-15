# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import slime.utils.external_utils.command_utils as U
from slime.utils.external_utils.command_utils import get_bool_env_var

# Standard OPD: requires a live teacher server during training.
# 4 GPUs for actor (TP=4), 2 GPUs for rollout, 2 GPUs for teacher server.
#
# Required env vars:
#   SFT_CHECKPOINT - path to the SFT checkpoint (HF format)

MODEL_NAME = "Qwen3-8B-Base-Open-Thoughts-Qwen3-32B-sft-3.5k"
TEACHER_MODEL_NAME = "Qwen3-32B"
TEACHER_IP = os.environ.get("MASTER_ADDR", "127.0.0.1")
TEACHER_PORT = 13141
MODEL_TYPE = "qwen3-8B"
NUM_GPUS = 8
SFT_CHECKPOINT = os.environ["SFT_CHECKPOINT"]


def deploy_teacher_model():
    import random, string
    random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    LOG_FILE = f"/tmp/sglang_{random_suffix}.log"
    external_ray = get_bool_env_var("SLIME_SCRIPT_EXTERNAL_RAY")

    U.exec_command(
        "pkill -9 sglang; "
        "sleep 3; "
        f"{'' if external_ray else 'ray stop --force; '}"
        f"{'' if external_ray else 'pkill -9 ray; '}"
        "pkill -9 slime; "
        "sleep 3; "
        f"{'' if external_ray else 'pkill -9 ray; '}"
        "pkill -9 slime; "
        "pkill -9 redis; "
        "true;"
    )

    U.exec_command(
        f"CUDA_VISIBLE_DEVICES=6,7 python3 -m sglang.launch_server "
        f"--model-path /root/models/{TEACHER_MODEL_NAME} "
        f"--host 0.0.0.0 "
        f"--port {TEACHER_PORT} "
        f"--tp 2 "
        f"--chunked-prefill-size 4096 "
        f"--mem-fraction-static 0.6 "
        f"--context-length 32768 "
        f"> {LOG_FILE} 2>&1 & "
    )

    U.exec_command(
        f"until curl -sf http://{TEACHER_IP}:{TEACHER_PORT}/health_generate > /dev/null; do "
        f"    echo 'Waiting for teacher model...'; "
        f"    tail -n 10 {LOG_FILE}; sleep 5; done; "
        f"echo 'Teacher model ready at {TEACHER_IP}:{TEACHER_PORT}.'; sleep 10;"
    )


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets/dapo-math-17k")
    U.exec_command(f"huggingface-cli download Qwen/{TEACHER_MODEL_NAME} --local-dir /root/models/{TEACHER_MODEL_NAME}")

    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=NUM_GPUS,
        hf_checkpoint=SFT_CHECKPOINT,
    )

    deploy_teacher_model()


def execute(rerun=True):
    load_save_path = f"/root/models/{MODEL_NAME}_ckpt__{Path(__file__).stem}/"

    ckpt_args = (
        f"--hf-checkpoint {SFT_CHECKPOINT} "
        f"--ref-load /root/models/{MODEL_NAME}_torch_dist "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        "--save-interval 5 "
        "--save-retain-interval 5 "
    )

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--num-rollout 3000 "
        "--rollout-batch-size 64 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 4096 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 256 "
        "--balance-data "
    )

    rm_args = (
        "--custom-rm-path slime.rollout.on_policy_distillation.reward_func "
        "--custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards "
        f"--rm-url http://{TEACHER_IP}:{TEACHER_PORT}/generate "
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
        "--max-tokens-per-gpu 8192 "
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
        "--actor-num-gpus-per-node 4 "
        "--rollout-num-gpus 2 "
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
