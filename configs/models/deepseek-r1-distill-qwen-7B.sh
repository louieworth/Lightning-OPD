# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# deepseek-ai/DeepSeek-R1-Distill-Qwen-7B (Qwen2 architecture: QKV bias, no QK-norm, untied output head).

MODEL_ARGS=(
   --swiglu
   --num-layers 28
   --hidden-size 3584
   --ffn-hidden-size 18944
   --num-attention-heads 28
   --group-query-attention
   --num-query-groups 4
   --use-rotary-position-embeddings
   --disable-bias-linear
   --add-qkv-bias
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --rotary-base 10000
   --vocab-size 152064
   --untie-embeddings-and-output-weights
)
