# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (Qwen2 architecture: QKV bias, no QK-norm, untied output head).

MODEL_ARGS=(
   --swiglu
   --num-layers 28
   --hidden-size 1536
   --ffn-hidden-size 8960
   --num-attention-heads 12
   --group-query-attention
   --num-query-groups 2
   --use-rotary-position-embeddings
   --disable-bias-linear
   --add-qkv-bias
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --rotary-base 10000
   --vocab-size 151936
   --untie-embeddings-and-output-weights
)
