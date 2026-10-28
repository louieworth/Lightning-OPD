# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Qwen3-0.6B and Qwen3-0.6B-Base share this architecture.
# Qwen3 uses an explicit head_dim=128, independent of hidden_size/num_heads.
MODEL_ARGS=(
   --swiglu
   --num-layers 28
   --hidden-size 1024
   --ffn-hidden-size 3072
   --num-attention-heads 16
   --group-query-attention
   --num-query-groups 8
   --use-rotary-position-embeddings
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --rotary-base "${MODEL_ARGS_ROTARY_BASE:-1000000}"
   --vocab-size 151936
   --kv-channels 128
   --qk-layernorm
)
