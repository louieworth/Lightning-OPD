# Qwen3 0.6B / 1.7B 普通在线 OPD

四组实验均使用本仓库 Slime 的在线 OPD，teacher 为
`Qwen/Qwen3-4B-Instruct-2507`。`method: opd` 选择
`configs/opd/qwen3-4b-opd.py`；学生在线生成回答，teacher 在线提供 logprob。

| Student | YAML | sbatch |
|---|---|---|
| `Qwen/Qwen3-0.6B-Base` | [Base 0.6B](qwen3_0_6b_base_opd_nt_teacher_4b_instruct_2507.yaml) | [提交脚本](../sbatch/qwen3_0_6b_base_opd_nt_teacher_4b_instruct_2507.sbatch) |
| `Qwen/Qwen3-1.7B-Base` | [Base 1.7B](qwen3_1_7b_base_opd_nt_teacher_4b_instruct_2507.yaml) | [提交脚本](../sbatch/qwen3_1_7b_base_opd_nt_teacher_4b_instruct_2507.sbatch) |
| `Qwen/Qwen3-0.6B`，non-thinking | [0.6B](qwen3_0_6b_opd_nt_teacher_4b_instruct_2507.yaml) | [提交脚本](../sbatch/qwen3_0_6b_opd_nt_teacher_4b_instruct_2507.sbatch) |
| `Qwen/Qwen3-1.7B`，non-thinking | [1.7B](qwen3_1_7b_opd_nt_teacher_4b_instruct_2507.yaml) | [提交脚本](../sbatch/qwen3_1_7b_opd_nt_teacher_4b_instruct_2507.sbatch) |

YAML 文件名、sbatch 文件名、Slurm job-name 和 W&B experiment_name 一致。
四组的训练及评估参数一致，学生模型 ID、模型架构和实验名称分别设置。

- 2 节点，每节点 4 张 H100，10 小时；2 actor + 4 rollout + 2 teacher GPU。
- 120 次训练更新，batch 128，每个训练 prompt 采样 1 条回答，学习率 `1e-6`。
- `apply_chat_template_kwargs: {enable_thinking: false}` 同时用于训练和评估。
- 显式使用 `<|im_end|>` / `<|endoftext|>` 作为停止 token，兼容 Base 的原生 EOS。
- 每 20 步评估，`val_n: 4`；测试集为 amc23、aime25、aime26、hmmt26、gpqa_diamond。
- W&B 记录 `student_model`、`teacher_model`、实际 `val_n`；每个测试集记录
  `eval_metric/<测试集>-avg@4`、`-pass@4`、`-response-length`，横轴为 `eval_step`。
- W&B 离线日志保存在当前 repo 的 `wandb/<experiment_name>/` 下。
  SDK 的 run 文件通常位于其 `wandb/offline-run-*/run-*.wandb` 子目录。
  YAML 的 `wandb_dir` 使用相对 repo 的路径；checkpoint 和评估 JSON 仍在 `output_dir`。
- `resume: auto` 恢复同名实验的完整 checkpoint。每 20 步同步保存并评估，
  新 checkpoint 保存且评估成功后删除旧 checkpoint，只保留最新一个。
  不再提前 20 分钟中止训练；评估 JSON 保留全部历史步数。

提交示例（其余实验替换为表格中的 sbatch 文件）：

```bash
sbatch configs/sbatch/qwen3_1_7b_base_opd_nt_teacher_4b_instruct_2507.sbatch

# 训练结束后，在能联网的登录节点同步
bash scripts/sync_wandb.sh qwen3_1_7b_base_opd_nt_teacher_4b_instruct_2507.sbatch
```

配置保持 `local_models_only: true`。四个学生模型和 teacher 均已缓存到
`/scratch/l/luli/hf/hub`。迁移环境后如果缺少 0.6B Base，可在能联网的登录节点缓存权重：

```bash
/scratch/l/luli/conda/envs/verl/bin/python -c \
  'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen3-0.6B-Base", cache_dir="/scratch/l/luli/hf/hub")'
```

该命令使用与 YAML 相同的 HF cache。模型的 Megatron 格式转换由作业内的
`auto_prepare_model: true` 执行。`configs/models/qwen3-0.6B.sh` 已包含 0.6B
所需的 1024 hidden size、3072 FFN、28 层和 128 head dimension。
