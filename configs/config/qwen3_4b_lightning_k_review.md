# Qwen3-4B Lightning OPD：固定数据和 K 次 policy rollout

当前 [Lightning YAML](qwen3_4b_lightning_opd.yaml) 使用此变体。[标准 OPD](qwen3_4b_opd.yaml) 保持原来的在线采样、训练和保存模式。此处的 K 是生成训练响应时使用当前 policy 的次数；初始评测和周期评测的生成不计入 K。

## 固定预算和配置

```yaml
num_rollout: 150
rollout_batch_size: 256
global_batch_size: 256
n_samples_per_prompt: 1
policy_rollout_k: 1
rollout_shuffle: false
rollout_seed: 42
eval_steps: 25
async_save: false
```

总训练数据位置固定为 `150 × 256 = 38,400`。先读取 `/scratch/l/luli/data/lightning_opd/train/dapo_math_17k.parquet`，渲染初始学生的 chat template，再过滤超过 2,048 tokens 的 prompt。`max_train_samples` 限制的是源数据池，不能改变这里的 38,400 条预算。

将过滤后的 DAPO 数据按 `rollout_seed + cycle` 打乱，每轮完整使用一次，重复到 38,400 条。重复的是 prompt，每个数据位置都有独立的 `lightning_sample_index` 和 sampling seed；每个位置生成自己的响应，不再循环读取同一个缓存响应来补足训练预算。

固定 prompt 序列的生成不依赖 K。同一源文件、模型 tokenizer、过滤设置、seed 和预算，在不同 K 实验中得到相同的 prompt 顺序。训练器不再额外 shuffle，block 内每条响应按固定顺序恰好参与一次更新。prompt 与 response 及教师打分都使用 Parquet。

K 只接受 `[1, num_rollout]` 范围内的整数。以完整 batch 为单位分块；不能整除时，前面的块多一个 batch，各块最多相差一步：

| K | 各块更新步数 | 各块数据条数 | 唯一保存时机 |
|---|---|---|---|
| 1 | 150 | 38,400 | 第 150 步 |
| 2 | 75、75 | 19,200、19,200 | 第 75 步 |
| 3 | 50、50、50 | 12,800、12,800、12,800 | 第 50 步 |
| 4 | 38、38、37、37 | 9,728、9,728、9,472、9,472 | 第 38 步 |
| 5 | 30 × 5 | 7,680 × 5 | 第 30 步 |
| 150 | 1 × 150 | 256 × 150 | 第 1 步 |

改变 K 时使用新的 `experiment_name`，例如 `qwen3_4b_lightning_opd_base_k2`。不能用另一种 K 的 checkpoint 续训。

## 实际执行顺序

提交入口保持不变：

```bash
sbatch configs/sbatch/qwen3_4b_lightning_opd.sbatch
```

资源保持两节点、每节点四张 H100、总计八张，八小时。

1. 一个节点准备／复用 DAPO 和 eval 的适配版 Parquet。
2. 两个节点核对环境、模型、输入和 allocation。
3. 一个节点通过 `--prepare-data` 构建固定 38,400 条 prompt 和 block plan。这一步仅准备 prompt，不提前用初始 policy 生成后续 block 的响应。
4. 转换／加载初始学生，启动 Ray 和训练 actor，同步推理权重。执行初始 eval。
5. 对每个 block 顺序执行：
   - 当前 student 的 SGLang 引擎为本块全部 prompt 生成响应，保存实际 response token IDs、response 和 loss mask。
   - 卸载训练和推理模型的 GPU 内存。主节点暂时用两张 GPU 启动 Qwen3-8B teacher，计算这些响应 token 的条件 logprob。
   - 写完本块的 `lightning_opd.parquet`，停止 teacher，再恢复 student 推理内存。
   - 训练器加载本块的缓存，每次 256 条、一次 optimizer update。每次重新计算当前学生对固定响应的 logprob，advantage 仍为 `log P_teacher - log P_student`。
     缓存采样器检查逐条绝对数据编号，不使用动态过滤、重采样或网络 RM，避免数据位置漂移；并记录已训练数据位置和 policy block 指标。
   - 在全局 eval 位置评测；仅第一个 block 结束时同步保存 checkpoint。
6. 下一 block 从刚更新的 student 重新生成训练响应。训练 actor、optimizer、scheduler、RNG 和全局 step 不会在 block 间重新创建。
7. 总共执行 150 次更新和最终 eval，检查唯一 checkpoint 与最终指标文件后标记成功。

每块 teacher 缓存保存的是实际 response token 的 log probability，不是全词表 logits。`K=2` 的第二块明确使用完成前 75 次更新后的学生权重，不使用初始模型或重新加载初始模型。

生成和打分按 1,024 条 row group、并发 32 写入，使用临时 Parquet 和原子重命名。请求失败时不会发布半成品。学生响应使用 SGLang 返回的 token IDs；teacher 打分校验 token IDs、长度和有限数值，避免重新分词错位。

## eval 的绝对位置

评测与 K 无关，也不会因 block 边界增加额外评测。初始评测以后仍在以下位置执行：

| 完成更新 | 已训练的数据位置数 | 指标文件 |
|---|---|---|
| 0 | 0 | `eval_metrics/0.json` |
| 25 | 6,400 | `eval_metrics/24.json` |
| 50 | 12,800 | `eval_metrics/49.json` |
| 75 | 19,200 | `eval_metrics/74.json` |
| 100 | 25,600 | `eval_metrics/99.json` |
| 125 | 32,000 | `eval_metrics/124.json` |
| 150 | 38,400 | `eval_metrics/149.json` |

例如 K=4，第一个 block 在第 38 步结束并保存，不在第 38 步额外 eval；下一次 eval 仍是第 50 步。评分代码、eval Parquet、sampling 参数和指标计算保持 YAML 原有设置。当前 Lightning YAML 的 eval 列表包含 8 个数据集，保留已有选择。

## 唯一 checkpoint 和 resume

按本次要求，**整个训练只在第一个 block 结束时保存一次**，不是每个 block 保存一次。Lightning 的此模式不使用 `save_model`／`save_eval_multiplier`；这两个参数仍用于标准 OPD。

保存完整 Megatron model、optimizer、scheduler、RNG 和数据读取状态。K=1 保存 `checkpoints/iter_0000149/`；K=2 保存 `checkpoints/iter_0000074/`。K>1 时最终 student 在内存中完成训练和第 150 步 eval，磁盘上保留的是第一个 block 结束后的 student。

```bash
sbatch configs/sbatch/qwen3_4b_lightning_opd.sbatch --resume
```

resume 总是从唯一 checkpoint 的下一步继续。K=2 从第 76 次更新开始，后 75 次更新中未保存的部分需要重跑。恢复时不改变绝对 eval 位置；checkpoint 恰好位于原定 eval 步且缺少 JSON 时才补评测。K=1 已保存最终 checkpoint 但最终 eval 中断时，resume 只补最终 eval。

初始 block 的 cache 可验证后复用。后续 block 的 cache 带有当前训练实例的 policy 标识；resume 时重新生成后续 block，防止使用上一次未保存训练轨迹产生的响应。K、训练预算、采样、源数据和主要优化参数变化会拒绝原实验的 resume。

## 文件与核心代码

默认结果目录为 `/scratch/l/luli/results/lightning-opd/qwen3_4b_lightning_opd_base_k1/`：

```text
config.yaml
initialization.json
data/fixed_prompts.parquet
data/fixed_prompts.manifest.json
data/block_plan.json
data/blocks/block_0000/student_rollouts.parquet
data/blocks/block_0000/student_rollouts.manifest.json
data/blocks/block_0000/lightning_opd.parquet
data/blocks/block_0000/lightning_opd.manifest.json
data/blocks/block_0000/teacher.log
checkpoints/iter_0000149/             K=1；其他 K 按第一块结束编号
checkpoints/rollout/                 续训数据状态
eval_metrics/{0,24,49,74,99,124,149}.json
wandb/
cluster-<job_id>.done
```

| 核心代码 | 作用 |
|---|---|
| [lightning_opd_blocks.py](../../slime/rollout/lightning_opd_blocks.py) | 固定数据、分块、当前 policy 生成、teacher 打分、缓存和显存交接 |
| [train.py](../../train.py) | 一个持续训练的全局循环、block 切换、固定 eval 位置、只保存第一块 |
| [rollout.py](../../slime/ray/rollout.py) | Ray 的 block 准备／加载 RPC 和数据状态保存 |
| [qwen3-4b-opd.py](../opd/qwen3-4b-opd.py) | YAML 校验、命令映射、resume、最终输出检查 |

W&B 项目仍是 `onestep-opd`，默认 offline，run name 跟随 `experiment_name`。同步命令仍使用当前 YAML：

```bash
bash configs/environment/sync_wandb.sh --config configs/config/qwen3_4b_lightning_opd.yaml
```

此变体完成 CPU 数据／时序回归检查和源码、shell 语法检查；没有提交 GPU 作业，尚未验证两节点的实际显存交接和训练性能。已有旧版小规模 GPU 检查不构成此变体的 GPU 验证。
