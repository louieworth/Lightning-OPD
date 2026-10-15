# Qwen3-4B Lightning OPD 训练流程审查文档

记录日期：2026-10-09。本文依据当前仓库源码、两个 Qwen3-4B YAML、Slurm 脚本及数据清单编写，采用迁移后的实际路径。文档中的命令供审查和后续使用。本次 K-block 改动完成 CPU 回归检查，没有提交训练、运行 GPU 测试或上传 WandB。

当前默认实验通过 `model_path: Qwen/Qwen3-4B-Base` 加载初始学生，使用 `teacher_model_path: Qwen/Qwen3-8B` 教师。当前 Lightning 流水线会构建固定 38,400 条 prompt，按 K 次当前 policy rollout 顺序准备 teacher logprob 并训练；block 之间保持训练状态。详细说明见 [固定数据与 K 变体审查](qwen3_4b_lightning_k_review.md)。两个 OPD 入口统一通过 `model_path` 加载已有模型，支持 HuggingFace 模型 ID 或本地 HF 模型目录。

当前状态：训练提示和 9 个评测集的适配版 Parquet 已准备；完整学生响应和教师 logprob 尚未生成，首次正式作业会自动生成。此前单节点 4 张 H100 的小规模完整流程已通过，双节点 8 卡完整训练尚未执行。数据和运行环境路径迁移后仅核对了文件及配置，没有重新执行 GPU 检查。

**1. 文件、路径与配置入口**

| 文件 | 用途 |
|---|---|
| [qwen3_4b_lightning_opd.yaml](qwen3_4b_lightning_opd.yaml) | Lightning OPD 的模型、数据、集群、训练和评测参数 |
| [qwen3_4b_opd.yaml](qwen3_4b_opd.yaml) | 标准在线 OPD 对照配置 |
| [Lightning OPD sbatch](../sbatch/qwen3_4b_lightning_opd.sbatch) | 完整流程的 Slurm 入口 |
| [OPD sbatch](../sbatch/qwen3_4b_opd.sbatch) | 选择 OPD YAML 后调用共同的作业流程 |
| [Lightning OPD Python 入口](../lightning_opd/qwen3-4b-lightning-opd.py) | 加载共同启动器，指定 `method=lightning_opd` |
| [共同启动器](../opd/qwen3-4b-opd.py) | 配置、数据适配、预计算、Ray 集群、参数映射、评分 |
| [原生训练入口](../../train.py) | 训练、权重同步、保存与评测循环 |
| [环境说明](../environment/README.md) | 安装信息、兼容性修改和 WandB 同步用法 |
| [环境记录](../environment/verl_runtime.json) | 版本、历史检查范围和迁移记录 |

当前目录关系：

```text
/scratch/l/luli/data/
├── train/dapo_math_17k.parquet          原始 verl 格式
├── eval/*.parquet                     原始 verl 格式
└── lightning_opd/
    ├── train/dapo_math_17k.parquet     本仓库实际读取的适配版
    ├── eval/*.parquet                 本仓库实际读取的适配版
    └── prepared_inputs.json           数据来源、路径和条数清单

/scratch/l/luli/src/Lightning-OPD/
├── .env                               本地凭据，Git 忽略，权限 600
├── .runtime/verl/                      Megatron 源码和历史检查记录
├── configs/config/                    YAML 和本文档
└── configs/sbatch/                    作业脚本

/scratch/l/luli/results/lightning-opd/
└── <experiment_name>/                 当前实验的生成数据、权重和结果
```

仓库原来的 `data/` 目录已经删除。实验结果中的 `data/` 是结果目录下的独立子目录，用于保存学生响应和教师 logprob。

**2. 提交命令和资源分配**

从仓库根目录提交默认 Lightning OPD 作业：

```bash
cd /scratch/l/luli/src/Lightning-OPD
sbatch configs/sbatch/qwen3_4b_lightning_opd.sbatch
```

| Slurm 设置 | 当前值 |
|---|---|
| Job name | `qwen3-lightning-opd` |
| Account | `aip-plbacon` |
| Partition | `gpubase_bynode_b2` |
| Wall time | `08:00:00` |
| Nodes | 2 |
| GPUs per node | `h100:4`，总共 8 张 |
| Tasks per node | 1，后续 Ray 创建 GPU worker |
| CPUs per task | 48，即每节点 48 |
| Memory | `--mem=0` |
| Exclusive | 开启 |
| Working directory | `/scratch/l/luli/src/Lightning-OPD` |
| Slurm log | `tmp/slurm-qwen3-lightning-opd-<job_id>.log` |

8 小时覆盖数据准备、预计算、转换、训练和全部评测。整个 allocation 从开始就占用两个节点，单节点准备阶段的另一节点也处于已分配状态。配置尚未经过完整作业的耗时测量，8 小时能否覆盖全部工作需要正式运行确认。

YAML 中 `nnodes: 2`、`n_gpus_per_node: 4` 与 sbatch 的资源数须一致。`preflight()` 会核对节点数量、可见 GPU、并行度和已有实验状态。

sbatch 直接调用 YAML 指定的 Python，不要求提交前手动 `conda activate`。`LIGHTNING_OPD_PYTHON` 可以覆盖启动解释器；自定义 YAML 可以作为 sbatch 脚本的第一个位置参数传入。

**3. 端到端作业顺序**

| 顺序 | 执行内容 | 执行范围 | 主要输出 |
|---|---|---|---|
| 1 | `--prepare-prompts`，准备或复用训练／评测 Parquet | 一个节点，CPU | 适配版 Parquet、`.source.json`、数据清单 |
| 2 | `--preflight`，核对环境、模型、输入、资源与实验状态 | 两个节点 | Slurm 中的检查信息 |
| 3 | 构建固定 38,400 条 prompt 和 K-block plan | 一个节点，CPU | `data/fixed_prompts.parquet`、`block_plan.json` |
| 4a | 将初始 HF 权重转成 Megatron | 主节点，4 GPU | `models/<hash>/torch_dist/` |
| 4b | 建立 Ray 集群、加载训练模型、同步推理权重 | 两个节点，8 GPU | Ray 日志与训练模型 |
| 4c | 初始 eval，按 K 次当前 student rollout→teacher logprob→block 训练循环，按全局位置 eval；训练完成或时限前保存 | 两个节点，8 GPU；teacher 暂用主节点两卡 | block Parquet、最新 checkpoint、eval JSON、offline WandB |

sbatch 的核心顺序为：

```bash
run_stage 1 --prepare-prompts "$@"
run_stage "$TASK_COUNT" --preflight "$@"
run_stage 1 --prepare-data "$@"
run_stage "$TASK_COUNT" "$@"
```

`run_stage()` 用 `srun` 启动 Python。作业使用 `set -euo pipefail` 和 `--kill-on-bad-exit=1`，任一阶段失败时后续阶段不启动。生产 sbatch 的这种行为与此前保持 30 分钟的交互 allocation 是两个不同入口。

指定 `--prepare-prompts`、`--prepare-data`、`--preflight`、`--check` 或 `--dry-run` 时，脚本只运行对应阶段。本文件给出的默认提交命令运行完整流程。

**4. 输入数据：Parquet 来源、适配和数量**

原始数据已经是 Parquet，无需从 JSONL 转格式。原始列为：

```text
prompt
reward_model
data_source
ability
extra_info
```

原始训练文件：

```text
/scratch/l/luli/data/train/dapo_math_17k.parquet
```

当前实际读取的训练文件：

```text
/scratch/l/luli/data/lightning_opd/train/dapo_math_17k.parquet
```

`prepare_prompts()` 按 `source_data_dir/<train|eval>/<文件名>` 定位原始文件；`convert_data()` 用 PyArrow 读入，保留原始列，并增加统一的 `label`、`metadata`。核心逻辑摘录如下，省略写入和错误处理：

```python
table = pq.read_table(source)
rows = table.to_pylist()

truth = row.get("label")
if truth is None:
    truth = (row.get("reward_model") or {}).get("ground_truth")

extra = row.get("extra_info") or {}
metadata = dict(row.get("metadata") or {})
metadata.update(
    data_source=str(row.get("data_source") or source.stem),
    sample_uid=str(extra.get("index", index)),
    extra_info=extra,
)
row.update(label=str(truth), metadata=metadata)
```

每个适配文件旁的 `.source.json` 记录原始路径、文件大小、修改时间、limit 和 adapter version。记录匹配时复用已有 Parquet。写入先经过临时文件再替换目标。

数据清单位于：

```text
/scratch/l/luli/data/lightning_opd/prepared_inputs.json
```

| 文件名 | 用途 | 当前条数 |
|---|---|---:|
| `dapo_math_17k.parquet` | 训练提示 | 17,916 |
| `amc23.parquet` | eval | 40 |
| `math500.parquet` | eval | 500 |
| `gsm8k.parquet` | eval | 1,319 |
| `aime25.parquet` | eval | 30 |
| `aime26.parquet` | eval | 30 |
| `hmmt26.parquet` | eval | 33 |
| `mmlu_college_chemistry.parquet` | eval | 100 |
| `mmlu_college_medicine.parquet` | eval | 173 |
| `gpqa_diamond.parquet` | eval | 198 |

9 个评测集合计 2,423 条。原始数据目录还有 `mmlu.parquet`、`amobench.parquet` 等文件，当前 YAML 未将它们加入 `val_files`。

`max_train_samples: 100000` 是源数据池上限。输入适配保留全部 17,916 条，渲染后仍按 2,048 token 限制过滤。固定序列准备器重复打乱后的有效源数据，恰好构建 38,400 个训练位置。已有 9 份 eval Parquet；当前 Lightning YAML 选择其中 8 份，未选择 aime25。

**5. 选择初始学生模型与教师模型**

默认选择：

```yaml
model_path: Qwen/Qwen3-4B-Base
teacher_model_path: Qwen/Qwen3-8B
```

`selected_models()` 从 `model_path` 和 `teacher_model_path` 加载学生和教师，解析 `/scratch/l/luli/hf` 中的本地 HF snapshot 或直接使用本地目录，并检查 config、tokenizer 和权重分片。`local_models_only: true` 时使用本地模型缓存。

需要加载本次独立训练的学生模型时，在训练完成后修改 YAML：

```yaml
model_path: /scratch/l/luli/results/onestep/model/qwen3_4b_sft_qwen3_8b
experiment_name: qwen3_4b_lightning_opd_trained
```

之后使用同一提交入口：

```bash
sbatch configs/sbatch/qwen3_4b_lightning_opd.sbatch
```

切换模型只需更改 `model_path`，同时使用新的 `experiment_name`，避免复用旧模型生成的离线数据和训练 checkpoint。选择用于论文复现的模型时，其 SFT 数据应来自与 OPD 阶段相同的教师，满足 teacher consistency 条件。启动器记录学生／教师路径，但无法仅凭外部 HF 权重验证其训练数据来源。

模型训练由独立的 `configs/sbatch/qwen3_4b_sft.sbatch` 完成，两个 OPD sbatch 只加载选定模型。默认 `model_path` 仍为 Base；论文实验应在独立训练完成后，将其替换为对应模型目录。

**6. 固定 prompt、K 次 policy rollout 与 teacher logprob**

共同启动器的 `prepare_lightning_data()` 在 `policy_rollout_k` 模式下仅调用 `prepare_fixed_prompts()`，使用过滤后的 DAPO 数据循环构建 38,400 条 prompt。顺序由 seed 确定，不依赖 K；训练器关闭额外 shuffle。每个重复 prompt 都有独立的绝对数据编号，并生成自己的响应。

[lightning_opd_blocks.py](../../slime/rollout/lightning_opd_blocks.py) 的 `schedule()` 按完整的 256 样本 batch 将 150 次更新分为 K 块。`BlockCoordinator.ensure_block()` 在每块开始时：

1. 使用刚同步的当前 student，为这一块全部 prompt 生成响应。K=2 的第二块使用训练完 75 步后的 policy。
2. 将 SGLang 返回的 response token IDs、loss mask 和 response 保存到本块 `student_rollouts.parquet`，sampling seed 为 `42 + 全局数据编号`。
3. 卸载训练与推理显存，在主节点暂时启动 TP=2 的 teacher。
4. teacher 对已保存的实际 token 计算 logprob，验证长度与 token IDs 完全对应，保存 `lightning_opd.parquet`。
5. 停止 teacher、恢复 student 推理显存，加载本块缓存进行训练。

student 使用现有 8 个 SGLang TP=1 引擎。每块生成和打分按 1,024 条 row group、并发 32 原子写入；不会在某个请求失败后发布部分数据。teacher 不参与训练期间的每次 forward。

```text
data/fixed_prompts.parquet
data/block_plan.json
data/blocks/block_0000/student_rollouts.parquet
data/blocks/block_0000/lightning_opd.parquet
```

block 缓存包含 `is_lightning_opd`、`response_tokens`、`loss_mask`、`response`、`teacher_log_probs`、`lightning_sample_index`、`policy_block_index`、`policy_update_step`。teacher 保存的是 response token 的条件 log probability，没有保存全词表 logits。

第一个 block 的初始 policy cache 可验证后复用。后续 cache 包含训练实例的 policy 标识；从 block 中间 resume 时，使用 checkpoint 中保存的 policy 标识继续读取同一份缓存。进入下一 block 时从已恢复并更新的 policy 重新生成响应。训练 actor、optimizer、scheduler、RNG 与全局步数在 block 之间连续保持。

**7. HF 转换、Ray 集群和训练参数映射**

`checkpoint_path()` 根据解析后的学生目录计算 12 位 SHA256 路径标识，转换产物位于：

```text
<output_dir>/<experiment_name>/models/<hash>/torch_dist/
```

`auto_prepare_model: true` 时，缺少转换 checkpoint 会使用 4 GPU 执行 `tools/convert_hf_to_torch_dist.py`。转换工具会使用其适合加载 HF 权重的并行布局；该布局与训练 TP=2 的布局可以不同，后续由分布式 checkpoint 加载处理。

架构参数来自 [qwen3-4B.sh](../models/qwen3-4B.sh)：

| 模型参数 | 当前值 |
|---|---|
| Layers | 36 |
| Hidden size | 2,560 |
| FFN hidden size | 9,728 |
| Query attention heads | 32 |
| KV groups | 8，GQA |
| Head dimension / KV channels | 128 |
| Vocabulary size | 151,936 |
| Normalization | RMSNorm，epsilon=`1e-6` |
| Position encoding | RoPE，默认 base=`1000000` |
| 其他 | SwiGLU、QK layernorm、关闭 linear bias |

`MODEL_ARGS_ROTARY_BASE` 可以覆盖架构脚本的 RoPE base；上表列出当前无覆盖时的值。

`cluster_nodes()` 使用 Slurm 节点列表，将短主机名转换为可路由的完整主机名，按数字 IP 排序。`train_cluster()` 在第一个节点启动 Ray head，其他节点加入同一集群，等待全部 8 GPU 注册，然后由主节点启动 `train.py`。

Lightning 的 actor 使用全部 8 GPU，TP=2、PP=1、CP=1，得到 DP=4。评测 SGLang 引擎与 actor 共用这些 GPU，`colocate: true` 会启用训练／推理显存卸载。`rollout_num_gpus: 8` 不会额外申请 8 张卡。

共同启动器的 `training_args()` 将 YAML 的非启动器字段转成 CLI，规则是 `underscore → hyphen`；true 生成开关，false／null 被省略。关键映射为：

| 配置／派生项 | 原生训练参数 |
|---|---|
| 所选学生 HF 目录 | `--hf-checkpoint` |
| 转换后的初始学生 checkpoint | `--ref-load` |
| 实验 `checkpoints/` | `--load`、`--save`（关闭保存时省略 `--save`） |
| 离线训练 Parquet | `--prompt-data` |
| `n_gpus_per_node` | `--num-gpus-per-node` |
| `eval_steps` | `--eval-interval` |
| `policy_rollout_k` | `--lightning-block-config` 与 block plan；block 切换不触发保存 |
| block 缓存读取 | `--rollout-function-path slime.rollout.lightning_opd_blocks.generate_cached_rollout` |
| `save_model` | `final` 派生 `--save-final-only` 和 `--save-interval 150` |
| `walltime_checkpoint_margin_seconds` | sbatch 用 Slurm EndTime 减去 1200 秒，传入 `--checkpoint-deadline` |
| 实验 `eval_metrics/` | `--eval-output-dir`，关闭保存时仍写入评测指标 |
| 派生 `eval.yaml` | `--eval-config` |
| 共同启动器 `evaluate()` | `--eval-function-path` |
| WandB 配置 | `--wandb-project`、`--wandb-team`、`--wandb-mode` 等 |

`ref_load` 指向初始学生权重。它与蒸馏教师是两个不同角色：教师指导来自离线 `teacher_log_probs`。

配置中的 false 当前依赖对应原生开关的默认值为 false。新增布尔参数时应核对原生 parser 的默认值和开关方向。

**8. 训练数据读取与核心损失**

数据读取链路为：

```text
RolloutDataSource
  → create_dataset() / Dataset
  → pandas.read_parquet(..., dtype_backend="pyarrow")
  → Sample
  → generate_cached_rollout() 校验绝对位置并恢复离线响应
  → Megatron actor
```

实现位置：

- [data_source.py](../../slime/rollout/data_source.py)：数据集初始化、打乱、循环读取和读取状态保存。
- [data.py](../../slime/utils/data.py)：Parquet 加载、prompt 处理、Sample 构造。
- [lightning_opd_blocks.py](../../slime/rollout/lightning_opd_blocks.py)：每次严格读取一个固定 batch，不使用动态过滤、重采样或训练时网络 RM。
- [sglang_rollout.py](../../slime/rollout/sglang_rollout.py)：`_handle_lightning_opd_sample()`，恢复离线 token 和响应。
- [on_policy_distillation.py](../../slime/rollout/on_policy_distillation.py)：加载教师 logprob，返回蒸馏用奖励数据。
- [actor.py](../../slime/backends/megatron_utils/actor.py)：计算当前学生 logprob 和执行训练。
- [loss.py](../../slime/backends/megatron_utils/loss.py)：advantage 与损失聚合。
- [ppo_utils.py](../../slime/utils/ppo_utils.py)：PPO 裁剪损失。

固定 prompt builder 按 seed 打乱源 DAPO 的重复轮次；训练器不再额外 shuffle 或循环当前 block。`apply_chat_template: false` 用于离线训练，因为 prompt 已在准备阶段渲染。缓存采样器逐条检查绝对数据编号并恢复保存的响应：

```python
sample.tokens = prompt_tokens + response_tokens
sample.loss_mask = loss_mask
sample.response = saved_response
```

训练中不重新生成这批固定响应，也不发送教师请求。每个批次更新前，训练 actor 重新计算当前学生对相同 token 序列的 logprob。

设 `q_t` 为教师对响应 token 的概率，`p_old,t` 为本批次更新前学生的概率。逐 token advantage 为：

```text
A_t = log(q_t) - log(p_old,t)
```

`compute_advantages_and_returns()` 的对应分支为：

```python
advantages = [
    teacher_log_prob - student_log_prob
    for teacher_log_prob, student_log_prob
    in zip(teacher_log_probs, student_log_probs, strict=False)
]
returns = advantages
```

之后使用 PPO 裁剪形式更新学生。以下摘录省略 tensor 拼接和聚合：

```python
ppo_kl = old_log_probs - new_log_probs
ratio = (-ppo_kl).exp()
loss_1 = -ratio * advantages
loss_2 = -ratio.clamp(0.8, 1.2) * advantages
policy_loss = torch.maximum(loss_1, loss_2)
```

损失按有效 response mask 聚合。当前 `calculate_per_token_loss` 未启用，使用框架默认的逐样本 response 平均再聚合，而非将所有序列的 token 直接混为一个统一平均。prompt 提供上下文，mask 仅针对响应部分计算训练目标。

总损失的结构为：

```text
loss = policy_loss - entropy_coef × entropy + kl_loss_coef × reference_KL
```

当前 `entropy_coef=0`、`kl_loss_coef=0`，因此教师作用通过 advantage 生效。`use_kl_loss: true` 仍会进入额外 reference KL 的计算／日志路径，但零系数不会将该项加入最终目标。

当前 OPD reward hook 返回标量 reward `0.0`，没有将数学答案正确性奖励加入 advantage。`include_verifiable_reward: true` 在 parser 中是日志相关开关；当前选择的 hook 未据此计算额外任务奖励，不能将它理解为训练答案奖励已开启。

`use_rollout_logprobs` 和 `use_tis` 默认关闭；本准备器也不生成行为策略的 `student_log_probs`。这里的 PPO ratio 是本批次更新前后学生的 ratio，不能将它描述成对“离线采样初始学生”和“当前学生”分布差异的完整重要性校正。

**9. 训练更新、评测与 checkpoint 编号**

当前 Lightning 设置：

```text
rollout_batch_size × n_samples_per_prompt = 256 × 1 = 256
global_batch_size = 256
num_rollout = 150
```

每个 rollout 批次对应一次优化器更新，150 次更新使用固定序列的全部 38,400 个训练位置，每个位置的缓存响应参与一次更新；重复 prompt 在不同位置生成独立响应。动态 microbatch 根据长度和 token budget 组织，这里的 256 是全局序列数，不是每 GPU 的 microbatch 大小。

原生 `train.py` 的主要顺序为：

```text
创建 Ray placement groups 和推理引擎
加载训练 actor 并同步初始权重
rollout_id=0 时初始评测
循环读取离线批次
计算当前学生 logprob、advantage、反向传播和 optimizer step
按条件保存 checkpoint 和数据读取状态
同步学生权重到 SGLang
按条件评测更新后的模型
训练结束释放推理服务
```

当前 YAML 显式设置 `num_rollout: 150`，因此创建 rollout manager 时返回的 `num_rollout_per_epoch` 为 `None`。当前新实验的评测按全局固定间隔触发：

```python
(rollout_id + 1) % interval == 0
```

原生 `train.py` 也包含 epoch 边界触发分支，但只有未指定 `num_rollout`、由 `num_epoch` 推导训练长度时才传入对应 epoch 长度。数据读取仍会循环和打乱；在当前配置中，数据循环到下一 epoch 不额外触发保存或评测。

默认新实验完成 150 次更新时执行 7 轮评测：一次初始评测和 6 次更新后的周期评测。当前 Lightning 选择的 8 个集合共有 2,393 条评测输入；若全部通过长度过滤，每轮生成 2,393 条答案，总计 16,751 条；这是答案数量计算，不是已完成的正式评测结果。

rollout 编号从 0 开始。固定每 25 次更新触发的编号是：

```text
24, 49, 74, 99, 124, 149
```

首次初始评测使用编号 0。最终第 150 次更新后对应 `eval_metrics/149.json`，所有 K 的最终 checkpoint 编号均为 149。启动器会检查最终 eval JSON 存在后才报告训练完成；时限前保存退出则记录为 `paused`，不要求该次 allocation 完成最终评测。

Lightning 使用以下分块策略：

```yaml
policy_rollout_k: 1
num_rollout: 150
rollout_batch_size: 256
eval_steps: 25
rollout_shuffle: false
save_model: final
walltime_checkpoint_margin_seconds: 1200
save_total_limit: 1
save_retain_interval: null
async_save: false
```

Lightning 和标准 OPD 均使用 `save_model: final`，在第 150 次更新结束时保存完整模型、optimizer、scheduler、RNG 和数据读取位置；若提前到达距 Slurm 时限 20 分钟的保存期限，则完成当前更新后同步保存并退出。block 边界和每 25 步评测不触发 checkpoint。`save_total_limit: 1` 保留最近一个完整 checkpoint，新保存及 rollout cursor 完成后再清理旧文件。

长评测也受同一保存期限约束：达到期限时取消未完成的评测请求，不写入部分评测结果，释放推理显存并恢复训练 worker，保存最近一个已完成更新及数据游标后退出。若该更新已经保存（包括最终第 150 步），直接保留完整断点。训练前初始评测若耗尽时限，则以 0 步暂停，不生成虚假的训练 checkpoint。

两种方法仍每 25 步 eval。标准 OPD 另支持 `save_model: active` 配合整数 `save_eval_multiplier` 的定期保存以及 `null` 关闭保存；当前配置使用 `final`。Lightning 启动器保留旧的首块保存模式供没有 `save_model` 的历史配置使用，当前配置已切换到 `final`。

训练权重同步给 SGLang 后直接在内存中评测，无需从磁盘重新加载或导出 HF。全部 eval JSON 和 W&B 指标继续保留。

论文 §4.1 和 Appendix B Table 6 明确 OPD 训练 150 步，§4.4 说明该预算足以收敛；论文未明确声明主表从各步挑选 best checkpoint。因此这里采用第 150 步结果作为默认复现口径，不宣称作者明确说明了 final-checkpoint 选择规则。主表的 bold 指不同方法之间的最优结果。[论文原文](https://arxiv.org/html/2604.13010v1#S4.SS1)

**10. 评测数据、评分代码和指标**

`prepare_eval()` 根据 YAML 生成实验目录中的 `eval.yaml`。评测读取适配后的 `prompt`、`label`、`metadata`，由共同启动器 `evaluate()` 设置评测专用 scorer，并启用 chat template。评测使用原始评测 prompt，因此需要在该阶段渲染模板。

评测参数为：

```yaml
eval_steps: 25
val_max_prompt_length: 4096
val_max_completion_length: 16384
val_n: 1
val_temperature: 0.7
val_top_p: 0.8
val_top_k: 20
save_eval_generations: false
```

评分引用 `/scratch/l/luli/src/opd-baselines/src/` 下的原实现：

| 代码 | 行为 |
|---|---|
| `baseline_reward.py` | 提取最后一个括号平衡的 `\boxed{...}`；数字精确比较、Math-Verify 数学匹配；MMLU／GPQA 比较 boxed A–D 选项 |
| `reward_async.py` | 通过 spawned 进程池执行评分，让 Math-Verify 的超时逻辑在进程主线程运行 |
| `eval_metrics.py` | 按 `data_source`、`sample_uid` 分组，计算准确率和 pass 指标 |

不通过时 `acc=0`，通过时 `acc=1`。每道题恰好生成 k 条答案时：

```text
avg@k  = 各题 k 个正确性值的均值，再对所有题平均
pass@k = 每题 k 个正确性值的最大值，再对所有题平均
```

当前 k=1，`avg@1` 与 `pass@1` 数值相同。JSON 中为 0–1 范围的值，例如 `0.75` 表示 75% 正确率。主要键为：

```text
eval/<data_source>/avg@1
eval/<data_source>/pass@1
eval/<data_source>/num_questions
```

`num_questions` 以实际加载并通过 prompt 长度过滤的题目数为准。当前未开启保存完整评测生成内容；指标会保存，并写入 offline WandB。

**11. 最终参数与隐式默认值**

完整 Lightning YAML 快照附在文末；实际运行仍以 YAML 文件为准。关键训练参数为：

| 参数 | 当前值 |
|---|---|
| Optimizer / LR | Adam / `2e-6`，constant |
| Weight decay | 0.1 |
| Adam beta1 / beta2 | 0.9 / 0.98 |
| Global batch / rollout batch | 256 / 256 |
| Response / prompt cap | 4,096 / 2,048 tokens |
| TP / DP / PP / CP | 2 / 4 / 1 / 1 |
| Sequence parallel | true |
| Recompute | full / uniform / 每次 1 层 |
| Dynamic batch token budget | 每 GPU 16,384 |
| Attention backend | fused，Transformer Engine cuDNN |
| Attention / hidden dropout | 0 / 0 |
| 梯度 AllReduce / attention softmax | FP32 / FP32 |
| KL / entropy 系数 | 0 / 0 |
| Seed / rollout seed | 42 / 42 |

以下是源码默认值或派生值，当前 YAML 未全部显式列出：

| 默认／派生参数 | 当前生效值 | 来源／说明 |
|---|---|---|
| `bf16` | true | `set_default_megatron_args()` 在未启用 fp16 时设置 |
| `use_distributed_optimizer` | true | Megatron backend 强制设置 |
| `eps_clip` | 0.2 | slime parser 默认 |
| `eps_clip_high` | 0.2 | 未设置时使用 `eps_clip` |
| `clip_grad` | 1.0 | slime parser 默认 |
| `normalize_advantages` | false | slime parser 默认 |
| `use_rollout_logprobs` | false | slime parser 默认 |
| `use_tis` | false | slime parser 默认 |
| `update_weights_interval` | 1 | 每次更新后同步推理权重 |
| `micro_batch_size` | 1 | parser 默认；动态 batch 开启时不作为固定 microbatch 大小使用 |
| `log_probs_max_tokens_per_gpu` | 16,384 | 未显式设置时继承 `max_tokens_per_gpu` |
| `offload_train` / `offload_rollout` | true / true | 当前 colocate 配置派生 |

与 `/scratch/l/luli/src/opd-baselines/configs/qwen3_4b_opd.yaml` 对齐的是模型选择、数据来源、评分实现、WandB offline／entity 和配置组织方式。优化器、生成长度、批次与步数采用本仓库当前 Lightning 设置，没有逐项复制 baseline 的所有训练超参数。

**12. 标准 OPD 对照配置**

提交入口：

```bash
sbatch configs/sbatch/qwen3_4b_opd.sbatch
```

主要区别如下。未列出的参数应直接查阅 OPD YAML：

| 项目 | Lightning OPD | 标准 OPD |
|---|---|---|
| 训练输入 | 固定响应和教师 logprob Parquet | 适配后的 prompt Parquet |
| 训练时生成 | 每块开始按当前 policy 生成，块内使用缓存 | 按当前学生在线生成 |
| 教师评分 | 每块开始前预计算，共 K 次 | 训练期间在线计算 |
| Actor GPUs | 8 | 2 |
| Rollout GPUs | 与 actor 共用 8 | 独立 4 |
| Teacher GPUs | 训练期间不预留 | 独立 2 |
| Actor TP / DP | 2 / 4 | 2 / 1 |
| 每批 prompt 数 | 256 | 64 |
| 每 prompt 响应数 | 1 | 4 |
| 每批响应总数 | 256 | 256 |
| `balance_data` | false | true |
| `apply_chat_template` | false，固定序列已渲染 | true，在线渲染 |
| `policy_rollout_k` | 1（可配置） | 每次在线 rollout 采样，共 150 次 |
| 保存模式 | `save_model: final` | `save_model: final` |
| `walltime_checkpoint_margin_seconds` | 1200 | 1200 |
| `save_total_limit` | 1 | 1 |
| `save_retain_interval` | null | null |
| 默认实验名 | `qwen3_4b_lightning_opd_base_k1` | `qwen3_4b_opd_base` |

两个实验的 GPU 分工和每题响应数不同。当前对照可比较这两个配置的端到端行为；若要报告严格受控的速度／质量对照，需要在审查后明确是否统一采样数量、初始化、评测和计时范围。

标准 OPD 的默认 teacher 在 IP 排序最后的节点使用最后两张 GPU，Ray 使用其余六张 GPU。外部 `teacher_url` 需要与对应 GPU 分配共同调整，现有 YAML 注释给出该模式的要求。

**13. 环境、.env 与 WandB offline 同步**

| 项目 | 当前路径／版本 |
|---|---|
| Python | `/scratch/l/luli/conda/envs/verl/bin/python` |
| Megatron source | `/scratch/l/luli/src/Lightning-OPD/.runtime/verl/src/Megatron-LM` |
| CUDA compiler | `/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/12.6.2` |
| PyTorch / runtime CUDA | 2.9.0 / 12.8 |
| SGLang / router | 0.5.6 / 0.3.1 |
| sgl-kernel | 0.3.18.post2 |
| TorchMemorySaver | 0.0.9 |
| Megatron core | 0.14.0，完整 editable 源码 |
| mbridge | 0.13.1 |
| Transformer Engine / FlashAttention2 | 2.6.0 / 2.8.3 |

`verl` 是 Conda 环境名称，训练后端是本仓库的 slime／Megatron／Ray，推理使用 SGLang。环境保留集群已有编译包；共享环境 `pip check` 仍存在发布依赖约束差异，具体情况见环境说明和 `.runtime/verl/after-pip-check.txt`。

当前使用 `attention_backend: fused`，因为 TE 2.6 的 FA2 支持范围不包含现有 FA2 2.8.3；训练使用其 cuDNN fused backend。Megatron 源码还应用了 [兼容补丁](../environment/megatron-core-v0.14-no-mtp.patch)，接受当前训练代码传入的空 `mtp_kwargs={}`；非空 MTP 参数仍报错。

`runtime_env()` 设置 HF 缓存、离线标志、CUDA compiler、Python/Megatron 搜索路径及 PyTorch CUDA runtime 的动态库路径。Python bytecode 使用节点本地临时目录，减轻共享 scratch 的 inode 使用。

两个 YAML 使用 `env_file: .env`。相对 env_file 从仓库根目录解析，加载采用 dotenv 赋值解析而非 shell 执行。API key 保存在根目录 `.env`，不会写进本文或 YAML；[.env.example](.env.example) 只有空值模板。

凭据的已导出环境变量优先于 `.env`；训练的 `WANDB_MODE`、entity 和 project 最终由 YAML 的运行配置决定。当前设置为：

```yaml
use_wandb: true
wandb_mode: offline
wandb_entity: li-jiang-opd
project_name: onestep-opd
```

训练日志保存到 `<实验目录>/wandb/`。结束后在能联网的登录节点执行：

```bash
cd /scratch/l/luli/src/Lightning-OPD

# 当前 Lightning 实验
bash configs/environment/sync_wandb.sh

# 当前结果根目录下的所有实验
bash configs/environment/sync_wandb.sh --all-experiments

# 标准 OPD 实验
bash configs/environment/sync_wandb.sh --config configs/config/qwen3_4b_opd.yaml
```

同步实现位于 [sync_wandb.py](../environment/sync_wandb.py)。它读取同一 YAML／`.env`，查找离线 `run-*.wandb`，默认跳过 `.synced` 标记存在的文件，并调用当前配置环境中的 `python -m wandb sync`。上传成功且 run 已结束时由 WandB 写入同步标记。本地日志不会被脚本删除，训练配置仍为 offline。

`--dry-run` 可以只列出同步命令，`--include-synced` 可以包含已同步文件；本轮没有执行这些命令或联网上传。

**14. 结果文件、恢复训练与 HF 导出**

默认实验目录：

```text
/scratch/l/luli/results/lightning-opd/qwen3_4b_lightning_opd_base_k1/
├── config.yaml
├── initialization.json
├── eval.yaml
├── data/
│   ├── fixed_prompts.parquet
│   ├── fixed_prompts.manifest.json
│   ├── block_plan.json
│   └── blocks/block_0000/{student_rollouts,lightning_opd}.parquet
├── models/<hash>/torch_dist/
├── checkpoints/
│   ├── latest_checkpointed_iteration.txt
│   ├── iter_0000149/                  所有 K 的最终模型；时限断点使用实际已完成的编号
│   └── rollout/                       数据读取状态
├── eval_metrics/
│   ├── 0.json                         初始评测
│   └── 149.json                       最终评测
├── wandb/
├── training_status.json              complete / paused 和已完成的全局更新次数
├── ray-node0.log
├── ray-node1.log
└── cluster-<job_id>.done
```

该目录结构描述正式作业的预期输出，完整正式训练尚未执行。正常结束该次 allocation 时 `cluster-<job_id>.done` 内容为 `success`；失败时主节点写入 `failed`，其他节点据此结束。训练是否完成应看 `training_status.json`；Slurm 总日志位于仓库 `tmp/`。

`config.yaml` 保存运行时 YAML 配置，`initialization.json` 保存学生／教师身份，两个 manifest 保存离线数据来源。默认 `resume: auto`：存在 checkpoint 就恢复，没有 checkpoint 则从初始化模型开始。重新提交同一脚本即可续训，也可以显式使用：

```bash
sbatch configs/sbatch/qwen3_4b_lightning_opd.sbatch --resume
```

`--resume` 恢复同一实验保存的模型、优化器／RNG 和数据读取状态。Lightning 支持从时限 checkpoint 所在的 block 中间恢复，使用原缓存的 policy ID 和绝对样本位置；后续 block 正常生成，不增加 block 边界 eval。`num_rollout: 150` 是总目标编号范围，恢复不会自动追加新的 150 次更新。使用其他初始化或教师时，应修改 `experiment_name`。

两种 OPD 均要求 checkpoint、rollout cursor 和保存的训练配置完整；训练数据、batch、seed、打乱方式和优化器等配方改变时拒绝直接 resume。标准 OPD 按保存的 epoch 和样本偏移恢复相同打乱顺序，从下一批 prompt 继续读取；不会重新从第一批开始。评测列表和保存频率可调整。

标准 OPD 使用 `sbatch configs/sbatch/qwen3_4b_opd.sbatch` 自动选择续训，也可以传 `--resume` 强制要求已有 checkpoint。若上次在原定 eval 步保存 checkpoint 后、评测写完前中断，`eval_on_resume: true` 会先补评测已载入的 checkpoint；非 eval 位置的时限 checkpoint 不额外触发评测。即使已保存第 150 次更新，也会补齐最终评测而不再进行训练更新；已有对应 JSON 时保留原来的评测。

训练权重是 Megatron 分布式格式；原生 eval 会同步这些权重到 SGLang，无需先导出 HF。需要部署或外部工具评测时，可以在训练完成后运行现有 [HF 导出脚本](../../scripts/convert_megatron_to_hf.sh)，或直接调用：

```bash
/scratch/l/luli/conda/envs/verl/bin/python tools/convert_torch_dist_to_hf.py \
  --input-dir /scratch/l/luli/results/lightning-opd/qwen3_4b_lightning_opd_base_k1/checkpoints/iter_0000149 \
  --output-dir /scratch/l/luli/results/lightning-opd/qwen3_4b_lightning_opd_base_k1/hf_final \
  --origin-hf-dir /absolute/path/to/the-selected-student-hf-snapshot
```

`origin-hf-dir` 使用实际选定的 Base 或 SFT 本地目录，用于配置和 tokenizer。上述导出步骤尚未在本轮执行，也不属于当前默认 sbatch 的自动阶段。

**15. 已验证范围与论文复现范围**

历史交互 allocation `514689` 在单节点 4 张 H100 上验证了：GPU kernels、显存卸载／恢复、学生生成、教师 logprob、HF→Megatron 转换、Ray 加载与权重同步、初始 eval、一次真实 optimizer 更新、checkpoint 保存及最终 eval。

该检查使用 2 条训练样本、2 条评测样本、32-token 响应上限，TP=2、DP=2。其分数用于检查可运行性，不用于论文性能对比。记录位置：

```text
.runtime/verl/checks/514689/passed.json
.runtime/verl/interactive-514689.log
configs/environment/verl_runtime.json
```

运行环境源码和检查产物从仓库 `data/runtime/` 迁移到 `.runtime/`，editable 安装路径已同步更新，记录保留迁移信息。迁移后未进行新 GPU 检查。

尚未完成的正式实验工作包括：完整离线响应／教师 logprob 准备、双节点完整训练与耗时测量、最终模型质量测量、HF 导出和 WandB 上传。

README 的论文配方包含同教师 SFT 初始化，并报告 AIME2024、AIME2025、HMMT2025 和 LiveCodeBench 等评测。当前默认配置跳过 SFT，Lightning 采用用户 baseline 的 8 个本地数学／选择题评测集，标准 OPD 为 9 个，不能直接将其结果填入 README 的原始论文结果表。

当前 `num_rollout: 150` 对应 README 的约 150 步收敛说明；原始 launcher 曾使用更大的训练上限。文档和当前脚本没有保证 Base 初始化、不同评测集合或两节点通信布局会得到相同论文分数。

**16. 完整 Lightning OPD YAML 快照**

以下是本次 K-block 改动后的配置快照；实际参数以 YAML 为准。

```yaml
# Qwen3-4B Lightning OPD. Launcher keys follow the opd-baselines naming;
# training keys map directly to slime flags (underscore -> hyphen).
method: lightning_opd
# Student initialization: HuggingFace model ID or local HF model directory.
model_path: Qwen/Qwen3-4B-Base
teacher_model_path: Qwen/Qwen3-8B
model_type: qwen3-4B
local_models_only: true
hf_home: /scratch/l/luli/hf
auto_prepare_model: true
conversion_num_gpus: 4

# Existing verl environment extended with the Lightning OPD runtime.
python_executable: /scratch/l/luli/conda/envs/verl/bin/python
# Full Megatron-LM v0.14.0 training source (pip megatron-core alone lacks training).
megatron_path: /scratch/l/luli/src/Lightning-OPD/.runtime/verl/src/Megatron-LM
# Compiler for FlashInfer JIT; PyTorch retains its own CUDA 12.8 runtime libraries.
cuda_home: /cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/12.6.2
# Local credentials; relative env_file paths are resolved from the repo root.
env_file: .env
use_wandb: true
wandb_mode: offline
wandb_entity: li-jiang-opd
project_name: onestep-opd
experiment_name: qwen3_4b_lightning_opd_base_k1  # Use a new name when changing K.
output_dir: /scratch/l/luli/results/lightning-opd
resume: auto

# Cluster allocation: 2 nodes x 4 H100s (8 total); sbatch wall time is 8 hours.
nnodes: 2
n_gpus_per_node: 4
startup_timeout: 3600
actor_num_nodes: 2
actor_num_gpus_per_node: 4
# Student inference engines share actor GPUs for block rollouts and evaluation.
# Actor/inference memory is offloaded before each block's temporary teacher scoring.
colocate: true
rollout_num_gpus: 8
rollout_num_gpus_per_engine: 1
sglang_mem_fraction_static: 0.4

# The default sbatch builds the fixed prompt sequence, then alternates K times:
# current-policy student rollout -> cached teacher logprobs -> block training.
# Prepared parquet inputs live outside the checkout under /scratch/l/luli/data/lightning_opd/.
# Rebuild with: python configs/lightning_opd/qwen3-4b-lightning-opd.py --prepare-prompts
source_data_dir: /scratch/l/luli/data
train_files: /scratch/l/luli/data/lightning_opd/train/dapo_math_17k.parquet
max_train_samples: 100000
# null -> <output_dir>/<experiment_name>/data/fixed_prompts.parquet
# Exactly num_rollout * rollout_batch_size rows, repeating shuffled DAPO decks.
# This prompt order is independent of K; responses/logprobs are cached per block.
lightning_fixed_data: null
input_key: prompt
label_key: label
metadata_key: metadata
# Offline prompts are already rendered; never apply a second chat template.
apply_chat_template: false
apply_chat_template_kwargs: {enable_thinking: true}
rollout_shuffle: false  # Keep fixed absolute positions; the builder shuffles DAPO decks.
rollout_seed: 42
seed: 42
num_rollout: 150  # README convergence setting; set 3000 to match the original launcher limit.
policy_rollout_k: 1  # Integer in [1, num_rollout]; number of current-policy rollout blocks.
rollout_batch_size: 256
n_samples_per_prompt: 1
rollout_max_prompt_len: 2048
rollout_max_response_len: 4096
rollout_temperature: 0.8
rollout_top_p: 1.0
rollout_top_k: -1
global_batch_size: 256
balance_data: false

# One-node CPU preparation builds the fixed 38,400-row sequence before training.
# Each block uses current Ray student engines; teacher scoring temporarily uses 2 GPUs.
precompute_concurrency: 32
precompute_port: 13142
precompute_tp_size: 2  # Legacy preparer only; block rollouts use rollout_num_gpus_per_engine.
teacher_num_gpus: 2  # Teacher scoring is temporary; all 8 GPUs remain available for training.
teacher_tp_size: 2
teacher_port: 13141
teacher_mem_fraction_static: 0.6
teacher_context_length: 32768
teacher_chunked_prefill_size: 4096
teacher_url: null

custom_rm_path: slime.rollout.on_policy_distillation.reward_func
custom_reward_post_process_path: slime.rollout.on_policy_distillation.post_process_rewards
include_verifiable_reward: true
advantage_estimator: on_policy_distillation
use_kl_loss: true
kl_loss_coef: 0.0
kl_loss_type: low_var_kl
entropy_coef: 0.0
optimizer: adam
lr: 2.0e-6
lr_decay_style: constant
weight_decay: 0.1
adam_beta1: 0.9
adam_beta2: 0.98

tensor_model_parallel_size: 2
sequence_parallel: true
pipeline_model_parallel_size: 1
context_parallel_size: 1
expert_model_parallel_size: 1
expert_tensor_parallel_size: 1
recompute_granularity: full
recompute_method: uniform
recompute_num_layers: 1
use_dynamic_batch_size: true
max_tokens_per_gpu: 16384
attention_dropout: 0.0
hidden_dropout: 0.0
accumulate_allreduce_grads_in_fp32: true
attention_softmax_in_fp32: true
# TE 2.6 supports packed GQA through cuDNN; its FA2 limit is 2.8.1,
# whereas the shared verl environment has FA2 2.8.3.
attention_backend: fused

# Answer accuracy on the same local validation sets as opd-baselines.
# Scoring reuses opd-baselines baseline_reward.py/reward_async.py/eval_metrics.py;
# these are the cluster's comparison sets, not the paper's full benchmark suite.
baseline_repo: /scratch/l/luli/src/opd-baselines
# Evaluation runs before training and every 25 steps, including the final step 150.
# Keep num_rollout divisible by eval_steps to evaluate the final trained weights.
eval_steps: 25
eval_on_resume: true  # Recover an evaluation interrupted after checkpoint saving.
val_files:
  - /scratch/l/luli/data/lightning_opd/eval/amc23.parquet
  - /scratch/l/luli/data/lightning_opd/eval/math500.parquet
  - /scratch/l/luli/data/lightning_opd/eval/gsm8k.parquet
  - /scratch/l/luli/data/lightning_opd/eval/aime26.parquet
  - /scratch/l/luli/data/lightning_opd/eval/hmmt26.parquet
  - /scratch/l/luli/data/lightning_opd/eval/mmlu_college_chemistry.parquet
  - /scratch/l/luli/data/lightning_opd/eval/mmlu_college_medicine.parquet
  - /scratch/l/luli/data/lightning_opd/eval/gpqa_diamond.parquet
val_max_prompt_length: 4096
val_max_completion_length: 16384
val_n: 1
val_temperature: 0.7
val_top_p: 0.8
val_top_k: 20
save_eval_generations: false
save_model: final
walltime_checkpoint_margin_seconds: 1200
# Resume inside a block uses its saved policy ID and cached teacher logprobs.
save_total_limit: 1  # null keeps all training checkpoints.
save_retain_interval: null  # Disable Megatron's separate interval-based pruning.
async_save: false
```
