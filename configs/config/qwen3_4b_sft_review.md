# Qwen3-4B SFT 训练流程与参数

本阶段先生成同教师 SFT 数据，再训练 Qwen3-4B-Base。模型保存到：

```text
/scratch/l/luli/results/onestep/model/qwen3_4b_sft_qwen3_8b/
```

配方依据是 [Lightning OPD 论文 Appendix B、Table 5](https://arxiv.org/html/2604.13010v1#A2)、仓库 `README.md` 的 Step 0–2，以及 `configs/sft/qwen3-4b-base-open-thoughts3-qwen3-8b.yaml`。SFT 完成后，再单独启动 Lightning OPD。

已于 2026-10-09 正式提交作业 **516328**，提交命令为 `sbatch --parsable configs/sbatch/qwen3_4b_sft.sbatch`。这是教师数据生成与 SFT 作业。提交前已准备并核对 300,000 行 prompt Parquet，训练入口的 CPU 依赖导入通过；没有提交 GPU 调试作业。日志文件为 `/scratch/l/luli/src/Lightning-OPD/tmp/slurm-qwen3-4b-sft-516328.log`。

## 1. 数据与模型

| 项目 | 设置 |
|---|---|
| Student 初始权重 | `Qwen/Qwen3-4B-Base`，读取已有 HF 缓存 |
| SFT teacher | `Qwen/Qwen3-8B`，读取已有 HF 缓存 |
| 后续 OPD teacher | 同一个 `Qwen/Qwen3-8B`，满足论文的 teacher consistency |
| SFT prompt 来源 | `open-thoughts/OpenThoughts3-1.2M` |
| 固定数据 revision | `61bcf9d4eb38b30295efc2021227a63cc5bb34c8` |
| Prompt 数量 | 从 1,200,000 条中随机抽取 300,000 条，seed 42 |
| Teacher 回答 | 对选中 prompt 重新生成；丢弃原始数据里的 QwQ-32B 回答 |
| 文件格式 | 原始数据、抽样 prompt、teacher 分片和合并训练数据均为 Parquet |

SFT 数据不使用 DAPO-Math-17k。DAPO 留到下一阶段，用 SFT 后的 student 生成 OPD rollouts。

所有数据存放在共享数据目录，不创建仓库内的相对 `data/`：

```text
/scratch/l/luli/data/lightning_opd/sft/
├── source/OpenThoughts3-1.2M/
│   ├── source_files.json
│   └── data/train-00000-of-00120.parquet … train-00119-of-00120.parquet
├── openthoughts3_300k_prompts.parquet
├── openthoughts3_300k_prompts.manifest.json
├── qwen3_8b_responses/
│   ├── manifest.json
│   └── rank00000/ … rank00007/ → shard-000000.parquet …
├── openthoughts3_300k_qwen3_8b.parquet
└── openthoughts3_300k_qwen3_8b.manifest.json
```

## 2. 环境加载

训练和教师生成都使用之前的 verl Python：

```text
/scratch/l/luli/conda/envs/verl/bin/python
```

SFT 补充依赖以单个压缩文件保存，避免占用 scratch 文件数量配额：

```bash
/scratch/l/luli/src/Lightning-OPD/.runtime/sft/packages.tar.gz
```

开始 SFT 时，各节点解压到 `/tmp/lightning-sft-deps-<job-id>-luli/`，并仅对 SFT 进程设置该目录为 `PYTHONPATH`。这里放 LlamaFactory、DeepSpeed 及其需要的兼容版本依赖。PyTorch 2.9.0、CUDA runtime、FlashAttention 和 Liger 从现有 verl 环境读取。教师生成进程不加载这层 SFT 依赖。

依赖记录在 `configs/environment/sft_requirements.txt`；需要重新准备时运行：

```bash
bash configs/environment/prepare_sft_dependencies.sh
```

该脚本在本机临时目录使用 venv 做依赖解析，再将补充包打包到上述压缩文件。提交后的训练解释器仍是 verl。没有额外提交 GPU 调试作业。

## 3. 提交与执行顺序

```bash
cd /scratch/l/luli/src/Lightning-OPD
sbatch configs/sbatch/qwen3_4b_sft.sbatch
```

申请 `aip-plbacon` 账户下的 `gpubase_bynode_b2`，2 节点 × 每节点 4 张 H100，总共 8 张，单次 allocation 8 小时。

一个作业按顺序执行以下阶段：

1. 准备或复用 30 万条 Parquet prompt。源码为 `scripts/prepare_sft_parquet.py`，抽样算法与原 `scripts/prepare_sft_prompts.py` 一致：`random.Random(42).sample(...)` 后排序。原始文件下载时校验大小及 SHA256。
2. 写入参数快照、模型路径和数据来源记录；生成 LlamaFactory YAML、DeepSpeed JSON、`dataset_info.json`。源码为 `scripts/qwen3_sft.py` 的 `configure()`。
3. 两个节点各启动 4 个独立 vLLM worker，每个 worker 使用 1 张 GPU。按 `prompts[rank::8]` 划分数据，每个 worker 处理 37,500 条。teacher 采样参数与原 `data_curation/pipeline.py` 默认值一致：temperature 0.7、top-p 0.9、最多 16,384 个新 token、每条 1 个回答。每 32 条保存一个原子写入的 Parquet 分片，每累积 1024 条合并成较大分片再删除冗余小文件；重新提交时复用已完成分片。这样保留 batch 级断点，共享目录长期保存约 296 个大分片，另加各 worker 当前尚未合并的小分片。
4. 校验总行数、sample ID 无重复且覆盖全部所选 prompt，然后流式合并为单个训练 Parquet。只有 300,000 条全部齐备才进入训练。
5. 每个节点启动一个 `torchrun`，每个 `torchrun` 启动 4 个训练进程，共 8 个训练 rank。`MASTER_ADDR` 使用第一个 Slurm 节点，所有节点共用端口，`SLURM_PROCID` 作为 node rank。调用 LlamaFactory `run_exp()` 完成全参数 SFT。

SFT 训练数据字段为：

```python
{
    "sample_id": original_source_row_index,
    "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "<think>...teacher response..."},
    ],
    "tokens": teacher_response_token_count,
}
```

`dataset_info.json` 使用 `sharegpt` 格式，并明确指定 `role/content/user/assistant/system` 标签。LlamaFactory 从绝对 Parquet 路径加载 `messages`，用 `qwen3` 模板编码；`train_on_prompt: false`，仅对 assistant token 计算交叉熵。启用 packing，序列上限为 16384。tokenized 缓存存放在模型目录的 `.pipeline/tokenized/`，供续跑复用。

本阶段没有设置验证集或“最佳 checkpoint”选择规则，保存训练 loss，在时限前或训练完成时保存完整 checkpoint。论文 benchmark 的生成评测属于后续独立工作；SFT loss 不能代替 AIME 等任务准确率。

## 4. 核心训练参数

| 参数 | 本次值 | 依据或说明 |
|---|---:|---|
| `max_steps` | 3000 | 论文 Table 5 |
| `global_batch_size` | 256 | 论文 Table 5 |
| `cutoff_len` | 16384 | 论文 Table 5 |
| `learning_rate` | `8e-5` | 论文 Table 5 |
| `lr_scheduler_type` | cosine | 论文 Table 5 |
| `warmup_ratio` | 0.1 | 论文 Table 5，300 个 optimizer steps |
| `packing` | true | 论文 Table 5 |
| `deepspeed_stage` | 2 | 论文 4B 使用 ZeRO-0；当前 8 卡使用 ZeRO-2，避免完整 Adam 状态初始化时显存不足 |
| `finetuning_type` | full | 原仓库 SFT YAML |
| `per_device_train_batch_size` | 4 | 原仓库 SFT YAML |
| `gradient_accumulation_steps` | 8 | 按当前 GPU 数调整，保持全局 batch |
| `bf16` | true | 原仓库 SFT YAML |
| `enable_liger_kernel` | true | 原仓库 SFT YAML |
| `flash_attn` | fa2 | 复用现有 FlashAttention，控制长序列显存 |
| `gradient_checkpointing` | true | LlamaFactory 默认启用，明确写出 |
| `optim` | adamw_torch | LlamaFactory 默认值，明确写出 |
| `adam_beta1 / adam_beta2` | 0.9 / 0.999 | SFT 默认值；不是 OPD 的 beta2 0.98 |
| `adam_epsilon` | `1e-8` | 默认值，明确写出 |
| `weight_decay` | 0.0 | SFT 默认值；不是 OPD 的 0.1 |
| `max_grad_norm` | 1.0 | 默认值，明确写出 |
| `save_strategy` | `"no"` | 关闭定期保存，由时限和训练完成回调触发完整 checkpoint |
| `save_total_limit` | 1 | 控制磁盘占用，只保留最新完整续训 checkpoint |
| `save_only_model` | false | 保存 optimizer 等完整状态以续跑 |
| `logging_steps` | 1 | 原仓库 SFT YAML |
| `seed` | 42 | 抽样和训练随机种子 |

原仓库默认 32 张 GPU：`32 × 4 × 2 = 256`。当前使用 8 张 GPU：`8 × 4 × 8 = 256`。单次 optimizer update 的 batch 不变，更新次数仍为 3000；吞吐和总耗时会与论文的 SFT 硬件设置不同。

**3000 步与 30 万条数据的关系：** 3000 是 optimizer 更新次数，训练加载完整 300,000 条教师回答。若不考虑 packing，`3000 × 256 = 768,000` 个样本位置，约为 2.56 轮；启用 packing 后，一个序列可以包含多个原始样本，因此实际 epoch 数取决于 tokenized 数据的序列数量，应以训练日志为准。这里按论文固定 3000 步，不改为只训练一轮，也不设置 `max_samples` 截断。

唯一的参数入口为 `configs/config/qwen3_4b_sft.yaml`。最终传给 LlamaFactory 的 YAML 和 DeepSpeed JSON 自动写到：

```text
/scratch/l/luli/results/onestep/model/qwen3_4b_sft_qwen3_8b/.pipeline/
├── config.yaml
├── llamafactory.yaml
├── deepspeed.json
├── dataset_info.json
└── status.json                 # 训练完成或到达时间断点后写入
```

## 5. 八小时限制与续跑

完整 30 万条教师生成加 3000 步 SFT 不保证在一次 8 小时 allocation 内完成。作业在结束前 20 分钟开始保存断点：生成阶段结束当前 batch 后停止；SFT 阶段结束当前 optimizer step 后触发完整 checkpoint。若训练未到 3000 步，状态会记录为 `paused`。

继续运行：

```bash
sbatch configs/sbatch/qwen3_4b_sft.sbatch
```

它会复用已完成教师分片，并从最新完整训练 checkpoint 恢复权重、optimizer、scheduler 和 RNG 状态。作业不会自动申请下一轮资源。Slurm 显示 `COMPLETED` 只表示这次 allocation 正常结束；SFT 是否完成应检查 `.pipeline/status.json` 的 `global_step == 3000` 和 `status == complete`。

默认 `save_strategy: "no"`、`save_total_limit: 1`，仅在时限前或完成 3000 步时保存完整续训状态；新 checkpoint 保存完成后清理更早的 checkpoint。恢复时允许修改 W&B 项目／模式、保存策略和保存数量，不需要重新生成数据；训练参数、初始化模型和教师／数据来源仍受不可变配方检查保护。配置修改从下一次提交生效。

## 6. 输出与 W&B

模型目录直接保存 HF 格式全参数权重、tokenizer、配置、训练状态和 loss；`checkpoint-<step>/` 保存可恢复训练的完整状态，正常训练完成后只保留 `checkpoint-3000/`。完成 3000 步后，将下一阶段 OPD YAML 的 `model_path` 指向该实验目录即可加载。

W&B 从仓库根目录 `.env` 加载凭据，训练期间 `WANDB_MODE=offline`，项目为 `onestep-opd`。离线记录在模型目录的 `wandb/`。需要上传时：

```bash
bash configs/environment/sync_wandb.sh --config configs/config/qwen3_4b_sft.yaml
```

上传脚本仅对同步子进程切换 online，训练配置保持 offline。

后续 Lightning OPD 将 `model_path` 指向本次完整模型，并使用新的 `experiment_name`；teacher 继续使用 `Qwen/Qwen3-8B`。两个 OPD 启动器通过 `model_path` 统一加载已有模型。本次提交只运行教师 SFT 数据生成和 SFT。
