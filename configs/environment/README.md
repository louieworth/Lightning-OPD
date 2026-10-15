# 集群运行环境

2026-10-09 已在交互 allocation `514689`（单节点四张 H100）通过完整小规模
检查，包括一次真实梯度／优化器更新、checkpoint 保存及更新权重后的最终
评测。版本、检查范围和日志位置见本目录的 `verl_runtime.json`。
双节点正式训练尚未启动。

两个 Qwen3-4B YAML 都直接使用现有环境：

```text
Python:       /scratch/l/luli/conda/envs/verl/bin/python
Megatron-LM:  /scratch/l/luli/src/Lightning-OPD/.runtime/verl/src/Megatron-LM
CUDA compiler:/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/12.6.2
```

sbatch 会自动设置 Python、Megatron 源码路径、CUDA 编译器和节点本地 Python
缓存目录。提交作业前无需执行 `conda activate`。交互使用时可以执行：

```bash
source /scratch/l/luli/conda/etc/profile.d/conda.sh
conda activate /scratch/l/luli/conda/envs/verl
```

本次补全了 SGLang 0.5.6、SGLang Router 0.3.1、sgl-kernel 0.3.18.post2、
torch-memory-saver 0.0.9、mbridge 0.13.1，以及 Megatron-LM core_v0.14.0
的完整训练源码（提交 `23e00ed0963c35382dfe8a5a94fb3cda4d21e133`）。
Megatron 以 editable 方式安装；请保留上面的源码目录。
安装前后包清单位于 `.runtime/verl/before-freeze.txt` 和
`.runtime/verl/after-freeze.txt`。

Megatron 源码应用了本目录的 `megatron-core-v0.14-no-mtp.patch`：当前仓库的
训练代码始终传入 `mtp_kwargs={}`，而 Megatron 0.14 尚未提供这个参数。
补丁接受这个空参数，计算路径保持原样；非空的 MTP 参数仍会报错。
重新下载同一提交时，可从源码目录使用 `git apply` 应用该补丁。

保留现有 PyTorch 2.9.0 / CUDA 12.8、vLLM 0.12.0 和集群编译的 Transformer
Engine 2.6.0。SGLang 的发布元数据固定了部分其他版本，包括 Torch 2.9.1，
因此这个共享环境的 `pip check` 仍会报告版本约束差异及未安装的多媒体／其他
评测依赖；它不是完全按 SGLang 官方依赖锁文件安装的环境。实际兼容性通过
下列 H100 检查验证，不以 `pip check` 通过代替 GPU 验证。

训练的 `attention_backend` 设置为 `fused`，使用 Transformer Engine 的 cuDNN
实现：现有 FA2 2.8.3 超出 TE 2.6 的 FA2 支持范围，无法使用其 `flash` 后端。
这项选择也可以在 YAML 中配置。独立的 FlashAttention 内核仍纳入 GPU 检查。

从仓库根目录申请保持 30 分钟的交互 allocation：

```bash
bash configs/environment/start_verl_interactive.sh
check_verl
```

检查失败会返回交互 shell，allocation 保留，可以修改后再次执行 `check_verl`。
Slurm 在 30 分钟时限到达后回收 allocation；中途执行 `exit` 会提前释放。

一次性检查也可使用：

```bash
sbatch configs/sbatch/check_verl_runtime.sbatch
```

该检查申请一节点四张 H100，依次检查 CUDA 内核、训练使用的 packed GQA
attention 前向／反向、显存卸载与恢复、两条 DAPO 样本的学生生成和教师
logprob，以及 Qwen3-4B 的 Megatron checkpoint 转换。完整通过后才写入
`.runtime/verl/checks/<job_id>/passed.json`。测试数据和 checkpoint 使用独立
目录；正式实验仍会生成完整离线训练数据。

在交互 shell 中执行 `check_verl --train-smoke`，还会用这两条离线样本运行一次
训练更新和两条评测，并保存 checkpoint／指标。这仅检查流水线可运行性，
不产生可用于论文比较的模型分数。

正式流水线（两节点，每节点四张 H100，共八张，八小时）：

```bash
sbatch configs/sbatch/qwen3_4b_lightning_opd.sbatch
```

顺序为准备／复用输入 parquet、检查两节点环境、构建固定 38,400 条 prompt、
转换／加载 checkpoint，然后按 K 次当前 student rollout→teacher logprob→block
训练循环，执行初始／周期／最终评测。数据和结果路径、
初始学生模型 `model_path`、教师模型 `teacher_model_path`、WandB 等设置均在
`configs/config/qwen3_4b_lightning_opd.yaml`；两个 YAML 都通过 `env_file: .env`
读取仓库根目录的 `.env`。该文件保存已有 WandB 凭据，权限为 `600`，并已加入
Git 忽略规则；`configs/config/.env.example` 只提供占位符模板。
模型／数据／运行环境路径和 WandB 项目、模式等参数仍在 YAML 中配置。

两个 OPD YAML 每 25 次更新评测，使用全局步数，独立于 Lightning 的 K。
Lightning 通过 `policy_rollout_k` 将固定数据按完整 batch 分块：K=1 是一块
38,400 条；K=2 是两块各 19,200 条，第二块使用完成前 75 步后的 policy。
actor、optimizer、scheduler 和全局步数保持连续。只在首块完成后保存一次：
K=1 第 150 步、K=2 第 75 步；K>1 的最终模型只在内存中评测。
详细审查见 [K-block 流程](../config/qwen3_4b_lightning_k_review.md)。

标准 OPD 的 `save_model: active`、`save_eval_multiplier: 2` 将保存间隔设为
50 步：150 步训练在第 50、100、150 步保存。`save_model`
可设为 `null`（关闭保存）、`final`（只保存最终步）或 `active`（按倍数保存）。
`save_eval_multiplier` 只接受 ≥1 的整数，在 `null`／`final` 模式不参与计算。
`active` 模式要求总步数可被保存间隔整除。`save_total_limit: 1` 在已保存的
最新模型评测成功后清理旧训练
checkpoint，保留完整续训状态。继续训练使用对应
sbatch 的 `--resume`；`num_rollout: 150` 是包含已完成更新的总目标。
`eval_on_resume: true` 会补齐原定 eval 步上保存后被中断的评测。
Lightning resume 从首块 checkpoint 继续；之后未保存的更新需重跑，重新生成
后续块，且非 eval 位置的 block 边界不额外触发 eval。更改 K 需要新的 `experiment_name`。
SFT 使用同名保存数量参数，默认保留一个完整 checkpoint，并允许在续跑时
修改 W&B 项目和保存数量。

原始数据已经是 Parquet：训练文件为
`/scratch/l/luli/data/train/dapo_math_17k.parquet`，评测文件位于
`/scratch/l/luli/data/eval/`。它们使用 verl 的 `reward_model`／`extra_info`
字段，适配后的 `label`／`metadata` 版本统一存放于
`/scratch/l/luli/data/lightning_opd/train/` 和
`/scratch/l/luli/data/lightning_opd/eval/`，并由两个 YAML 直接读取。
数据清单为 `/scratch/l/luli/data/lightning_opd/prepared_inputs.json`。
仓库原来的 `data/` 已删除；运行环境源码和历史检查记录迁到 `.runtime/`，
Conda 中 Megatron editable 安装的路径也已同步更新。
正式实验生成的学生响应和教师 logprob 继续写入 YAML 结果目录下的 `data/`。

SFT、OPD 和 Lightning OPD 的 WandB 项目统一默认使用 `onestep-opd`。
OPD 和 Lightning OPD 的数据集 Avg@1 单独记录为
`eval_metric/<数据集>-avg@1`（例如 `eval_metric/math500-avg@1`），
该分组每个数据集只有一个指标，横轴使用隐藏的 `eval_step`，对应已完成更新数。
初始评测为 0，之后每 25 步为 25、50、75、100、125、150；旧的
24、49、74 等从 0 编号的索引在同步时转换。上传前检查已有评测记录，
每个数据集的同一评测 step 只上传一次，重复同步不会追加第二轮评测曲线。
WandB 不再记录 `system/`、`perf/`、`lightning/` 和 `eval/`，并关闭自动系统监控。
本地评测 JSON、训练日志及 TensorBoard 仍保留诊断信息。
WandB 默认 `offline`，训练日志保存到
`<output_dir>/<experiment_name>/wandb/` 下。训练结束后，在能联网的登录节点
从仓库根目录手动同步：

```bash
bash configs/environment/sync_wandb.sh
```

脚本读取同一 YAML 和 `.env`，递归查找当前实验的离线 run 文件，默认跳过
已同步的文件。上传时生成过滤副本，移除上述分组并将旧 `eval/step` 转为
`eval_step`，不改动训练进程的原始文件。全部进程日志上传后，脚本恢复
`eval_metric/*` 的默认横轴绑定，避免最后一个进程覆盖定义而导致曲线错位。
同一 run 的多个进程日志使用追加同步，避免覆盖已有记录。
重建云端 run 后，`tmp/wandb_run_id_map.json` 保存旧 ID 到新 ID 的映射。
同步原离线文件及从原 checkpoint 续训时会自动使用新 ID；请保留该映射文件。
上传成功并且 run 已结束后由 WandB 写入 `.synced` 标记。
上传子进程使用 online 模式，训练配置保持 offline。其他用法：

```bash
# 标准 OPD 实验
bash configs/environment/sync_wandb.sh --config configs/config/qwen3_4b_opd.yaml

# output_dir 下所有实验
bash configs/environment/sync_wandb.sh --all-experiments

# 只列出将要同步的文件和命令
bash configs/environment/sync_wandb.sh --dry-run

# 只恢复已有 run 的 Avg@1 横轴定义，不重新上传历史
bash configs/environment/sync_wandb.sh --repair-eval-axis RUN_ID

# 指定某个实验的 WandB 目录
bash configs/environment/sync_wandb.sh /scratch/l/luli/results/lightning-opd/qwen3_4b_lightning_opd_base_k1/wandb
```
