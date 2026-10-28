# 集群运行环境

当前 `configs/config/*.yaml` 默认使用当前 Python，不再绑定 `.runtime/` 或
`/scratch/l/luli/conda/envs/verl`。训练入口和 `slime` 代码仍来自本仓库。

## 当前依赖加载方式

- `python_executable: null`：sbatch 使用提交时 PATH 中的 Python；可通过
  `LIGHTNING_OPD_PYTHON=/path/to/python` 指定，或在 YAML 填写解释器路径。
  读取 YAML 的启动解释器也需要安装 PyYAML。
- `megatron_path: null`：从所选环境导入完整 Megatron，包括
  `megatron.core` 和 `megatron.training`。单独安装 `megatron-core` 未必提供
  `megatron.training`。若源码位于其他目录，设置 `MEGATRON_PATH` 或 YAML
  的 `megatron_path`；不需要把源码放在本仓库的 `.runtime` 中。
- `cuda_home: null`：沿用环境中的 `CUDA_HOME` / PATH；CUDA 编译器仍需
  可供 FlashInfer 等 JIT 操作使用。
- SFT 的 `generation_python: null` 默认沿用训练解释器；
  `sft_dependency_archive: null` 直接加载环境内已安装的 LlamaFactory 等包。
  压缩包只作为可选 overlay，可通过 YAML 或 `SFT_DEPENDENCY_ARCHIVE` 指定。
- W&B 同步和 GPU 检查脚本同样使用当前 Python 或 `LIGHTNING_OPD_PYTHON`。
  检查脚本的旧文件名保留兼容性，新的 GPU 检查结果写入 `tmp/runtime-checks/`。

上游安装方式见
[Lightning-OPD Installation](https://github.com/jet-ai-projects/Lightning-OPD#installation)：
数据生成、SFT 和 OPD 使用不同依赖环境，OPD 在预装 Megatron、PyTorch、SGLang
的 Docker 镜像内运行。本仓库以固定 digest 的上游镜像为基础，在构建阶段安装
本地代码和新增依赖。当前两节点 H100 入口为：

```bash
bash run_docker.sh build
bash run_docker.sh submit --validate-only
bash run_docker.sh submit
```

默认实验为 Qwen3-1.7B-Base / Qwen3-4B-Instruct-2507，2 节点 × 4 H100。
Tamia 使用 Apptainer 运行 Docker 派生镜像；完整说明见
[容器入口](../../containers/README.md)。容器内使用 `/opt/Megatron-LM`，不引用
旧的 `.runtime` 或 `verl` 环境，运行阶段不会安装包。

也可以在自行准备的 Python 环境内安装本仓库及训练依赖；环境名称任意，无须
安装或导入 `verl` 框架。`pip install -e .` 会安装本仓库和 requirements 中的包，
完整 Megatron、PyTorch、SGLang 及相应 CUDA 扩展仍需由所选环境提供。
SFT 可在单独的兼容环境内安装 `sft_requirements.txt` 中的依赖。

YAML 中的模型缓存、数据和结果路径，以及 sbatch 的集群资源配置仍保留本机
设置。使用容器或其他机器时需挂载相应目录或调整这些路径。旧 Megatron 0.14
环境还应用过本目录的兼容补丁，具体版本记录见下方；更换依赖后应重新运行
GPU 检查，不能把旧环境的验证结果用于新环境。

删除旧依赖前，确认所选解释器及 `megatron.core`、`megatron.training` 的源码位置
均已独立于旧目录。旧环境的 editable 安装可能仍指向 `.runtime`，仅清空 YAML
路径不会迁移这些安装。可先在新环境中检查：

```bash
python -c 'import sys,slime,megatron.core,megatron.training; print(sys.executable); print(slime.__file__); print(megatron.core.__file__); print(megatron.training.__file__)'
# 在 GPU allocation 内验证 CUDA 内核；完整训练检查可加 --train-smoke。
bash configs/environment/run_runtime_check.sh --kernels-only
```

## 历史环境验证记录

以下内容保留旧环境的安装和检查信息，其中的 `verl`、`.runtime` 路径及旧
配置文件名描述历史状态，不再是当前训练 YAML 的默认依赖。

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

默认直接读取 YAML 的 `train_files`／`val_files`，检查两节点环境、构建固定 prompt 序列、
转换／加载 checkpoint，然后按 K 次当前 student rollout→teacher logprob→block
训练循环，执行初始／周期／最终评测。数据和结果路径、
初始学生模型 `model_path`、教师模型 `teacher_model_path` 等设置均在
`configs/config/qwen3_4b_lightning_opd.yaml`；两个 YAML 都通过 `env_file: .env`
读取仓库根目录的 `.env`。该文件保存已有 WandB 凭据，权限为 `600`，并已加入
Git 忽略规则；`configs/config/.env.example` 只提供占位符模板。
模型／数据／运行环境路径在 YAML 中配置。WandB 的 `WANDB_API_KEY`、
`WANDB_ENTITY`、`WANDB_PROJECT`、`WANDB_MODE` 统一从 `env_file`（默认 `.env`）读取，
覆盖进程中已有的同名值；YAML 中的旧 WandB 项目、entity、模式字段不再参与配置。
训练参数、SFT 和离线同步共用这套设置；同步上传子进程单独使用 online 模式。

两个 OPD YAML 每 25 次更新评测，使用全局步数，独立于 Lightning 的 K。
Lightning 通过 `policy_rollout_k` 将固定数据按完整 batch 分块：K=1 是一块
38,400 条；K=2 是两块各 19,200 条，第二块使用完成前 75 步后的 policy。
actor、optimizer、scheduler 和全局步数保持连续。checkpoint 按全局评估步数
保存，与 K 和 block 边界无关；最终更新也会保存并评估。
详细审查见 [K-block 流程](../config/qwen3_4b_lightning_k_review.md)。

OPD 和 Lightning OPD 默认使用 `save_model: active`、`save_eval_multiplier: 1`，
每次评估前同步保存完整的模型、optimizer、scheduler、RNG 和数据位置。例如
每 20 步评估时，在第 20、40、60 步保存；第 40 步保存且评估成功后删除第 20 步
checkpoint。`save_model`
可设为 `null`（关闭保存）、`final`（只保存最终步）或 `active`（按倍数保存）。
`save_eval_multiplier` 只接受 ≥1 的整数，在 `null`／`final` 模式不参与计算。
最终更新不要求是保存间隔的整数倍。`save_total_limit: 1` 在已保存的
最新模型评测成功后清理旧训练
checkpoint，保留完整续训状态。继续训练使用对应
sbatch 的 `--resume`；`num_rollout: 150` 是包含已完成更新的总目标。
`eval_on_resume: true` 会补齐原定 eval 步上保存后被中断的评测。
Lightning resume 从最新 checkpoint 继续，在 block 中间恢复时复用该块原来的
policy 和缓存 teacher logprobs。非 eval 位置的 block 边界不额外触发保存或 eval。
OPD/Lightning 不再按作业时限提前 20 分钟保存退出或取消评估。
更改 K 需要新的 `experiment_name`。
SFT 使用同名保存数量参数，默认保留一个完整 checkpoint，并允许在续跑时
修改 W&B 项目和保存数量。

原始数据已经是 Parquet：训练文件为
`/scratch/l/luli/data/train/dapo_math_17k.parquet`，评测文件位于
`/scratch/l/luli/data/eval/`。它们使用 verl 的 `reward_model`／`extra_info`
字段，适配后的 `label`／`metadata` 版本统一存放于
`/scratch/l/luli/data/lightning_opd/train/` 和
`/scratch/l/luli/data/lightning_opd/eval/`，并由两个 YAML 直接读取。
数据清单为 `/scratch/l/luli/data/lightning_opd/prepared_inputs.json`。
默认 sbatch 不转换或重写输入文件；`--preflight` 直接校验 `train_files`／
`val_files`：须非空且包含 `prompt`、`label`、`metadata` 列。
只有需要从 `source_data_dir` 转换原始数据时才显式运行 `--convert-datasets`
（兼容旧名 `--prepare-prompts`）；源文件缺失时允许复用合格的已有输入。
Lightning 的 `--prepare-data` 仍会从 `train_files` 构建固定采样序列和 K-block 计划。
仓库原来的 `data/` 已删除；运行环境源码和历史检查记录迁到 `.runtime/`，
Conda 中 Megatron editable 安装的路径也已同步更新。
正式实验生成的学生响应和教师 logprob 继续写入 YAML 结果目录下的 `data/`。

SFT、OPD 和 Lightning OPD 的 WandB 项目统一默认使用 `onestep-opd`。
OPD 和 Lightning OPD 的数据集平均准确率按实际采样数记录为
`eval_metric/<数据集>-avg@<k>`；`val_n: 4` 时为 `eval_metric/<数据集>-avg@4`。
每题至少一次回答正确的比例记录为 `eval_metric/<数据集>-pass@<k>`，
例如 `eval_metric/amc23-pass@4`。两类准确率都由评估器的原始结果直接记录，
不会用 Avg@4 推算 Pass@4。
各测试集的平均生成长度记录为 `eval_metric/<数据集>-response-length`，
单位为生成 token，不包含 prompt，平均覆盖该测试集所有评估采样。
平均准确率、Pass@k 和长度都使用隐藏的 `eval_step` 作为横轴，对应已完成更新数。
长度也写入本地 `eval_metrics/*.json`；旧训练日志中的
`eval/<数据集>/response_len/mean` 在同步时映射到上述长度指标。
WandB Config 同时记录 `student_model`、`teacher_model`、`policy_rollout_k`、
`val_n` 以及各测试集的 `eval_samples_per_dataset`。`val_n` 会同时传给
CLI 和生成的 eval 配置；同一采样数的测试集会将顶层
`n_samples_per_eval_prompt` 显示为实际值。`avg@4` 为每题 4 次回答的正确率
取平均，再对测试集问题取平均；`pass@4` 为每题 4 次中至少一次回答正确的比例。
旧日志中误标为 `avg@1` 的均值在同步时按该实验的 `val_n` 修正名称，数值不变。
`student_model` 和 `teacher_model` 使用完整 Hugging Face 模型 ID，作为 Config
固定参数，支持在 W&B runs 表中筛选 Base 与非 Base 模型。
初始评测为 0，之后每 25 步为 25、50、75、100、125、150；旧的
24、49、74 等从 0 编号的索引在同步时转换。上传前检查已有评测记录，
每个数据集的同一评测 step 只上传一次，重复同步不会追加第二轮评测曲线。
WandB 不再记录 `system/`、`perf/`、`lightning/` 和 `eval/`，并关闭自动系统监控。
本地评测 JSON、训练日志及 TensorBoard 仍保留诊断信息。
WandB 默认 `offline`，训练日志保存到
当前 repo 的 `wandb/<experiment_name>/` 下，可用 YAML 的 `wandb_dir` 指定路径；
相对路径始终从 repo 根目录解析。SDK 通常在其中创建 `wandb/offline-run-*` 子目录。
同步脚本同时查找旧的 `<output_dir>/<experiment_name>/wandb/`，可继续同步旧 run。
训练结束后，在能联网的登录节点
从仓库根目录手动同步：

```bash
bash scripts/sync_wandb.sh qwen3_1_7b_base_nt_teacher_4b_instruct_2507_k40.sbatch
```

推荐使用上述按 sbatch 名称同步的入口，支持 Lightning OPD 和相邻仓库的标准 OPD。
它先合并各进程流、评估 JSON 和 Slurm 日志，按照真实 step 补齐缺失指标，
记录 student/teacher model 和实际 val_n，并在上传后核验；重复执行不追加重复数据。
完整用法见 [scripts/README_wandb_sync.md](../../scripts/README_wandb_sync.md)。
以下是原始离线流工具的用法，适用于需要直接检查或修复原始流的情况。

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

# 只恢复已有 run 的评估横轴定义，不重新上传历史
bash configs/environment/sync_wandb.sh --repair-eval-axis RUN_ID

# 指定某个实验的 WandB 目录
bash configs/environment/sync_wandb.sh /scratch/l/luli/results/lightning-opd/qwen3_4b_lightning_opd_base_k1/wandb
```
