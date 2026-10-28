# 按 sbatch 名称同步 W&B

训练结束后，在能联网的登录节点执行；不需要 GPU，也不会启动训练或重新评测。

```bash
# 文件名可带或不带 .sbatch，也可以给完整路径
bash scripts/sync_wandb.sh qwen3_1_7b_base_nt_teacher_4b_instruct_2507_k40.sbatch

# 一次同步多个实验；也接受 experiment_name
bash scripts/sync_wandb.sh \
  qwen3_1_7b_base_opd_teacher_8b \
  qwen3_1_7b_non_thinking_opd_teacher_4b_instruct_2507

# 只检查本地数据，不连接 W&B
bash scripts/sync_wandb.sh qwen3_1_7b_base_nt_teacher_4b_instruct_2507_k40 --dry-run

# 只读取云端并核验，不修改云端
bash scripts/sync_wandb.sh qwen3_1_7b_base_nt_teacher_4b_instruct_2507_k40 --verify-only
```

名称可以是 sbatch 文件名、`#SBATCH --job-name` 或 YAML 的 `experiment_name`。
默认搜索本仓库和相邻的 `opd-baselines`；其他仓库用 `--repo /path/to/repo`。
同一个 job-name 对应多个配置时，用具体 sbatch 文件名消除歧义。
标准 OPD 同一个配置若有多个独立结果目录，默认分别同步；用 `--job-id 626215`
或 `--run-dir /path/to/results` 可限定一次提交。

脚本读取**实际结果目录保存的 `config.yaml`**，记录 `student_model`、
`teacher_model`、`model_path`、`teacher_model_path`、实际 `val_n` 和
`eval_samples_per_dataset` 等固定参数。后续修改 YAML 不会改变旧实验的模型身份或采样数。
API key、entity 和 project 默认来自仓库 `.env`；可用 `--env-file`、
`--entity`、`--project` 覆盖。Python 默认沿用 `verl` 环境，可通过
`LIGHTNING_OPD_PYTHON=/path/to/python` 覆盖，需要安装 `wandb` 和 `PyYAML`。

本仓库的训练日志默认保存在 `<repo>/wandb/<experiment_name>/` 下，SDK 通常再创建
`wandb/offline-run-*` 子目录。YAML 可设置 `wandb_dir`，相对路径从 repo 根目录解析。
同步时同时检查该目录和旧的 `<output_dir>/<experiment_name>/wandb/`，兼容已有 run。

每个测试集使用以下评估指标；`N` 取实际 `val_n`，目前这些实验是 4：

```text
eval_metric/<测试集>-avg@N
eval_metric/<测试集>-pass@N
eval_metric/<测试集>-response-length
```

评估横轴是 `eval_step`（已完成的训练更新数）。训练和 rollout 分别使用
`train/step`、`rollout/step`。Slime 的旧评估索引会转换为完成步数，
原生 verl 的 `eval/step` 已经是完成步数，直接保留。
`response-length` 是回答生成的平均 token 数。Pass@N 来自原始评测结果，
不会由 Avg@N 推算。

同步前会合并多个进程的原始 `.wandb` 流、评估 JSON 和可定位到的 Slurm 日志，
以恢复进程退出前未刷盘的数据。旧实验的 JSON 若缺少回答长度，可以通过
`--log /path/to/original-slurm.log` 补充原始日志。缺失必需评估指标、源数据冲突、
云端冲突或任务仍在 Slurm 队列时会报错，保留原始记录。

默认复用原始 W&B run ID，并遵循 `tmp/wandb_run_id_map.json` 中的替换关系。
同一指标、同一真实 step 只补录一次，重复执行不会追加重复曲线。
旧名称及 `system/`、`perf/`、`lightning/`、`eval/` 分组会隐藏并从 Summary 移除，
已有历史保留。脚本不修改原始训练日志和 `.synced` 标记。
上传后检查模型参数、指标数值、横轴和历史保留情况，成功时输出 `VERIFIED` 和 run URL。
本地计划、上传前快照及核验报告保存在 `tmp/wandb-sync/<目标run哈希>/`。

如果同一个结果目录包含多个未合并的原始 W&B run ID，先用 `--job-id` 选择
一次提交；确认它们确属同一个实验后，也可以用 `--run-id TARGET_ID` 显式合并。
请保留已有的 `tmp/wandb_run_id_map.json`，以继续同步此前重建过的 run。
