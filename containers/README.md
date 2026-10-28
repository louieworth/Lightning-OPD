# 两节点 H100 容器运行

默认配置是 `qwen3_1_7b_base_nt_teacher_4b_instruct_2507_k1.yaml`：
Qwen3-1.7B-Base 学生、Qwen3-4B-Instruct-2507 教师、K=1，2 节点 × 4 H100。
训练超参数取自原 YAML。
正式训练默认使用原实验名加 `_container` 的独立目录，首次从所选 Base 模型
开始，后续提交按原 YAML 的 `resume: auto` 恢复这个容器实验。原结果保留。
可用 `--experiment-name NAME` 指定目录名，或 `--experiment-name original`
明确沿用 YAML 的原目录。生成的 YAML 位于当前作业的检查目录，原 YAML 不改。

## Tamia

在仓库根目录执行：

```bash
bash run_docker.sh build                  # 一次性构建镜像
bash run_docker.sh submit --validate-only  # 30 分钟两节点八卡验证
bash run_docker.sh submit --train-smoke    # 独立的一步训练、checkpoint 和评估
bash run_docker.sh submit                 # 正式训练，默认 8 小时
```

`bash run_docker.sh` 也会提交默认训练，镜像不存在时先构建。宿主机仅使用系统
Python 标准库、Slurm 和容器运行时；训练不使用宿主机 Conda 或 `.runtime`，
运行阶段不安装包。模型与数据仍需存在于 YAML 指定的位置。

Tamia 提供 `StdEnv/2023` 下的 `apptainer/1.4.5`；当前登录环境没有 Docker
客户端和 daemon。`--runtime auto` 优先选择可用的 Docker，否则用 Apptainer
运行 Docker 基础镜像构建的 SIF。显式 `--runtime docker` 不会自动切换运行时。

镜像：`tmp/containers/lightning-opd-h100.sif`。
作业日志：`tmp/slurm-<job-name>-<job-id>.log`。
验证报告：`tmp/containers/checks/<job-id>/`，每节点一个
`imports-node-*.json`，每 GPU 一个 `gpu-rank-*.json`。

只有 NCCL/Gloo、FlashAttention、SGLang RMSNorm、Transformer Engine 前后向、
packed GQA 和内存卸载/恢复全部成功，才写出 GPU 报告。日志还需出现
`Two-node H100 container validation passed`，才表示模型、数据及训练状态的
preflight 也通过。正式作业默认先验证，再准备数据、训练；确认环境后可用
`--skip-distributed-check` 省略八卡检查，依赖导入和 preflight 仍会执行。

`--train-smoke` 默认申请 45 分钟，在 `tmp/containers/checks/<job-id>/smoke`
生成独立 YAML 和结果目录：8 个训练样本、1 次训练迭代、32 token 响应，以及
每个评估集 2 条输入。它不改原 YAML 或原实验 checkpoint。只有训练迭代、
checkpoint 和最终评估完成，才生成 `training-passed.json`。

调试时可独立保留 1 小时的两节点 allocation，失败后继续在同一作业内运行：

```bash
TASK_DEBUG_JOB=$(sbatch --parsable configs/sbatch/qwen3_h100_container_keepalive.sbatch)
/usr/bin/python3 -I -S -B containers/debug.py "$TASK_DEBUG_JOB" --wait --train-smoke
# 修复后重复上一条检查命令；不需要重新申请 allocation。
srun --jobid="$TASK_DEBUG_JOB" --nodes=2 --ntasks=2 --ntasks-per-node=1 hostname
```

保活作业与检查进程独立，`srun` 失败不会结束保活作业。到 1 小时时限或显式
`scancel "$TASK_DEBUG_JOB"` 时释放资源。重复完整检查前会把上次的测试目录
归档到 `tmp/containers/debug/<job-id>/`，正式训练结果不受影响。

## 配置与 sbatch

YAML 可设置 `epoch`，默认 `1`；`num_rollout` 表示每轮更新次数。
例如 `epoch: 2`、`num_rollout: 120`、`policy_rollout_k: 80` 会准备
30,720 条固定提示（batch 128），共训练 240 步，每 3 步刷新一次策略。
K 是全部 epoch 的 block 总数，由 YAML 显式指定，不自动翻倍。
在线 OPD 同样按 `num_rollout * epoch` 更新，数据游标正常循环。
改变 epoch/K 时使用新的 `experiment_name`，旧实验结果保留。
`--train-smoke` 始终只执行 1 步，不受正式配置的 epoch 影响。

```bash
bash run_docker.sh submit --config configs/config/qwen3_4b_base_nt_teacher_8b.yaml
bash run_docker.sh submit --time 10:00:00
bash run_docker.sh submit --experiment-name my-container-experiment

# 原有 OPD/Lightning OPD sbatch 可显式接入容器。
export LIGHTNING_OPD_CONTAINER="$PWD/tmp/containers/lightning-opd-h100.sif"
export LIGHTNING_OPD_CONTAINER_RUNTIME=apptainer
sbatch configs/sbatch/qwen3_1_7b_base_nt_teacher_4b_instruct_2507.sbatch
```

入口面向 OPD/Lightning OPD，YAML 必须为 `nnodes: 2`、`n_gpus_per_node: 4`。
SFT 的 LlamaFactory/vLLM 流程仍使用其独立入口。集群 account 和 partition
沿用已有 sbatch；没有更改模型身份、学习率、批量或步骤数。

## Docker daemon 可用的服务器

```bash
bash run_docker.sh build --runtime docker
bash run_docker.sh submit --runtime docker
```

两节点须事先提供同名的 `lightning-opd:h100` 镜像；Docker 镜像不会像共享 SIF
一样自动出现在另一节点。工作进程使用 host 网络、host IPC、GPU、RDMA 设备
（若存在）和宿主机用户身份，退出时清理自己创建的唯一命名容器。

自己管理两台 GPU 服务器、不使用 Slurm 时，两台各运行一个工作进程，传入
相同的两节点 IP 和数字 job ID，分别使用 rank 0/1：

```bash
# 示例 IP 需要替换；另一台运行相同命令，将 rank 改为 1。
bash run_docker.sh run --runtime docker --node-ips 10.0.0.10,10.0.0.11 \
  --node-rank 0 --job-id 10001 --stage distributed
```

正式训练前在 rank 0 执行 `--stage prepare-data`，随后两台同时执行
`--stage train`。两台的仓库、YAML、镜像和模型必须一致，结果目录必须是共享
存储。仅在第一台启动 Docker 不会自动在第二台启动；Tamia 上使用 `submit`
让 Slurm 完成两节点启动。

## 镜像和挂载

基础镜像固定为上游 `tonyhao96/jetmoe:v0.2` 的 digest：
`sha256:d79db3699f58172088c8897efd4e21aa7960cbd5530c81b02a6bf07132695313`。
保留其已编译的原生 CUDA 组件，新增包见 `requirements-opd.txt`。Megatron 的
core、training 和工具源码安装到 `/opt/Megatron-LM`，本仓库安装到
`/opt/Lightning-OPD`。
补充 Python 包安装在镜像内的 `/opt/opd-venv`，继承基础镜像的原生 CUDA 包，
该路径属于镜像，宿主机无需创建 Python 环境。Torch、CUDA、SGLang 和 TE
保留基础镜像版本；TorchMemorySaver 0.0.9 从基础镜像原用的 Git 提交
`9b8b788fdeb9c2ee528183214cef65a99b71e7d5` 重编译，源码归档校验 SHA256，
回移上游 `7078bc36ebc9` 的 cudaFree 修复：退出 region 后释放 TMS 管理的
虚拟地址时，仍经过其地址归属检查，普通 CUDA 地址照常转发释放。
修复后的原生库指纹保存在 `/opt/lightning-tms-build.json`。
GPU 报告只是单项检查记录，仍须作业成功退出；完整短测还须存在
`smoke/training-passed.json`。

运行时挂载当前仓库并将它放在 PYTHONPATH 最前面，因此训练使用本地 `slime`。
环境经过清理，不继承宿主机 Python 路径。`.env`、`.runtime`、模型和数据不会
打进镜像。默认挂载仓库、`/scratch` 和节点本地 `/tmp`，保持 YAML 绝对路径
不变；其他存储可重复添加 `--bind /path:/path`。

凭据从挂载仓库的 `.env` 或显式传入的环境变量读取，命令日志不打印凭据值。
构建需要网络和足够磁盘空间；日常运行使用已有镜像。

Tamia 的构建命令先在登录节点下载基础镜像，再自动提交 CPU 作业（8 核、
32 GiB、最多 1 小时），使用本地 `SLURM_TMPDIR` 离线解包；构建日志在
`tmp/containers/build-<job-id>.log`。新增 Python 包在登录节点安装进镜像，
不要求计算节点连接 Docker Hub 或 PyPI。
转换后的基础文件由构建用户拥有，Apptainer 安装步骤以普通用户写入 overlay，
无需 Docker daemon、管理员权限或 fakeroot/subuid 映射。

训练 preflight 会检查结果目录所在文件系统的可用空间，预留下一份完整训练
checkpoint（BF16 模型文件体积的约 8 倍，包含 Adam 状态）、尚未生成的初始化
checkpoint 和 5 GiB 工作空间；每次保存前会再次检查。上一份 checkpoint
仍保留到新一轮评估成功后。并行实验共用 scratch 配额，需要为它们各自预留
空间；空间检查不会为其他作业锁定容量。`df -h` 显示的空间也可能受 Lustre
项目配额限制，可用 `lfs project -d "$PWD"` 查项目 ID，再用
`lfs quota -p <project-id> /scratch` 核对。

重复调试会在 `tmp/containers/debug/<job-id>/previous-*/smoke/runs/` 留下
模型和 checkpoint，每份 1.7B 的短测约 26 GiB。确认不再需要的旧短测可以
清理其中的 `models/` 和 `checkpoints/iter_*/`，保留日志、配置、评估指标和
最终验证证据。`tmp/containers/cache/` 是可重新下载的 OCI 构建缓存；运行
已有 `lightning-opd-h100.sif` 不依赖它。正式训练结果不在这些临时目录下。
安装阶段需要登录节点 `/tmp` 至少约 32 GiB 空闲空间；完成后仅将最终 SIF
保存到共享 scratch，临时镜像自动清理。
其他环境的 Apptainer 构建使用临时 ext4 文件和 FUSE 挂载承载解包文件，避免
耗尽共享 scratch 的 inode 配额，需要至少 100 GiB 空闲空间。临时文件在构建
完成后清理，下载的 OCI 层保留在 `tmp/containers/cache`，供重试复用。
