"""Run tools/student_teacher_overlap.py for each pair on its own head-node GPU, held through Ray."""
import json, os, subprocess, sys
import ray
ROOT = "/opt/tiger/Lightning-OPD"; M = "/opt/tiger/models"; O = ROOT + "/outputs"
B = "data/blocks/block_0000/student_rollouts.parquet"
PAIRS = [
    ("A Qwen3-1.7B <- 4B-Instruct-2507", "Qwen3-1.7B", "Qwen3-4B-Instruct-2507", "qwen3_1_7b_nt_teacher_4b_instruct_2507_k40_byte16"),
    ("B Qwen3-1.7B <- Qwen3-4B", "Qwen3-1.7B", "Qwen3-4B", "qwen3_1_7b_nt_teacher_4b_k40_byte16"),
    ("C R1-1.5B <- JustRL-1.5B", "DeepSeek-R1-Distill-Qwen-1.5B", "JustRL-DeepSeek-1.5B", "r1_distill_qwen_1_5b_teacher_justrl_1_5b_k40_byte16"),
    ("D R1-7B <- Light-R1-14B", "DeepSeek-R1-Distill-Qwen-7B", "Light-R1-14B-DS", "r1_distill_qwen_7b_teacher_light_r1_14b_k40_byte16"),
    ("0.6B-Base <- 4B-Instruct-2507", "Qwen3-0.6B-Base", "Qwen3-4B-Instruct-2507", "qwen3_0_6b_base_nt_teacher_4b_instruct_2507_k1_byte16_v2"),
    ("1.7B-Base <- 4B-Instruct-2507", "Qwen3-1.7B-Base", "Qwen3-4B-Instruct-2507", "qwen3_1_7b_base_nt_teacher_4b_instruct_2507_k1_byte16_v2"),
    ("4B-Base <- 4B-Instruct-2507", "Qwen3-4B-Base", "Qwen3-4B-Instruct-2507", "qwen3_4b_base_nt_teacher_4b_instruct_2507_k1_byte16_v2"),
]
n, dtype = sys.argv[1], sys.argv[2]
ray.init(address="auto", logging_level="ERROR", log_to_driver=False)
here = ray.util.get_node_ip_address()

@ray.remote(num_gpus=1, num_cpus=4)
def run(label, student, teacher, rollouts):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(str(g) for g in ray.get_gpu_ids()), OMP_NUM_THREADS="4")
    r = subprocess.run([sys.executable, ROOT + "/tools/student_teacher_overlap.py", "--student", f"{M}/{student}",
                        "--teacher", f"{M}/{teacher}", "--rollouts", rollouts, "--n", n, "--device", "cuda",
                        "--dtype", dtype, "--label", label], capture_output=True, text=True, env=env, cwd=ROOT)
    lines = [l for l in r.stdout.splitlines() if l.startswith("{")]
    return lines[-1] if lines else json.dumps(dict(label=label, error=r.stderr[-800:]))

jobs = []
for label, s, t, exp in PAIRS:
    path = f"{O}/{exp}/{B}" if exp else None
    if path is None or not os.path.exists(path):
        print(json.dumps(dict(label=label, skipped="no initial-student rollouts on disk"))); continue
    jobs.append(run.options(resources={f"node:{here}": 0.001}).remote(label, s, t, path))
for line in ray.get(jobs):
    print(line, flush=True)
