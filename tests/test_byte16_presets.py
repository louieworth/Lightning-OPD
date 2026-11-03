"""The 1102-cluster presets (byte16_*.yaml: 2 pods x 8 H100, pre-started Ray) reproduce
"When Do We Need On-Policy Distillation?" (arXiv 2610.11291) for four teacher-student pairs.

Each pair runs K=1 (Lightning OPD / Semi-OPD, method lightning_opd), K=40 (Lightning OPD with 40
current-policy rollout blocks; user request 11/03) and K=120 (standard OPD, method opd). Besides paths, models and the 16-GPU layout, the user approved changing only the
evaluation (AIME24/25/26 + HMMT Feb26, avg@8, 32k response limit, a step-0 evaluation once) and
generation throughput. Every training setting must stay the original Slurm preset's.
K=120 doubles the source preset's 2:4:2 GPU split (user decision 11/02, after a trial of rollout 10 /
teacher TP2 / SGLang memory 0.85-0.7 sped up no training step).
"""

from pathlib import Path
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs/config"
MODELS = Path("/opt/tiger/models")
K1_SOURCE = "qwen3_1_7b_nt_teacher_4b_instruct_2507_k1.yaml"
K120_SOURCE = "qwen3_1_7b_opd_nt_teacher_4b_instruct_2507.yaml"
# name prefix -> (student, teacher, model_type, chat template kwargs): paper Table 3 rows a, d, i, l.
PAIRS = {
    "qwen3_1_7b_nt_teacher_4b_instruct_2507": ("Qwen3-1.7B", "Qwen3-4B-Instruct-2507", "qwen3-1.7B", {"enable_thinking": False}),
    "qwen3_1_7b_nt_teacher_4b": ("Qwen3-1.7B", "Qwen3-4B", "qwen3-1.7B", {"enable_thinking": False}),
    "r1_distill_qwen_1_5b_teacher_justrl_1_5b": (
        "DeepSeek-R1-Distill-Qwen-1.5B", "JustRL-DeepSeek-1.5B", "deepseek-r1-distill-qwen-1.5B", {}),
    "r1_distill_qwen_7b_teacher_light_r1_14b": (
        "DeepSeek-R1-Distill-Qwen-7B", "Light-R1-14B-DS", "deepseek-r1-distill-qwen-7B", {}),
}
PRESETS = {f"byte16_{pair}_{k}.yaml": (pair, k) for pair in PAIRS for k in ("k1", "k40", "k120")}
# Lightning presets and their number of current-policy rollout blocks (policy_rollout_k).
LIGHTNING_K = {"k1": 1, "k40": 40}
PATHS_AND_LAYOUT = {"model_path", "teacher_model_path", "model_type", "hf_home", "output_dir", "experiment_name",
                    "wandb_dir", "source_data_dir", "train_files", "ray_cluster", "nnodes", "n_gpus_per_node",
                    "actor_num_nodes", "actor_num_gpus_per_node", "colocate", "rollout_num_gpus",
                    "teacher_num_gpus", "teacher_tp_size", "apply_chat_template_kwargs"}
APPROVED_EVAL = {"val_files", "val_n", "val_max_completion_length", "val_before_train"}
# User-approved 2026-11-02: generation throughput only (requests in flight per SGLang engine and
# client threads for block generation / teacher scoring); sampling seeds stay per data position.
APPROVED_THROUGHPUT = {"sglang_max_running_requests": 64, "sglang_server_concurrency": 64, "precompute_concurrency": 1024}
# User-approved 2026-11-02: D k120's 7B actor (4 GPUs, TP2) failed step 3 allocating 4.64 GiB of logits
# while PyTorch held 10.28 GiB reserved but unallocated. Allocator only, training processes only.
APPROVED_MEMORY = {"byte16_r1_distill_qwen_7b_teacher_light_r1_14b_k120.yaml": {
    "train_env_vars": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}}}


def load(name):
    return yaml.safe_load((CONFIGS / name).read_text())


def source(k):
    return K1_SOURCE if k in LIGHTNING_K else K120_SOURCE


class Byte16PresetTests(unittest.TestCase):
    def test_the_twelve_presets_exist(self):
        self.assertEqual(sorted(path.name for path in CONFIGS.glob("byte16_*.yaml")), sorted(PRESETS))

    def test_k1_and_k40_are_lightning_opd_and_k120_is_standard_opd(self):
        for name, (_, k) in PRESETS.items():
            with self.subTest(config=name):
                cfg = load(name)
                if k in LIGHTNING_K:
                    self.assertEqual((cfg["method"], cfg["policy_rollout_k"]), ("lightning_opd", LIGHTNING_K[k]))
                else:
                    self.assertEqual(cfg["method"], "opd")
                    self.assertNotIn("policy_rollout_k", cfg)
                self.assertEqual(cfg["num_rollout"], 120)

    def test_training_settings_are_the_source_presets(self):
        for name, (_, k) in PRESETS.items():
            with self.subTest(config=name):
                cfg, original = load(name), load(source(k))
                changed = {key for key in cfg.keys() | original.keys() if cfg.get(key) != original.get(key)}
                memory = APPROVED_MEMORY.get(name, {})
                k_blocks = {"policy_rollout_k"} if k == "k40" else set()
                self.assertEqual(changed - PATHS_AND_LAYOUT - APPROVED_EVAL - set(APPROVED_THROUGHPUT) - set(memory) - k_blocks,
                                 set())
                self.assertEqual({key: cfg[key] for key in APPROVED_THROUGHPUT}, APPROVED_THROUGHPUT)

    def test_only_d_k120_training_uses_expandable_segments(self):
        for name, (_, k) in PRESETS.items():
            with self.subTest(config=name):
                expected = APPROVED_MEMORY.get(name, {}).get("train_env_vars", load(source(k)).get("train_env_vars"))
                self.assertEqual(load(name).get("train_env_vars"), expected)

    def test_pairs_models_and_approved_evaluation(self):
        evaluation = [f"/opt/tiger/Lightning-OPD/data/lightning_opd/eval/{name}.parquet"
                      for name in ("aime24", "aime25", "aime26", "hmmt26")]
        for name, (pair, k) in PRESETS.items():
            with self.subTest(config=name):
                cfg = load(name)
                student, teacher, model_type, chat = PAIRS[pair]
                self.assertEqual((cfg["model_path"], cfg["teacher_model_path"], cfg["model_type"]),
                                 (str(MODELS / student), str(MODELS / teacher), model_type))
                self.assertEqual(cfg["apply_chat_template_kwargs"], chat)
                self.assertEqual((cfg["val_files"], cfg["val_n"], cfg["val_max_completion_length"]), (evaluation, 8, 32768))
                self.assertEqual((cfg["ray_cluster"], cfg["nnodes"], cfg["n_gpus_per_node"]), ("existing", 2, 8))
                self.assertEqual(cfg["experiment_name"], f"{pair}_{k}_byte16")

    def test_opd_doubles_the_source_gpu_split(self):
        def split(cfg):
            return (cfg["actor_num_nodes"] * cfg["actor_num_gpus_per_node"], cfg["rollout_num_gpus"],
                    cfg["teacher_num_gpus"], cfg["teacher_tp_size"])

        doubled = tuple(2 * gpus for gpus in split(load(K120_SOURCE)))
        for name, (_, k) in PRESETS.items():
            if k == "k120":
                with self.subTest(config=name):
                    self.assertEqual(split(load(name)), doubled)

    def test_k40_uses_the_k1_layout(self):
        keys = ("actor_num_nodes", "actor_num_gpus_per_node", "colocate", "rollout_num_gpus", "teacher_num_gpus", "teacher_tp_size")
        for pair in PAIRS:
            with self.subTest(pair=pair):
                k1, k40 = load(f"byte16_{pair}_k1.yaml"), load(f"byte16_{pair}_k40.yaml")
                self.assertEqual({key: k40[key] for key in keys}, {key: k1[key] for key in keys})

    def test_step_zero_is_evaluated_once_per_student(self):
        evaluated = sorted(name for name in PRESETS if load(name)["val_before_train"])
        self.assertEqual(evaluated, ["byte16_qwen3_1_7b_nt_teacher_4b_instruct_2507_k1.yaml",
                                     "byte16_r1_distill_qwen_1_5b_teacher_justrl_1_5b_k1.yaml",
                                     "byte16_r1_distill_qwen_7b_teacher_light_r1_14b_k1.yaml"])


if __name__ == "__main__":
    unittest.main()
