"""Megatron model-argument scripts (configs/models/<model_type>.sh) match the HF checkpoints they train.

A wrong head count, RoPE base or bias flag still converts and trains, just as a different model,
so every 1102 preset's model_type is checked against its local student's config.json.
"""

import json
from pathlib import Path
import subprocess
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]


def model_args(model_type):
    script = ROOT / "configs/models" / f"{model_type}.sh"
    output = subprocess.run(["bash", "-c", 'source "$1"; printf "%s\\0" "${MODEL_ARGS[@]}"', "bash", str(script)],
                            check=True, capture_output=True).stdout.decode().rstrip("\0").split("\0")
    flags, values = set(), {}
    for item, following in zip(output, output[1:] + [""]):
        if item.startswith("--"):
            flags.add(item)
            if not following.startswith("--"):
                values[item] = following
    return flags, values


def expected(hf):
    heads = hf["num_attention_heads"]
    qwen2 = hf["architectures"] == ["Qwen2ForCausalLM"]
    return {
        "--num-layers": str(hf["num_hidden_layers"]), "--hidden-size": str(hf["hidden_size"]),
        "--ffn-hidden-size": str(hf["intermediate_size"]), "--num-attention-heads": str(heads),
        "--num-query-groups": str(hf["num_key_value_heads"]), "--vocab-size": str(hf["vocab_size"]),
        "--rotary-base": str(int(hf["rope_theta"])), "--norm-epsilon": str(float(hf["rms_norm_eps"])),
    }, {
        "--untie-embeddings-and-output-weights": not hf.get("tie_word_embeddings", False),
        "--add-qkv-bias": qwen2, "--qk-layernorm": not qwen2,
    }


class ModelArgsTests(unittest.TestCase):
    def test_every_1102_preset_trains_the_architecture_of_its_student(self):
        checked = 0
        for preset in sorted((ROOT / "configs/config").glob("byte16_*.yaml")):
            cfg = yaml.safe_load(preset.read_text())
            hf_config = Path(cfg["model_path"]) / "config.json"
            if not hf_config.is_file():
                continue  # Model not downloaded on this machine.
            with self.subTest(preset=preset.name, model_type=cfg["model_type"]):
                flags, values = model_args(cfg["model_type"])
                values = {key: value.strip('"') for key, value in values.items()}
                values["--rotary-base"] = values.get("--rotary-base", "").replace("${MODEL_ARGS_ROTARY_BASE:-", "").rstrip("}")
                values["--norm-epsilon"] = str(float(values.get("--norm-epsilon", "nan")))
                numbers, switches = expected(json.loads(hf_config.read_text()))
                self.assertEqual({key: values.get(key) for key in numbers}, numbers)
                self.assertEqual({key: key in flags for key in switches}, switches)
                checked += 1
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()
