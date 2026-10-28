"""Evaluation JSON retains completion lengths even if a W&B worker exits early."""

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test_eval_logging import load_function


class EvalLengthPersistenceTests(unittest.TestCase):
    def test_persists_mean_generated_tokens_and_uses_dataset_sample_count(self):
        samples = [SimpleNamespace(effective_response_length=length,
                                   metadata={"data_source": "amc23", "sample_uid": "q"},
                                   reward={"acc": reward})
                   for length, reward in zip((0, 2, 4, 6), (0, 1, 0, 1))]
        output = SimpleNamespace(metrics={}, data={"amc23": {"samples": samples}})
        rollout = ModuleType("slime.rollout.sglang_rollout")
        rollout.GenerateState = Mock()
        rollout.generate_rollout = Mock(return_value=output)
        metrics = Mock()
        metrics.compute_eval_metrics.return_value = {"eval/amc23/avg@4": 0.5}
        evaluate = load_function("configs/opd/qwen3-4b-opd.py", "evaluate",
                                 {"copy": copy, "Path": Path, "json": json,
                                  "MODULE": "configs.opd.qwen3-4b-opd",
                                  "baseline_module": lambda name: metrics})
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                eval_datasets=[SimpleNamespace(name="amc23", n_samples_per_eval_prompt=4)],
                n_samples_per_eval_prompt=1, eval_completed_steps=20,
                eval_output_dir=str(Path(directory) / "eval_metrics"))
            with patch.dict(sys.modules, {rollout.__name__: rollout}):
                result = evaluate(args, 19, object())
            saved = json.loads((Path(args.eval_output_dir) / "19.json").read_text())
        self.assertEqual(result.metrics["eval_metric/amc23-response-length"], 3.0)
        self.assertEqual(saved["eval_metric/amc23-response-length"], 3.0)
        self.assertEqual(saved["_eval_completed_steps"], 20)
        self.assertEqual(metrics.compute_eval_metrics.call_args.kwargs["k"], 4)


if __name__ == "__main__":
    unittest.main()
