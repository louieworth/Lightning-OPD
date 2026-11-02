"""Every OPD / Lightning OPD preset reads training prompts in source-file order."""

from pathlib import Path
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]


def algorithm_presets():
    presets = {}
    for path in sorted((ROOT / "configs/config").glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        if cfg.get("method") in ("opd", "lightning_opd"):
            presets[path.name] = cfg
    return presets


class AlgorithmPresetTests(unittest.TestCase):
    def test_no_algorithm_preset_shuffles_training_prompts(self):
        shuffle = {name: cfg.get("rollout_shuffle") for name, cfg in algorithm_presets().items()}
        self.assertTrue(shuffle)
        self.assertEqual({name: value for name, value in shuffle.items() if value is not False}, {})

    def test_every_algorithm_preset_stops_at_chatml_end_of_turn_and_end_of_text(self):
        stops = {name: cfg.get("rollout_stop_token_ids") for name, cfg in algorithm_presets().items()}
        self.assertTrue(stops)
        self.assertEqual({name: value for name, value in stops.items() if value != [151645, 151643]}, {})


if __name__ == "__main__":
    unittest.main()
