"""Prevent old offline files and checkpoints from recreating deleted cloud runs."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("wandb_run_ids", ROOT / "slime/utils/wandb_run_ids.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class WandbRunIdTests(unittest.TestCase):
    def test_redirects_old_checkpoint_id_and_respects_project_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.json"
            path.write_text(json.dumps({"runs": {"team/project": {"old": "new", "new": "latest"}}}))
            self.assertEqual(module.resolve_wandb_run_id("old", "team", "project", path), "latest")
            self.assertEqual(module.resolve_wandb_run_id("old", "team", "another", path), "old")
            self.assertEqual(module.resolve_wandb_run_id(None, "team", "project", path), None)
            self.assertEqual(module.resolve_wandb_run_id("other", "team", "project", path), "other")

    def test_no_mapping_keeps_original_id_and_cycles_fail_before_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.json"
            self.assertEqual(module.resolve_wandb_run_id("old", "team", "project", path), "old")
            path.write_text(json.dumps({"runs": {"team/project": {"old": "new", "new": "old"}}}))
            with self.assertRaisesRegex(ValueError, "redirect cycle"):
                module.resolve_wandb_run_id("old", "team", "project", path)


if __name__ == "__main__":
    unittest.main()
