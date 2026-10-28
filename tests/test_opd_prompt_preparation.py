"""CPU regression checks for direct dataset validation and explicit conversion."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prompt_launcher", ROOT / "configs/opd/qwen3-4b-opd.py")
LAUNCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCHER)


class PromptPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="opd-prompts-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.train = self.root / "prepared/train/dapo_math_17k_en.parquet"
        self.evaluation = self.root / "prepared/eval/amc23.parquet"
        self.source = self.root / "source"
        self.cfg = dict(source_data_dir=str(self.source), train_files=str(self.train),
                        val_files=[str(self.evaluation)])

    def write_parquet(self, path, *, prepared=True, label="1", empty=False):
        row = dict(prompt="one", reward_model=dict(ground_truth=label), extra_info=dict(index=0))
        if prepared:
            row.update(label=label, metadata=dict(sample_uid="0"))
        table = pa.Table.from_pylist([row])
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table.slice(0, 0) if empty else table, path)

    def test_reuses_train_and_eval_without_original_sources(self):
        for path in (self.train, self.evaluation):
            self.write_parquet(path)
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (self.train, self.evaluation)}

        records = LAUNCHER.convert_datasets(self.cfg)

        self.assertEqual([(record["split"], record["rows"]) for record in records], [("train", 1), ("eval", 1)])
        self.assertTrue(all(record["source"] is None for record in records))
        self.assertEqual(json.loads((self.train.parent.parent / "prepared_inputs.json").read_text()), records)
        for path, identity in before.items():
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), identity)

    def test_generates_missing_targets_from_original_sources(self):
        for split, target in (("train", self.train), ("eval", self.evaluation)):
            self.write_parquet(self.source / split / target.name, prepared=False)

        records = LAUNCHER.convert_datasets(self.cfg)

        for record in records:
            table = pq.read_table(record["path"])
            self.assertTrue({"prompt", "label", "metadata"}.issubset(table.column_names))
            self.assertEqual(table["label"].to_pylist(), ["1"])
            self.assertTrue(Path(record["source"]).is_file())

    def test_refreshes_cached_targets_when_original_source_changes(self):
        self.cfg["val_files"] = []
        source = self.source / "train" / self.train.name
        self.write_parquet(source, prepared=False)
        LAUNCHER.convert_datasets(self.cfg)
        original_mtime = self.train.stat().st_mtime_ns
        LAUNCHER.convert_datasets(self.cfg)
        self.assertEqual(self.train.stat().st_mtime_ns, original_mtime)

        self.write_parquet(source, prepared=False, label="changed")
        LAUNCHER.convert_datasets(self.cfg)

        self.assertEqual(pq.read_table(self.train)["label"].to_pylist(), ["changed"])

    def test_rejects_unprepared_or_empty_targets_without_source(self):
        self.cfg["val_files"] = []
        for options in (dict(prepared=False), dict(empty=True)):
            with self.subTest(options=options):
                self.write_parquet(self.train, **options)
                with self.assertRaisesRegex(ValueError, "expected nonempty prepared"):
                    LAUNCHER.convert_datasets(self.cfg)

    def test_reports_both_paths_when_target_and_source_are_missing(self):
        with self.assertRaises(FileNotFoundError) as raised:
            LAUNCHER.convert_datasets(self.cfg)

        self.assertIn(str(self.train), str(raised.exception))
        self.assertIn(str(self.source / "train" / self.train.name), str(raised.exception))

    def test_preflight_reads_configured_inputs_without_source_data_or_conversion(self):
        self.cfg.pop("source_data_dir")
        self.cfg.update(method="lightning_opd", policy_rollout_k=1, eval_steps=1,
                        output_dir=str(self.root), experiment_name="direct-inputs")
        for path in (self.train, self.evaluation):
            self.write_parquet(path)
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (self.train, self.evaluation)}

        LAUNCHER.check_inputs(self.cfg, preparation=True)

        self.assertFalse((self.train.parent.parent / "prepared_inputs.json").exists())
        for path, identity in before.items():
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), identity)

    def test_preflight_rejects_unprepared_and_empty_configured_inputs(self):
        self.cfg.update(method="opd", eval_steps=1)
        for path in (self.train, self.evaluation):
            self.write_parquet(path)
        for target in (self.train, self.evaluation):
            for options, error in ((dict(prepared=False), "prepared prompt/label/metadata"),
                                   (dict(empty=True), "dataset is empty")):
                with self.subTest(target=target, options=options):
                    self.write_parquet(target, **options)
                    with self.assertRaisesRegex(ValueError, error):
                        LAUNCHER.check_inputs(self.cfg, preparation=True)
                    self.write_parquet(target)

    def test_conversion_command_and_legacy_alias_run_only_explicit_conversion(self):
        for flag in ("--convert-datasets", "--prepare-prompts"):
            with self.subTest(flag=flag), patch("sys.argv", ["launcher", flag]), \
                    patch.object(LAUNCHER, "read_config", return_value=self.cfg), \
                    patch.object(LAUNCHER, "convert_datasets") as conversion:
                LAUNCHER.main(method="lightning_opd")
                conversion.assert_called_once_with(self.cfg)


if __name__ == "__main__":
    unittest.main()
