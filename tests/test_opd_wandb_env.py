"""W&B configuration must come from .env throughout startup, training and sync."""

import contextlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/config/qwen3_1_7b_base_nt_teacher_4b_instruct_2507_k1.yaml"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LAUNCHER = load_module("wandb_env_launcher", "configs/opd/qwen3-4b-opd.py")
SFT = load_module("wandb_env_sft", "scripts/qwen3_sft.py")
SYNC = load_module("wandb_env_sync", "configs/environment/sync_wandb.py")


class WandbEnvironmentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="opd-wandb-env-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dotenv = self.root / ".env"
        self.settings = dict(WANDB_API_KEY="unit-test-key", WANDB_ENTITY="test-entity",
                             WANDB_PROJECT="test-project", WANDB_MODE="online")
        self.dotenv.write_text("".join(f"{key}={value}\n" for key, value in self.settings.items()))
        self.cfg = LAUNCHER.read_config(CONFIG, "lightning_opd")
        for key in ("wandb_entity", "project_name", "wandb_mode"):
            self.cfg.pop(key, None)
        self.cfg.update(env_file=str(self.dotenv), output_dir=str(self.root), experiment_name="test")
        environment = patch.dict(os.environ, {"PATH": "/usr/bin:/bin"}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def assert_training_scope(self, cfg):
        args = LAUNCHER.training_args(cfg, "student", None)
        for flag, key in (("--wandb-team", "WANDB_ENTITY"), ("--wandb-project", "WANDB_PROJECT"),
                          ("--wandb-mode", "WANDB_MODE")):
            self.assertEqual(args[args.index(flag) + 1], self.settings[key])
        self.assertNotIn(self.settings["WANDB_API_KEY"], " ".join(args))

    def test_runtime_and_training_read_all_dotenv_settings_without_yaml_fields(self):
        LAUNCHER.load_env(self.dotenv)

        env = LAUNCHER.runtime_env(self.cfg)

        for key, value in self.settings.items():
            self.assertEqual(env[key], value)
        self.assert_training_scope(self.cfg)

    def test_dotenv_overrides_inherited_values_and_legacy_yaml_settings(self):
        os.environ.update({key: "stale" for key in self.settings})
        os.environ["OTHER_SETTING"] = "inherited"
        self.dotenv.write_text(self.dotenv.read_text() + "OTHER_SETTING=from-file\n")
        self.cfg.update(wandb_entity="yaml-team", project_name="yaml-project", wandb_mode="invalid-yaml-mode")

        LAUNCHER.load_env(self.dotenv)
        env = LAUNCHER.runtime_env(self.cfg)

        for key, value in self.settings.items():
            self.assertEqual(env[key], value)
        self.assertEqual(os.environ["OTHER_SETTING"], "inherited")
        self.assert_training_scope(self.cfg)

    def test_missing_entity_uses_sdk_default_scope_without_key_error(self):
        env = LAUNCHER.runtime_env(self.cfg)
        args = LAUNCHER.training_args(self.cfg, "student", None)

        self.assertNotIn("WANDB_ENTITY", env)
        self.assertNotIn("--wandb-team", args)
        self.assertEqual(env["WANDB_MODE"], "offline")
        self.assertEqual(env["WANDB_PROJECT"], "onestep-opd")

    def test_invalid_dotenv_mode_fails_before_model_setup(self):
        self.dotenv.write_text("WANDB_MODE=invalid\n")
        argv = ["launcher", "--config", str(CONFIG), "--env-file", str(self.dotenv), "--dry-run"]
        with patch("sys.argv", argv), patch.object(LAUNCHER, "selected_models") as models:
            with self.assertRaisesRegex(ValueError, "WANDB_MODE"):
                LAUNCHER.main(method="lightning_opd")
            models.assert_not_called()

    def test_startup_loads_dotenv_before_runtime_and_training_argument_construction(self):
        output = io.StringIO()
        argv = ["launcher", "--config", str(CONFIG), "--env-file", str(self.dotenv), "--dry-run"]
        with patch("sys.argv", argv), patch.object(LAUNCHER, "selected_models", return_value=("student", "teacher")), \
                contextlib.redirect_stdout(output):
            LAUNCHER.main(method="lightning_opd")

        for key in ("WANDB_ENTITY", "WANDB_PROJECT", "WANDB_MODE"):
            self.assertIn(self.settings[key], output.getvalue())
        self.assertNotIn(self.settings["WANDB_API_KEY"], output.getvalue())

    def test_sft_uses_the_same_dotenv_settings(self):
        self.cfg.update(wandb_entity="yaml-team", project_name="yaml-project", wandb_mode="disabled")
        self.cfg["wandb_dir"] = str(self.root / "wandb/sft")

        SFT.setup_env(self.cfg)

        for key, value in self.settings.items():
            self.assertEqual(os.environ[key], value)
        self.assertEqual(os.environ["WANDB_DIR"], self.cfg["wandb_dir"])

    def test_wandb_paths_agree_between_environment_cli_and_sync(self):
        from slime.utils.wandb_paths import wandb_source_roots

        for override in (None, "wandb/custom-experiment", str(self.root / "absolute-wandb")):
            with self.subTest(override=override):
                cfg = dict(self.cfg)
                if override:
                    cfg["wandb_dir"] = override
                expected = (ROOT / (override or "wandb/test")).resolve()
                env = LAUNCHER.runtime_env(cfg)
                args = LAUNCHER.training_args(cfg, "student", None)
                self.assertEqual(env["WANDB_DIR"], str(expected))
                self.assertEqual(args[args.index("--wandb-dir") + 1], str(expected))
                self.assertEqual(wandb_source_roots(cfg, ROOT, LAUNCHER.run_dir(cfg)),
                                 [expected, LAUNCHER.run_dir(cfg) / "wandb"])

    def test_eval_sample_count_matches_launcher_and_dataset_defaults(self):
        paths = sorted((ROOT / "configs/config").glob("qwen3_0_6b_nt_teacher_4b_instruct_2507*.yaml"))
        paths += sorted((ROOT / "configs/config").glob("qwen3_1_7b_nt_teacher_4b_instruct_2507*.yaml"))
        paths += sorted((ROOT / "configs/config").glob("qwen3_1_7b_base_nt_teacher_4b_instruct_2507*.yaml"))
        for path in paths:
            name = path.stem
            with self.subTest(config=name):
                cfg = LAUNCHER.read_config(path, "lightning_opd")
                cfg["output_dir"] = str(self.root)
                with patch.object(LAUNCHER, "require_parquet", side_effect=Path):
                    eval_path = LAUNCHER.prepare_eval(cfg)
                args = LAUNCHER.training_args(cfg, "student", None, eval_path)
                self.assertEqual(cfg["val_n"], 4)
                self.assertEqual(args[args.index("--n-samples-per-eval-prompt") + 1], "4")
                defaults = yaml.safe_load(Path(eval_path).read_text())["eval"]["defaults"]
                self.assertEqual(defaults["n_samples_per_eval_prompt"], 4)

    def test_base_and_non_base_configs_share_training_and_eval_hyperparameters(self):
        paths = sorted((ROOT / "configs/config").glob("qwen3_0_6b_nt_teacher_4b_instruct_2507*.yaml"))
        paths += sorted((ROOT / "configs/config").glob("qwen3_1_7b_nt_teacher_4b_instruct_2507*.yaml"))
        paths += sorted((ROOT / "configs/config").glob("qwen3_1_7b_base_nt_teacher_4b_instruct_2507*.yaml"))
        identity = {"model_path", "model_type", "experiment_name", "policy_rollout_k", "epoch"}
        expected = yaml.safe_load(CONFIG.read_text())
        expected = {key: value for key, value in expected.items() if key not in identity}
        for path in paths:
            with self.subTest(config=path.name):
                cfg = yaml.safe_load(path.read_text())
                self.assertEqual({key: value for key, value in cfg.items() if key not in identity}, expected)

    def sync_config(self):
        path = self.root / "config.yaml"
        path.write_text(yaml.safe_dump(dict(env_file=str(self.dotenv), output_dir=str(self.root),
                                            experiment_name="test", python_executable=sys.executable)))
        return path

    def test_offline_sync_dry_run_uses_dotenv_scope_without_yaml_fields(self):
        log = self.root / "test/wandb/offline-run-test/run-test.wandb"
        log.parent.mkdir(parents=True)
        log.write_bytes(b"not-read-in-dry-run")
        output = io.StringIO()
        argv = ["sync", "--config", str(self.sync_config()), "--dry-run"]
        with patch("sys.argv", argv), patch.object(SYNC.subprocess, "run") as process, \
                contextlib.redirect_stdout(output):
            SYNC.main()
            process.assert_not_called()

        self.assertIn("--project test-project", output.getvalue())
        self.assertIn("--entity test-entity", output.getvalue())
        self.assertNotIn(self.settings["WANDB_API_KEY"], output.getvalue())

    def test_sync_repair_uses_dotenv_scope_without_yaml_fields(self):
        argv = ["sync", "--config", str(self.sync_config()), "--repair-eval-axis", "test", "--dry-run"]
        with patch("sys.argv", argv), patch.object(SYNC, "repair_eval_metric_axes") as repair:
            SYNC.main()
            repair.assert_called_once_with(["test"], "test-project", "test-entity", True)


if __name__ == "__main__":
    unittest.main()
