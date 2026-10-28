"""Exercise local-code and dependency selection without GPUs or old runtimes."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LAUNCHER = load_module("environment_launcher", "configs/opd/qwen3-4b-opd.py")
SFT_SETTINGS = load_module("environment_sft_settings", "configs/environment/sft_settings.py")


class EnvironmentSelectionTests(unittest.TestCase):
    def test_configs_default_to_installed_dependencies(self):
        paths = list((ROOT / "configs/config").glob("*.yaml"))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(config=path.name):
                cfg = yaml.safe_load(path.read_text())
                for key in ("python_executable", "megatron_path", "sft_dependency_archive", "generation_python"):
                    self.assertIsNone(cfg.get(key))

    def test_local_slime_is_first_with_optional_external_megatron(self):
        config = ROOT / "configs/config/qwen3_4b_base_nt_teacher_8b.yaml"
        with tempfile.TemporaryDirectory(prefix="external-megatron-") as directory:
            for override in ("", directory):
                with self.subTest(override=override), patch.dict(os.environ, {"MEGATRON_PATH": override}):
                    cfg = LAUNCHER.read_config(config, "lightning_opd")
                    paths = LAUNCHER.runtime_env(cfg)["PYTHONPATH"].split(os.pathsep)
                    self.assertEqual(paths[0], str(ROOT))
                    self.assertEqual(cfg["megatron_path"], str(Path(directory).resolve()) if override else None)
                    if override:
                        self.assertEqual(paths[1], str(Path(directory).resolve()))
            with patch.dict(os.environ, {"MEGATRON_PATH": directory}):
                cfg["megatron_path"] = str(Path(directory) / "explicit")
                # An explicit YAML checkout overrides the environment variable.
                custom = Path(directory) / "config.yaml"
                custom.write_text(yaml.safe_dump(cfg))
                self.assertEqual(LAUNCHER.read_config(custom, "lightning_opd")["megatron_path"], cfg["megatron_path"])

    def test_sft_defaults_to_current_interpreter_without_archive(self):
        cfg = yaml.safe_load((ROOT / "configs/config/qwen3_4b_sft.yaml").read_text())
        with patch.dict(os.environ, {"LIGHTNING_OPD_PYTHON": "", "SFT_DEPENDENCY_ARCHIVE": ""}):
            settings = SFT_SETTINGS.settings(cfg)
        self.assertEqual(settings[:2], [sys.executable, sys.executable])
        self.assertEqual(settings[7], "")


class SFTBatchEnvironmentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="environment-batch-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.log = self.root / "calls.jsonl"
        self.write_command("python", f"""#!{sys.executable}
import json, os, subprocess, sys
args = sys.argv[1:]
if args[:2] == ['-B', 'configs/environment/sft_settings.py']:
    raise SystemExit(subprocess.call([sys.executable, *args]))
if args[:2] == ['-B', '-c']:
    exec(args[2])
else:
    with open(os.environ['TEST_CALL_LOG'], 'a') as stream:
        stream.write(json.dumps(dict(command='python', args=args, pythonpath=os.environ.get('PYTHONPATH'))) + '\\n')
""")
        self.write_command("srun", """#!/bin/bash
while [[ "$1" == --* ]]; do shift; done
exec "$@"
""")
        self.write_command("scontrol", """#!/bin/bash
case "$2" in
    job) echo 'EndTime=2099-01-01T00:00:00' ;;
    hostnames) echo test-node ;;
esac
""")
        self.write_command("tar", f"""#!{sys.executable}
import json, os, subprocess, sys
with open(os.environ['TEST_CALL_LOG'], 'a') as stream:
    stream.write(json.dumps(dict(command='tar', args=sys.argv[1:])) + '\\n')
raise SystemExit(subprocess.call([{shutil.which('tar')!r}, *sys.argv[1:]]))
""")
        self.job = self.root.name
        dependencies = Path('/tmp') / f'lightning-sft-deps-{self.job}-environment-test'
        self.addCleanup(lambda: shutil.rmtree(dependencies, ignore_errors=True))

    def write_command(self, name, contents):
        path = self.bin / name
        path.write_text(contents)
        path.chmod(0o755)

    def run_batch(self, script, *, archive=False, corrupt=False):
        self.log.unlink(missing_ok=True)
        cfg_name = 'qwen3_4b_sft_downloaded.yaml' if 'downloaded' in script else 'qwen3_4b_sft.yaml'
        cfg = yaml.safe_load((ROOT / 'configs/config' / cfg_name).read_text())
        if archive:
            source = self.root / 'overlay_marker.py'
            source.write_text('# optional SFT package overlay\n')
            package = self.root / 'packages.tar.gz'
            with tarfile.open(package, 'w:gz') as stream:
                stream.add(source, arcname=source.name)
            if corrupt:
                package.write_text('invalid archive')
            cfg['sft_dependency_archive'] = str(package)
        config = self.root / 'config.yaml'
        config.write_text(yaml.safe_dump(cfg))
        env = dict(os.environ, PATH=f'{self.bin}:{os.environ["PATH"]}',
                   LIGHTNING_OPD_PYTHON=str(self.bin / 'python'), SFT_DEPENDENCY_ARCHIVE='',
                   SLURM_JOB_ID=self.job, SLURM_JOB_NUM_NODES='2', SLURM_JOB_NODELIST='test-node',
                   SLURM_PROCID='0', USER='environment-test', TEST_CALL_LOG=str(self.log))
        result = subprocess.run(['bash', str(ROOT / 'configs/sbatch' / script), str(config)],
                                cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
        calls = [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []
        return result, calls

    def test_both_sft_pipelines_use_installed_packages_or_optional_archive(self):
        for script in ('qwen3_4b_sft.sbatch', 'qwen3_4b_sft_downloaded.sbatch'):
            for archive in (False, True):
                with self.subTest(script=script, archive=archive):
                    result, calls = self.run_batch(script, archive=archive)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    workers = [call for call in calls if 'torch.distributed.run' in call['args']]
                    self.assertEqual(len(workers), 1)
                    self.assertEqual(workers[0]['pythonpath'].split(os.pathsep)[0], str(ROOT))
                    if not archive:
                        self.assertEqual(workers[0]['pythonpath'], str(ROOT))
                    extractions = [call for call in calls if call['command'] == 'tar']
                    self.assertEqual(len(extractions), int(archive))

    def test_invalid_optional_archive_stops_before_training(self):
        result, calls = self.run_batch('qwen3_4b_sft_downloaded.sbatch', archive=True, corrupt=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any('torch.distributed.run' in call['args'] for call in calls))


if __name__ == '__main__':
    unittest.main()
