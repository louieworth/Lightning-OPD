"""Exercise the actual batch scripts with Slurm commands replaced by CPU stubs."""

import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "configs/sbatch/qwen3_4b_base_nt_teacher_8b.sbatch"


class BatchPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="opd-batch-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.log = self.root / "stages.txt"
        self.write_command("python", """#!/bin/bash
if [[ "$1" == -B ]]; then printf '%s\\n' "$0"; exit 0; fi
if [[ "$1" != -c ]]; then exit 2; fi
echo "$TEST_OPD_METHOD"
""")
        self.write_command("srun", """#!/bin/bash
printf '%s\\t' "$@" >> "$TEST_STAGE_LOG"
printf '\\n' >> "$TEST_STAGE_LOG"
for arg in "$@"; do
    if [[ -n "${TEST_FAIL_STAGE:-}" && "$arg" == "$TEST_FAIL_STAGE" ]]; then exit 7; fi
done
""")

    def write_command(self, name, contents):
        path = self.root / name
        path.write_text(contents)
        path.chmod(0o755)

    def run_batch(self, *, method="lightning_opd", args=(), script=BATCH, fail_stage="", use_override=True):
        self.log.unlink(missing_ok=True)
        env = dict(os.environ, PATH=f"{self.root}:{os.environ.get('PATH', '')}",
                   LIGHTNING_OPD_PYTHON=str(self.root / "python"), SLURM_JOB_ID="123",
                   SLURM_JOB_NUM_NODES="2", TEST_OPD_METHOD=method, TEST_STAGE_LOG=str(self.log),
                    TEST_FAIL_STAGE=fail_stage)
        if not use_override:
            env.pop("LIGHTNING_OPD_PYTHON")
        result = subprocess.run(["bash", str(script), *args], env=env, cwd=ROOT,
                                capture_output=True, text=True, timeout=15)
        stages = [line.rstrip("\t").split("\t") for line in self.log.read_text().splitlines()]
        return result, stages

    def test_default_uses_active_python_without_a_named_conda_environment(self):
        result, stages = self.run_batch(use_override=False)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(stages), 3)
        for stage in stages:
            self.assertIn(str(self.root / "python"), stage)

    def test_lightning_default_validates_then_builds_fixed_data_then_trains(self):
        result, stages = self.run_batch()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(stages), 3)
        self.assertIn("--preflight", stages[0])
        self.assertIn("--nodes=2", stages[0])
        self.assertIn("--prepare-data", stages[1])
        self.assertIn("--nodes=1", stages[1])
        self.assertIn("--nodes=2", stages[2])
        for stage in stages:
            self.assertNotIn("--prepare-prompts", stage)
            self.assertNotIn("--convert-datasets", stage)

    def test_standard_opd_validates_then_trains_without_data_preparation(self):
        result, stages = self.run_batch(method="opd")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(stages), 2)
        self.assertIn("--preflight", stages[0])
        self.assertNotIn("--preflight", stages[1])
        for stage in stages:
            self.assertIn("--nodes=2", stage)
            self.assertNotIn("--prepare-data", stage)
            self.assertNotIn("--prepare-prompts", stage)
            self.assertNotIn("--convert-datasets", stage)

    def test_conversion_is_a_single_explicit_stage_for_both_names(self):
        for flag in ("--convert-datasets", "--prepare-prompts"):
            with self.subTest(flag=flag):
                result, stages = self.run_batch(args=(flag,))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(len(stages), 1)
                self.assertIn(flag, stages[0])
                self.assertIn("--nodes=1", stages[0])

    def test_wrapper_forwards_alternate_yaml_and_resume_to_every_stage(self):
        alternate = str(self.root / "alternate config.yaml")
        wrapper = ROOT / "configs/sbatch/qwen3_1_7b_nt_teacher_4b_instruct_2507_k1.sbatch"

        result, stages = self.run_batch(script=wrapper, args=(alternate, "--resume"))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(stages), 3)
        for stage in stages:
            self.assertEqual(stage[stage.index("--config") + 1], alternate)
            self.assertIn("--resume", stage)

    def test_failed_preflight_stops_before_preparation_and_training(self):
        result, stages = self.run_batch(fail_stage="--preflight")

        self.assertEqual(result.returncode, 7)
        self.assertEqual(len(stages), 1)
        self.assertIn("--preflight", stages[0])

    def test_shared_pipeline_wrappers_default_to_existing_yaml_files(self):
        wrappers = [path for path in (ROOT / "configs/sbatch").glob("*.sbatch")
                    if path == BATCH or "exec bash " + str(BATCH) in path.read_text()]
        self.assertTrue(wrappers)
        for wrapper in wrappers:
            with self.subTest(wrapper=wrapper.name):
                match = re.search(r"^TASK_CONFIG=(\S+)", wrapper.read_text(), re.MULTILINE)
                self.assertIsNotNone(match)
                self.assertTrue((ROOT / match.group(1)).is_file(), match.group(1))


if __name__ == "__main__":
    unittest.main()
