"""Launching on a pre-started Ray cluster (`ray_cluster: existing`).

No Ray/Slurm management; workers use the launcher's venv. The nodes share no filesystem, so the
rollout manager stays on the launcher's node, the converted initial checkpoint is mirrored to the
other nodes, and the student model must already exist on every node.
"""

import contextlib
import importlib.util
import io
import json
import os as module_os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import yaml

from slime.utils import ray_job

ROOT = Path(__file__).resolve().parents[1]
PRESET = ROOT / "configs/config/qwen3_1_7b_nt_teacher_4b_instruct_2507_k1.yaml"
NODE = "[fd00::1]"
SECRETS = {"WANDB_API_KEY": "secret-key", "proxy_pwd": "secret-pwd", "http_proxy": "http://u:secret-pw@proxy:1"}
OTHER = "[fd00::2]"


def launcher():
    spec = importlib.util.spec_from_file_location("existing_cluster_launcher", ROOT / "configs/opd/qwen3-4b-opd.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def single_node(**changes):
    cfg = yaml.safe_load(PRESET.read_text())
    cfg.update(ray_cluster="existing", nnodes=1, n_gpus_per_node=8, actor_num_nodes=1,
               actor_num_gpus_per_node=8, rollout_num_gpus=8, conversion_num_gpus=8)
    cfg.update(changes)
    return cfg


def fake_ray(free_gpus, other_free=8.0):
    free = {"me": {"GPU": free_gpus}, "other": {"GPU": other_free}}
    addresses = {"me": NODE, "other": OTHER}
    nodes = [dict(NodeID=node, Alive=True, Resources={"GPU": 8.0}, NodeManagerAddress=addresses[node]) for node in free]
    return types.SimpleNamespace(
        init=lambda **kwargs: None, shutdown=lambda: None, nodes=lambda: nodes,
        util=types.SimpleNamespace(get_node_ip_address=lambda: NODE),
        get_runtime_context=lambda: types.SimpleNamespace(get_node_id=lambda: "me"),
        _private=types.SimpleNamespace(state=types.SimpleNamespace(available_resources_per_node=lambda: free)),
    )


class ExistingClusterConfigTests(unittest.TestCase):
    def read(self, cfg, method="lightning_opd"):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.yaml"
            path.write_text(yaml.safe_dump(dict(cfg, output_dir=directory)))
            return launcher().read_config(path, method)

    def test_single_node_job_is_accepted(self):
        self.assertEqual(self.read(single_node())["ray_cluster"], "existing")

    def test_job_may_span_every_node_of_the_cluster(self):
        cfg = self.read(single_node(nnodes=2, actor_num_nodes=2, rollout_num_gpus=16))
        self.assertEqual((cfg["nnodes"], cfg["ray_cluster"]), (2, "existing"))

    def test_unknown_cluster_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "ray_cluster"):
            self.read(single_node(ray_cluster="shared"))


class ConvertedCheckpointTests(unittest.TestCase):
    def test_experiments_with_the_same_student_share_one_conversion(self):
        module = launcher()
        first = dict(output_dir="/runs", experiment_name="pair-a-k1")
        second = dict(output_dir="/runs", experiment_name="pair-b-k120")
        self.assertEqual(module.checkpoint_path(first, "/models/student"), module.checkpoint_path(second, "/models/student"))
        self.assertNotEqual(module.checkpoint_path(first, "/models/student"), module.checkpoint_path(first, "/models/other"))
        self.assertFalse(str(module.checkpoint_path(first, "/models/student")).startswith("/runs/pair-a-k1/"))


class ExistingClusterLaunchTests(unittest.TestCase):
    def launch(self, free_gpus, other_free=8.0, missing=(), **changes):
        module = launcher()
        calls, mirrored, self.secrets = [], [], {}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "torch_dist"
            checkpoint.mkdir()
            (checkpoint / "latest_checkpointed_iteration.txt").write_text("release")
            cfg = single_node(output_dir=directory, experiment_name="test", save_interval=None, eval_steps=None)
            cfg.update(changes)
            with patch.dict(sys.modules, {"ray": fake_ray(free_gpus, other_free)}), \
                    patch.object(ray_job, "mirror", side_effect=lambda path, **kw: mirrored.append(Path(path))), \
                    patch.object(ray_job, "missing_on_nodes", return_value=list(missing)), \
                    patch.object(ray_job, "write_worker_secrets",
                                 side_effect=lambda secret, path: self.secrets.update(secret)), \
                    patch.object(module, "check_training_storage"), patch.object(module, "check_inputs"), \
                    patch.object(module, "check_run_state"), \
                    patch.object(module, "checkpoint_path", return_value=checkpoint), \
                    patch.object(module.lightning_blocks, "prepare_fixed_prompts"), \
                    patch.object(module, "model_args", return_value=[]), \
                    patch.object(module, "training_args", return_value=[]), \
                    patch.object(module, "background") as background, \
                    patch.object(module.subprocess, "run", side_effect=lambda command, **kw: calls.append((command, kw))), \
                    contextlib.redirect_stdout(io.StringIO()):
                module.train_cluster(cfg, "student", "teacher", dict(SECRETS, PATH="/usr/bin", CUDA_VISIBLE_DEVICES="0"))
        return calls, background, mirrored, checkpoint

    def test_single_node_job_trains_here_without_starting_ray_or_slurm_services(self):
        calls, background, mirrored, _ = self.launch(free_gpus=8.0)
        self.assertEqual(mirrored, [])
        background.assert_not_called()
        [(command, options)] = calls
        self.assertIn("train.py", command)
        env = options["env"]
        self.assertEqual(env[ray_job.NODE_ENV], NODE)
        self.assertEqual(env[ray_job.SINGLE_NODE_ENV], "1")
        runtime = json.loads(env[ray_job.RUNTIME_ENV])
        self.assertEqual(runtime["py_executable"], sys.executable)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", runtime["env_vars"])
        for no_proxy in (env["no_proxy"], env["NO_PROXY"], runtime["env_vars"]["no_proxy"]):
            # SGLang servers bind 0.0.0.0 and send their startup warmup request to that address.
            self.assertTrue({"127.0.0.1", "localhost", "0.0.0.0", "fd00::1"}.issubset(no_proxy.split(",")))
        self.assert_httpx_bypasses_this_node(env["no_proxy"])

    def test_credentials_reach_workers_through_a_file_not_the_logged_runtime_env(self):
        calls, *_ = self.launch(free_gpus=8.0)
        [(command, options)] = calls
        runtime_text = options["env"][ray_job.RUNTIME_ENV]
        self.assertFalse(any(value in runtime_text for value in SECRETS.values()))
        runtime = json.loads(runtime_text)
        self.assertEqual(runtime["worker_process_setup_hook"], "slime.utils.ray_job.load_worker_secrets")
        self.assertIn(ray_job.SECRETS_ENV, runtime["env_vars"])
        self.assertEqual(self.secrets, SECRETS)
        self.assertTrue(all(options["env"][key] == value for key, value in SECRETS.items()))  # the driver keeps them

    def assert_httpx_bypasses_this_node(self, no_proxy):
        """slime's httpx client is built from these variables; a bracketed IPv6 entry breaks it."""
        import httpx

        with patch.dict(module_os.environ, {"http_proxy": "http://proxy.invalid:3128", "no_proxy": no_proxy}):
            client = httpx.Client(trust_env=True)
        bypassed = {pattern.pattern for pattern, transport in client._mounts.items() if transport is None}
        self.assertIn("all://[fd00::1]", bypassed)

    def test_refuses_to_start_when_this_node_lacks_free_gpus(self):
        with self.assertRaisesRegex(RuntimeError, "free GPUs"):
            self.launch(free_gpus=4.0)

    def test_two_node_job_mirrors_code_initial_checkpoint_and_teacher_and_spans_nodes(self):
        calls, _, mirrored, checkpoint = self.launch(free_gpus=8.0, nnodes=2, actor_num_nodes=2, rollout_num_gpus=16)
        # Lightning OPD scores its blocks with the teacher on every node.
        self.assertEqual(mirrored, [ROOT, checkpoint, Path("teacher")])
        [(command, options)] = calls
        self.assertEqual(options["env"][ray_job.NODE_ENV], NODE)
        self.assertNotIn(ray_job.SINGLE_NODE_ENV, options["env"])
        # The router on this node health-checks engines on the other node directly, not through a proxy.
        self.assertIn("fd00::2", options["env"]["no_proxy"].split(","))

    def test_opd_reserves_teacher_gpus_through_ray_on_this_node_while_training(self):
        module = launcher()
        events = []

        @contextlib.contextmanager
        def gpu_process(command, env, num_gpus, resources, log_path):
            events.append(("teacher up", command, num_gpus, resources))
            yield
            events.append(("teacher down",))

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "torch_dist"
            checkpoint.mkdir()
            (checkpoint / "latest_checkpointed_iteration.txt").write_text("release")
            cfg = single_node(method="opd", output_dir=directory, experiment_name="test", nnodes=2, actor_num_nodes=1,
                              actor_num_gpus_per_node=4, colocate=False, rollout_num_gpus=10, teacher_url=None)
            with patch.dict(sys.modules, {"ray": fake_ray(8.0)}), patch.object(ray_job, "mirror"), \
                    patch.object(ray_job, "missing_on_nodes", return_value=[]), \
                    patch.object(ray_job, "gpu_process", gpu_process), patch.object(ray_job, "write_worker_secrets"), \
                    patch.object(module, "check_training_storage"), patch.object(module, "check_inputs"), \
                    patch.object(module, "check_run_state"), \
                    patch.object(module, "checkpoint_path", return_value=checkpoint), \
                    patch.object(module, "run_training",
                                 side_effect=lambda cfg, s, t, env, url: events.append(("train", url))):
                module.train_cluster(cfg, "student", "teacher-model", {"PATH": "/usr/bin"})
        (up, command, gpus, resources), train, down = events
        self.assertEqual((up, down), ("teacher up", ("teacher down",)))
        self.assertIn("teacher-model", command)
        self.assertEqual((gpus, resources), (cfg["teacher_num_gpus"], {f"node:{NODE}": 0.001}))
        self.assertEqual(train, ("train", f"http://127.0.0.1:{cfg['teacher_port']}/generate"))

    def test_two_node_job_needs_free_gpus_on_every_node(self):
        with self.assertRaisesRegex(RuntimeError, "free GPUs"):
            self.launch(free_gpus=8.0, other_free=2.0, nnodes=2, actor_num_nodes=2, rollout_num_gpus=16)

    def test_two_node_job_needs_the_student_on_every_node(self):
        with self.assertRaisesRegex(FileNotFoundError, "student"):
            self.launch(free_gpus=8.0, missing=["[fd00::2]"], nnodes=2, actor_num_nodes=2, rollout_num_gpus=16)


if __name__ == "__main__":
    unittest.main()
