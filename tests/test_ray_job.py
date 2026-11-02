"""A training job on a pre-started Ray cluster runs in the launcher's venv.

The cluster's nodes share no filesystem. The rollout manager reads and writes run files, so it
always runs on the launcher's node; GPU bundles are pinned there too when the job uses one node.
"""

import json
import os
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

from slime.utils import ray_job

PINNED = {ray_job.NODE_ENV: "[fd00::1]"}
SINGLE_NODE = dict(PINNED, **{ray_job.SINGLE_NODE_ENV: "1"})


def unset(*names):
    return {name: os.environ.pop(name) for name in names if name in os.environ}


class RayJobTests(unittest.TestCase):
    def setUp(self):
        saved = unset(ray_job.NODE_ENV, ray_job.SINGLE_NODE_ENV, ray_job.RUNTIME_ENV)
        self.addCleanup(os.environ.update, saved)

    def test_unpinned_job_requests_plain_gpu_bundles(self):
        self.assertEqual(ray_job.gpu_bundles(2), [{"GPU": 1, "CPU": 1}] * 2)
        self.assertEqual(ray_job.node_resources(), {})

    def test_single_node_job_requests_its_node_in_every_gpu_bundle(self):
        with patch.dict(os.environ, SINGLE_NODE):
            self.assertEqual(ray_job.gpu_bundles(2), [{"GPU": 1, "CPU": 1, "node:[fd00::1]": 0.001}] * 2)
            self.assertEqual(ray_job.node_resources(), {"node:[fd00::1]": 0.001})

    def test_multi_node_job_lets_gpu_bundles_span_nodes(self):
        with patch.dict(os.environ, PINNED):
            self.assertEqual(ray_job.gpu_bundles(2), [{"GPU": 1, "CPU": 1}] * 2)
            self.assertEqual(ray_job.node_resources(), {"node:[fd00::1]": 0.001})

    def test_connect_starts_the_job_with_its_runtime_env(self):
        runtime = {"py_executable": "/venv/bin/python", "env_vars": {"NO_PROXY": "127.0.0.1"}}
        with patch.dict(os.environ, {ray_job.RUNTIME_ENV: json.dumps(runtime), "RAY_ADDRESS": "auto"}), \
                patch("ray.is_initialized", return_value=False), patch("ray.init") as init:
            ray_job.connect()
        init.assert_called_once_with(address="auto", runtime_env=runtime)

    def test_connect_without_a_job_runtime_env_leaves_ray_alone(self):
        with patch("ray.init") as init:
            ray_job.connect()
        init.assert_not_called()


class WorkerSecretTests(unittest.TestCase):
    def test_credentials_are_split_from_the_forwarded_environment(self):
        env = {"PATH": "/bin", "WANDB_API_KEY": "k", "WANDB_PROJECT": "p", "proxy_pwd": "x",
               "http_proxy": "http://u:pw@proxy:1", "HTTPS_PROXY": "http://u:pw@proxy:1", "no_proxy": "127.0.0.1"}
        public, secret = ray_job.split_secrets(env)
        self.assertEqual(public, {"PATH": "/bin", "WANDB_PROJECT": "p", "no_proxy": "127.0.0.1"})
        self.assertEqual(set(secret), {"WANDB_API_KEY", "proxy_pwd", "http_proxy", "HTTPS_PROXY"})

    def test_worker_hook_loads_secrets_without_overriding_its_environment(self):
        import tempfile

        saved = unset("WANDB_API_KEY", "http_proxy", ray_job.SECRETS_ENV)
        self.addCleanup(os.environ.update, saved)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "secrets.json")
            with open(path, "w") as stream:
                json.dump({"WANDB_API_KEY": "k", "http_proxy": "from-file"}, stream)
            with patch.dict(os.environ, {ray_job.SECRETS_ENV: path, "http_proxy": "already-set"}):
                ray_job.load_worker_secrets()
                self.assertEqual((os.environ["WANDB_API_KEY"], os.environ["http_proxy"]), ("k", "already-set"))


class PinnedSchedulingTests(unittest.TestCase):
    def test_rollout_manager_is_pinned_with_the_job(self):
        from slime.ray import placement_group

        captured = {}

        class Manager:
            @staticmethod
            def options(**options):
                captured.update(options)
                return types.SimpleNamespace(remote=lambda *a: types.SimpleNamespace())

        args = types.SimpleNamespace(num_rollout=1, check_weight_update_equal=False, offload_rollout=False)
        with patch.dict(os.environ, PINNED), patch.object(placement_group, "RolloutManager", Manager):
            placement_group.create_rollout_manager(args, pg=None)
        self.assertEqual(captured.get("resources"), {"node:[fd00::1]": 0.001})


def local_ray(nodes=(), here=None, gpu_ids=()):
    """A stand-in for ray that runs remote actors in this process and records how they were requested."""
    requested = {}

    class Handle:
        def __init__(self, instance):
            self.instance = instance

        def __getattr__(self, name):
            return types.SimpleNamespace(remote=getattr(self.instance, name))

    def remote(**options):
        requested.update(options)
        return lambda cls: types.SimpleNamespace(
            options=lambda **more: requested.update(more) or types.SimpleNamespace(
                remote=lambda *args: Handle(cls(*args))))

    ray = types.SimpleNamespace(remote=remote, get=lambda value: value, kill=lambda handle: None,
                                get_gpu_ids=lambda: list(gpu_ids), nodes=lambda: list(nodes),
                                get_runtime_context=lambda: types.SimpleNamespace(get_node_id=lambda: here))
    return ray, requested


def node(node_id, address, gpus=8.0, alive=True):
    return dict(NodeID=node_id, NodeManagerAddress=address, Alive=alive, Resources={"GPU": gpus} if gpus else {})


class GpuNodeTests(unittest.TestCase):
    def test_alive_gpu_nodes_with_this_node_first(self):
        nodes = [node("cpu", "[fd00::9]", gpus=0), node("other", "[fd00::2]"), node("dead", "[fd00::3]", alive=False),
                 node("here", "[fd00::1]")]
        ray, _ = local_ray(nodes, here="here")
        with patch.dict(sys.modules, {"ray": ray}):
            self.assertEqual(ray_job.gpu_nodes(2), ["[fd00::1]", "[fd00::2]"])
            self.assertEqual(ray_job.gpu_nodes(1), ["[fd00::1]"])


class GpuProcessTests(unittest.TestCase):
    def run_process(self, devices=None, gpu_ids=(), num_gpus=0):
        ray, requested = local_ray(gpu_ids=gpu_ids)
        command = [sys.executable, "-c", "import os, sys, time; print(os.environ['CUDA_VISIBLE_DEVICES']); time.sleep(.5); sys.exit(3)"]
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"ray": ray}):
            log = Path(directory) / "not-yet-made/teacher.log"
            with ray_job.gpu_process(command, None, num_gpus, {"node:[fd00::2]": 0.001}, log, devices) as process:
                deadline = time.monotonic() + 30
                while process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.05)
            return process.returncode, log.read_text().strip(), requested

    def test_a_process_on_gpus_its_caller_owns_reserves_none_through_ray(self):
        # Teacher scoring runs while every Ray model is offloaded but their placement group still holds all GPUs.
        returncode, visible, requested = self.run_process(devices="2,3")
        self.assertEqual((returncode, visible), (3, "2,3"))
        self.assertEqual((requested["num_gpus"], requested["resources"]), (0, {"node:[fd00::2]": 0.001}))

    def test_a_process_without_devices_gets_the_gpus_ray_assigned(self):
        returncode, visible, requested = self.run_process(gpu_ids=(4, 5), num_gpus=2)
        self.assertEqual((returncode, visible, requested["num_gpus"]), (3, "4,5", 2))


if __name__ == "__main__":
    unittest.main()
