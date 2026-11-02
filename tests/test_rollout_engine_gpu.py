"""Each SGLang rollout engine runs on the GPU that Ray assigned to its placement-group bundle.

slime used to derive the engine's GPU as (actor GPUs + rank) % GPUs per node, which assumes the
actor holds the first GPUs of every node and the engines follow contiguously. On the 1102 cluster the
K=120 teacher holds head GPUs 0-1 through Ray, the actor bundles get GPUs 2-5 and the first two engines
get GPUs 6-7, so that formula put engines 0 and 1 on the actor's GPUs 4-5 (CUDA out of memory).
"""

import os
from types import SimpleNamespace
import unittest
from unittest import mock

from slime.backends.sglang_utils.sglang_engine import _compute_server_args


def engine_args():
    return SimpleNamespace(
        hf_checkpoint="/models/student", seed=42, offload_rollout=False, rollout_num_gpus_per_engine=1,
        num_gpus_per_node=8, colocate=False, debug_rollout_only=False, actor_num_nodes=1,
        actor_num_gpus_per_node=4, use_critic=False, sglang_dp_size=1, sglang_pp_size=1, sglang_ep_size=1,
        use_rollout_routing_replay=False, fp16=False)


def base_gpu_id(ray_gpu_ids, visible_devices=None, rank=0):
    environment = {key: value for key, value in os.environ.items() if key != "CUDA_VISIBLE_DEVICES"}
    if visible_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = visible_devices
    with mock.patch("ray.get_gpu_ids", return_value=ray_gpu_ids), mock.patch.dict(os.environ, environment, clear=True):
        kwargs, _ = _compute_server_args(engine_args(), rank, "127.0.0.1:1", 2, "127.0.0.1", 3)
    return kwargs["base_gpu_id"]


class RolloutEngineGpuTests(unittest.TestCase):
    def test_engine_runs_on_its_bundle_gpu_when_the_teacher_holds_the_first_gpus(self):
        self.assertEqual(base_gpu_id([6], rank=0), 6)
        self.assertEqual(base_gpu_id([7], rank=1), 7)

    def test_bundle_gpu_is_indexed_among_the_devices_the_engine_sees(self):
        self.assertEqual(base_gpu_id([7], visible_devices="6,7"), 1)


if __name__ == "__main__":
    unittest.main()
