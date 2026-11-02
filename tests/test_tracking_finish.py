"""Ray actors upload their last W&B metrics before the job exits.

The rollout manager and the main training rank log into the driver's W&B run as secondary processes.
Ray kills them when the driver exits, and every 11/02 run on 1102 (A k1, B k1, A k120) lost its
final evaluation that way: the rollout manager logged it about 20 seconds before the exit, and it never
reached the local run file or the server. dispose() is the last call each actor gets.
"""

import tempfile
import unittest
from pathlib import Path

import wandb
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal import datastore

from slime.backends.megatron_utils.actor import MegatronTrainRayActor
from slime.ray.rollout import RolloutManager

FINAL_EVAL = {"eval_metric/aime24-avg@8": 0.35, "eval_step": 120}


def logged_keys(run_file):
    store, keys = datastore.DataStore(), set()
    store.open_for_scan(str(run_file))
    while (data := store.scan_data()) is not None:
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if record.WhichOneof("record_type") == "history":
            keys |= {item.key or "/".join(item.nested_key) for item in record.history.item}
    return keys


def rollout_manager():
    manager = object.__new__(RolloutManager.__ray_metadata__.modified_class)
    manager._metric_checker = None
    return manager


def training_rank():
    return object.__new__(MegatronTrainRayActor)


class TrackingFinishTests(unittest.TestCase):
    def tearDown(self):
        if wandb.run is not None:
            wandb.finish()

    def test_dispose_writes_the_last_logged_metrics(self):
        for name, make in (("rollout manager", rollout_manager), ("training rank", training_rank)):
            with self.subTest(actor=name), tempfile.TemporaryDirectory() as directory:
                wandb.init(mode="offline", dir=directory, settings=wandb.Settings(x_disable_stats=True, console="off"))
                wandb.log(FINAL_EVAL)
                make().dispose()
                self.assertIsNone(wandb.run)
                (run_file,) = Path(directory).glob("wandb/offline-run-*/run-*.wandb")
                self.assertLessEqual(set(FINAL_EVAL), logged_keys(run_file))


if __name__ == "__main__":
    unittest.main()
