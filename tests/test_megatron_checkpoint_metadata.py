"""CPU regressions for PyTorch dropping MCore's dynamic checkpoint metadata."""

import pickle
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter, save
from torch.distributed.checkpoint.metadata import Metadata, TensorProperties, TensorStorageMetadata

from slime.backends.megatron_utils.checkpoint_metadata import install_metadata_compatibility, recover_mcore_metadata


def tensor_metadata(shape):
    return TensorStorageMetadata(TensorProperties(), torch.Size(shape), [])


class CheckpointMetadataTests(unittest.TestCase):
    def setUp(self):
        self.reader = FileSystemReader.read_metadata
        self.writer = FileSystemWriter.finish
        self.addCleanup(setattr, FileSystemReader, "read_metadata", self.reader)
        self.addCleanup(setattr, FileSystemWriter, "finish", self.writer)

    def test_writer_preserves_mcore_fields_on_disk(self):
        install_metadata_compatibility()
        metadata = Metadata({})
        metadata.mcore_data = {"moment": {"nd_reformulated_orig_global_shape": (4, 6)}}
        metadata.all_local_plans = ["cached plan"]
        with tempfile.TemporaryDirectory(prefix="mcore-metadata-test-") as directory:
            writer = FileSystemWriter(directory)
            writer.set_up_storage_writer(True)
            writer.finish(metadata, [])
            with (Path(directory) / ".metadata").open("rb") as stream:
                restored = pickle.load(stream)
        self.assertEqual(restored.mcore_data, metadata.mcore_data)
        self.assertEqual(restored.all_local_plans, metadata.all_local_plans)

    def test_recovers_shapes_from_saved_model_without_touching_disk(self):
        metadata = Metadata({"weight": tensor_metadata((4, 6)), "norm": tensor_metadata((6,))})
        for state in ("exp_avg", "exp_avg_sq", "fp32_param"):
            metadata.state_dict_metadata[f"optimizer.state.{state}.weight"] = tensor_metadata((2, 1, 12))
            metadata.state_dict_metadata[f"optimizer.state.{state}.norm"] = tensor_metadata((1, 6))
        with tempfile.TemporaryDirectory(prefix="mcore-metadata-test-") as directory:
            path = Path(directory) / ".metadata"
            path.write_bytes(pickle.dumps(metadata))
            before = path.read_bytes()
            install_metadata_compatibility()
            restored = FileSystemReader(directory).read_metadata()
            self.assertEqual(path.read_bytes(), before)
        self.assertEqual(len(restored.mcore_data), 6)
        self.assertEqual(restored.mcore_data["optimizer.state.exp_avg.weight"],
                         {"nd_reformulated_orig_global_shape": (4, 6)})
        self.assertEqual(restored.mcore_data["optimizer.state.fp32_param.norm"],
                         {"nd_reformulated_orig_global_shape": (6,)})

    def test_preserves_existing_metadata_and_legacy_unflattened_tensors(self):
        metadata = Metadata({})
        metadata.mcore_data = {"existing": {"nd_reformulated_orig_global_shape": (2, 3)}}
        self.assertIs(recover_mcore_metadata(metadata), metadata)
        self.assertIn("existing", metadata.mcore_data)
        legacy = Metadata({"norm": tensor_metadata((6,)),
                           "optimizer.state.exp_avg.norm": tensor_metadata((6,))})
        self.assertEqual(recover_mcore_metadata(legacy).mcore_data, {})

    def test_rejects_inconsistent_shape_recovery(self):
        for stored in ((2, 1, 11), (3, 1, 8), (24,)):
            with self.subTest(stored=stored), self.assertRaisesRegex(ValueError, "MCore shape metadata"):
                recover_mcore_metadata(Metadata({
                    "weight": tensor_metadata((4, 6)),
                    "optimizer.state.exp_avg.weight": tensor_metadata(stored),
                }))

    def test_megatron_loads_optimizer_values_after_metadata_loss(self):
        from megatron.core.dist_checkpointing import ShardedTensor
        from megatron.core.dist_checkpointing.strategies import torch as strategy
        from torch.distributed._shard.metadata import ShardMetadata

        install_metadata_compatibility()
        with tempfile.TemporaryDirectory(prefix="mcore-load-test-") as directory:
            root = Path(directory)
            torch.distributed.init_process_group("gloo", init_method=f"file://{root / 'rendezvous'}",
                                                 rank=0, world_size=1, timeout=timedelta(seconds=30))
            try:
                key = "optimizer.state.exp_avg.weight"
                expected = torch.arange(24, dtype=torch.float32).reshape(2, 1, 12)
                save({"weight": torch.zeros(4, 6), key: expected}, checkpoint_id=root / "checkpoint")
                path = root / "checkpoint/.metadata"
                with path.open("rb") as stream:
                    raw = pickle.load(stream)
                self.assertFalse(hasattr(raw, "mcore_data"))
                # Emulate both flattened TP shards on one CPU process.
                shards = [ShardedTensor.from_rank_offsets_flat(
                    key, torch.empty(12), (2, 6), (0, rank, 2), flattened_range=slice(0, 12)
                ) for rank in range(2)]
                # MCore hardcodes CUDA placements; only adapt that device label
                # so its actual checkpoint conversion and loading can run on CPU.
                def cpu_placement(offsets, sizes, placement):
                    return ShardMetadata(offsets, sizes, placement.replace("/cuda", "/cpu"))

                with patch.object(strategy, "ShardMetadata", side_effect=cpu_placement):
                    result = strategy.TorchDistLoadShardedStrategy().load({"moments": shards}, root / "checkpoint")
                for rank, moment in enumerate(result["moments"]):
                    torch.testing.assert_close(moment, expected[rank].flatten())
            finally:
                torch.distributed.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
