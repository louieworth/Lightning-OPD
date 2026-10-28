"""Preserve and recover MCore metadata with PyTorch 2.9 checkpoint writers."""

import logging
import math
from dataclasses import dataclass, field, fields
from functools import wraps

from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.metadata import Metadata

logger = logging.getLogger(__name__)


@dataclass
class MCoreMetadata(Metadata):
    # PyTorch 2.9 FileSystemWriter.finish uses dataclasses.replace, which drops
    # MCore's dynamically attached fields unless they are dataclass fields.
    mcore_data: dict = field(default_factory=dict)
    all_local_plans: object = None


def recover_mcore_metadata(metadata):
    """Infer lost optimizer shapes from the saved model, without changing disk files."""
    if getattr(metadata, "mcore_data", None) is not None:
        return metadata

    recovered = {}
    for key, tensor in metadata.state_dict_metadata.items():
        if not key.startswith("optimizer.state.") or not hasattr(tensor, "size"):
            continue
        parts = key.split(".", 3)
        if len(parts) != 4:
            continue
        model_tensor = metadata.state_dict_metadata.get(parts[3])
        if model_tensor is None or not hasattr(model_tensor, "size"):
            continue
        original_shape = tuple(model_tensor.size)
        stored_shape = tuple(tensor.size)
        if stored_shape == original_shape:
            continue  # Legacy un-reformulated tensors use MCore's existing loader.
        if len(stored_shape) != len(original_shape) + 1:
            raise ValueError(f"Cannot recover MCore shape metadata for {key}: {stored_shape} vs {original_shape}")
        fragments = stored_shape[:-1]
        if (
            any(fragment <= 0 or size % fragment for size, fragment in zip(original_shape, fragments))
            or math.prod(stored_shape) != math.prod(original_shape)
        ):
            raise ValueError(f"Inconsistent MCore shape metadata for {key}: {stored_shape} vs {original_shape}")
        recovered[key] = {"nd_reformulated_orig_global_shape": original_shape}

    metadata.mcore_data = recovered
    if recovered:
        logger.warning("Recovered missing MCore shape metadata for %d optimizer tensors", len(recovered))
    return metadata


def install_metadata_compatibility():
    """Install once per worker; also covers deferred async checkpoint finalization."""
    if getattr(FileSystemWriter.finish, "_slime_mcore_metadata", False):
        return

    read_metadata = FileSystemReader.read_metadata
    finish = FileSystemWriter.finish

    @wraps(read_metadata)
    def read_with_mcore_metadata(self):
        return recover_mcore_metadata(read_metadata(self))

    @wraps(finish)
    def finish_with_mcore_metadata(self, metadata, results):
        if hasattr(metadata, "mcore_data") and not isinstance(metadata, MCoreMetadata):
            preserved = MCoreMetadata(**{item.name: getattr(metadata, item.name) for item in fields(Metadata)})
            preserved.__dict__.update(vars(metadata))
            metadata = preserved
        return finish(self, metadata, results)

    finish_with_mcore_metadata._slime_mcore_metadata = True
    FileSystemReader.read_metadata = read_with_mcore_metadata
    FileSystemWriter.finish = finish_with_mcore_metadata
