"""Megatron's common checkpoint state loads under torch's default weights_only=True.

Megatron 0.14 calls torch.load(common.pt) without weights_only. Its saved args hold enum members
(e.g. attention_backend), which pickle protocol 2 rebuilds through builtins.getattr; torch 2.8's
weights-only unpickler rejects that global unless it is registered.
"""

from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

import torch

from megatron.core.transformer.enums import AttnBackend
from slime.utils.checkpoint_globals import register_checkpoint_safe_globals


class CheckpointSafeGlobalsTests(unittest.TestCase):
    def test_megatron_common_state_with_enum_args_loads_weights_only(self):
        register_checkpoint_safe_globals()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "common.pt"
            torch.save({"args": Namespace(attention_backend=AttnBackend.fused, params_dtype=torch.bfloat16)}, path)
            state = torch.load(path, map_location="cpu")
        self.assertEqual(state["args"].attention_backend, AttnBackend.fused)


if __name__ == "__main__":
    unittest.main()
