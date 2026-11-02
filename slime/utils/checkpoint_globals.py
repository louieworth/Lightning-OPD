"""Globals that torch's weights-only unpickler must accept in this job's own checkpoints.

Megatron 0.14 loads common.pt with torch.load's default weights_only=True (torch >= 2.6). Its saved
args hold enum members, which pickle protocol 2 rebuilds as getattr(EnumClass, name); torch 2.8's
weights-only unpickler rejects builtins.getattr unless registered. getattr can then only reach
objects that are themselves allowed (megatron.core registers its enum classes).
"""

import builtins

import torch


def register_checkpoint_safe_globals():
    import slime.utils.eval_config

    torch.serialization.add_safe_globals([slime.utils.eval_config.EvalDatasetConfig, builtins.getattr])
