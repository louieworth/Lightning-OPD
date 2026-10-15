"""Let long evaluations return control before the training checkpoint deadline."""

import asyncio
import time


class CheckpointDeadlineReached(RuntimeError):
    """Evaluation stopped so the driver can save the last completed update."""


async def evaluation_before_deadline(awaitable, deadline):
    if deadline is None:
        return await awaitable
    task = asyncio.ensure_future(awaitable)
    done, _ = await asyncio.wait([task], timeout=max(0, deadline - time.time()))
    if task in done:
        return task.result()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    raise CheckpointDeadlineReached("Evaluation reached the training checkpoint deadline.")
