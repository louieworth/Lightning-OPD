"""Exercise real spawned graders and their explicit shutdown before Ray exits."""

import unittest

from slime.eval import reward_async


class RewardPoolShutdownTests(unittest.IsolatedAsyncioTestCase):
    async def test_shutdown_joins_workers_and_allows_a_fresh_pool(self):
        self.addCleanup(reward_async.shutdown)
        previous = None
        for _ in range(2):
            score = await reward_async.compute_score("amc23", r"\boxed{2}", "2")
            self.assertEqual(score["acc"], 1.0)
            pool = reward_async._pool
            self.assertIsNot(pool, previous)
            workers = list(pool._processes.values())
            self.assertTrue(workers)
            self.assertTrue(all(worker.is_alive() for worker in workers))
            reward_async.shutdown()
            reward_async.shutdown()
            self.assertIsNone(reward_async._pool)
            self.assertTrue(all(not worker.is_alive() for worker in workers))
            previous = pool
