"""The Megatron actor offloads and restores its GPU state only on real transitions.

Block Lightning keeps the actor on the GPUs between updates that no engine needs. train() still
calls wake_up() first, which must then neither rebuild the NCCL groups (seconds per group here)
nor resume memory; sleeping twice must not pause twice.
"""

import types
import unittest
from unittest.mock import MagicMock, patch

from slime.backends.megatron_utils import actor as actor_module


class ActorOffloadStateTests(unittest.TestCase):
    def actor(self):
        actor = object.__new__(actor_module.MegatronTrainRayActor)
        actor.args = types.SimpleNamespace(offload_train=True)
        actor._asleep = False
        return actor

    def run_calls(self, *calls):
        memory_saver = MagicMock()
        destroy, reload = MagicMock(), MagicMock()
        with patch.object(actor_module, "torch_memory_saver", memory_saver), \
                patch.object(actor_module, "destroy_process_groups", destroy), \
                patch.object(actor_module, "reload_process_groups", reload), \
                patch.object(actor_module, "clear_memory"), patch.object(actor_module, "print_memory"), \
                patch("slime.backends.megatron_utils.opd.clear_teacher_head_cache"):
            actor = self.actor()
            for call in calls:
                getattr(actor, call)()
        return memory_saver, destroy, reload

    def test_waking_an_awake_actor_changes_nothing(self):
        memory_saver, destroy, reload = self.run_calls("wake_up", "wake_up")
        memory_saver.resume.assert_not_called()
        reload.assert_not_called()

    def test_saving_keeps_an_awake_actor_s_groups_and_an_asleep_one_s_teardown(self):
        for asleep, expected in ((False, (0, 0)), (True, (1, 1))):
            with self.subTest(asleep=asleep):
                destroy, reload = MagicMock(), MagicMock()
                actor = self.actor()
                actor._asleep = asleep
                actor.args.debug_rollout_only, actor.args.async_save = False, False
                actor.model = actor.optimizer = actor.opt_param_scheduler = None
                with patch.object(actor_module, "destroy_process_groups", destroy), \
                        patch.object(actor_module, "reload_process_groups", reload), \
                        patch.object(actor_module, "save"):
                    actor.save_model(3, force_sync=True)
                self.assertEqual((reload.call_count, destroy.call_count), expected)

    def test_each_real_transition_happens_once(self):
        memory_saver, destroy, reload = self.run_calls("sleep", "sleep", "wake_up", "wake_up")
        self.assertEqual((memory_saver.pause.call_count, destroy.call_count), (1, 1))
        self.assertEqual((memory_saver.resume.call_count, reload.call_count), (1, 1))


if __name__ == "__main__":
    unittest.main()
