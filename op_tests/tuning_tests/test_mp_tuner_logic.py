# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Unit tests for mp_tuner polling loop logic.

Simulates async_result behavior without GPU/multiprocessing to verify:
1. consecutive_timeouts tracks correctly and resets on success
2. half-GPU threshold triggers break at the right time
3. KeyError ("PID not in gpu_map") tasks request a pool restart, are retried on the
   rebuilt pool, and are dropped once MAPPING_ERROR_MAX_RETRIES is reached

TestRealMpTunerMappingError drives the real mp_tuner() loop with a fake pool, so it
covers the shipped code rather than the simulation below.

Run: python3 -m unittest op_tests.tuning_tests.test_mp_tuner_logic -v
"""

import time
import unittest
from unittest import mock
from multiprocessing import TimeoutError as MPTimeoutError

from aiter.utility import mp_tuner as mp_tuner_mod
from aiter.utility.mp_tuner import MAPPING_ERROR_MAX_RETRIES


class FakeAsyncResult:
    """Simulates multiprocessing.AsyncResult for testing polling logic."""

    def __init__(self, behavior, value=None):
        """
        behavior: "ok", "timeout_pending", "timeout_expired", "keyerror", "accelerator"
        value: return value for "ok"
        """
        self.behavior = behavior
        self.value = value

    def get(self, timeout=10):
        if self.behavior == "ok":
            return self.value
        elif self.behavior in ("timeout_pending", "timeout_expired"):
            raise MPTimeoutError("timeout")
        elif self.behavior == "keyerror":
            raise KeyError("12345")
        elif self.behavior == "accelerator":
            raise type("AcceleratorError", (Exception,), {})("GPU fault")


def simulate_poll_round(
    remaining_tasks, task_start_times, mp_num, timeout, mapping_error_retries=None
):
    """
    Simulate one round of the mp_tuner polling loop.

    mapping_error_retries: per-task KeyError counter, carried across rounds by the
    caller exactly as mp_tuner does. Omit it for single-round tests.

    Returns (completed, dummy_failed, pool_restart_needed, broke_early)
    """
    if mapping_error_retries is None:
        mapping_error_retries = {}
    completed_this_round = []
    dummy_failed_tasks = []
    consecutive_timeouts = 0
    half_gpu = max(1, (mp_num + 1) // 2)
    pool_restart_needed = False
    broke_early = False

    for k, async_result in remaining_tasks:
        try:
            if timeout is not None:
                elapsed = time.time() - task_start_times[k]
                remaining_time = timeout - elapsed
                actual_timeout = max(1, min(10, remaining_time))
            else:
                actual_timeout = 10

            async_result.get(timeout=actual_timeout)
            completed_this_round.append((k, async_result))
            consecutive_timeouts = 0

        except MPTimeoutError:
            if timeout is not None:
                elapsed = time.time() - task_start_times[k]
                if elapsed > timeout:
                    consecutive_timeouts += 1
                    completed_this_round.append((k, async_result))
                    pool_restart_needed = True

                    if consecutive_timeouts >= half_gpu:
                        broke_early = True
                        break
                else:
                    consecutive_timeouts = 0

        except Exception as e:
            error_type = type(e).__name__
            is_mapping_error = error_type == "KeyError"

            if is_mapping_error:
                mapping_error_retries[k] = mapping_error_retries.get(k, 0) + 1
                dummy_failed_tasks.append((k, "mapping error"))
                pool_restart_needed = True
                if mapping_error_retries[k] >= MAPPING_ERROR_MAX_RETRIES:
                    completed_this_round.append((k, async_result))
            elif error_type == "AcceleratorError":
                completed_this_round.append((k, async_result))
                pool_restart_needed = True
                broke_early = True
                break
            else:
                completed_this_round.append((k, async_result))

    return completed_this_round, dummy_failed_tasks, pool_restart_needed, broke_early


class TestConsecutiveTimeouts(unittest.TestCase):

    def test_single_timeout_no_break_8gpu(self):
        """1 stuck GPU out of 8: should NOT break early."""
        mp_num = 8
        timeout = 0.0
        now = time.time()
        remaining = [
            (0, FakeAsyncResult("timeout_expired")),
            (1, FakeAsyncResult("ok", [("info", 1.0, 0.0)])),
            (2, FakeAsyncResult("timeout_expired")),
            (3, FakeAsyncResult("ok", [("info", 2.0, 0.0)])),
        ]
        start_times = {k: now - 10 for k, _ in remaining}

        completed, dummy, restart, broke = simulate_poll_round(
            remaining, start_times, mp_num, timeout
        )
        self.assertFalse(broke, "Should NOT break early with interleaved success")
        self.assertTrue(restart, "Should still need restart (at least 1 timeout)")
        self.assertEqual(len(completed), 4, "All tasks should be processed")

    def test_half_gpu_consecutive_triggers_break(self):
        """4 consecutive timeouts with 8 GPUs (half=4): should break."""
        mp_num = 8
        timeout = 0.0
        now = time.time()
        remaining = [
            (0, FakeAsyncResult("timeout_expired")),
            (1, FakeAsyncResult("timeout_expired")),
            (2, FakeAsyncResult("timeout_expired")),
            (3, FakeAsyncResult("timeout_expired")),
            (4, FakeAsyncResult("ok", [("info", 1.0, 0.0)])),
        ]
        start_times = {k: now - 10 for k, _ in remaining}

        completed, dummy, restart, broke = simulate_poll_round(
            remaining, start_times, mp_num, timeout
        )
        self.assertTrue(broke, "Should break after 4 consecutive timeouts (half of 8)")
        self.assertTrue(restart)
        self.assertEqual(len(completed), 4, "Task 4 not polled due to break")

    def test_success_resets_consecutive(self):
        """Success in between resets counter: 3 timeouts, 1 ok, 3 timeouts != break for 8 GPU."""
        mp_num = 8
        timeout = 0.0
        now = time.time()
        remaining = [
            (0, FakeAsyncResult("timeout_expired")),
            (1, FakeAsyncResult("timeout_expired")),
            (2, FakeAsyncResult("timeout_expired")),
            (3, FakeAsyncResult("ok", [("info", 1.0, 0.0)])),
            (4, FakeAsyncResult("timeout_expired")),
            (5, FakeAsyncResult("timeout_expired")),
            (6, FakeAsyncResult("timeout_expired")),
        ]
        start_times = {k: now - 10 for k, _ in remaining}

        completed, dummy, restart, broke = simulate_poll_round(
            remaining, start_times, mp_num, timeout
        )
        self.assertFalse(broke, "Should NOT break: success at task 3 resets counter")
        self.assertTrue(restart, "Still need restart from timeouts")
        self.assertEqual(len(completed), 7)

    def test_2gpu_half_is_1(self):
        """2 GPUs: half=1, single consecutive timeout triggers break."""
        mp_num = 2
        timeout = 0.0
        now = time.time()
        remaining = [
            (0, FakeAsyncResult("timeout_expired")),
            (1, FakeAsyncResult("ok", [("info", 1.0, 0.0)])),
        ]
        start_times = {k: now - 10 for k, _ in remaining}

        completed, dummy, restart, broke = simulate_poll_round(
            remaining, start_times, mp_num, timeout
        )
        self.assertTrue(broke, "2 GPUs: half=1, first timeout should break")
        self.assertEqual(len(completed), 1)

    def test_pending_timeout_resets_consecutive(self):
        """Task not yet expired (still running) resets consecutive count."""
        mp_num = 4
        timeout = 100.0
        now = time.time()
        remaining = [
            (0, FakeAsyncResult("timeout_expired")),
            (1, FakeAsyncResult("timeout_pending")),
            (2, FakeAsyncResult("timeout_expired")),
        ]
        start_times = {
            0: now - 200,
            1: now,
            2: now - 200,
        }

        completed, dummy, restart, broke = simulate_poll_round(
            remaining, start_times, mp_num, timeout
        )
        self.assertFalse(broke, "Pending task resets consecutive, so no break")
        self.assertEqual(len(completed), 2)


class TestKeyErrorHandling(unittest.TestCase):

    def test_keyerror_stays_in_remaining_until_retries_exhausted(self):
        """A first KeyError keeps the task queued but must request a pool restart.

        gpu_map is only rebuilt by a restart, so without one the task can never
        succeed; it is kept queued so the retry lands on the rebuilt pool.
        """
        mp_num = 4
        timeout = 0.0
        now = time.time()
        remaining = [
            (0, FakeAsyncResult("keyerror")),
            (1, FakeAsyncResult("ok", [("info", 1.0, 0.0)])),
        ]
        start_times = {k: now - 10 for k, _ in remaining}

        completed, dummy, restart, broke = simulate_poll_round(
            remaining, start_times, mp_num, timeout
        )
        completed_ids = {k for k, _ in completed}
        self.assertNotIn(0, completed_ids, "KeyError task should be retried, not dropped")
        self.assertIn(1, completed_ids, "OK task should be completed")
        self.assertEqual(len(dummy), 1, "KeyError task should be in dummy_failed")
        self.assertTrue(restart, "KeyError must request the restart that rebuilds gpu_map")

    def test_keyerror_with_timeout_gets_resubmitted(self):
        """KeyError tasks wait for root-cause timeout to trigger restart."""
        mp_num = 2
        timeout = 0.0
        now = time.time()
        remaining = [
            (0, FakeAsyncResult("keyerror")),
            (1, FakeAsyncResult("timeout_expired")),
        ]
        start_times = {k: now - 10 for k, _ in remaining}

        completed, dummy, restart, broke = simulate_poll_round(
            remaining, start_times, mp_num, timeout
        )
        completed_ids = {k for k, _ in completed}
        self.assertNotIn(0, completed_ids, "KeyError task stays for resubmit")
        self.assertIn(1, completed_ids, "Root-cause timeout is completed")
        self.assertTrue(restart, "Timeout should trigger restart")


        new_remaining = [(k, ar) for k, ar in remaining if k not in completed_ids]
        self.assertEqual(len(new_remaining), 1)
        self.assertEqual(
            new_remaining[0][0], 0, "Only KeyError task remains for resubmit"
        )

    def test_keyerror_only_tasks_still_terminate(self):
        """KeyError-only tasks must drain, not poll forever.

        A KeyError is raised immediately, so it never reaches the MPTimeoutError
        branch and the task's timeout is never evaluated. Before the retry bound
        existed these tasks were never completed and never triggered a restart, so
        the loop spun at 1 Hz forever whenever no other task supplied a root cause.
        """
        mp_num = 4
        timeout = 100.0
        now = time.time()
        remaining = [
            (0, FakeAsyncResult("keyerror")),
            (1, FakeAsyncResult("keyerror")),
        ]
        start_times = {k: now for k, _ in remaining}
        retries = {}

        for round_idx in range(MAPPING_ERROR_MAX_RETRIES):
            completed, dummy, restart, broke = simulate_poll_round(
                remaining, start_times, mp_num, timeout, retries
            )
            self.assertTrue(restart, "every mapping error must request a restart")
            for item in completed:
                remaining.remove(item)

        self.assertEqual(
            remaining, [], "all KeyError tasks must be drained within the retry bound"
        )


class TestAcceleratorError(unittest.TestCase):

    def test_accelerator_breaks_immediately(self):
        """AcceleratorError should break immediately and trigger restart."""
        mp_num = 4
        timeout = 100.0
        now = time.time()
        remaining = [
            (0, FakeAsyncResult("ok", [("info", 1.0, 0.0)])),
            (1, FakeAsyncResult("accelerator")),
            (2, FakeAsyncResult("ok", [("info", 2.0, 0.0)])),
        ]
        start_times = {k: now for k, _ in remaining}

        completed, dummy, restart, broke = simulate_poll_round(
            remaining, start_times, mp_num, timeout
        )
        self.assertTrue(broke, "AcceleratorError should break")
        self.assertTrue(restart, "AcceleratorError should trigger restart")
        completed_ids = {k for k, _ in completed}
        self.assertIn(0, completed_ids)
        self.assertIn(1, completed_ids)
        self.assertNotIn(2, completed_ids, "Task 2 not reached due to break")


class FakePool:
    """Stands in for mp.Pool so the real mp_tuner loop can run without GPUs.

    script: task index -> list of FakeAsyncResult behaviors, consumed one entry per
    submission, so a task can behave differently before and after a pool restart.
    The last entry is reused if the task is submitted more times than scripted.
    """

    instances = 0

    def __init__(self, script, submit_counts, processes=None, **kwargs):
        self.script = script
        # Shared across pool restarts: a resubmitted task must advance to its next
        # scripted behavior, not replay the first one on the fresh pool.
        self.submit_counts = submit_counts
        FakePool.instances += 1

    def apply_async(self, func, args=()):
        if func.__name__ == "get_pid":
            # Distinct fake PIDs so mp_tuner can build gpu_map.
            return FakeAsyncResult("ok", value=10000 + FakePool.instances)
        # work_group submission: args = (gpu_map, fast_mode, err_ratio, in_data,
        # task_group[k], verbose); recover k from the task payload's info field.
        info = args[4][0]
        k = int(info.split("_")[1])
        n = self.submit_counts.get(k, 0)
        self.submit_counts[k] = n + 1
        behaviors = self.script[k]
        return FakeAsyncResult(behaviors[min(n, len(behaviors) - 1)], value=[(info, 1.0, 0.0)])

    def close(self):
        pass

    def terminate(self):
        pass

    def join(self):
        pass


class TestRealMpTunerMappingError(unittest.TestCase):
    """Drives the shipped mp_tuner() loop, not the simulation above."""

    def _run(self, script, n_tasks):
        tasks = [(f"task_{k}",) for k in range(n_tasks)]
        in_datas = [(n_tasks, None)]
        FakePool.instances = 0
        submit_counts = {}

        def make_pool(processes=None, **kwargs):
            return FakePool(script, submit_counts, processes=processes, **kwargs)

        with mock.patch.object(
            mp_tuner_mod.torch.cuda, "device_count", return_value=2
        ), mock.patch.object(mp_tuner_mod.mp, "set_start_method"), mock.patch.object(
            mp_tuner_mod.mp, "Pool", side_effect=make_pool
        ):
            return mp_tuner_mod.mp_tuner(tasks, in_datas, mp_num=2, timeout=None)

    def test_permanent_mapping_error_terminates(self):
        """The regression: a task whose PID is never in gpu_map must not spin forever.

        timeout=None disables the only other exit, and a KeyError raises instantly so
        it never reaches the timeout branch. Before the retry bound this loop never
        exited. Returning at all is the assertion; the values just confirm the task is
        reported as failed rather than silently dropped.
        """
        script = {0: ["keyerror"], 1: ["keyerror"]}
        t0 = time.time()
        out = self._run(script, 2)
        self.assertLess(time.time() - t0, 60, "mp_tuner must terminate, not hang")
        self.assertEqual(len(out), 2, "one result per task, cardinality preserved")
        for info, latency, err in out:
            self.assertEqual(latency, float("inf"), "failed task reported as inf")

    def test_mapping_error_recovers_after_restart(self):
        """A transient mapping error must not cost the task its real result.

        Task 0 hits a stale gpu_map once, then succeeds on the rebuilt pool. The
        retry bound must not turn that into a dummy failure.
        """
        script = {0: ["keyerror", "ok"], 1: ["ok"]}
        out = self._run(script, 2)
        self.assertEqual(len(out), 2)
        latencies = {info: latency for info, latency, _ in out}
        self.assertEqual(
            latencies["task_0"], 1.0, "task 0 should keep its post-restart result"
        )
        self.assertEqual(latencies["task_1"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
