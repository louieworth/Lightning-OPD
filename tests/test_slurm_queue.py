"""History must show finished allocations by end time with correct run durations."""

from datetime import datetime
import importlib.util
import io
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("slurm_queue", ROOT / "scripts/slurm_queue.py")
QUEUE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUEUE)


def record(job_id, *, state="COMPLETED", end="2026-10-18T02:51:50", elapsed="00:00:14"):
    return (f"{job_id}|job-{job_id}|{state}|0:0|2026-10-18T02:51:36|{end}|{elapsed}|2|"
            "cpu=96,gres/gpu=8,gres/gpu:h100=8|/scratch/l/luli/src/Lightning-OPD")


class SlurmQueueTests(unittest.TestCase):
    def test_orders_by_end_time_and_excludes_active_jobs_steps_and_unknown_end(self):
        output = "\n".join([
            record("999", end="2026-10-17T02:51:50"),
            record("100", state="FAILED"),
            record("200.batch"),
            record("300", state="RUNNING"),
            record("400", state="PENDING"),
            record("500", end="Unknown"),
            record("600", state="CANCELLED by 1234", end="2026-10-18T02:00:00"),
        ])

        jobs = QUEUE.finished_jobs(output)

        self.assertEqual([job["job_id"] for job in jobs], ["100", "600", "999"])
        self.assertEqual(jobs[1]["state"], "CANCELLED")

    def test_deduplicates_allocations_and_keeps_latest_end(self):
        jobs = QUEUE.finished_jobs(record("100", end="2026-10-17T02:51:50") + "\n" + record("100"))

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["end"], "2026-10-18T02:51:50")

    def test_gpu_count_avoids_double_counting_typed_and_total_resources(self):
        self.assertEqual(QUEUE.gpu_count("gres/gpu=8,gres/gpu:h100=8"), "8")
        self.assertEqual(QUEUE.gpu_count("gres/gpu:h100=4,gres/gpu:a100=2"), "6")
        self.assertEqual(QUEUE.gpu_count("cpu=16,mem=32G"), "0")

    def test_expands_lookback_then_selects_latest_ten_by_end(self):
        recent = record("1")
        older = "\n".join(record(str(index), end=f"2026-10-{index:02}T02:51:50") for index in range(1, 13))
        results = [subprocess.CompletedProcess([], 0, stdout=recent),
                   subprocess.CompletedProcess([], 0, stdout=older)]
        with patch.object(QUEUE.subprocess, "run", side_effect=results) as query:
            jobs = QUEUE.recent_jobs("test-user", 10, now=datetime(2026, 10, 18))

        self.assertEqual([job["job_id"] for job in jobs], [str(index) for index in range(12, 2, -1)])
        self.assertEqual(query.call_count, 2)
        first_command = query.call_args_list[0].args[0]
        self.assertIn("-X", first_command)
        self.assertEqual(first_command[first_command.index("-u") + 1], "test-user")
        self.assertEqual(first_command[first_command.index("-S") + 1], "2026-09-18T00:00:00")
        self.assertEqual(query.call_args_list[0].kwargs["env"]["SLURM_TIME_FORMAT"], "standard")

    def test_queue_mode_preserves_original_format_and_forwards_arguments(self):
        with patch.object(QUEUE.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run, \
                patch.object(QUEUE.getpass, "getuser", return_value="test-user"):
            self.assertEqual(QUEUE.main(["--states=RUNNING"]), 0)

        self.assertEqual(run.call_args.args[0], [
            "squeue", "-u", "test-user", "--format=" + QUEUE.QUEUE_FORMAT, "--states=RUNNING",
        ])

    def test_history_shows_actual_elapsed_start_end_and_state(self):
        jobs = QUEUE.finished_jobs(record("100", state="FAILED", elapsed="1-02:03:04"))
        with patch.object(QUEUE, "recent_jobs", return_value=jobs) as history, \
                patch.object(QUEUE.getpass, "getuser", return_value="test-user"), patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(QUEUE.main(["-h"]), 0)

        history.assert_called_once_with("test-user", 10)
        for value in ("ELAPSED", "1-02:03:04", "FAILED", "2026-10-18 02:51:36", "2026-10-18 02:51:50", "Lightning-OPD"):
            self.assertIn(value, output.getvalue())


if __name__ == "__main__":
    unittest.main()
