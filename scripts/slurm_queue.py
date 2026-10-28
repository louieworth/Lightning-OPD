#!/usr/bin/python3 -S
"""Keep sq's queue view and add sq -h [count] for finished Slurm jobs."""

import argparse
from datetime import datetime, timedelta
import getpass
import os
from pathlib import Path
import subprocess
import sys


QUEUE_FORMAT = "%.10i %40j %.8T %.11M %.11L %.9l %.5D %.5C %.14b %.8m %.19S %R"
TERMINAL_STATES = (
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "PREEMPTED",
    "BOOT_FAIL", "DEADLINE", "OUT_OF_MEMORY",
)
FIELDS = (
    "JobID%80,JobName%256,State%40,ExitCode,Start,End,Elapsed,"
    "NNodes,AllocTRES%512,WorkDir%2048"
)


def gpu_count(tres):
    resources = dict(part.split("=", 1) for part in tres.split(",") if "=" in part)
    if "gres/gpu" in resources:
        return resources["gres/gpu"]
    return str(sum(int(value) for key, value in resources.items()
                   if key.startswith("gres/gpu:") and value.isdigit()))


def finished_jobs(output):
    jobs = {}
    keys = ("job_id", "name", "state", "exit", "start", "end", "elapsed", "nodes", "tres", "workdir")
    for line in output.splitlines():
        fields = line.split("|")
        if len(fields) != len(keys):
            continue
        job = dict(zip(keys, (field.strip() for field in fields)))
        job["state"] = job["state"].split()[0].rstrip("+") if job["state"] else ""
        if "." in job["job_id"] or job["state"] not in TERMINAL_STATES:
            continue
        try:
            ended = datetime.fromisoformat(job["end"])
        except ValueError:
            continue
        job["ended"] = ended
        previous = jobs.get(job["job_id"])
        if previous is None or ended > previous["ended"]:
            jobs[job["job_id"]] = job
    return sorted(jobs.values(), key=lambda job: (job["ended"], job["job_id"]), reverse=True)


def recent_jobs(user, count, now=None):
    now = now or datetime.now()
    env = dict(os.environ, SLURM_TIME_FORMAT="standard")
    # Expand the query when the user has fewer than count recent finished jobs.
    for days in (30, 90, 365, None):
        start = (now - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S") if days else "1970-01-01"
        command = [
            "sacct", "-X", "-n", "-P", "--array", "-u", user,
            "-S", start, "-E", "now", "--state=" + ",".join(TERMINAL_STATES),
            "--format=" + FIELDS,
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True, env=env)
        jobs = finished_jobs(result.stdout)
        if len(jobs) >= count or days is None:
            return jobs[:count]
    return []


def history_table(jobs):
    rows = [["JOBID", "JOBNAME", "STATE", "EXIT", "ELAPSED", "START", "END", "NODES", "GPUS", "REPO"]]
    for job in jobs:
        rows.append([
            job["job_id"], job["name"], job["state"], job["exit"], job["elapsed"],
            job["start"].replace("T", " ") if job["start"] not in {"None", "Unknown", ""} else "-",
            job["end"].replace("T", " "), job["nodes"], gpu_count(job["tres"]),
            Path(job["workdir"]).name or "-",
        ])
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    return "\n".join("  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip() for row in rows)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    user = getpass.getuser()
    if argv == ["--help"]:
        print("sq                Show your current Slurm queue.\n"
              "sq -h             Show your 10 most recently finished jobs.\n"
              "sq -h COUNT       Show COUNT finished jobs, newest end time first.\n"
              "ELAPSED is job runtime, excluding queue wait; timestamps use cluster local time.\n"
              "Other arguments are forwarded to squeue.")
        return 0
    if not argv or argv[0] != "-h":
        return subprocess.run(["squeue", "-u", user, "--format=" + QUEUE_FORMAT, *argv]).returncode
    parser = argparse.ArgumentParser(prog="sq -h", description="Show finished Slurm jobs, newest end time first.")
    parser.add_argument("count", nargs="?", type=int, default=10, help="Number of jobs (default: 10)")
    args = parser.parse_args(argv[1:])
    if args.count < 1:
        parser.error("count must be positive")
    jobs = recent_jobs(user, args.count)
    if not jobs:
        print(f"No finished Slurm jobs found for {user}.")
        return 0
    print(f"Last {len(jobs)} finished jobs for {user} (newest end time first):")
    print(history_table(jobs))
    print("\nELAPSED = job runtime, excluding queue wait. Times use cluster local time.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as error:
        print(f"sq: {error.stderr.strip() or 'sacct query failed'}", file=sys.stderr)
        sys.exit(error.returncode or 1)
    except OSError as error:
        print(f"sq: {error}", file=sys.stderr)
        sys.exit(1)
