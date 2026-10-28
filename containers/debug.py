"""Run container checks with srun inside an existing keepalive allocation."""

import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]


def job_record(job):
    result = subprocess.run(['scontrol', 'show', 'job', str(job), '--oneliner'],
                            check=True, text=True, capture_output=True)
    return dict(item.split('=', 1) for item in shlex.split(result.stdout) if '=' in item)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', type=int)
    parser.add_argument('--wait', action='store_true', help='Wait for the existing allocation to start')
    cli, arguments = parser.parse_known_args()
    previous = None
    while True:
        job = job_record(cli.job)
        state = job['JobState']
        if state == 'RUNNING':
            break
        if state not in ('PENDING', 'CONFIGURING') or not cli.wait:
            raise RuntimeError(f'Allocation {cli.job} is {state}; use a running keepalive job.')
        status = (state, job.get('Reason'), job.get('StartTime'))
        if status != previous:
            print(f'Waiting for allocation {cli.job}: {status}', flush=True)
            previous = status
        time.sleep(15)
    directory = ROOT / 'tmp/containers/debug' / str(cli.job)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'controller.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        retention = directory / 'failure-retention.json'
        if not retention.is_file():
            print('Checking keepalive retention with an intentional srun exit status 42.', flush=True)
            with (directory / 'failure-retention.log').open('w') as log:
                result = subprocess.run(['srun', '--jobid=' + str(cli.job), '--mpi=none',
                                         '--nodes=1', '--ntasks=1', '--cpus-per-task=1',
                                         '--kill-on-bad-exit=1', '/bin/bash', '-c', 'exit 42'],
                                        stdout=log, stderr=subprocess.STDOUT)
            after = job_record(cli.job)
            if result.returncode != 42 or after['JobState'] != 'RUNNING':
                raise RuntimeError('The allocation did not survive the expected failing srun step.')
            retention.write_text(json.dumps({'job': cli.job, 'step_exit_code': result.returncode,
                                              'allocation_state_after_failure': after['JobState']}, indent=2) + '\n')
            print('Keepalive retention verified: the failed step left the allocation RUNNING.', flush=True)
        checks = ROOT / 'tmp/containers/checks' / str(cli.job)
        if checks.exists():
            archive = directory / f'previous-{time.time_ns()}'
            checks.rename(archive)
            print('Previous debug attempt preserved at ' + str(archive), flush=True)
        os.environ.update(SLURM_JOB_ID=str(cli.job), SLURM_JOBID=str(cli.job),
                          SLURM_JOB_NODELIST=job['NodeList'], SLURM_NODELIST=job['NodeList'],
                          SLURM_JOB_NUM_NODES=job['NumNodes'].split('-')[0],
                          SLURM_NNODES=job['NumNodes'].split('-')[0],
                          SLURM_NTASKS='2', SLURM_NTASKS_PER_NODE='1',
                          SLURM_CPUS_PER_TASK=job.get('CPUs/Task', '48'),
                          SLURM_SUBMIT_DIR=str(ROOT), SLURM_EXPORT_ENV='ALL')
        spec = importlib.util.spec_from_file_location('container_launch', ROOT / 'containers/launch.py')
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        try:
            launcher.main(['batch', *arguments])
        finally:
            print(f'Debug controller finished; it does not cancel allocation {cli.job}.', flush=True)


if __name__ == '__main__':
    main()
