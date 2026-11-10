"""Start, stop or inspect tools/gpu_keepalive.py on every node of the pre-started Ray cluster.

    python3 tools/gpu_keepalive_ray.py start [-- <gpu_keepalive.py args>]
    python3 tools/gpu_keepalive_ray.py status
    python3 tools/gpu_keepalive_ray.py stop

One detached actor per node (named gpu_keepalive_<node>) holds the keep-alive process. The actors
reserve no Ray GPUs, so training jobs still find every GPU free, and they outlive this command.
"""

import sys
from pathlib import Path

import ray

ROOT = Path(__file__).resolve().parents[1]


@ray.remote(num_cpus=0, num_gpus=0)
class KeepAlive:
    def __init__(self, command, log):
        import os
        import subprocess

        env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}  # Ray sets it to ""
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        self.log = open(log, "ab")
        self.process = subprocess.Popen(command, env=env, cwd=str(ROOT), stdout=self.log, stderr=subprocess.STDOUT,
                                        start_new_session=True)

    def status(self):
        import subprocess

        logs = sorted((ROOT / "tmp/gpu_keepalive").glob("gpu*.log"))
        last = [subprocess.run(["tail", "-1", str(p)], capture_output=True, text=True).stdout.strip() for p in logs]
        return self.process.poll(), last

    def stop(self):
        import os
        import signal

        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            self.process.wait(timeout=30)
        return self.process.returncode


def nodes():
    return [n for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU", 0) > 0]


def name(node):
    return "gpu_keepalive_" + node["NodeManagerAddress"].strip("[]").replace(":", "_")


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else "status"
    extra = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ray.init(address="auto", logging_level="ERROR", log_to_driver=False, namespace="gpu_keepalive")
    for node in nodes():
        actor_name = name(node)
        try:
            actor = ray.get_actor(actor_name)
        except ValueError:
            actor = None
        if action == "start":
            if actor is not None:
                print(actor_name, "already running")
                continue
            command = [sys.executable, str(ROOT / "tools/gpu_keepalive.py"), *extra]
            KeepAlive.options(name=actor_name, lifetime="detached",
                              resources={f"node:{node['NodeManagerAddress']}": 0.001}).remote(
                command, str(ROOT / "tmp/gpu_keepalive/keepalive.out"))
            print(actor_name, "started")
        elif action == "stop":
            if actor is None:
                print(actor_name, "not running")
                continue
            print(actor_name, "stopped, exit", ray.get(actor.stop.remote()))
            ray.kill(actor)
        else:
            if actor is None:
                print(actor_name, "not running")
                continue
            code, last = ray.get(actor.status.remote())
            print(actor_name, "process exit code" if code is not None else "running", code if code is not None else "")
            for line in last:
                print("   ", line)


if __name__ == "__main__":
    main()
