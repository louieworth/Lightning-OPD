"""Run one training job on a pre-started Ray cluster whose nodes share no filesystem.

The launcher sets these variables for train.py:
- LIGHTNING_OPD_RAY_RUNTIME_ENV: JSON job runtime env (venv interpreter, env vars) for every Ray worker.
- LIGHTNING_OPD_RAY_NODE: the launcher's Ray node IP (as Ray formats it, e.g. "[fd00::1]"). The rollout
  manager reads and writes run files, so it always runs there.
- LIGHTNING_OPD_RAY_SINGLE_NODE=1: the job uses one node, so every GPU bundle is pinned there as well.
Without them, slime keeps its usual implicit ray.init and unpinned scheduling.
"""

import contextlib
import json
import os
from pathlib import Path
import re

NODE_ENV = "LIGHTNING_OPD_RAY_NODE"
SECRETS_ENV = "LIGHTNING_OPD_WORKER_SECRETS"
# Ray logs a job's runtime env in plain text, so credentials reach workers through a 0600 file instead.
SECRET_NAME = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|_PWD)$|^(https?|all|ftp)_proxy$", re.IGNORECASE)
SINGLE_NODE_ENV = "LIGHTNING_OPD_RAY_SINGLE_NODE"
RUNTIME_ENV = "LIGHTNING_OPD_RAY_RUNTIME_ENV"


def node_resources():
    node = os.environ.get(NODE_ENV)
    return {f"node:{node}": 0.001} if node else {}


def gpu_bundles(num_gpus):
    pinned = node_resources() if os.environ.get(SINGLE_NODE_ENV) == "1" else {}
    return [{"GPU": 1, "CPU": 1, **pinned} for _ in range(num_gpus)]


def split_secrets(env):
    """(public, secret) variables: credentials and credential-bearing proxy URLs are secret."""
    public, secret = {}, {}
    for key, value in env.items():
        (secret if SECRET_NAME.search(key) else public)[key] = value
    return public, secret


def load_worker_secrets():
    """Ray worker_process_setup_hook: add the job's credentials without overriding the worker's own values."""
    path = os.environ.get(SECRETS_ENV)
    if path and os.path.isfile(path):
        with open(path) as stream:
            for key, value in json.load(stream).items():
                os.environ.setdefault(key, value)


def write_worker_secrets(secret, path):
    """Write the credentials file (mode 0600) at `path` on this node and every other alive node."""
    import ray

    def write(values, target):
        import json
        import os

        os.makedirs(os.path.dirname(target), mode=0o700, exist_ok=True)
        descriptor = os.open(target + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(values, stream)
        os.replace(target + ".tmp", target)

    write(secret, str(path))
    ray.get([_pinned(ray.remote(num_cpus=0)(write), node).remote(secret, str(path)) for node in _other_nodes()])


def connect():
    import ray

    runtime_env = os.environ.get(RUNTIME_ENV)
    if runtime_env and not ray.is_initialized():
        ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), runtime_env=json.loads(runtime_env))


def _other_nodes():
    import ray

    here = ray.get_runtime_context().get_node_id()
    return [node for node in ray.nodes() if node["Alive"] and node["NodeID"] != here]


def gpu_nodes(count):
    """Ray addresses (e.g. "[fd00::1]") of the first `count` alive GPU nodes, this node first."""
    import ray

    here = ray.get_runtime_context().get_node_id()
    nodes = [node for node in ray.nodes() if node["Alive"] and node["Resources"].get("GPU", 0) > 0]
    nodes.sort(key=lambda node: node["NodeID"] != here)
    return [node["NodeManagerAddress"] for node in nodes[:count]]


def _pinned(function, node):
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    return function.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node["NodeID"], soft=False))


# Remote helpers are built inside these functions so Ray ships them by value: other nodes may not
# have this repository's code yet (mirroring it is one of their uses).


def missing_on_nodes(path):
    """Ray node addresses (other than this one) where `path` does not exist."""
    import ray

    def exists(target):
        import os

        return os.path.exists(target)

    nodes = _other_nodes()
    found = ray.get([_pinned(ray.remote(num_cpus=0)(exists), node).remote(str(path)) for node in nodes])
    return [node["NodeManagerAddress"] for node, present in zip(nodes, found, strict=True) if not present]


SKIP = {".git", "__pycache__", ".pytest_cache"}


def mirror(root, exclude=()):
    """Copy directory `root` to the same path on every other alive node.

    Files with equal size and mtime are skipped; top-level entries named in `exclude` and
    .git/__pycache__ anywhere are not copied. Symlinks are not followed or copied.
    """
    import ray

    def files(top, excluded):
        import os
        from pathlib import Path

        top, found = Path(top), {}
        if not top.is_dir():
            return found
        for directory, dirs, names in os.walk(top):
            here = Path(directory)
            dirs[:] = [d for d in dirs if d not in SKIP and not (here == top and d in excluded)]
            for name in names:
                path = here / name
                if path.is_file() and not path.is_symlink() and not (here == top and name in excluded):
                    found[str(path.relative_to(top))] = (path.stat().st_size, path.stat().st_mtime_ns)
        return found

    def write(top, relative, data, mode, mtime_ns):
        import os
        from pathlib import Path

        path = Path(top) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".mirror-tmp")
        temporary.write_bytes(data)
        os.chmod(temporary, mode)
        os.utime(temporary, ns=(mtime_ns, mtime_ns))
        os.replace(temporary, path)

    root, excluded = Path(root).resolve(), set(exclude)
    source = files(root, excluded)
    remote_files, remote_write = ray.remote(num_cpus=0)(files), ray.remote(num_cpus=0)(write)
    for node in _other_nodes():
        existing = ray.get(_pinned(remote_files, node).remote(str(root), excluded))
        pending = []
        for name, (size, mtime_ns) in source.items():
            if existing.get(name) == (size, mtime_ns):
                continue
            path = root / name
            pending.append(_pinned(remote_write, node).remote(
                str(root), name, ray.put(path.read_bytes()), path.stat().st_mode & 0o7777, mtime_ns))
            if len(pending) >= 8:
                done, pending = ray.wait(pending, num_returns=len(pending) - 4)
                ray.get(done)
        ray.get(pending)


@contextlib.contextmanager
def gpu_process(command, env, num_gpus, resources, log_path, devices=None):
    """Run `command` on the node `resources` selects, while Ray holds `num_gpus` GPUs for it.

    The process gets the GPUs Ray assigned via CUDA_VISIBLE_DEVICES, or `devices` when the caller already owns
    GPUs that Ray has handed out (then num_gpus is 0). env=None runs it in the Ray worker's environment.
    Yields a Popen-like view (poll, returncode); the process is stopped on exit.
    """
    import ray

    class Holder:
        def __init__(self, command, env, log_path, devices):
            import os
            from pathlib import Path
            import subprocess

            gpus = devices if devices is not None else ",".join(str(gpu) for gpu in ray.get_gpu_ids())
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            self.log = open(log_path, "ab")
            self.process = subprocess.Popen(command, env=dict(os.environ if env is None else env, CUDA_VISIBLE_DEVICES=gpus),
                                            stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)

        def poll(self):
            return self.process.poll()

        def stop(self):
            import os
            import signal
            import subprocess

            if self.process.poll() is None:
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait()
            self.log.close()

    class Process:
        returncode = None

        def poll(self):
            self.returncode = ray.get(holder.poll.remote())
            return self.returncode

    holder = ray.remote(num_cpus=0, num_gpus=num_gpus)(Holder).options(resources=resources).remote(
        command, env, str(log_path), devices)
    try:
        process = Process()
        if process.poll() is not None:
            raise RuntimeError(f"GPU process exited at startup; inspect {log_path}.")
        yield process
    finally:
        try:
            ray.get(holder.stop.remote())
        finally:
            ray.kill(holder)
