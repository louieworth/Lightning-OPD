"""Build and run the OPD image on two exclusive four-H100 nodes.

Host-side code deliberately uses only Python's standard library. YAML,
PyTorch and every training dependency are loaded inside the image.
"""

import argparse
import contextlib
import fcntl
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.parse
import uuid

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'tmp/containers'
BASE = 'tonyhao96/jetmoe@sha256:d79db3699f58172088c8897efd4e21aa7960cbd5530c81b02a6bf07132695313'
DEFAULT_CONFIG = ROOT / 'configs/config/qwen3_1_7b_base_nt_teacher_4b_instruct_2507_k1.yaml'
SOURCE_ITEMS = ('containers', 'slime', 'slime_plugins', 'configs', 'scripts', 'tools', 'data_curation',
                'train.py', 'setup.py', 'pyproject.toml', 'requirements.txt', 'README.md',
                'LICENSE', 'THIRD_PARTY_NOTICES.md')


def run(command, *, capture=False, env=None):
    if not capture:
        print('+ ' + shlex.join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), check=True, cwd=ROOT, env=env,
                          text=True, stdout=subprocess.PIPE if capture else None)


def apptainer_binary():
    command = os.environ.get('APPTAINER_BIN') or shutil.which('apptainer')
    if command:
        return command
    # Tamia's StdEnv/2023 provides this version; 1.5.x needs StdEnv/2026.
    result = subprocess.run(['bash', '-lc', 'module load apptainer/1.4.5 && command -v apptainer'],
                            capture_output=True, text=True, timeout=60)
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip().splitlines()[-1]
    raise RuntimeError('No container runtime found. Tamia: module load StdEnv/2023 apptainer/1.4.5')


def runtime_name(requested):
    if requested == 'apptainer':
        apptainer_binary()
        return requested
    docker = shutil.which('docker')
    if docker:
        try:
            if subprocess.run([docker, 'info'], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, timeout=15).returncode == 0:
                return 'docker'
        except subprocess.TimeoutExpired:
            pass
    if requested == 'docker':
        raise RuntimeError('Docker is unavailable on this host. Tamia provides Apptainer; use --runtime apptainer.')
    apptainer_binary()
    return 'apptainer'


def image_name(runtime, value):
    if runtime == 'docker':
        return value or 'lightning-opd:h100'
    return str(Path(value or ARTIFACTS / 'lightning-opd-h100.sif').expanduser().resolve())


def source_context(destination):
    """Whitelist source and exclude credentials, data, environments and caches."""
    def ignore(directory, names):
        return [name for name in names if name.startswith('.env') or name in
                {'.git', '__pycache__', '.pytest_cache', '.ruff_cache'}
                or name.endswith(('.pyc', '.sif', '.sqsh'))]
    for name in SOURCE_ITEMS:
        source, target = ROOT / name, destination / name
        if source.is_dir():
            shutil.copytree(source, target, ignore=ignore)
        elif source.is_file():
            shutil.copy2(source, target)
    checksum = hashlib.sha256()
    for path in sorted(destination.rglob('*')):
        if path.is_file():
            checksum.update(str(path.relative_to(destination)).encode())
            checksum.update(path.read_bytes())
    return checksum.hexdigest()


def filesystem_tool(name):
    command = shutil.which(name)
    if command:
        return command
    candidate = Path('/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/bin') / name
    if candidate.is_file():
        return str(candidate)
    raise RuntimeError(f'{name} is required for inode-safe Apptainer builds; use the cluster software module.')


@contextlib.contextmanager
def extraction_directory():
    """Keep hundreds of thousands of OCI files inside one scratch file."""
    local = os.environ.get('SLURM_TMPDIR')
    if local and Path(local).is_dir() and shutil.disk_usage(local).free >= 100 * 1024 ** 3:
        with tempfile.TemporaryDirectory(prefix='opd-image-', dir=local) as directory:
            yield Path(directory)
        return
    directory = Path(tempfile.mkdtemp(prefix='build-', dir=ARTIFACTS))
    try:
        volume, mount = directory / 'temporary.ext4', directory / 'root'
        mount.mkdir()
        mkfs, fuse = filesystem_tool('mkfs.ext4'), filesystem_tool('fuse2fs')
        filesystem_tool('fusermount3')
        if not Path('/dev/fuse').exists():
            raise RuntimeError('Apptainer image conversion needs /dev/fuse for inode-safe temporary storage.')
        if shutil.disk_usage(ARTIFACTS).free < 100 * 1024 ** 3:
            raise RuntimeError('Apptainer image conversion requires at least 100 GiB free scratch space.')
        with volume.open('wb') as stream:
            stream.truncate(160 * 1024 ** 3)
        run([mkfs, '-q', '-F', '-N', '1000000', '-E',
             f'root_owner={os.getuid()}:{os.getgid()},lazy_itable_init=1,lazy_journal_init=1', volume])
        run([fuse, '-o', 'fakeroot', volume, mount])
        if not os.path.ismount(mount):
            raise RuntimeError('The temporary ext4 filesystem was not mounted.')
        yield mount
    finally:
        if os.path.ismount(directory / 'root'):
            run([filesystem_tool('fusermount3'), '-u', directory / 'root'])
            # fuse2fs may still have the unlinked volume open briefly.
            for _ in range(50):
                if not os.path.ismount(directory / 'root'):
                    break
                time.sleep(0.1)
        if os.path.ismount(directory / 'root'):
            raise RuntimeError(f'Build filesystem remains mounted at {directory}; preserve it for manual unmount.')
        shutil.rmtree(directory)


def base_layout():
    """Stage the public pinned OCI image before entering offline compute nodes."""
    repository, digest = BASE.split('@')
    for layout in (ARTIFACTS / 'oci', ARTIFACTS / 'cache/cache/blob'):
        manifest = layout / 'blobs/sha256' / digest.split(':')[1]
        if (layout / 'index.json').is_file() and (layout / 'oci-layout').is_file() and manifest.is_file() and \
                hashlib.sha256(manifest.read_bytes()).hexdigest() == digest.split(':')[1]:
            value = json.loads(manifest.read_text())
            blobs = [value['config'], *value['layers']]
            if all((layout / 'blobs/sha256' / blob['digest'].split(':')[1]).is_file() and
                   (layout / 'blobs/sha256' / blob['digest'].split(':')[1]).stat().st_size == blob['size']
                   for blob in blobs):
                return layout
    layout = ARTIFACTS / 'oci'
    storage = layout / 'blobs/sha256'
    storage.mkdir(parents=True, exist_ok=True)
    query = urllib.parse.urlencode({'service': 'registry.docker.io', 'scope': f'repository:{repository}:pull'})
    with urllib.request.urlopen('https://auth.docker.io/token?' + query, timeout=60) as response:
        token = json.load(response)['token']
    headers = {'Authorization': 'Bearer ' + token,
               'Accept': 'application/vnd.docker.distribution.manifest.v2+json'}
    url = 'https://registry-1.docker.io/v2/' + repository
    with urllib.request.urlopen(urllib.request.Request(url + '/manifests/' + digest, headers=headers), timeout=60) as response:
        manifest = response.read()
    if hashlib.sha256(manifest).hexdigest() != digest.split(':')[1]:
        raise RuntimeError('The base image manifest does not match its pinned digest.')
    value = json.loads(manifest)
    (storage / digest.split(':')[1]).write_bytes(manifest)
    for blob in [value['config'], *value['layers']]:
        path = storage / blob['digest'].split(':')[1]
        if path.is_file() and path.stat().st_size == blob['size']:
            continue
        print(f'Downloading OCI layer {blob["digest"][:23]} ({blob["size"] / 1024 ** 2:.0f} MiB)', flush=True)
        temporary = path.with_suffix('.downloading')
        try:
            checksum = hashlib.sha256()
            request = urllib.request.Request(url + '/blobs/' + blob['digest'], headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as stream:
                while chunk := response.read(4 * 1024 ** 2):
                    stream.write(chunk)
                    checksum.update(chunk)
            if checksum.hexdigest() != path.name or temporary.stat().st_size != blob['size']:
                raise RuntimeError('The downloaded OCI layer failed digest/size verification.')
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    (layout / 'oci-layout').write_text('{"imageLayoutVersion":"1.0.0"}\n')
    (layout / 'index.json').write_text(json.dumps({'schemaVersion': 2, 'manifests': [
        {'mediaType': value['mediaType'], 'size': len(manifest), 'digest': digest}]}))
    return layout


def build_base():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    base = ARTIFACTS / 'jetmoe-v0.2.sif'
    if base.is_file():
        return base
    layout = base_layout()
    if '.tamia.' in socket.getfqdn() and not os.environ.get('SLURM_JOB_ID'):
        run(['sbatch', '--wait', '--export=ALL', '--output=' + str(ARTIFACTS / 'build-%j.log'),
             ROOT / 'configs/sbatch/opd_container_build.sbatch'])
        if not base.is_file():
            raise RuntimeError('The CPU build did not produce the base image; inspect its build log.')
        return base
    pending = base.with_suffix('.sif.building')
    pending.unlink(missing_ok=True)
    try:
        with extraction_directory() as extraction:
            env = dict(os.environ, APPTAINER_TMPDIR=str(extraction))
            run([apptainer_binary(), 'build', '--disable-cache', '--fix-perms', '--mksquashfs-args',
                 '-processors 2 -mem 2G -comp zstd -Xcompression-level 1', pending,
                 'oci:' + str(layout) + '@' + BASE.split('@')[1]], env=env)
        pending.replace(base)
    finally:
        pending.unlink(missing_ok=True)
    return base


def build_image(runtime, image):
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with (ARTIFACTS / 'build.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # Snapshot sources after the potentially lengthy base conversion.
        base = build_base() if runtime == 'apptainer' else None
        with tempfile.TemporaryDirectory(prefix='source-', dir=ARTIFACTS) as directory:
            context = Path(directory)
            source_hash = source_context(context)
            (context / 'containers/source.json').write_text(json.dumps(
                {'base': BASE, 'source_sha256': source_hash}, indent=2) + '\n')
            if runtime == 'docker':
                run(['docker', 'build', '--progress=plain', '-t', image,
                     '-f', context / 'containers/Dockerfile', context])
            else:
                tool = apptainer_binary()
                env = os.environ.copy()
                for name in ('build', 'cache'):
                    (ARTIFACTS / name).mkdir(exist_ok=True)
                env['APPTAINER_TMPDIR'] = os.environ.get('SLURM_TMPDIR', '/tmp')
                env['APPTAINER_CACHEDIR'] = str(ARTIFACTS / 'cache')
                output = Path(image)
                output.parent.mkdir(parents=True, exist_ok=True)
                local = Path(os.environ.get('SLURM_TMPDIR', '/tmp'))
                if shutil.disk_usage(local).free < base.stat().st_size + 12 * 1024 ** 3:
                    raise RuntimeError('The image install needs at least 32 GiB free local temporary disk.')
                with tempfile.TemporaryDirectory(prefix='opd-image-install-', dir=local) as install:
                    temporary = Path(install) / 'image.sif'
                    # Local disk makes FUSE metadata writes practical and keeps
                    # package files out of the shared scratch inode quota.
                    run(['cp', '--reflink=auto', '--sparse=always', base, temporary])
                    run([tool, 'overlay', 'create', '--sparse', '--size', '8192', temporary], env=env)
                    # Rootless conversion owns files as the current user.
                    # Tamia forbids root mappings; no fakeroot is needed.
                    run([tool, 'exec', '--writable', '--cleanenv', '--no-home',
                         '--no-mount', 'cwd', '--pwd', '/', '--bind', f'{context}:/mnt:ro', temporary,
                         '/bin/bash', '-c', 'set -euo pipefail; mkdir -p /opt/Lightning-OPD; '
                         'cp -a --no-preserve=ownership /mnt/. /opt/Lightning-OPD/; '
                         'bash /opt/Lightning-OPD/containers/install.sh'], env=env)
                    pending = output.with_name(output.name + '.building')
                    pending.unlink(missing_ok=True)
                    try:
                        run(['cp', '--reflink=auto', '--sparse=always', temporary, pending])
                        pending.replace(output)
                    finally:
                        pending.unlink(missing_ok=True)
            manifest = {'image': image, 'runtime': runtime, 'base': BASE, 'source_sha256': source_hash}
            (ARTIFACTS / 'last-build.json').write_text(json.dumps(manifest, indent=2) + '\n')


def resolve_hosts(names):
    domain = socket.getfqdn().partition('.')[2]
    result = []
    for name in names:
        qualified = f'{name}.{domain}' if domain and '.' not in name else name
        address = socket.gethostbyname(qualified)
        if ipaddress.ip_address(address).is_loopback:
            raise ValueError(f'{qualified} resolves to loopback; a routable node address is required.')
        result.append((address, name))
    result.sort(key=lambda item: ipaddress.ip_address(item[0]))
    if len({ip for ip, _ in result}) != len(result):
        raise ValueError('Training nodes must have distinct IP addresses.')
    return result


def forwarded_environment(ips, rank, job):
    keys = {'CUDA_VISIBLE_DEVICES', 'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN',
            'NCCL_SOCKET_IFNAME', 'NCCL_IB_HCA', 'NCCL_IB_DISABLE', 'NCCL_IB_GID_INDEX',
            'NCCL_NET_GDR_LEVEL', 'GLOO_SOCKET_IFNAME', 'OMP_NUM_THREADS'}
    result = {key: value for key, value in os.environ.items()
              if key in keys or key.startswith(('SLURM_', 'WANDB_'))}
    result.update(LIGHTNING_OPD_REPO=str(ROOT), LIGHTNING_OPD_NODE_IPS=','.join(ips),
                  LIGHTNING_OPD_NODE_RANK=str(rank), LIGHTNING_OPD_JOB_ID=str(job))
    return result


def local_interface(address):
    """Avoid Gloo selecting a hostname mapped to loopback in /etc/hosts."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        for _, name in socket.if_nameindex():
            try:
                info = fcntl.ioctl(probe.fileno(), 0x8915, struct.pack('256s', name.encode()[:15]))
            except OSError:
                continue
            if socket.inet_ntoa(info[20:24]) == address:
                return name
    raise RuntimeError(f'The node IP {address} is not assigned to a local IPv4 interface.')


def container_command(runtime, image, payload, values, extra_binds=(), name=None):
    # Keep absolute YAML/model/data/checkpoint paths identical on every node.
    binds = [str(ROOT), '/tmp']
    if Path('/scratch').is_dir():
        binds.append('/scratch')
    binds.extend(extra_binds)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('APPTAINERENV_', 'SINGULARITYENV_'))
           and key not in ('APPTAINER_BIND', 'APPTAINER_BINDPATH', 'SINGULARITY_BINDPATH')}
    if runtime == 'docker':
        command = ['docker', 'run', '--rm', '--name', name or 'lightning-' + uuid.uuid4().hex,
                   '--network=host', '--ipc=host', '--uts=host', '--gpus=all', '--ulimit=memlock=-1',
                   '--user', f'{os.getuid()}:{os.getgid()}', '--workdir', str(ROOT),
                   '--entrypoint', '/bin/bash']
        for group in os.getgroups():
            if group != os.getgid():
                command += ['--group-add', str(group)]
        if Path('/dev/infiniband').exists():
            command += ['--device', '/dev/infiniband']
        for bind in dict.fromkeys(binds):
            command += ['--volume', bind if ':' in bind else f'{bind}:{bind}']
        env.update(values)
        for key in sorted(values):
            command += ['--env', key]  # Never expose secret values in argv/logs.
        command += [image]
    else:
        command = [apptainer_binary(), 'exec', '--nv', '--cleanenv', '--no-home', '--pwd', str(ROOT)]
        for bind in dict.fromkeys(binds):
            command += ['--bind', bind]
        env.update({'APPTAINERENV_' + key: value for key, value in values.items()})
        command += [image, '/bin/bash']
    command += [str(ROOT / 'containers/entrypoint.sh'), *payload]
    return command, env


def managed_container(command, env, docker_name=None):
    process = subprocess.Popen(command, env=env, cwd=ROOT, start_new_session=True)
    def stop(signum, frame):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        raise SystemExit(128 + signum)
    handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        status = process.wait()
        if status:
            raise subprocess.CalledProcessError(status, command)
    finally:
        if docker_name:
            # Only the uniquely named container created by this worker.
            subprocess.run(['docker', 'rm', '-f', docker_name], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=30)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def worker(cli, extra):
    ips = cli.node_ips.split(',') if cli.node_ips else []
    if len(ips) != 2 or len(set(ips)) != 2 or any(ipaddress.IPv4Address(ip).is_loopback for ip in ips):
        raise ValueError('Two distinct routable --node-ips are required.')
    if ips != sorted(ips, key=ipaddress.IPv4Address):
        raise ValueError('--node-ips must be in numeric IP order.')
    rank = cli.node_rank
    if rank is None:
        names = cli.node_names.split(',') if cli.node_names else []
        current = os.environ.get('SLURMD_NODENAME', socket.gethostname()).split('.')[0]
        try:
            rank = [name.split('.')[0] for name in names].index(current)
        except ValueError as error:
            raise ValueError(f'This node ({current}) is absent from the resolved allocation: {names}') from error
    if rank not in (0, 1):
        raise ValueError('--node-rank must be 0 or 1.')
    job = cli.job_id or os.environ.get('SLURM_JOB_ID')
    if not job or not str(job).isdigit():
        raise ValueError('A shared numeric --job-id or SLURM_JOB_ID is required on both nodes.')
    runtime = runtime_name(cli.runtime)
    image = image_name(runtime, cli.image)
    if runtime == 'apptainer' and not Path(image).is_file():
        raise FileNotFoundError(f'Container image is missing: {image}. Run bash run_docker.sh build first.')
    values = forwarded_environment(ips, rank, job)
    if not all(values.get(key) for key in ('GLOO_SOCKET_IFNAME', 'NCCL_SOCKET_IFNAME')):
        interface = local_interface(ips[rank])
        if not values.get('GLOO_SOCKET_IFNAME'):
            values['GLOO_SOCKET_IFNAME'] = interface
        if not values.get('NCCL_SOCKET_IFNAME'):
            values['NCCL_SOCKET_IFNAME'] = '=' + interface
    values['NCCL_CUMEM_ENABLE'] = '0'
    report = str(ARTIFACTS / 'checks' / str(job))
    if cli.stage in ('metadata', 'smoke-config', 'verify-smoke'):
        payload = ['python', str(ROOT / 'containers/session.py'), cli.stage, '--config', str(cli.config)]
        if cli.stage == 'metadata':
            payload += ['--experiment-name', cli.experiment_name]
    elif cli.stage == 'imports':
        payload = ['python', str(ROOT / 'containers/doctor.py'), '--imports', '--report-dir', report]
    elif cli.stage == 'distributed':
        payload = ['python', '-m', 'torch.distributed.run', '--nnodes=2', '--nproc-per-node=4',
                   f'--node-rank={rank}', f'--master-addr={ips[0]}',
                   f'--master-port={40000 + int(job) % 10000}',
                   str(ROOT / 'containers/doctor.py'), '--distributed', '--report-dir', report]
    else:
        flags = [] if cli.stage == 'train' else ['--' + cli.stage]
        payload = ['python', str(ROOT / 'containers/session.py'), 'launch', '--config', str(cli.config),
                   '--experiment-name', cli.experiment_name,
                   '--', *flags, *extra]
    name = f'lightning-{job}-{rank}-{uuid.uuid4().hex[:8]}' if runtime == 'docker' else None
    command, env = container_command(runtime, image, payload, values, cli.bind, name)
    managed_container(command, env, name)


def batch(cli, extra):
    if not os.environ.get('SLURM_JOB_NODELIST'):
        raise ValueError('batch requires a Slurm allocation. Use submit, or run with explicit node IPs/rank/job ID.')
    names = run(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']], capture=True).stdout.split()
    nodes = resolve_hosts(names)
    if len(nodes) != 2:
        raise ValueError(f'Expected two H100 nodes, got {len(nodes)}.')
    ips, names = map(list, zip(*nodes))
    runtime = runtime_name(cli.runtime)
    image = image_name(runtime, cli.image)
    def stage(name, count=2, capture=False):
        command = ['srun', '--mpi=none', f'--nodes={count}', f'--ntasks={count}', '--ntasks-per-node=1',
                   '--cpu-bind=none', '--gpu-bind=none', '--kill-on-bad-exit=1', '--unbuffered',
                   '--nodelist=' + ','.join(names[:count]), '/usr/bin/python3', '-I', '-S', '-B',
                   str(ROOT / 'containers/launch.py'), 'worker', '--runtime', runtime, '--image', image,
                   '--config', str(cli.config), '--node-ips', ','.join(ips), '--node-names', ','.join(names),
                   '--stage', name]
        if getattr(cli, 'experiment_name', None):
            command += ['--experiment-name', cli.experiment_name]
        for bind in cli.bind:
            command += ['--bind', bind]
        return run([*command, '--', *extra], capture=capture)
    metadata = json.loads(stage('metadata', 1, True).stdout)
    cli.config = Path(metadata.get('config', cli.config))
    if (metadata['nnodes'], metadata['n_gpus_per_node']) != (2, 4):
        raise ValueError('This runner allocates 2 nodes x 4 H100s; the YAML must match.')
    print(json.dumps({'runtime': runtime, 'image': image, 'nodes': nodes, 'recipe': metadata}), flush=True)
    modes = {'--check': 'check', '--dry-run': 'dry-run', '--prepare-data': 'prepare-data',
             '--preflight': 'preflight', '--convert-datasets': 'convert-datasets',
             '--prepare-prompts': 'convert-datasets'}
    single = [modes[flag] for flag in extra if flag in modes]
    if len(set(single)) > 1:
        raise ValueError('Choose only one single-stage mode.')
    if single:
        stage(single[0], 2 if single[0] == 'preflight' else 1)
        return
    stage('imports')
    compare_node_reports(os.environ['SLURM_JOB_ID'])
    if not cli.skip_distributed_check:
        stage('distributed')
    stage('preflight')
    if cli.validate_only:
        print('Two-node H100 container validation passed. Training was not started.', flush=True)
        return
    if cli.train_smoke:
        cli.config = Path(json.loads(stage('smoke-config', 1, True).stdout)['config'])
        cli.experiment_name = 'original'
        stage('preflight')
    if metadata['method'] == 'lightning_opd':
        stage('prepare-data', 1)
    stage('train')
    if cli.train_smoke:
        stage('verify-smoke', 1)


def compare_node_reports(job):
    directory = ARTIFACTS / 'checks' / str(job)
    reports = [json.loads((directory / f'imports-node-{rank}.json').read_text()) for rank in (0, 1)]
    for field in ('schema', 'base', 'source_sha256', 'versions', 'pip_freeze_sha256', 'tms_native_patch'):
        if reports[0].get(field) != reports[1].get(field):
            raise RuntimeError(f'The two nodes have different container {field}; use the same image on both.')


def submit(cli, extra):
    runtime = runtime_name(cli.runtime)
    image = image_name(runtime, cli.image)
    if runtime == 'apptainer' and not Path(image).is_file():
        build_image(runtime, image)
    elif runtime == 'docker' and subprocess.run(['docker', 'image', 'inspect', image],
                                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        build_image(runtime, image)
    (ROOT / 'tmp').mkdir(exist_ok=True)
    walltime = cli.time or ('00:30:00' if cli.validate_only else '00:45:00' if cli.train_smoke else '08:00:00')
    command = ['sbatch', '--parsable', '--export=ALL', '--time=' + walltime,
               '--job-name=' + ('opd-container-check' if cli.validate_only or cli.train_smoke else 'qwen3-1.7b-container'),
               str(ROOT / 'configs/sbatch/qwen3_h100_container.sbatch'),
               '--config', str(cli.config), '--runtime', runtime, '--image', image]
    command += ['--experiment-name', cli.experiment_name]
    if cli.validate_only:
        command += ['--validate-only']
    if cli.train_smoke:
        command += ['--train-smoke']
    if cli.skip_distributed_check:
        command += ['--skip-distributed-check']
    for bind in cli.bind:
        command += ['--bind', bind]
    job = run([*command, '--', *extra], capture=True).stdout.strip()
    print(f'Submitted {job}; logs: {ROOT}/tmp/slurm-*-{job.split(";")[0]}.log', flush=True)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if '--' in argv:
        separator = argv.index('--')
        argv, extra = argv[:separator], argv[separator + 1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'submit', 'batch', 'run', 'worker'])
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--experiment-name', default='auto',
                        help='auto adds _container; original keeps the YAML name; otherwise use the supplied name')
    parser.add_argument('--runtime', choices=['auto', 'docker', 'apptainer'], default='auto')
    parser.add_argument('--image')
    parser.add_argument('--base-only', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--bind', action='append', default=[])
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--train-smoke', action='store_true', help='Run one isolated training iteration/checkpoint/eval')
    parser.add_argument('--skip-distributed-check', action='store_true')
    parser.add_argument('--time', help='Slurm wall time, e.g. 00:30:00')
    parser.add_argument('--node-ips', help='Two IPv4 addresses in numeric order, comma separated')
    parser.add_argument('--node-names')
    parser.add_argument('--node-rank', type=int)
    parser.add_argument('--job-id')
    parser.add_argument('--stage', default='train', choices=['metadata', 'smoke-config', 'verify-smoke', 'imports', 'distributed', 'preflight',
                                                           'prepare-data', 'train', 'check', 'dry-run', 'convert-datasets'])
    cli = parser.parse_args(argv)
    if cli.runtime == 'auto' and cli.image and Path(cli.image).expanduser().is_file():
        cli.runtime = 'apptainer'
    if cli.validate_only and cli.train_smoke:
        parser.error('--validate-only and --train-smoke are mutually exclusive.')
    if (cli.validate_only or cli.train_smoke) and cli.skip_distributed_check:
        parser.error('Validation/smoke must include the distributed check.')
    uses_allocation = cli.action == 'batch' or (cli.action == 'run' and
                      os.environ.get('SLURM_JOB_ID') and cli.node_rank is None)
    if (cli.validate_only or cli.train_smoke) and cli.action != 'submit' and not uses_allocation:
        parser.error('Validation/smoke needs submit or a Slurm allocation; manual node runs use --stage distributed.')
    if (cli.validate_only or cli.train_smoke) and any(flag in extra for flag in
            ('--check', '--dry-run', '--prepare-data', '--preflight', '--convert-datasets', '--prepare-prompts')):
        parser.error('Validation/smoke cannot be combined with a single-stage mode.')
    cli.config = cli.config.expanduser().resolve()
    if cli.action == 'build':
        runtime = runtime_name(cli.runtime)
        if cli.base_only:
            build_base()
        else:
            build_image(runtime, image_name(runtime, cli.image))
    elif cli.action == 'submit':
        submit(cli, extra)
    elif cli.action == 'batch' or cli.action == 'run' and os.environ.get('SLURM_JOB_ID') and cli.node_rank is None:
        batch(cli, extra)
    else:
        worker(cli, extra)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        sys.exit(f'Container launch failed: {error}')
