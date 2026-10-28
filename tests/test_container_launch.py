"""Host orchestration, node mapping and credential isolation without GPUs."""

import importlib.util
import contextlib
import io
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = load('container_runner', 'containers/launch.py')


class ContainerLaunchTests(unittest.TestCase):
    def test_free_outside_a_region_handles_both_tms_and_ordinary_cuda_pointers(self):
        fix = load('container_memory_free_fix', 'containers/patch_memory_saver.py')
        compiler = shutil.which('g++')
        self.assertIsNotNone(compiler, 'The image needs a C++ compiler to build TorchMemorySaver.')
        # Exercise the exact intercepted function against CUDA/TMS API stubs.
        # Calling real cudaFree on a TMS VMM address models cudaErrorInvalidValue.
        program = '''using cudaError_t = int;
namespace APIForwarder {
cudaError_t call_real_cuda_free(void* ptr) { return ptr == reinterpret_cast<void*>(1) ? 9 : 0; }
}
struct TorchMemorySaver {
static TorchMemorySaver& instance() { static TorchMemorySaver value; return value; }
cudaError_t free(void* ptr) { return ptr == reinterpret_cast<void*>(1) ? 0 : APIForwarder::call_real_cuda_free(ptr); }
};
struct { bool is_interesting_region() { return false; } } thread_local_config;
''' + fix.ANCHOR + '''
int main() {
    auto tracked = cudaFree(reinterpret_cast<void*>(1));
    if (tracked) return tracked;
    return cudaFree(reinterpret_cast<void*>(2));
}
'''
        with tempfile.TemporaryDirectory() as directory:
            source, binary = Path(directory) / 'free.cpp', Path(directory) / 'free'
            for code, expected in ((program, 9), (fix.patched_source(program), 0)):
                source.write_text(code)
                subprocess.run([compiler, '-std=c++17', str(source), '-o', str(binary)],
                               check=True, capture_output=True, timeout=30)
                result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)

    def test_memory_free_patch_is_idempotent_and_rejects_unexpected_source(self):
        fix = load('container_memory_exit_guard', 'containers/patch_memory_saver.py')
        source = fix.patched_source(fix.ANCHOR)
        self.assertEqual(fix.patched_source(source), source)
        with self.assertRaisesRegex(RuntimeError, 'Unexpected TorchMemorySaver'):
            fix.patched_source('unrecognized source')

    def test_memory_hook_is_found_in_the_inherited_cuda_stack(self):
        doctor = load('container_memory_doctor', 'containers/doctor.py')
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory)
            package = library / 'torch_memory_saver'
            package.mkdir()
            hook = library / 'torch_memory_saver_hook_mode_preload.abi3.so'
            hook.write_bytes(b'test-hook')
            with patch.dict(os.environ, {}, clear=True), \
                    patch.object(doctor.importlib.util, 'find_spec', return_value=types.SimpleNamespace(origin=str(package / '__init__.py'))), \
                    patch.object(doctor.os, 'execve') as execute:
                doctor.preload_memory_hook()
            self.assertEqual(execute.call_args.args[2]['LD_PRELOAD'], str(hook))
            self.assertEqual(execute.call_args.args[2]['LIGHTNING_OPD_CHECK_PRELOADED'], '1')

    def test_container_experiment_forks_only_the_name_and_preserves_original_results(self):
        import yaml

        session = load('container_name_session', 'containers/session.py')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'original.yaml'
            original = yaml.safe_load(RUNNER.DEFAULT_CONFIG.read_text())
            text = yaml.safe_dump(original)
            config.write_text(text)
            def metadata(path, name):
                output = io.StringIO()
                with patch.object(session, 'ROOT', root), \
                        patch.dict(os.environ, {'LIGHTNING_OPD_JOB_ID': '123'}), \
                        patch.object(session.sys, 'argv', ['session', 'metadata', '--config', str(path),
                                                          '--experiment-name', name]), \
                        contextlib.redirect_stdout(output):
                    session.main()
                return json.loads(output.getvalue())
            report = metadata(config, 'auto')
            generated = Path(report['config'])
            expected = dict(original, experiment_name=original['experiment_name'] + '_container')
            self.assertEqual(yaml.safe_load(generated.read_text()), expected)
            self.assertEqual(config.read_text(), text)
            self.assertEqual(metadata(generated, 'auto')['experiment_name'], expected['experiment_name'])
            self.assertEqual(metadata(config, 'original')['config'], str(config))

    def test_apptainer_build_uses_the_ordinary_user_without_root_mappings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / 'repo'
            (repo / 'containers').mkdir(parents=True)
            artifacts = root / 'artifacts'
            artifacts.mkdir()
            base = artifacts / 'base.sif'
            base.write_bytes(b'base')
            image = artifacts / 'result.sif'
            commands = []
            def run(command, **kwargs):
                commands.append(command)
                if command[0] == 'cp':
                    Path(command[-1]).write_bytes(b'assembled')
            with patch.object(RUNNER, 'ROOT', repo), patch.object(RUNNER, 'ARTIFACTS', artifacts), \
                    patch.object(RUNNER, 'build_base', return_value=base), \
                    patch.object(RUNNER, 'apptainer_binary', return_value='/apptainer'), \
                    patch.object(RUNNER, 'run', side_effect=run):
                RUNNER.build_image('apptainer', str(image))
            self.assertTrue(image.is_file())
            self.assertFalse(any('--fakeroot' in command for command in commands))
            creation = next(command for command in commands if 'create' in command)
            self.assertIn('--sparse', creation)
            execution = next(command for command in commands if 'exec' in command)
            self.assertIn('--writable', execution)

    def test_cached_base_build_is_offline_and_selects_the_pinned_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            layout = root / 'oci'
            blobs = layout / 'blobs/sha256'
            blobs.mkdir(parents=True)
            (layout / 'index.json').write_text('{}')
            (layout / 'oci-layout').write_text('{"imageLayoutVersion":"1.0.0"}')
            config = b'{"architecture":"amd64"}'
            config_hash = hashlib.sha256(config).hexdigest()
            (blobs / config_hash).write_bytes(config)
            manifest = json.dumps({'config': {'digest': 'sha256:' + config_hash,
                                              'size': len(config)}, 'layers': []}).encode()
            digest = 'sha256:' + hashlib.sha256(manifest).hexdigest()
            (blobs / digest.split(':')[1]).write_bytes(manifest)
            commands = []
            def run(command, **kwargs):
                commands.append(command)
                Path(command[-2]).write_bytes(b'test-sif')
            with patch.object(RUNNER, 'ARTIFACTS', root), \
                    patch.object(RUNNER, 'BASE', 'test/image@' + digest), \
                    patch.object(RUNNER.socket, 'getfqdn', return_value='compute.example.test'), \
                    patch.object(RUNNER, 'apptainer_binary', return_value='/apptainer'), \
                    patch.object(RUNNER, 'extraction_directory', return_value=contextlib.nullcontext(root)), \
                    patch.object(RUNNER.urllib.request, 'urlopen', side_effect=AssertionError('Network is unavailable')), \
                    patch.object(RUNNER, 'run', side_effect=run):
                image = RUNNER.build_base()
            self.assertTrue(image.is_file())
            self.assertEqual(commands[0][-1], 'oci:' + str(layout) + '@' + digest)
            self.assertIn('--disable-cache', commands[0])
            self.assertFalse(image.with_suffix('.sif.building').exists())

    def test_docker_has_host_network_mounts_gpu_access_and_no_secret_argv(self):
        values = RUNNER.forwarded_environment(['10.1.1.1', '10.1.1.2'], 1, '123')
        values['WANDB_API_KEY'] = 'secret-for-unit-test'
        command, env = RUNNER.container_command('docker', 'test:image', ['python', 'train.py'], values, name='unit-test')
        self.assertIn('--network=host', command)
        self.assertIn('--ipc=host', command)
        self.assertIn('--gpus=all', command)
        self.assertIn(f'{ROOT}:{ROOT}', command)
        self.assertNotIn(values['WANDB_API_KEY'], ' '.join(command))
        self.assertEqual(env['WANDB_API_KEY'], values['WANDB_API_KEY'])
        self.assertEqual(env['LIGHTNING_OPD_NODE_RANK'], '1')
        self.assertNotIn('--privileged', command)

    def test_apptainer_cleans_host_python_and_preserves_gpu_rank_and_slurm(self):
        with patch.dict(os.environ, {'APPTAINERENV_LD_PRELOAD': '/host/bad.so', 'SLURM_JOB_ID': '123'}), \
                patch.object(RUNNER, 'apptainer_binary', return_value='/bin/apptainer'):
            values = RUNNER.forwarded_environment(['10.1.1.1', '10.1.1.2'], 0, '123')
            command, env = RUNNER.container_command('apptainer', '/image.sif', ['python', 'train.py'], values)
        for flag in ('--nv', '--cleanenv', '--no-home'):
            self.assertIn(flag, command)
        self.assertNotIn('APPTAINERENV_LD_PRELOAD', env)
        self.assertEqual(env['APPTAINERENV_SLURM_JOB_ID'], '123')
        self.assertEqual(env['APPTAINERENV_LIGHTNING_OPD_NODE_IPS'], '10.1.1.1,10.1.1.2')

    def test_image_context_excludes_credentials_and_host_environments(self):
        with tempfile.TemporaryDirectory() as directory:
            source, context = Path(directory) / 'repo', Path(directory) / 'context'
            (source / 'configs/config').mkdir(parents=True)
            context.mkdir()
            for name in ['.env', 'configs/config/.env', 'configs/config/recipe.yaml', 'train.py']:
                (source / name).write_text('private' if '.env' in name else 'source')
            (source / '.runtime').mkdir()
            (source / '.runtime/secret').write_text('host dependency')
            with patch.object(RUNNER, 'ROOT', source):
                digest = RUNNER.source_context(context)
            self.assertEqual(len(digest), 64)
            self.assertTrue((context / 'configs/config/recipe.yaml').exists())
            self.assertFalse((context / '.env').exists())
            self.assertFalse((context / 'configs/config/.env').exists())
            self.assertFalse((context / '.runtime').exists())

    def test_host_resolution_uses_fqdn_and_numeric_ip_order(self):
        addresses = {'node-a.example.test': '10.1.1.20', 'node-b.example.test': '10.1.1.3'}
        with patch.object(RUNNER.socket, 'getfqdn', return_value='login.example.test'), \
                patch.object(RUNNER.socket, 'gethostbyname', side_effect=addresses.__getitem__):
            self.assertEqual(RUNNER.resolve_hosts(['node-a', 'node-b']),
                             [('10.1.1.3', 'node-b'), ('10.1.1.20', 'node-a')])
        with patch.object(RUNNER.socket, 'getfqdn', return_value='login.example.test'), \
                patch.object(RUNNER.socket, 'gethostbyname', return_value='127.0.1.1'):
            with self.assertRaisesRegex(ValueError, 'loopback'):
                RUNNER.resolve_hosts(['node-a'])

    def test_network_interface_uses_the_routable_address_instead_of_the_hostname(self):
        responses = [b'\0' * 20 + RUNNER.socket.inet_aton('127.0.0.1'),
                     b'\0' * 20 + RUNNER.socket.inet_aton('10.1.1.2')]
        with patch.object(RUNNER.socket, 'if_nameindex', return_value=[(1, 'lo'), (2, 'eno1')]), \
                patch.object(RUNNER.fcntl, 'ioctl', side_effect=responses):
            self.assertEqual(RUNNER.local_interface('10.1.1.2'), 'eno1')

    def batch(self, *, validate=False, extra=(), fail='', topology=(2, 4)):
        cli = types.SimpleNamespace(runtime='apptainer', image='/image.sif', config=Path('/config.yaml'),
                                    bind=[], validate_only=validate, train_smoke=False, skip_distributed_check=False)
        calls = []
        def run(command, **kwargs):
            calls.append(command)
            if command[0] == 'scontrol':
                return types.SimpleNamespace(stdout='node-a\nnode-b\n')
            stage = command[command.index('--stage') + 1]
            if stage == fail:
                raise subprocess.CalledProcessError(7, command)
            metadata = dict(method='lightning_opd', nnodes=topology[0], n_gpus_per_node=topology[1], experiment_name='test')
            return types.SimpleNamespace(stdout=json.dumps(metadata) if stage == 'metadata' else '')
        with patch.dict(os.environ, {'SLURM_JOB_NODELIST': 'node-[a,b]', 'SLURM_JOB_ID': '123'}), \
                patch.object(RUNNER, 'runtime_name', return_value='apptainer'), \
                patch.object(RUNNER, 'resolve_hosts', return_value=[('10.1.1.1', 'node-a'), ('10.1.1.2', 'node-b')]), \
                patch.object(RUNNER, 'compare_node_reports'), \
                patch.object(RUNNER, 'run', side_effect=run):
            RUNNER.batch(cli, list(extra))
        return [call for call in calls if call[0] == 'srun']

    def test_training_validates_both_nodes_before_preparing_and_training(self):
        calls = self.batch()
        self.assertEqual([call[call.index('--stage') + 1] for call in calls],
                         ['metadata', 'imports', 'distributed', 'preflight', 'prepare-data', 'train'])
        for call in calls:
            stage = call[call.index('--stage') + 1]
            self.assertIn('--nodes=1' if stage in ('metadata', 'prepare-data') else '--nodes=2', call)
            self.assertIn('/usr/bin/python3', call)
            self.assertIn('-S', call)
            self.assertNotIn('-c', call)

    def test_validation_only_never_starts_training(self):
        calls = self.batch(validate=True)
        self.assertEqual([call[call.index('--stage') + 1] for call in calls],
                         ['metadata', 'imports', 'distributed', 'preflight'])

    def test_bad_topology_and_failed_communication_stop_before_training(self):
        with self.assertRaisesRegex(ValueError, '2 nodes x 4'):
            self.batch(topology=(1, 8))
        with self.assertRaises(subprocess.CalledProcessError):
            self.batch(fail='distributed')

    def test_mismatched_node_images_are_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(RUNNER, 'ARTIFACTS', Path(directory)):
            reports = Path(directory) / 'checks/123'
            reports.mkdir(parents=True)
            first = dict(schema=1, base='test-base', source_sha256='same', versions={'torch': '2.9.0'},
                         pip_freeze_sha256='same-versions')
            for rank in (0, 1):
                (reports / f'imports-node-{rank}.json').write_text(json.dumps(first))
            RUNNER.compare_node_reports('123')
            (reports / 'imports-node-1.json').write_text(json.dumps(dict(first, source_sha256='different')))
            with self.assertRaisesRegex(RuntimeError, 'different container'):
                RUNNER.compare_node_reports('123')
            (reports / 'imports-node-1.json').write_text(json.dumps(dict(first, pip_freeze_sha256='different-versions')))
            with self.assertRaisesRegex(RuntimeError, 'pip_freeze_sha256'):
                RUNNER.compare_node_reports('123')
            (reports / 'imports-node-1.json').write_text(json.dumps(dict(first, tms_native_patch={'library_sha256': 'different'})))
            with self.assertRaisesRegex(RuntimeError, 'tms_native_patch'):
                RUNNER.compare_node_reports('123')

    def test_container_host_mapping_does_not_call_scontrol(self):
        launcher = load('container_cluster_launcher', 'configs/opd/qwen3-4b-opd.py')
        with patch.dict(os.environ, {'LIGHTNING_OPD_NODE_IPS': '10.1.1.1,10.1.1.2',
                                     'LIGHTNING_OPD_NODE_RANK': '1', 'SLURM_JOB_NODELIST': 'ignored'}), \
                patch.object(launcher.subprocess, 'check_output') as scontrol:
            self.assertEqual(launcher.cluster_nodes({'nnodes': 2}), (['10.1.1.1', '10.1.1.2'], 1))
            scontrol.assert_not_called()

    def test_validation_cannot_skip_the_communication_check(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            RUNNER.main(['submit', '--validate-only', '--skip-distributed-check'])
        self.assertEqual(error.exception.code, 2)

    def test_default_shell_entrypoint_uses_system_python_and_submits(self):
        with tempfile.TemporaryDirectory() as directory:
            wrapper = Path(directory) / 'python'
            log = Path(directory) / 'args'
            wrapper.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$TEST_ARGUMENT_LOG"\n')
            wrapper.chmod(0o755)
            env = dict(os.environ, LIGHTNING_OPD_HOST_PYTHON=str(wrapper), TEST_ARGUMENT_LOG=str(log))
            subprocess.run(['bash', str(ROOT / 'run_docker.sh')], check=True, env=env)
            args = log.read_text().splitlines()
            self.assertEqual(args[:3], ['-I', '-S', '-B'])
            self.assertEqual(args[-1], 'submit')

    def test_smoke_uses_a_separate_recipe_and_requires_checkpoint_and_final_eval(self):
        import yaml
        import pyarrow as pa
        import pyarrow.parquet as pq

        session = load('container_test_session', 'containers/session.py')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = root / 'amc23.parquet'
            pq.write_table(pa.table({'prompt': ['p1', 'p2', 'p3'], 'label': ['1', '2', '3']}), evaluation)
            original = yaml.safe_load(RUNNER.DEFAULT_CONFIG.read_text())
            original.update(epoch=2, num_rollout=240, num_rollout_per_epoch=120)
            original['val_files'] = [str(evaluation)]
            config = root / 'original.yaml'
            text = yaml.safe_dump(original)
            config.write_text(text)
            with patch.object(session, 'ROOT', root), \
                    patch.dict(os.environ, {'LIGHTNING_OPD_JOB_ID': '123'}), \
                    patch.object(session.sys, 'argv', ['session', 'smoke-config', '--config', str(config)]), \
                    contextlib.redirect_stdout(io.StringIO()):
                session.main()
            smoke = root / 'tmp/containers/checks/123/smoke/config.yaml'
            cfg = yaml.safe_load(smoke.read_text())
            self.assertEqual(config.read_text(), text)
            self.assertEqual((cfg['nnodes'], cfg['n_gpus_per_node']), (2, 4))
            self.assertEqual((cfg['num_rollout'], cfg['global_batch_size']), (1, 8))
            from slime.utils.training_epochs import normalize_epochs
            self.assertEqual(normalize_epochs(cfg)['num_rollout'], 1)
            self.assertEqual((cfg['epoch'], cfg['num_rollout_per_epoch']), (1, 1))
            self.assertEqual(cfg['lr_warmup_iters'], 0)
            self.assertEqual(cfg['save_model'], 'final')
            self.assertNotEqual(cfg['experiment_name'], original['experiment_name'])
            self.assertEqual(pq.ParquetFile(cfg['val_files'][0]).metadata.num_rows, 2)
            with patch.object(session.sys, 'argv', ['session', 'verify-smoke', '--config', str(smoke)]):
                with self.assertRaises(RuntimeError):
                    session.main()
                output = Path(cfg['output_dir']) / cfg['experiment_name']
                (output / 'checkpoints').mkdir(parents=True)
                (output / 'checkpoints/latest_checkpointed_iteration.txt').write_text('0')
                (output / 'eval_metrics').mkdir()
                (output / 'eval_metrics/0.json').write_text('{}')
                with contextlib.redirect_stdout(io.StringIO()):
                    session.main()
            self.assertTrue((smoke.parent / 'training-passed.json').is_file())


if __name__ == '__main__':
    unittest.main()
