"""Backport the upstream cudaFree routing fix to pinned TorchMemorySaver 0.0.9.

0.0.9 sends cudaFree straight to CUDA when outside an interesting region,
including MemPool destruction. TMS-owned virtual addresses must instead pass
through its ownership lookup; unowned addresses are forwarded there as usual.
Upstream fix: https://github.com/fzyzcjy/torch_memory_saver/commit/7078bc36ebc926ced1cd6d314b869c0a0d3d1f18
"""

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import urllib.request


# Match the base image's direct_url.json, including its memory-margin API.
# The PyPI 0.0.9 sdist predates that Git revision despite sharing its version.
SOURCE_COMMIT = '9b8b788fdeb9c2ee528183214cef65a99b71e7d5'
UPSTREAM_FIX = '7078bc36ebc926ced1cd6d314b869c0a0d3d1f18'
SOURCE_URL = 'https://codeload.github.com/fzyzcjy/torch_memory_saver/tar.gz/' + SOURCE_COMMIT
SOURCE_SHA256 = 'd06d96a6a3b17fe4b3cba2bbbd23939d6adc30aa74d2daa4606a6bc036697c69'
BUILD_MARKER = Path('/opt/lightning-tms-build.json')
MARKER = '// LIGHTNING_OPD_TMS_TRACKED_FREE'
FIX_NAME = 'cuda-free-owned-pointer-outside-region'
ANCHOR = '''cudaError_t cudaFree(void *ptr) {
    if (thread_local_config.is_interesting_region()) {
        return TorchMemorySaver::instance().free(ptr);
    } else {
        return APIForwarder::call_real_cuda_free(ptr);
    }
}'''
REPLACEMENT = '''cudaError_t cudaFree(void *ptr) {
    ''' + MARKER + '''
    return TorchMemorySaver::instance().free(ptr);
}'''


def patched_source(source):
    if MARKER in source:
        return source
    if source.count(ANCHOR) != 1:
        raise RuntimeError('Unexpected TorchMemorySaver source; refusing to apply the cudaFree fix.')
    return source.replace(ANCHOR, REPLACEMENT, 1)


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-archive', type=Path, help='Use a cached source archive without network access')
    cli = parser.parse_args()
    if metadata.version('torch-memory-saver') != '0.0.9':
        raise RuntimeError('This backport is verified only for torch-memory-saver==0.0.9.')
    if BUILD_MARKER.is_file():
        record = json.loads(BUILD_MARKER.read_text())
        library = Path(record['library'])
        if (record.get('fix') == FIX_NAME and record.get('source_sha256') == SOURCE_SHA256
                and library.is_file() and checksum(library) == record['library_sha256']):
            print('TorchMemorySaver native cudaFree fix already installed', flush=True)
            return
    with tempfile.TemporaryDirectory(prefix='opd-tms-build-') as directory:
        build = Path(directory)
        archive = cli.source_archive
        if archive is None:
            archive = build / 'source.tar.gz'
            urllib.request.urlretrieve(SOURCE_URL, archive)
        if checksum(archive) != SOURCE_SHA256:
            raise RuntimeError('TorchMemorySaver source archive failed SHA256 verification.')
        with tarfile.open(archive) as source:
            # Python < 3.11.4 (e.g. Debian's 3.11.2) has no extraction filters; the archive is SHA256-pinned.
            if hasattr(tarfile, 'data_filter'):
                source.extractall(build, filter='data')
            else:
                source.extractall(build)
        package = build / ('torch_memory_saver-' + SOURCE_COMMIT)
        entrypoint = package / 'csrc/entrypoint.cpp'
        entrypoint.write_text(patched_source(entrypoint.read_text()))
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir',
                        '--disable-pip-version-check', '--no-deps',
                        '--no-build-isolation', '--force-reinstall', str(package)], check=True)
    package = metadata.distribution('torch-memory-saver')
    library = Path(package.locate_file('torch_memory_saver_hook_mode_preload.abi3.so')).resolve()
    import site
    roots = [Path(sys.prefix).resolve(), Path(site.getusersitepackages()).resolve()]  # non-root hosts use --user
    if not any(library.is_relative_to(root) for root in roots):
        raise RuntimeError('The patched TorchMemorySaver must be installed inside the image Python environment.')
    record = {'fix': FIX_NAME, 'version': package.version, 'source_sha256': SOURCE_SHA256,
              'base_commit': SOURCE_COMMIT, 'upstream_fix': UPSTREAM_FIX,
              'library': str(library), 'library_sha256': checksum(library)}
    BUILD_MARKER.write_text(json.dumps(record, indent=2) + '\n')
    manifest = Path('/opt/lightning-container.json')
    if manifest.is_file():
        value = json.loads(manifest.read_text())
        value['tms_native_patch'] = record
        manifest.write_text(json.dumps(value, indent=2) + '\n')
    print('TorchMemorySaver native cudaFree fix installed: ' + str(library), flush=True)


if __name__ == '__main__':
    main()
