"""In-container adapter for the repository's YAML launcher."""

import argparse
import copy
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile

import yaml

ROOT = Path(os.environ.get('LIGHTNING_OPD_REPO', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))


def main():
    arguments = sys.argv[1:]
    extra = []
    if '--' in arguments:
        separator = arguments.index('--')
        arguments, extra = arguments[:separator], arguments[separator + 1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['metadata', 'launch', 'smoke-config', 'verify-smoke'])
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--experiment-name', default='original')
    cli = parser.parse_args(arguments)
    cfg = yaml.safe_load(cli.config.read_text())
    if cfg.get('method') not in ('lightning_opd', 'opd'):
        raise ValueError('The H100 container runner requires an OPD/Lightning OPD YAML.')
    if cli.action in ('metadata', 'launch') and cli.experiment_name != 'original':
        name = cli.experiment_name
        if name == 'auto':
            name = cfg['experiment_name']
            if not name.endswith('_container'):
                name += '_container'
        if not name or Path(name).name != name or name in ('.', '..'):
            raise ValueError('--experiment-name must be a single directory name.')
        if name != cfg['experiment_name']:
            cfg = copy.deepcopy(cfg)
            cfg['experiment_name'] = name
            directory = ROOT / 'tmp/containers/checks' / os.environ['LIGHTNING_OPD_JOB_ID']
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / 'run-config.yaml'
            with tempfile.NamedTemporaryFile(mode='w', dir=directory, delete=False) as stream:
                yaml.safe_dump(cfg, stream, sort_keys=False)
                temporary = Path(stream.name)
            temporary.replace(target)
            cli.config = target
    if cli.action == 'metadata':
        print(json.dumps({'config': str(cli.config), **{key: cfg[key] for key in
                         ('method', 'nnodes', 'n_gpus_per_node', 'experiment_name')}}))
        return
    if cli.action == 'smoke-config':
        import pyarrow.parquet as pq

        cfg = copy.deepcopy(cfg)
        directory = ROOT / 'tmp/containers/checks' / os.environ['LIGHTNING_OPD_JOB_ID'] / 'smoke'
        directory.mkdir(parents=True, exist_ok=True)
        cfg.update(experiment_name='container-smoke-' + os.environ['LIGHTNING_OPD_JOB_ID'],
                   output_dir=str(directory / 'runs'), resume=False, use_wandb=False,
                   epoch=1, num_rollout_per_epoch=1,
                   num_rollout=1, rollout_batch_size=8, global_batch_size=8, max_train_samples=8,
                   lr_warmup_iters=0,
                   rollout_max_response_len=32, max_tokens_per_gpu=4096,
                   eval_steps=1, val_n=1, val_max_completion_length=32,
                   save_model='final', save_interval=1, save_eval_multiplier=1,
                   save_retain_interval=None, save_total_limit=1, async_save=False)
        if cfg['method'] == 'lightning_opd':
            cfg['policy_rollout_k'] = 1
            cfg['lightning_fixed_data'] = str(directory / 'fixed_prompts.parquet')
        evaluation = directory / 'eval'
        evaluation.mkdir(exist_ok=True)
        files = []
        for value in cfg['val_files']:
            source = Path(value)
            target = evaluation / source.name
            pq.write_table(pq.read_table(source).slice(0, 2), target)
            files.append(str(target))
        cfg['val_files'] = files
        output = directory / 'config.yaml'
        output.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(json.dumps({'config': str(output)}))
        return
    if cli.action == 'verify-smoke':
        output = Path(cfg['output_dir']) / cfg['experiment_name']
        tracker = output / 'checkpoints/latest_checkpointed_iteration.txt'
        final_eval = output / 'eval_metrics/0.json'
        if not tracker.is_file() or tracker.read_text().strip() != '0' or not final_eval.is_file():
            raise RuntimeError('The training smoke did not finish its update, checkpoint and final evaluation.')
        report = dict(config=str(cli.config), checkpoint_tracker=str(tracker), final_eval=str(final_eval))
        (cli.config.parent / 'training-passed.json').write_text(json.dumps(report, indent=2) + '\n')
        print('Two-node training smoke passed: ' + json.dumps(report), flush=True)
        return
    sys.argv = [str(Path(__file__)), '--config', str(cli.config), *extra]
    importlib.import_module('configs.opd.qwen3-4b-opd').main(method=cfg['method'])


if __name__ == '__main__':
    main()
