"""Sync this experiment's offline WandB runs using the same YAML and .env."""

import argparse
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from slime.utils.wandb_metrics import EVAL_STEP_METRIC, EXCLUDED_WANDB_PREFIXES, filter_wandb_metrics
from slime.utils.wandb_run_ids import resolve_wandb_run_id


def eval_metric_axis_definitions(definitions, metric_names, fields, axis_name=EVAL_STEP_METRIC):
    """Restore Avg@1 axes without moving indices used by other metrics."""
    result = deepcopy(definitions)
    name_key = fields["name"]
    glob_key = fields["glob_name"]
    step_key = fields["step_metric"]
    index_key = fields["step_metric_index"]
    names = {name for name in metric_names if name.startswith("eval_metric/")}
    names.update(metric[name_key] for metric in result
                 if metric.get(name_key, "").startswith("eval_metric/"))
    if not names:
        return result
    axis_index = next((i + 1 for i, metric in enumerate(result)
                       if metric.get(name_key) == axis_name), None)
    if axis_index is None:
        result.append({name_key: axis_name})
        axis_index = len(result)
    if "options" in fields:
        options = result[axis_index - 1].setdefault(fields["options"], [])
        if fields["hidden_option"] not in options:
            options.append(fields["hidden_option"])
    for metric in result:
        name = metric.get(name_key, metric.get(glob_key, ""))
        if name.startswith("eval_metric/"):
            metric.pop(step_key, None)
            metric[index_key] = axis_index
            if "step_sync_option" in fields:
                options = metric.setdefault(fields["options"], [])
                if fields["step_sync_option"] in options:
                    options.remove(fields["step_sync_option"])
    existing = {metric.get(name_key) for metric in result}
    for name in sorted(names - existing):
        result.append({name_key: name, index_key: axis_index})
    return result


def hide_excluded_metric_definitions(definitions, metric_names, fields):
    """Hide previously recorded diagnostics without changing retained axes."""
    result = deepcopy(definitions)
    name_key, glob_key = fields["name"], fields["glob_name"]
    options_key, hidden = fields["options"], fields["hidden_option"]
    existing = {metric.get(name_key): metric for metric in result if name_key in metric}
    for metric in result:
        if metric.get(name_key, metric.get(glob_key, "")).startswith(EXCLUDED_WANDB_PREFIXES):
            options = metric.setdefault(options_key, [])
            if hidden not in options:
                options.append(hidden)
    for name in sorted(set(metric_names) - set(existing)):
        if name.startswith(EXCLUDED_WANDB_PREFIXES):
            result.append({name_key: name, options_key: [hidden]})
    return result


def deduplicate_eval_metrics(metrics, seen):
    """Keep one score per dataset/completed step across streams and syncs."""
    retained = dict(metrics)
    step = retained.get(EVAL_STEP_METRIC)
    for key, value in metrics.items():
        if not key.startswith("eval_metric/"):
            continue
        if step is None:
            raise ValueError("Evaluation metric lacks its completed-step axis")
        identity = (step, key)
        if identity in seen:
            if seen[identity] != value:
                raise ValueError(f"Conflicting evaluation score at step {step}: {key}")
            del retained[key]
        else:
            seen[identity] = value
    if not any(key.startswith("eval_metric/") for key in retained):
        retained.pop(EVAL_STEP_METRIC, None)
    return retained


def filtered_run_file(log, destination, seen_eval_points=None):
    """Upload a filtered snapshot; never edit a training process's original log."""
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    target = destination / log.parent.name / log.name
    target.parent.mkdir(parents=True)
    if (log.parent / "files").exists():
        shutil.copytree(log.parent / "files", target.parent / "files")
    source, writer = DataStore(), DataStore()
    source.open_for_scan(str(log))
    writer.open_for_write(str(target))
    size = log.stat().st_size
    metric_names = []
    if seen_eval_points is None:
        seen_eval_points = {}
    try:
        while source.get_offset() < size:
            try:
                data = source.scan_data()
            except AssertionError:
                if source.in_last_block():
                    break
                raise
            if data is None or source.get_offset() > size:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            kind = record.WhichOneof("record_type")
            if kind == "stats":
                continue
            if kind in {"history", "summary"}:
                items = record.history.item if kind == "history" else record.summary.update
                values = {(item.key or "/".join(item.nested_key)): json.loads(item.value_json)
                          for item in items}
                retained = filter_wandb_metrics(values)
                if kind == "history":
                    retained = deduplicate_eval_metrics(retained, seen_eval_points)
                elif seen_eval_points:
                    # Replaying an older stream must not regress the summary.
                    latest = {}
                    for (step, key), value in seen_eval_points.items():
                        if key not in latest or step > latest[key][0]:
                            latest[key] = (step, value)
                    for key in list(retained):
                        if key.startswith("eval_metric/") and key in latest:
                            retained[key] = latest[key][1]
                    if EVAL_STEP_METRIC in retained:
                        retained[EVAL_STEP_METRIC] = max(step for step, key in seen_eval_points)
                if kind == "history" and not any(
                    not key.startswith("_") and key not in {"train/step", "rollout/step", EVAL_STEP_METRIC}
                    for key in retained
                ):
                    continue
                del items[:]
                for key, value in retained.items():
                    item = items.add()
                    item.key, item.value_json = key, json.dumps(value)
            elif kind == "metric":
                metric = record.metric
                original_name = metric.name or metric.glob_name
                if metric.name and metric.name not in metric_names:
                    metric_names.append(metric.name)
                if original_name == "eval/step":
                    metric.name = EVAL_STEP_METRIC
                    metric.options.hidden = True
                elif original_name.startswith(EXCLUDED_WANDB_PREFIXES):
                    continue
                if metric.step_metric_index:
                    metric.step_metric = metric_names[metric.step_metric_index - 1]
                    metric.ClearField("step_metric_index")
                if metric.step_metric == "eval/step":
                    metric.step_metric = EVAL_STEP_METRIC
            writer.write(record)
    finally:
        source.close()
        writer.close()
    return target


def eval_metric_names(history_keys, prefixes=("eval_metric/",)):
    """Read W&B's history-key catalogue, including older JSON-string responses."""
    if isinstance(history_keys, str):
        history_keys = json.loads(history_keys)
    names = set()
    if isinstance(history_keys, dict):
        for key, value in history_keys.items():
            if key.startswith(prefixes):
                names.add(key)
            if isinstance(value, (dict, list)):
                names.update(eval_metric_names(value, prefixes))
    elif isinstance(history_keys, list):
        for value in history_keys:
            if isinstance(value, str) and value.startswith(prefixes):
                names.add(value)
            elif isinstance(value, (dict, list)):
                names.update(eval_metric_names(value, prefixes))
    return names


def repair_eval_metric_axes(run_ids, project, entity, dry_run=False):
    """Restore evaluation axes and hide diagnostics after all streams sync."""
    import wandb
    from wandb.proto import wandb_internal_pb2

    api = wandb.Api(api_key=os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_KEY"), timeout=60)
    entity = entity or api.default_entity
    fields = {field.name: str(field.number) for field in wandb_internal_pb2.MetricRecord.DESCRIPTOR.fields}
    options = wandb_internal_pb2.MetricRecord.DESCRIPTOR.fields_by_name["options"].message_type
    fields["hidden_option"] = options.fields_by_name["hidden"].number
    fields["step_sync_option"] = options.fields_by_name["step_sync"].number
    for run_id in dict.fromkeys(run_ids):
        run_id = resolve_wandb_run_id(run_id, entity, project)
        path = f"{entity}/{project}/{run_id}"
        run = api.run(path)
        internal = run.rawconfig.get("_wandb", {})
        previous = internal.get("m", [])
        names = eval_metric_names(run._attrs.get("historyKeys", {}))
        if not names:
            for row in run.scan_history(page_size=1000):
                names.update(key for key, value in row.items()
                             if key.startswith("eval_metric/") and value is not None)
        repaired = eval_metric_axis_definitions(previous, names, fields)
        excluded = eval_metric_names(run._attrs.get("historyKeys", {}), EXCLUDED_WANDB_PREFIXES)
        excluded.update(key for key in run.summary.keys() if key.startswith(EXCLUDED_WANDB_PREFIXES))
        excluded.update("system/" + key for key in run.system_metrics)
        repaired = hide_excluded_metric_definitions(repaired, excluded, fields)
        excluded_summary = [key for key in run.summary.keys()
                            if key.startswith(EXCLUDED_WANDB_PREFIXES)]
        if repaired == previous and not excluded_summary:
            print(f"Eval metric axes already correct: {path}", flush=True)
            continue
        if dry_run:
            print(f"Would bind {len(names)} eval_metric metrics to {EVAL_STEP_METRIC}: {path}", flush=True)
            continue
        # Append definitions rather than renumbering existing step_metric_index values.
        run.rawconfig.setdefault("_wandb", {})["m"] = repaired
        for key in excluded_summary:
            del run.summary[key]
        run.update()
        run.load(force=True)
        if run.rawconfig.get("_wandb", {}).get("m") != repaired:
            raise RuntimeError(f"W&B did not retain repaired eval metric axes: {path}")
        print(f"Bound {len(names)} eval_metric metrics to {EVAL_STEP_METRIC}: {path}", flush=True)


def offline_logs(roots, include_synced):
    logs = set()
    for root in roots:
        candidates = [root] if root.is_file() else root.rglob("run-*.wandb")
        for candidate in candidates:
            if candidate.suffix != ".wandb" or not candidate.parent.name.startswith("offline-run-"):
                continue
            if include_synced or not Path(str(candidate) + ".synced").exists():
                logs.add(candidate.resolve())
    return sorted(logs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Optional run files, run directories or WandB directories")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/config/qwen3_4b_lightning_opd.yaml")
    parser.add_argument("--env-file", type=Path, help="Override the YAML env_file")
    parser.add_argument("--all-experiments", action="store_true", help="Sync all experiment WandB directories in output_dir")
    parser.add_argument("--include-synced", action="store_true", help="Also upload files already marked as synced")
    parser.add_argument("--dry-run", action="store_true", help="Print upload commands without contacting WandB")
    parser.add_argument("--repair-eval-axis", nargs="+", metavar="RUN_ID",
                        help="Repair existing runs' eval_metric axes without uploading history")
    args = parser.parse_args()
    if args.paths and args.all_experiments:
        parser.error("Use explicit paths or --all-experiments, not both.")

    cfg = yaml.safe_load(args.config.read_text())
    if not isinstance(cfg, dict):
        parser.error("The configuration must be a YAML mapping.")
    spec = importlib.util.spec_from_file_location("qwen3_4b_opd_launcher", ROOT / "configs/opd/qwen3-4b-opd.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    launcher.load_env(args.env_file or cfg.get("env_file") or ROOT / ".env")

    if args.repair_eval_axis:
        if args.paths or args.all_experiments:
            parser.error("Use --repair-eval-axis without run paths or --all-experiments.")
        repair_eval_metric_axes(args.repair_eval_axis, cfg["project_name"],
                                cfg.get("wandb_entity") or os.environ.get("WANDB_ENTITY"), args.dry_run)
        return

    if args.paths:
        roots = [path.expanduser().resolve() for path in args.paths]
        for root in roots:
            if not root.exists():
                parser.error(f"Run path does not exist: {root}")
    elif args.all_experiments:
        output = Path(cfg["output_dir"]).expanduser().resolve()
        roots = sorted(output.glob("*/wandb"))
    else:
        roots = [launcher.run_dir(cfg) / "wandb"]

    logs = offline_logs(roots, args.include_synced)
    if not logs:
        print("No unsynced offline WandB runs found in: " + ", ".join(map(str, roots)))
        return

    # Only the upload subprocess uses online mode; training stays offline.
    env = os.environ.copy()
    env["WANDB_MODE"] = "online"
    if not env.get("WANDB_API_KEY") and env.get("WANDB_KEY"):
        env["WANDB_API_KEY"] = env["WANDB_KEY"]
    env.pop("WANDB_KEY", None)
    python = env.get("LIGHTNING_OPD_PYTHON") or cfg["python_executable"]
    project = cfg["project_name"]
    entity = cfg.get("wandb_entity") or env.get("WANDB_ENTITY")
    print(f"{'Would sync' if args.dry_run else 'Syncing'} {len(logs)} offline run file(s) to {entity or '<default entity>'}/{project}.")
    snapshots = tempfile.TemporaryDirectory(prefix="lightning-opd-wandb-filtered-")
    eval_points = {}
    api = None
    for log in logs:
        original_id = log.stem.removeprefix("run-")
        run_id = resolve_wandb_run_id(original_id, entity, project)
        if not args.dry_run and run_id not in eval_points:
            import wandb
            if api is None:
                api = wandb.Api(api_key=env.get("WANDB_API_KEY"), timeout=60)
            try:
                remote = api.run(f"{entity or api.default_entity}/{project}/{run_id}")
            except wandb.errors.CommError as error:
                if not any(message in str(error).lower() for message in ("not found", "not find run", "could not find")):
                    raise
                eval_points[run_id] = {}
            else:
                eval_points[run_id] = {
                    (row[EVAL_STEP_METRIC], key): value
                    for row in remote.scan_history(page_size=1000)
                    for key, value in row.items()
                    if row.get(EVAL_STEP_METRIC) is not None
                    and key.startswith("eval_metric/") and value is not None
                }
        upload = log if args.dry_run else filtered_run_file(
            log, Path(snapshots.name), eval_points[run_id])
        command = [python, "-m", "wandb", "sync", "--append", "--no-sync-tensorboard", "--mark-synced", "--project", project]
        if entity:
            command.extend(["--entity", entity])
        if run_id != original_id:
            command.extend(["--id", run_id])
        command.append(str(upload))
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, env=env, check=True)
            if Path(str(upload) + ".synced").exists():
                Path(str(log) + ".synced").touch()

    # Each process stream can replace W&B's metric definitions. Restore the Avg@1
    # binding once the last stream has uploaded, using the configured interpreter.
    run_ids = sorted({log.stem.removeprefix("run-") for log in logs})
    command = [python, str(Path(__file__).resolve()), "--config", str(args.config),
               "--repair-eval-axis", *run_ids]
    if args.env_file:
        command.extend(["--env-file", str(args.env_file)])
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, KeyError, ValueError, subprocess.CalledProcessError) as error:
        print(f"WandB sync failed: {error}", file=sys.stderr)
        sys.exit(1)
