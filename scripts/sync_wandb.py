"""Sync completed OPD experiments by sbatch filename, job name, or experiment name.

The default uploads and verifies. --dry-run reads local sources only; --verify-only
compares them with W&B without changing the cloud. No training or evaluation runs.
"""

import argparse
import ast
from collections import defaultdict
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slime.utils.wandb_metrics import EXCLUDED_WANDB_PREFIXES, filter_wandb_metrics  # noqa: E402
from slime.utils.wandb_run_ids import resolve_wandb_run_id  # noqa: E402
from slime.utils.wandb_paths import wandb_source_roots  # noqa: E402

AXES = {"train/step", "rollout/step", "eval_step"}
ANSI = re.compile(r"\x1b\[[0-9;]*m")
LIGHTNING_ROW = re.compile(r"(?:model|rollout|data)\.py:\d+ - (step|perf|eval|rollout) (\d+): (\{.*\})")
VERL_ROW = re.compile(r"\bstep:(\d+) - (.*)")
CONFIG_REFERENCE = re.compile(r"configs/(?:config/)?[\w.-]+\.ya?ml")
# Fixed training settings are useful for filtering; credentials and arbitrary
# process environment/configuration are never copied into a newly created run.
CONFIG_KEYS = set("""method model_path teacher_model_path model_type experiment_name
policy_rollout_k val_n learning_rate warmup_steps weight_decay max_grad_norm
per_device_batch_size global_batch_size batch_size ppo_mini_batch_size group_size
n_gpus_per_node num_steps total_training_steps num_epochs train_epochs
temperature top_p top_k_sampling max_prompt_length max_completion_length
val_max_prompt_length val_max_completion_length val_temperature val_top_p val_top_k
val_do_sample val_batch_size eval_steps val_before_train mode distillation_loss_mode
distillation_topk opd_loss_token_mode opd_token_top_k""".split())


def equal(left, right):
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        return (math.isnan(left) and math.isnan(right)) or math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12)
    return left == right


def read_yaml(path):
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def find_recipe(selector, repos):
    """Inspect sbatch text; never execute it or its shell substitutions."""
    supplied = Path(selector).expanduser()
    name = supplied.name.removesuffix(".sbatch")
    matches = set()
    for repo in repos:
        configs = list((repo / "configs/config").glob("*.yaml")) + list((repo / "configs").glob("*.yaml"))
        for config in configs:
            cfg = yaml.safe_load(config.read_text())
            if not isinstance(cfg, dict):
                continue
            if name in {config.stem, cfg.get("experiment_name")}:
                matches.add((repo, config))
        for folder in (repo / "configs/sbatch", repo / "scripts/sbatch"):
            for sbatch in folder.glob("*.sbatch"):
                text = sbatch.read_text()
                job = re.search(r"^#SBATCH\s+--job-name(?:=|\s+)(\S+)", text, re.M)
                selected = sbatch.resolve() == supplied.resolve() if supplied.is_file() else name == sbatch.stem or (job and name == job[1])
                if not selected:
                    continue
                references = {repo / item for item in CONFIG_REFERENCE.findall(text)}
                matches.update((repo, path) for path in references if path.is_file())
    # A wrapper's explicit YAML and the matching experiment YAML are the same path.
    matches = sorted(matches)
    if len(matches) != 1:
        detail = ", ".join(str(config) for _, config in matches) or "none"
        raise ValueError(f"Cannot resolve one recipe for {selector!r} (matches: {detail}). Use the sbatch filename/path or experiment name.")
    return matches[0]


def find_run_dirs(repo, recipe, explicit=None, job_id=None):
    cfg = read_yaml(recipe)
    name = cfg["experiment_name"]
    def matches(folder):
        saved = folder / "config.yaml"
        if not saved.is_file() or read_yaml(saved).get("experiment_name") != name:
            return False
        # Preflight attempts can leave a saved config but no training/evaluation.
        # An explicit selection is kept so build_source can report its problem.
        return bool(explicit or job_id or next((folder / "eval_metrics").glob("*.json"), None))
    if explicit:
        folders = [explicit.expanduser().resolve()]
    elif cfg.get("method") == "lightning_opd":
        folders = [Path(cfg["output_dir"]).expanduser() / name]
    else:
        markers = repo / "wandb/.run_roots"
        files = [markers / f"slurm-{job_id}.txt"] if job_id else sorted(markers.glob("*.txt"))
        folders = [Path(marker.read_text().strip()) for marker in files if marker.is_file()]
        if not job_id and not any(matches(folder) for folder in folders):
            folders = [path.parent for path in Path(cfg["output_dir"]).expanduser().glob("*/config.yaml")]
    result = []
    for folder in sorted(set(folders)):
        if matches(folder):
            result.append(folder.resolve())
    if not result:
        raise ValueError(f"No saved results for {name}; check --repo / --run-dir / --job-id. The script never launches an sbatch job.")
    return result


def read_stream(path):
    """Read complete protobuf records, tolerating only an unfinished last block."""
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    before = signature(path)
    result = {"rows": [], "job_ids": set(), "run_id": path.stem.removeprefix("run-")}
    store = DataStore()
    store.open_for_scan(str(path))
    size = path.stat().st_size
    try:
        while store.get_offset() < size:
            try:
                data = store.scan_data()
            except AssertionError:
                if store.in_last_block():
                    break
                raise
            if data is None or store.get_offset() > size:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            kind = record.WhichOneof("record_type")
            if kind == "history":
                result["rows"].append({(item.key or "/".join(item.nested_key)): json.loads(item.value_json)
                                       for item in record.history.item})
            elif kind in {"run", "config"}:
                config = record.run.config if kind == "run" else record.config
                for item in config.update:
                    key = item.key or "/".join(item.nested_key)
                    if key in {"SLURM_JOB_ID", "env_vars/SLURM_JOB_ID"}:
                        result["job_ids"].add(str(json.loads(item.value_json)))
                    elif key == "env_vars":
                        job = json.loads(item.value_json).get("SLURM_JOB_ID")
                        if job is not None:
                            result["job_ids"].add(str(job))
    finally:
        store.close()
    if signature(path) != before:
        raise ValueError(f"WandB stream changed while reading; wait for training to finish: {path}")
    result["signature"] = before
    if result["run_id"].startswith("slurm-"):
        result["job_ids"].add(result["run_id"].removeprefix("slurm-"))
    return result


def console_rows(path, method):
    with path.open(errors="replace") as handle:
        for line in handle:
            line = ANSI.sub("", line)
            if method == "lightning_opd":
                match = LIGHTNING_ROW.search(line)
                if not match:
                    continue
                kind, index, payload = match.groups()
                row = ast.literal_eval(payload)
                if not isinstance(row, dict):
                    raise ValueError(f"Invalid metric record in {path}")
                if kind == "eval":
                    row.setdefault("eval_step", 0 if int(index) == 0 else int(index) + 1)
                elif kind in {"perf", "rollout"}:
                    row.setdefault("rollout/step", int(index))
                else:
                    row.setdefault("train/step", int(index))
                yield row
            else:
                match = VERL_ROW.search(line)
                if not match:
                    continue
                row = {}
                for item in match[2].split(" - "):
                    key, separator, value = item.partition(":")
                    if separator:
                        try:
                            row[key] = json.loads(value)
                        except ValueError:
                            continue
                if row:
                    row["_step"] = int(match[1])
                    yield row


def normalized(row, method, counts):
    row = {key: value for key, value in row.items() if value is not None}
    if method != "lightning_opd":
        # Verl already logs the number of completed updates. Never add one.
        if "eval/step" in row:
            row.setdefault("eval_step", row["eval/step"])
        if "train/step" not in row and any(key.startswith("actor/") for key in row):
            step = row.get("training/global_step", row.get("_step"))
            if step is None:
                raise ValueError("Native OPD training metric lacks its completed step")
            row["train/step"] = max(0, int(step) - 1)
    return filter_wandb_metrics(row, counts)


def metric_axis(key, method):
    if key.startswith("eval_metric/"):
        return "eval_step"
    if key.startswith("train/") or method != "lightning_opd":
        return "train/step"
    return "rollout/step"


def merge_rows(records, method, counts):
    """One record per real axis/step; SDK _step ordering is independent."""
    merged = {}
    for raw in records:
        row = normalized(raw, method, counts)
        for key, value in row.items():
            if key.startswith("_") or key in AXES:
                continue
            axis = metric_axis(key, method)
            if axis not in row:
                raise ValueError(f"Metric lacks its real axis {axis}: {key}")
            point = (axis, int(row[axis]))
            target = merged.setdefault(point, {axis: point[1]})
            if key in target and not equal(target[key], value):
                raise ValueError(f"Conflicting source values at {point}: {key}")
            target[key] = value
    def order(point):
        axis, step = point
        return step + (axis != "eval_step"), axis
    return [merged[point] for point in sorted(merged, key=order)]


def build_source(repo, folder, job_id=None, log_paths=()):
    config_signature = signature(folder / "config.yaml")
    cfg = read_yaml(folder / "config.yaml")  # Saved recipe, not today's edited YAML.
    if cfg["method"] not in {"lightning_opd", "opd"}:
        raise ValueError(f"Unsupported method: {cfg['method']}")
    # This repository also has ordinary on-policy OPD recipes. They still use
    # Slime's zero-based console/evaluation indices, unlike native verl OPD.
    method = "lightning_opd" if repo.resolve() == ROOT or cfg["method"] == "lightning_opd" else "opd"
    count = cfg["val_n"]
    if type(count) is not int or count < 1:
        raise ValueError("Saved val_n must be a positive integer")
    counts = {Path(path).stem: count for path in cfg["val_files"]}
    if not counts or len(counts) != len(cfg["val_files"]):
        raise ValueError("val_files must have distinct dataset names")
    paths = {path for location in wandb_source_roots(cfg, repo, folder)
             for path in location.rglob("run-*.wandb")}
    markers = repo / "wandb/.run_roots"
    jobs = set()
    if method == "opd":
        for marker in markers.glob("*.txt"):
            if Path(marker.read_text().strip()).resolve() == folder.resolve():
                identity = marker.stem
                jobs.add(identity.removeprefix("slurm-"))
                paths.update((repo / "wandb").glob(f"*/run-{identity}.wandb"))
    records, run_ids, used = [], set(), {str(folder / "config.yaml"): config_signature}
    for path in sorted(paths):
        stream = read_stream(path)
        if job_id and str(job_id) not in stream["job_ids"]:
            continue
        records.extend(stream["rows"])
        jobs.update(stream["job_ids"])
        run_ids.add(stream["run_id"])
        used[str(path)] = stream["signature"]
    if job_id:
        if not run_ids:
            raise ValueError(f"No WandB stream for job {job_id} in {folder}")
        jobs = {str(job_id)}
    if not run_ids:
        raise ValueError(f"No original WandB streams found for {folder}")
    logs = set(log_paths)
    for job in jobs:
        logs.update((repo / "tmp").glob(f"slurm-*-{job}.log"))
    for path in sorted(logs):
        used[str(path)] = signature(path)
        records.extend(console_rows(path, method))
    evaluations = sorted((folder / "eval_metrics").glob("*.json"))
    for path in evaluations:
        used[str(path)] = signature(path)
        data = json.loads(path.read_text())
        if "_eval_completed_steps" in data:
            step = int(data["_eval_completed_steps"])
        elif method == "opd":
            step = int(path.stem)
        else:
            index = int(path.stem)
            step = 0 if index == 0 else index + 1
        records.append(dict(data, eval_step=step))
    rows = merge_rows(records, method, counts)
    expected = {f"eval_metric/{dataset}-{statistic}" for dataset, n in counts.items()
                for statistic in (f"avg@{n}", f"pass@{n}", "response-length")}
    evals = [row for row in rows if "eval_step" in row]
    if not evals:
        raise ValueError(f"No evaluations found in {folder}")
    for row in evals:
        missing = expected - row.keys()
        if missing:
            raise ValueError(f"Incomplete evaluation at step {row['eval_step']}: {sorted(missing)}. Supply the original Slurm log with --log if needed; metrics are never estimated.")
        for dataset, n in counts.items():
            average, passed, length = (row[f"eval_metric/{dataset}-{statistic}"]
                                       for statistic in (f"avg@{n}", f"pass@{n}", "response-length"))
            if not (0 <= average <= passed <= 1 and math.isfinite(length) and length >= 0):
                raise ValueError(f"Invalid evaluation for {dataset} at step {row['eval_step']}")
    metadata = {key: cfg[key] for key in CONFIG_KEYS if key in cfg}
    metadata.update(student_model=cfg["model_path"], teacher_model=cfg["teacher_model_path"],
                    n_samples_per_eval_prompt=count, eval_samples_per_dataset=counts)
    result = {"folder": str(folder), "name": cfg["experiment_name"], "method": method,
            "metadata": metadata, "rows": rows, "run_ids": sorted(run_ids),
            "job_ids": sorted(jobs), "files": used}
    check_stable(result)
    return result


def signature(path):
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns]


def check_stable(source):
    for path, previous in source["files"].items():
        if signature(Path(path)) != previous:
            raise ValueError(f"Source changed during sync; retry after training finishes: {path}")


def check_jobs(source):
    if not source["job_ids"] or not shutil.which("squeue"):
        return
    result = subprocess.run(["squeue", "--noheader", "--jobs", ",".join(source["job_ids"]), "--format=%i %T"],
                            text=True, capture_output=True, timeout=20)
    if result.returncode and "invalid job id" not in result.stderr.lower():
        raise ValueError("Cannot check Slurm job state: " + result.stderr.strip())
    if result.stdout.strip():
        raise ValueError("Sync after these jobs leave the queue: " + result.stdout.strip())


def index_history(rows, method):
    result = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if key.startswith("_") or key in AXES or key.startswith(EXCLUDED_WANDB_PREFIXES) or value is None:
                continue
            axis = metric_axis(key, method)
            if row.get(axis) is not None:
                result[axis, int(row[axis]), key].append(value)
    return result


def missing_rows(source, history):
    indexed = index_history(history, source["method"])
    maximum = {}
    known = set()
    for row in source["rows"]:
        axis = next(key for key in AXES if key in row)
        maximum[axis] = max(maximum.get(axis, row[axis]), row[axis])
        known.update(row.keys() - AXES)
    if any(key in known and step > maximum.get(axis, step)
           for axis, step, key in indexed):
        raise ValueError("Cloud history extends beyond the selected local source; select the complete result directory to avoid regressing Summary")
    pending = []
    for row in source["rows"]:
        axis = next(key for key in AXES if key in row)
        retained = {axis: row[axis]}
        for key, value in row.items():
            if key == axis:
                continue
            present = indexed.get((axis, row[axis], key), [])
            if present and not all(equal(value, item) for item in present):
                raise ValueError(f"Cloud/source conflict at {axis}={row[axis]}: {key}. Original history was preserved; no upload attempted.")
            if key.startswith("eval_metric/") and len(present) > 1:
                raise ValueError(f"Duplicate cloud evaluation at {row[axis]}: {key}")
            if not present:
                retained[key] = value
        if len(retained) > 1:
            pending.append(retained)
    return pending


def latest_summary(source):
    latest = {}
    for row in source["rows"]:
        latest.update(row)
    return latest


def legacy_names(history, source, system_metrics=()):
    names = {key for row in history for key in row}
    names.update("system/" + key for key in system_metrics)
    return {name for name in names if name.startswith(EXCLUDED_WANDB_PREFIXES)
            or (name.startswith("eval_metric/") and next(iter(filter_wandb_metrics(
                {name: 0}, source["metadata"]["eval_samples_per_dataset"]))) != name)}


def metric_fields():
    from wandb.proto import wandb_internal_pb2
    descriptor = wandb_internal_pb2.MetricRecord.DESCRIPTOR
    fields = {field.name: str(field.number) for field in descriptor.fields}
    fields["hidden_option"] = descriptor.fields_by_name["options"].message_type.fields_by_name["hidden"].number
    return fields


def definitions_correct(run, names, hidden, fields, extra_axes=None):
    definitions = run.rawconfig.get("_wandb", {}).get("m", [])
    by_name = {item.get(fields["name"]): item for item in definitions}
    visible = {name: "eval_step" for name in names}
    visible.update(extra_axes or {})
    for name in visible.keys() | hidden:
        metric = by_name.get(name)
        if metric is None:
            return False
        is_hidden = fields["hidden_option"] in metric.get(fields["options"], [])
        if is_hidden != (name in hidden):
            return False
        if name in visible:
            axis = metric.get(fields["step_metric"])
            if metric.get(fields["step_metric_index"]):
                axis = definitions[metric[fields["step_metric_index"]] - 1].get(fields["name"])
            if axis != visible[name]:
                return False
    return True


def load_remote(api, path, wandb):
    try:
        return api.run(path)
    except wandb.errors.CommError as error:
        if any(text in str(error).lower() for text in ("not found", "not find run", "could not find")):
            return None
        raise


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


@contextmanager
def sync_lock(folder):
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (folder / "sync.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"Another sync is already using {folder}") from error
        yield


def sync_source(source, api, wandb, entity, project, run_id, audit, verify_only=False):
    path = f"{entity}/{project}/{run_id}"
    remote = load_remote(api, path, wandb)
    if remote is not None and remote.state == "running":
        raise ValueError(f"Cloud run is still running: {path}")
    history = list(remote.scan_history(page_size=1000)) if remote else []
    if remote:
        for key in ("student_model", "teacher_model", "model_path", "teacher_model_path"):
            if key in remote.config and remote.config[key] != source["metadata"][key]:
                raise ValueError(f"Model provenance conflict in {path}: {key}")
    pending = missing_rows(source, history)
    names = {key for row in source["rows"] for key in row if key.startswith("eval_metric/")}
    extra_axes = {key: metric_axis(key, source["method"]) for row in source["rows"] for key in row
                  if key not in AXES and not key.startswith(("train/", "rollout/", "eval_metric/"))}
    hidden = legacy_names(history + [dict(remote.summary) if remote else {}], source,
                          remote.system_metrics if remote else ())
    summary = latest_summary(source)
    fields = metric_fields()
    metadata_changed = remote is None or any(remote.config.get(key) != value for key, value in source["metadata"].items())
    axes_changed = remote is None or not definitions_correct(remote, names, hidden, fields, extra_axes)
    needs_cleanup = remote is not None and bool(hidden.intersection(remote.summary.keys()))
    needs_summary = remote is None or any(not equal(remote.summary.get(key), value) for key, value in summary.items())
    plan = {"run": path, "name": source["name"], "metadata": source["metadata"],
            "new_rows": len(pending), "new_points": sum(len(row) - 1 for row in pending),
            "eval_steps": sorted(row["eval_step"] for row in source["rows"] if "eval_step" in row),
            "metadata_changed": metadata_changed, "axes_changed": axes_changed}
    print(json.dumps(plan, ensure_ascii=False), flush=True)
    save_json(audit / "plan.json", plan)
    check_stable(source)
    if verify_only:
        if pending or metadata_changed or axes_changed or needs_cleanup or needs_summary:
            raise ValueError(f"Run does not yet meet the standard: {path}; run without --verify-only to sync")
    else:
        check_jobs(source)
        if pending or metadata_changed or axes_changed:
            save_json(audit / "cloud-before.json", {"history": history, "summary": dict(remote.summary) if remote else {},
                                                   "metadata": {key: remote.config.get(key) for key in source["metadata"]} if remote else {}})
            sdk = audit / "sdk"
            sdk.mkdir(exist_ok=True)
            with wandb.init(entity=entity, project=project, id=run_id,
                            resume="must" if remote else "never", name=source["name"],
                            mode="online", dir=str(sdk),
                            settings=wandb.Settings(console="off", quiet=True, x_disable_stats=True)) as target:
                target.config.update(source["metadata"], allow_val_change=True)
                for axis in sorted(AXES):
                    target.define_metric(axis, hidden=axis == "eval_step")
                for prefix, axis in (("train/*", "train/step"), ("rollout/*", "rollout/step")):
                    target.define_metric(prefix, step_metric=axis, step_sync=False)
                for name in sorted(hidden):
                    target.define_metric(name, hidden=True)
                for name in sorted(names):
                    target.define_metric(name, step_metric="eval_step", step_sync=False, hidden=False)
                for name, axis in sorted(extra_axes.items()):
                    target.define_metric(name, step_metric=axis, step_sync=False, hidden=False)
                for row in pending:
                    target.log(row)
                target.summary.update(summary)
    for attempt in range(4):
        api.flush()
        checked = api.run(path)
        if not verify_only and (needs_cleanup or needs_summary or pending):
            for name in hidden.intersection(checked.summary.keys()):
                del checked.summary[name]
            checked.summary.update(summary)
            api.flush()
            checked = api.run(path)
        observed = list(checked.scan_history(page_size=1000))
        remaining = missing_rows(source, observed)
        valid_summary = not hidden.intersection(checked.summary.keys()) and all(
            equal(checked.summary.get(key), value) for key, value in summary.items())
        valid_config = all(checked.config.get(key) == value for key, value in source["metadata"].items())
        if not remaining and valid_summary and valid_config and definitions_correct(checked, names, hidden, fields, extra_axes):
            break
        if attempt == 3:
            raise ValueError(f"Cloud verification incomplete: {path}; rerun safely to retry")
        time.sleep(2)
    original_by_step = {row["_step"]: row for row in observed}
    for row in history:
        if any(not equal(original_by_step.get(row["_step"], {}).get(key), value) for key, value in row.items()):
            raise ValueError(f"Original cloud history changed at _step={row['_step']}")
    check_stable(source)
    report = {"url": checked.url, "run": path, "state": checked.state,
              "student_model": source["metadata"]["student_model"], "teacher_model": source["metadata"]["teacher_model"],
              "eval_steps": plan["eval_steps"], "new_points": plan["new_points"],
              "verified_points": sum(len(row) - 1 for row in source["rows"]), "original_history_preserved": True}
    save_json(audit / "verification.json", report)
    print("VERIFIED " + json.dumps(report, ensure_ascii=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+", help="Sbatch basename/path, #SBATCH job-name, or experiment_name; accepts several")
    parser.add_argument("--repo", type=Path, help="Also search this repository (default: Lightning-OPD and sibling opd-baselines)")
    parser.add_argument("--job-id", help="Select one Slurm submission")
    parser.add_argument("--run-dir", type=Path, help="Select an actual saved result directory")
    parser.add_argument("--run-id", help="Explicit target W&B ID; normally inferred from original streams and redirects")
    parser.add_argument("--log", type=Path, action="append", default=[], help="Additional original Slurm log (repeatable)")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--entity", help="Override W&B entity from .env")
    parser.add_argument("--project", help="Override W&B project from .env")
    parser.add_argument("--audit-dir", type=Path, default=ROOT / "tmp/wandb-sync")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="Inspect local sources only; no network or uploads")
    modes.add_argument("--verify-only", action="store_true", help="Read-only check against W&B")
    args = parser.parse_args(argv)
    if len(args.names) != 1 and (args.run_dir or args.run_id or args.job_id or args.log):
        parser.error("--run-dir / --run-id / --job-id / --log require exactly one name")
    repos = list(dict.fromkeys([ROOT, (args.repo or ROOT.parent / "opd-baselines").expanduser().resolve()]))
    spec = importlib.util.spec_from_file_location("wandb_sync_launcher", ROOT / "configs/opd/qwen3-4b-opd.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    launcher.load_env(args.env_file)
    env = launcher.wandb_env()
    entity = args.entity or env.get("WANDB_ENTITY")
    project = args.project or env["WANDB_PROJECT"]
    sources = {}
    for name in args.names:
        repo, recipe = find_recipe(name, repos)
        for folder in find_run_dirs(repo, recipe, args.run_dir, args.job_id):
            source = build_source(repo, folder, args.job_id, args.log)
            resolved = {resolve_wandb_run_id(identity, entity, project) for identity in source["run_ids"]}
            if not args.run_id and len(resolved) != 1:
                raise ValueError(f"Multiple distinct W&B runs in {folder}: {sorted(resolved)}; select --job-id or --run-id")
            source["target_id"] = args.run_id or next(iter(resolved))
            sources[str(folder)] = source
    # Resolve and validate every local input before any cloud mutation.
    for source in sources.values():
        print(json.dumps({key: source[key] for key in ("name", "folder", "job_ids", "run_ids", "target_id")}, ensure_ascii=False), flush=True)
        if args.dry_run:
            print(json.dumps({"student_model": source["metadata"]["student_model"], "teacher_model": source["metadata"]["teacher_model"],
                              "val_n": source["metadata"]["val_n"], "eval_steps": [row["eval_step"] for row in source["rows"] if "eval_step" in row],
                              "source_rows": len(source["rows"]), "source_points": sum(len(row) - 1 for row in source["rows"]),
                              "destination": f"{entity or '<default entity>'}/{project}/{source['target_id']}"}, ensure_ascii=False), flush=True)
    if args.dry_run:
        return
    os.environ.update(env)
    os.environ.update(WANDB_MODE="online", WANDB_DISABLE_GIT="true", WANDB_DISABLE_CODE="true")
    os.environ.pop("WANDB_KEY", None)
    import wandb
    api = wandb.Api(api_key=env.get("WANDB_API_KEY"), timeout=60)
    entity = entity or api.default_entity
    if not entity:
        raise ValueError("Set WANDB_ENTITY in .env or use --entity")
    for source in sources.values():
        # Resolve redirects again once the default entity is known.
        identities = {resolve_wandb_run_id(identity, entity, project) for identity in source["run_ids"]}
        if not args.run_id and len(identities) != 1:
            raise ValueError("Multiple run IDs; use --job-id or --run-id")
        run_id = args.run_id or next(iter(identities))
        digest = hashlib.sha256(f"{entity}/{project}/{run_id}".encode()).hexdigest()[:16]
        audit = args.audit_dir.expanduser().resolve() / digest
        with sync_lock(audit):
            save_json(audit / "source.json", source)
            sync_source(source, api, wandb, entity, project, run_id, audit, args.verify_only)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"W&B sync failed: {error}", file=sys.stderr)
        sys.exit(1)
