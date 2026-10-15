"""The metric subset retained in W&B; local diagnostics remain available."""

EXCLUDED_WANDB_PREFIXES = ("system/", "perf/", "lightning/", "eval/")
EVAL_STEP_METRIC = "eval_step"


def completed_eval_step(rollout_index):
    """Translate legacy evaluation indices: initial 0, then 24 -> 25, etc."""
    return 0 if rollout_index == 0 else rollout_index + 1


def filter_wandb_metrics(metrics):
    result = {key: value for key, value in metrics.items()
              if not key.startswith(EXCLUDED_WANDB_PREFIXES)}
    if "eval/step" in metrics and EVAL_STEP_METRIC not in result:
        result[EVAL_STEP_METRIC] = completed_eval_step(metrics["eval/step"])
    return result


def canonical_wandb_history(records, preserve_conflicts=False):
    """Merge process streams into one row per completed training update."""
    rows = {}
    additional = []
    for record in records:
        metrics = filter_wandb_metrics({key: value for key, value in record.items() if value is not None})
        groups = {}
        for key, value in metrics.items():
            if key.startswith("_") or key in {"train/step", "rollout/step", EVAL_STEP_METRIC}:
                continue
            if key.startswith("eval_metric/"):
                axis, offset = EVAL_STEP_METRIC, 0
            elif key.startswith("train/"):
                axis, offset = "train/step", 1
            else:
                axis, offset = "rollout/step", 1
            if axis not in metrics:
                raise ValueError(f"Metric lacks its real step: {key}")
            step = int(metrics[axis]) + offset
            group = groups.setdefault(step, {})
            group[axis] = metrics[axis]
            group[key] = value
        for step, values in groups.items():
            row = rows.setdefault(step, {"_step": step})
            conflicts = {}
            for key, value in values.items():
                if key in row and row[key] != value:
                    if not preserve_conflicts or key.startswith("eval_metric/"):
                        raise ValueError(f"Conflicting metric at completed step {step}: {key}")
                    conflicts[key] = value
                    continue
                row[key] = value
            if conflicts:
                conflicts.update({key: value for key, value in values.items()
                                  if key in {"train/step", "rollout/step"}})
                additional.append(conflicts)
            if record.get("_timestamp", 0) >= row.get("_timestamp", 0):
                row.update({key: record[key] for key in ("_timestamp", "_runtime") if key in record})
    result = [rows[step] for step in sorted(rows)]
    for values in additional:
        values["_step"] = result[-1]["_step"] + 1 if result else 0
        result.append(values)
    return result
