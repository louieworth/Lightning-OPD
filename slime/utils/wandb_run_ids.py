"""Keep offline logs and checkpoint resumes attached to replacement W&B runs."""

import json
import os
from pathlib import Path


DEFAULT_RUN_ID_MAP = Path(__file__).resolve().parents[2] / "tmp" / "wandb_run_id_map.json"


def resolve_wandb_run_id(run_id, entity, project, map_path=None):
    if not run_id or not project:
        return run_id
    path = Path(map_path) if map_path is not None else DEFAULT_RUN_ID_MAP
    if not path.exists():
        return run_id
    mapping = json.loads(path.read_text())
    scope = f"{entity or os.environ.get('WANDB_ENTITY', '')}/{project}"
    redirects = mapping.get("runs", {}).get(scope, {})
    visited = set()
    while run_id in redirects:
        if run_id in visited:
            raise ValueError(f"W&B run ID redirect cycle in {scope}")
        visited.add(run_id)
        run_id = redirects[run_id]
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(f"Invalid W&B replacement run ID in {scope}")
    return run_id
