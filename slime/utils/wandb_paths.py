"""Shared local W&B storage paths for launchers and offline sync."""

from pathlib import Path


def wandb_directory(config, repo_root):
    """Resolve optional YAML paths against the repository, independently of CWD."""
    path = Path(config.get("wandb_dir") or Path("wandb") / config["experiment_name"]).expanduser()
    return (path if path.is_absolute() else Path(repo_root) / path).resolve()


def wandb_source_roots(config, repo_root, result_dir):
    """Also discover logs written by the earlier result-directory layout."""
    return list(dict.fromkeys([wandb_directory(config, repo_root),
                              (Path(result_dir) / "wandb").resolve()]))
