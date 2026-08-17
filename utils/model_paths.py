from pathlib import Path


def resolve_model_path(cache_dir: Path | str, repo_id: str) -> Path:
    cache_dir = Path(cache_dir)
    legacy_path = cache_dir.joinpath(*repo_id.split("/"))
    snapshot_root = cache_dir / "models" / repo_id.replace("/", "--") / "snapshots"
    candidates = [legacy_path]
    if snapshot_root.is_dir():
        candidates.extend(path for path in snapshot_root.iterdir() if path.is_dir())

    for path in candidates:
        if (path / "config.json").is_file():
            return path

    raise FileNotFoundError(
        f"Local model '{repo_id}' was not found in '{cache_dir}'. "
        "Run download_model.py before starting the application."
    )
