from datetime import datetime
from pathlib import Path


def resolve_project_path(path: str | Path, root_dir: str | Path = ".") -> Path:
    """
    Resolve a path relative to the project root.

    Example:
        resolve_project_path("artifacts/runs")
        -> Path("artifacts/runs").resolve()
    """
    path = Path(path)

    if path.is_absolute():
        return path

    return (Path(root_dir) / path).resolve()


def ensure_dir(path: str | Path) -> Path:
    """
    Create a directory if it does not already exist.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_run_id(
    model: str,
    experiment: str,
    seed: int,
    reward: str | None = None,
) -> str:
    """
    Create a readable run ID.

    Example:
        2026-04-26_diffdock_baseline_seed42
        2026-04-26_diffdock_posttraining_negative-rmsd_seed42
    """
    date = datetime.now().strftime("%Y-%m-%d")

    parts = [date, model, experiment]

    if reward:
        parts.append(reward)

    parts.append(f"seed{seed}")

    return "_".join(parts)


def create_run_dir(
    base_dir: str | Path,
    run_id: str,
    exist_ok: bool = False,
) -> Path:
    """
    Create a run directory.

    By default, this fails if the run already exists to avoid accidentally
    overwriting experiment artifacts.
    """
    run_dir = Path(base_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=exist_ok)
    return run_dir