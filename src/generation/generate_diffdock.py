"""
Thin DiffDock inference wrapper.

This module only handles:
- ComplexInput records
- DiffDock inference command execution
- raw DiffDock output collection
- standardized GeneratedPose records

Reward scoring, RMSD, evaluation, and reranking belong in later pipeline stages.
"""

from __future__ import annotations

import shutil
import subprocess
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from src.generation.contract import validate_generated_pose_records
from src.utils.schemas import ComplexInput, GeneratedPose


Runner = Callable[..., Any]


def _resolve_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def _get_python_executable() -> str:
    return os.environ.get("DIFFDOCK_PYTHON", "python")


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _tail_text(text: str, max_lines: int = 40) -> str:
    lines = text.strip().splitlines()
    if not lines:
        return ""
    return "\n".join(lines[-max_lines:])


def _format_command(
    command_template: Sequence[str],
    record: ComplexInput,
    raw_output_dir: Path,
    num_samples: int,
    repo_dir: Path | None = None,
    config_path: Path | None = None,
) -> list[str]:
    values = {
        "complex_id": record.complex_id,
        "protein_path": _resolve_path(record.protein_path),
        "ligand_path": _resolve_path(record.ligand_path),
        "ground_truth_pose_path": _resolve_path(record.ground_truth_pose_path),
        "raw_output_dir": _resolve_path(raw_output_dir),
        "output_dir": _resolve_path(raw_output_dir),
        "num_samples": str(num_samples),
        "config_path": _resolve_path(config_path) if config_path is not None else "",
        "repo_dir": _resolve_path(repo_dir) if repo_dir is not None else "",
        "python_executable": _get_python_executable(),
    }

    return [part.format(**values) for part in command_template]


def preflight_diffdock_generation(
    records: list[ComplexInput],
    repo_dir: str | Path,
    config_path: str | Path,
    num_samples: int,
    command_template: Sequence[str],
) -> None:
    """
    Fail fast on common setup issues before starting expensive DiffDock runs.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be greater than 0")

    if not command_template:
        raise ValueError("command_template must not be empty")

    executable = str(
        command_template[0].format(python_executable=_get_python_executable())
    )
    executable_path = Path(executable)
    if executable_path.parent != Path("."):
        command_exists = executable_path.exists()
    else:
        command_exists = shutil.which(executable) is not None

    if not command_exists:
        raise FileNotFoundError(f"DiffDock command executable not found: {executable}")

    repo_dir = Path(repo_dir)
    config_path = Path(config_path)

    if not repo_dir.is_dir():
        raise FileNotFoundError(f"DiffDock repo_dir not found: {repo_dir}")

    if not config_path.is_file():
        raise FileNotFoundError(f"DiffDock config_path not found: {config_path}")

    for record in records:
        protein_path = Path(record.protein_path)
        ligand_path = Path(record.ligand_path)
        ground_truth_pose_path = Path(record.ground_truth_pose_path)

        for path, field_name in [
            (protein_path, "protein_path"),
            (ligand_path, "ligand_path"),
            (ground_truth_pose_path, "ground_truth_pose_path"),
        ]:
            if not path.is_file():
                raise FileNotFoundError(
                    f"{field_name} for {record.complex_id} not found: {path}"
                )

        if protein_path.suffix.lower() != ".pdb":
            raise ValueError(
                f"protein_path for {record.complex_id} must be a .pdb file: "
                f"{protein_path}"
            )

        if ligand_path.suffix.lower() != ".sdf":
            raise ValueError(
                f"ligand_path for {record.complex_id} must be a .sdf file: "
                f"{ligand_path}"
            )


def run_diffdock_command(
    command: Sequence[str],
    cwd: str | Path,
    stdout_path: str | Path,
    stderr_path: str | Path,
    timeout_seconds: int | None,
) -> subprocess.CompletedProcess[str]:
    """
    Run one DiffDock command, persist logs, and raise clear execution errors.
    """
    stdout_path = Path(stdout_path)
    stderr_path = Path(stderr_path)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            list(command),
            cwd=Path(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        stdout_path.write_text(_as_text(error.stdout), encoding="utf-8")
        stderr_path.write_text(_as_text(error.stderr), encoding="utf-8")
        raise TimeoutError(
            f"DiffDock command timed out after {timeout_seconds} seconds"
        ) from error

    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")

    if result.returncode != 0:
        stderr_tail = _tail_text(result.stderr)
        stdout_tail = _tail_text(result.stdout)
        details = [
            f"DiffDock command failed with return code {result.returncode}: "
            f"{' '.join(command)}",
            f"cwd: {Path(cwd)}",
            f"stdout log: {stdout_path}",
            f"stderr log: {stderr_path}",
        ]

        if stderr_tail:
            details.append(f"stderr tail:\n{stderr_tail}")
        elif stdout_tail:
            details.append(f"stdout tail:\n{stdout_tail}")

        raise RuntimeError(
            "\n".join(details)
        )

    return result


def _standardize_diffdock_outputs(
    record: ComplexInput,
    raw_output_dir: Path,
    output_dir: Path,
    num_samples: int,
) -> list[GeneratedPose]:
    raw_pose_paths = sorted(raw_output_dir.rglob("*.sdf"))

    if len(raw_pose_paths) < num_samples:
        raise FileNotFoundError(
            f"DiffDock produced {len(raw_pose_paths)} SDF files for "
            f"{record.complex_id}, expected at least {num_samples}: {raw_output_dir}"
        )

    generated = []

    for sample_id, raw_pose_path in enumerate(raw_pose_paths[:num_samples]):
        standardized_path = output_dir / f"{record.complex_id}_sample_{sample_id}.sdf"
        shutil.copyfile(raw_pose_path, standardized_path)

        generated.append(
            GeneratedPose(
                complex_id=record.complex_id,
                sample_id=sample_id,
                pose_path=str(standardized_path),
                confidence_score=None,
            )
        )

    return generated


def generate_diffdock_poses(
    records: list[ComplexInput],
    output_dir: str | Path,
    num_samples: int,
    command_template: Sequence[str],
    raw_output_dir: str | Path | None = None,
    repo_dir: str | Path | None = None,
    config_path: str | Path | None = None,
    log_dir: str | Path | None = None,
    timeout_seconds: int | None = None,
    runner: Runner = run_diffdock_command,
) -> list[GeneratedPose]:
    """
    Run DiffDock once per complex and return standardized pose records.

    `command_template` supports these placeholders:
    `{complex_id}`, `{protein_path}`, `{ligand_path}`,
    `{ground_truth_pose_path}`, `{output_dir}`, `{num_samples}`.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be greater than 0")

    if not command_template:
        raise ValueError("command_template must not be empty")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_repo_dir = Path(repo_dir).resolve() if repo_dir is not None else Path.cwd()
    resolved_config_path = (
        Path(config_path).resolve() if config_path is not None else None
    )

    if repo_dir is not None and config_path is not None:
        preflight_diffdock_generation(
            records=records,
            repo_dir=resolved_repo_dir,
            config_path=resolved_config_path,
            num_samples=num_samples,
            command_template=command_template,
        )

    raw_output_root = (
        Path(raw_output_dir)
        if raw_output_dir is not None
        else output_dir.parent / "raw_diffdock_outputs"
    )
    raw_output_root.mkdir(parents=True, exist_ok=True)

    log_root = Path(log_dir) if log_dir is not None else output_dir.parent / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    generated: list[GeneratedPose] = []

    for record in records:
        complex_raw_output_dir = raw_output_root / record.complex_id
        complex_raw_output_dir.mkdir(parents=True, exist_ok=True)

        command = _format_command(
            command_template=command_template,
            record=record,
            raw_output_dir=complex_raw_output_dir,
            num_samples=num_samples,
            repo_dir=resolved_repo_dir,
            config_path=resolved_config_path,
        )

        runner(
            command=command,
            cwd=resolved_repo_dir,
            stdout_path=log_root / f"{record.complex_id}.stdout.log",
            stderr_path=log_root / f"{record.complex_id}.stderr.log",
            timeout_seconds=timeout_seconds,
        )

        generated.extend(
            _standardize_diffdock_outputs(
                record=record,
                raw_output_dir=complex_raw_output_dir,
                output_dir=output_dir,
                num_samples=num_samples,
            )
        )

    validate_generated_pose_records(
        records=records,
        generated=generated,
        num_samples=num_samples,
        output_dir=output_dir,
    )

    return generated
