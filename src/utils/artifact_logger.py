# Artifact logging helpers for writing run outputs to disk.

# Use this module to persist common experiment artifacts in a consistent way:
# - `save_json(obj, path)`: single JSON object (configs, metrics, summaries)
# - `save_records_json(records, path)`: list of records as JSON
# - `save_csv(rows, path)`: tabular records as CSV (requires non-empty rows)
# - `save_text(text, path)`: plain text files (logs, notes, summaries)

# Behavior guarantees:
# - Parent directories are created automatically.
# - Dataclasses are converted to dictionaries before serialization.
# - Files are written as UTF-8 for portability across environments.


from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def _to_serializable(obj: Any) -> Any:
    """
    Convert dataclasses to dictionaries so they can be saved as JSON/CSV.
    """
    if is_dataclass(obj):
        return asdict(obj)

    return obj

def read_json(path: str | Path) -> Any:
    """
    Read a JSON file.

    Used for:
    - dataset manifests
    - metrics files
    - generated sample manifests
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: str | Path) -> None:
    """
    Write an object to a JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def read_text(path: str | Path) -> str:
    """
    Read a text file.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {path}")

    return path.read_text(encoding="utf-8")


def write_text(text: str, path: str | Path) -> None:
    """
    Write text to a file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(text, encoding="utf-8")


def file_exists(path: str | Path) -> bool:
    """
    Check whether a file exists.
    """
    return Path(path).exists()


def require_file(path: str | Path) -> Path:
    """
    Return a path if it exists, otherwise raise FileNotFoundError.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    return path

def save_json(obj: Any, path: str | Path) -> None:
    """
    Save a JSON artifact.

    Used for:
    - config snapshots
    - metrics.json
    - run summaries
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    obj = _to_serializable(obj)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_records_json(records: list[Any], path: str | Path) -> None:
    """
    Save a list of records as JSON.

    Records can be dictionaries or dataclasses.
    """
    serialized = [_to_serializable(record) for record in records]

    save_json(serialized, path)


def save_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    """
    Save a list of dictionaries as a CSV file.

    Used for:
    - rewards.csv
    - per-sample metrics
    - training logs
    """
    if not rows:
        raise ValueError("Cannot save empty CSV.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_text(text: str, path: str | Path) -> None:
    """
    Save a plain text artifact.

    Used for:
    - errors.log
    - summary.md
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        f.write(text)