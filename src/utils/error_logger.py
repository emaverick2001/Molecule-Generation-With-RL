# Error logging helpers for run-level diagnostics and failure tracking.

# Use this module to append structured error information to a persistent
# `errors.log` file during pipeline execution (baseline, rerank, reward-filtering,
# post-training).

# Functions:
# - `append_error(error_log_path, message, context=None)`:
#   - logs a timestamped error message
#   - optionally logs key/value context (e.g., complex_id, stage, file_path)
# - `append_exception(error_log_path, exc, context=None)`:
#   - logs exception type and message
#   - logs optional context
#   - appends traceback text for debugging root cause

# Behavior guarantees:
# - Parent directories are created automatically.
# - Logs are appended (never overwrite existing history).
# - Entries are UTF-8 text and human-readable.

# Typical usage:
# - In `except` blocks around sample-level processing to avoid losing run progress.
# - For non-fatal validation/data issues where execution should continue.

from __future__ import annotations

import traceback
from datetime import datetime
from pathlib import Path


def append_error(
    error_log_path: str | Path,
    message: str,
    context: dict | None = None,
) -> None:
    """
    Append a simple error message to errors.log.
    """
    error_log_path = Path(error_log_path)
    error_log_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat(timespec="seconds")

    lines = [
        f"[{timestamp}] ERROR",
        f"message: {message}",
    ]

    if context:
        lines.append("context:")
        for key, value in context.items():
            lines.append(f"  {key}: {value}")

    lines.append("")

    with error_log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def append_exception(
    error_log_path: str | Path,
    exc: Exception,
    context: dict | None = None,
) -> None:
    """
    Append an exception with traceback to errors.log.
    """
    error_log_path = Path(error_log_path)
    error_log_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat(timespec="seconds")

    lines = [
        f"[{timestamp}] EXCEPTION",
        f"type: {type(exc).__name__}",
        f"message: {str(exc)}",
    ]

    if context:
        lines.append("context:")
        for key, value in context.items():
            lines.append(f"  {key}: {value}")

    lines.append("traceback:")
    lines.append(traceback.format_exc())
    lines.append("")

    with error_log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")