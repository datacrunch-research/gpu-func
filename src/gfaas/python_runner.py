"""Remote harness for self-contained Python scripts submitted by the CLI."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .artifacts import scratch_path


def run_script(
    *,
    source: str,
    filename: str,
    program_args: list[str] | None = None,
) -> dict[str, Any]:
    """Run one Python source file and stream its output through the Call log."""
    safe_name = Path(filename).name
    if safe_name != filename or not safe_name:
        raise ValueError("Python script filename must be a base name")

    output_root = os.environ.get("GFAAS_OUTPUT_ROOT")
    if not output_root:
        raise RuntimeError("Python script output directory is unavailable")

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="gfaas-python-", dir=scratch_path()) as directory:
        script = Path(directory) / safe_name
        script.write_text(source, encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, "-u", str(script), *(program_args or [])],
            cwd=output_root,
            env=environment,
            check=False,
        )

    if completed.returncode != 0:
        raise RuntimeError(f"Python script stopped with status {completed.returncode}")

    return {
        "phase": "run",
        "returncode": completed.returncode,
        "duration_ms": int((time.monotonic() - started) * 1000),
    }
