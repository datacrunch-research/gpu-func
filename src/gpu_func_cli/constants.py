"""Shared constants for CLI defaults and process exit codes."""

from __future__ import annotations

# ``--gpu`` remains as a compatibility shortcut. New scripts should use
# ``--gpu-type`` and ``--arch`` explicitly or let the worker detect its CUDA
# architecture. The pool names match the current gfaas capability vocabulary.
GPU_DEFAULTS = {
    "B200": ("b200", "sm_100a"),
    "GB300": ("gb300", "sm_103"),
    "B300": ("gb300", "sm_103"),
    "H200": ("h200", "sm_90a"),
    "H100": ("h100", "sm_90a"),
    "A100": ("a100", "sm_80"),
    "RTX6000": ("rtx6000", "sm_89"),
}

RC_OK = 0
RC_COMPILE = 1
RC_CRASH = 2
RC_WRONG = 3
RC_TIMEOUT = 4
RC_SETUP = 5

_CHECKOUT_SKIP_DIRS = {
    "__pycache__",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "build",
    "dist",
}

MAX_WORKSPACE_FILES = 10_000
MAX_WORKSPACE_BYTES = 1024**3
