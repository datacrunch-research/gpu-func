"""CLI-specific exception type carrying an intended process exit code."""

from __future__ import annotations

from .constants import RC_SETUP


class CliError(RuntimeError):
    def __init__(self, message: str, exit_code: int = RC_SETUP) -> None:
        super().__init__(message)
        self.exit_code = exit_code
