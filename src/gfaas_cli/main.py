"""Public entry point for the unified gfaas command."""

# PYTHON_ARGCOMPLETE_OK

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from typing import NoReturn

import argcomplete
import httpx

from gfaas import Client
from gfaas.errors import GfaasError

from . import general
from .client import GfaasClient, sdk_client_from_args
from .commands import _cmd_custom, _cmd_exercise, _cmd_exercise_mode, _cmd_workers
from .constants import RC_SETUP
from .errors import CliError
from .parser import EXERCISE_MODES
from .parser import build_parser as _build_parser
from .reports import _cmd_report


def main(
    argv: Sequence[str] | None = None,
    *,
    client_factory: Callable[[], Client] | None = None,
) -> int:
    """Run one CLI command and return its process exit status."""
    try:
        return _main(argv, client_factory=client_factory)
    except CliError as exc:
        print(f"gfaas: error: {exc}", file=sys.stderr)
        return exc.exit_code
    except (GfaasError, httpx.HTTPError, OSError, UnicodeError, ValueError) as exc:
        print(f"gfaas: error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ngfaas: interrupted", file=sys.stderr)
        return 130


def _main(
    argv: Sequence[str] | None,
    *,
    client_factory: Callable[[], Client] | None,
) -> int:
    command_line, program_args = general._split_program_args(
        argv if argv is not None else sys.argv[1:]
    )
    parser = _build_parser()  # parser.py: the argparse grammar
    if argv is None:
        argcomplete.autocomplete(parser)
    args = parser.parse_args(command_line)
    if program_args and not general.accepts_program_args(args):
        parser.error("program arguments after -- require the run command")

    if args.command_name in general.GENERAL_COMMANDS:
        factory = client_factory or (lambda: sdk_client_from_args(args))
        try:
            return general.dispatch(args, program_args, client_factory=factory)
        except CliError as exc:
            # The general commands historically use status 1 for runtime errors.
            raise CliError(str(exc), exit_code=1) from exc

    # Remote workflows submit a durable Call. Report inspection stays local.
    if args.command_name in {"workers", "pools"}:
        with GfaasClient.from_args(args) as client:
            return _cmd_workers(client)
    if args.command_name == "exercise":
        return _cmd_exercise(args)  # explicit course exercise
    if args.command_name in EXERCISE_MODES:
        return _cmd_exercise_mode(args)  # top-level shortcut: cwd-detected exercise
    if args.command_name == "custom":
        return _cmd_custom(args)  # arbitrary kernel (+ optional harness)
    if args.command_name == "report":
        return _cmd_report(args)  # local: summary / feedback on a .ncu-rep
    return RC_SETUP


def entrypoint() -> NoReturn:
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()
