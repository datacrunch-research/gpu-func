"""Public CLI entry point for gpu-func."""

from __future__ import annotations

import sys

from .client import GfaasClient
from .commands import _cmd_custom, _cmd_exercise, _cmd_exercise_mode, _cmd_workers
from .constants import RC_SETUP
from .errors import CliError
from .parser import EXERCISE_MODES
from .parser import build_parser as _build_parser
from .reports import _cmd_report


def main(argv: list[str] | None = None) -> int:
    # Single funnel for the whole CLI: any CliError raised deep in a journey
    # surfaces here as a message + its exit code (constants.RC_*).
    try:
        return _main(argv)
    except CliError as exc:
        print(f"gpu-func: {exc}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("gpu-func: interrupted", file=sys.stderr)
        return 130


def _main(argv: list[str] | None = None) -> int:
    parser = _build_parser()  # parser.py: the argparse grammar
    args = parser.parse_args(argv)
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
    parser.print_help()
    return RC_SETUP
