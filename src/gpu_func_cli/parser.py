"""Argparse construction for the public gpu-func command surface."""

from __future__ import annotations

import argparse
import os
import re

# The exercise actions, usable both as `exercise <id> <mode>` and as a
# top-level `gpu-func <mode>` that auto-detects the exercise from the cwd.
EXERCISE_MODES = ["compile", "test", "benchmark", "sanitizer", "profile", "grade"]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must not be negative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must not be negative")
    return parsed


def _byte_size(value: str) -> int:
    match = re.fullmatch(r"([0-9]+)([KMGTP]i?B|B)?", value, re.IGNORECASE)
    if match is None:
        raise argparse.ArgumentTypeError("use bytes or a size such as 512MiB or 4GiB")
    amount = int(match.group(1))
    suffix = (match.group(2) or "B").upper()
    powers = {
        "B": 0,
        "KB": 1,
        "KIB": 1,
        "MB": 2,
        "MIB": 2,
        "GB": 3,
        "GIB": 3,
        "TB": 4,
        "TIB": 4,
        "PB": 5,
        "PIB": 5,
    }
    return amount * 1024 ** powers[suffix]


def _environment(value: str) -> tuple[str, str]:
    name, separator, setting = value.partition("=")
    if not separator or not name:
        raise argparse.ArgumentTypeError("environment variables must use NAME=VALUE")
    return name, setting


def _add_remote_opts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--gpu",
        help="legacy GPU label such as B200 or GB300; prefer --gpu-type and --arch",
    )
    parser.add_argument(
        "--gpu-type",
        default=os.environ.get("GFAAS_GPU_TYPE"),
        help="gfaas GPU pool; the only configured pool is selected automatically",
    )
    parser.add_argument("--gpu-count", type=_positive_int, default=1)
    parser.add_argument("--image", default=os.environ.get("GFAAS_IMAGE", "cuda-nvcc"))
    parser.add_argument("--arch")
    parser.add_argument("--timeout", type=_positive_int, default=600)
    parser.add_argument("--capacity-wait", type=_nonnegative_int)
    parser.add_argument("--wait-timeout", type=_nonnegative_float)
    parser.add_argument("--cpu-millicores", type=_positive_int)
    parser.add_argument("--memory", type=_byte_size)
    parser.add_argument("--storage", type=_byte_size)
    parser.add_argument("--shared-memory", type=_byte_size)
    parser.add_argument("--max-log", type=_byte_size)
    parser.add_argument("--max-output", type=_byte_size)
    parser.add_argument("--env", action="append", type=_environment, default=[])
    parser.add_argument("--idempotency-key")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument(
        "--json-events",
        action="store_true",
        help="write durable Call events as JSON Lines while waiting",
    )


def _add_common_exercise_opts(p: argparse.ArgumentParser) -> None:
    """Options shared by `exercise` and every top-level mode command.

    Kept identical between the two surfaces so they behave the same; the only
    difference is how the exercise is located (positional id vs. cwd auto-detect).
    """
    p.add_argument("specs", nargs="*")
    p.add_argument("--file", dest="source_file")
    p.add_argument(
        "--course-root",
        default=os.environ.get("CUDA_COURSE_REPO"),
        help="path to a cuda-course checkout (or set CUDA_COURSE_REPO). "
        "Default: auto-detect from --file / the cwd.",
    )
    p.add_argument(
        "--exercise-dir",
        help="path to a flat exercise dir (run.py + runner/ side by side, e.g. an "
        "unzipped exercise). Runs it directly, bypassing the cuda-course layout.",
    )
    _add_remote_opts(p)
    p.add_argument("--json", dest="json_path")
    p.add_argument("--artifact-dir")
    p.add_argument("--verbose", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpu-func")
    parser.add_argument("--api-base", default=os.environ.get("GFAAS_API_BASE"))
    parser.add_argument("--request-timeout", type=_positive_float)
    parser.add_argument("--poll-interval", type=_nonnegative_float)

    sub = parser.add_subparsers(dest="command_name")

    sub.add_parser("workers", aliases=["pools"], help="List configured gfaas GPU pools")

    exercise = sub.add_parser("exercise", help="Run a course exercise action")
    exercise.add_argument("exercise_id")
    exercise.add_argument("exercise_command", choices=EXERCISE_MODES)
    _add_common_exercise_opts(exercise)

    # Top-level shortcuts: `gpu-func benchmark [specs...]` auto-detects the
    # exercise from the cwd (an unzipped exercise: run.py + runner/ siblings), so
    # the `exercise <id>` prefix and `--exercise-dir` become optional. Passing
    # --exercise-dir still works from anywhere. With no specs, the runner runs
    # every test/benchmark for that mode.
    for mode in EXERCISE_MODES:
        mp = sub.add_parser(
            mode,
            help=f"Run the {mode} action on the exercise in the cwd (or --exercise-dir)",
        )
        mp.add_argument(
            "--exercise-id",
            help="exercise id for reporting (default: the exercise dir name)",
        )
        _add_common_exercise_opts(mp)
        mp.set_defaults(exercise_command=mode)

    custom = sub.add_parser("custom", help="Compile, run, or profile a custom CUDA program")
    custom.add_argument("custom_command", choices=["compile", "run", "profile"])
    custom.add_argument("source", help="CUDA source file containing the kernel or host wrapper")
    custom.add_argument("--harness", help="Optional CUDA/C++ source file containing main()")
    custom.add_argument("--output", default="custom_kernel")
    custom.add_argument("--arg", action="append", default=[], help="Program argument, repeatable")
    custom.add_argument("--nvcc-flags", default="-std=c++20 -O3 -lineinfo")
    _add_remote_opts(custom)
    custom.add_argument("--json", dest="json_path")
    custom.add_argument("--artifact-dir")
    custom.add_argument("--ncu-args", default="--set basic")
    custom.add_argument("--nvtx-range", default="profile_kernel")
    custom.add_argument("--no-nvtx-filter", action="store_true")
    custom.add_argument(
        "--report-name",
        help="base name for the profile .ncu-rep (default: source file stem)",
    )
    custom.add_argument("--verbose", action="store_true")

    report = sub.add_parser("report", help="Inspect local Nsight Compute reports")
    report_sub = report.add_subparsers(dest="report_command", required=True)
    summary = report_sub.add_parser(
        "summary",
        help="Print a generic metric summary from a local .ncu-rep",
    )
    summary.add_argument("report", help="Path to a local .ncu-rep file")
    summary.add_argument("--per-kernel", action="store_true")
    summary.add_argument("--json", dest="json_path")

    feedback = report_sub.add_parser(
        "feedback",
        help="Run CUDA course feedback rules against a local .ncu-rep",
    )
    feedback.add_argument("report", help="Path to a local .ncu-rep file")
    feedback.add_argument(
        "--course-dir",
        default=os.environ.get("CUDA_COURSE_DIR"),
        help="CUDA course checkout containing runner/ and exercises/ (or set CUDA_COURSE_DIR)",
    )
    feedback.add_argument("--exercise", default="01-haxpy")
    feedback.add_argument("--benchmark", default="benchmarks/01_aligned_small.txt")
    feedback.add_argument("--json", dest="json_path")
    feedback.add_argument("--verbose", action="store_true")
    feedback.add_argument(
        "--trust-course-code",
        action="store_true",
        help="allow report feedback to import and execute the course exercise run.py locally",
    )
    return parser
