"""Local Nsight Compute report summary and course feedback commands."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from .constants import RC_OK
from .errors import CliError


def _cmd_report(args: argparse.Namespace) -> int:
    if args.report_command == "summary":
        return _cmd_report_summary(args)
    if args.report_command == "feedback":
        return _cmd_report_feedback(args)
    raise CliError(f"unknown report command {args.report_command!r}")


def _cmd_report_summary(args: argparse.Namespace) -> int:
    report_path = _validate_ncu_report_path(args.report)
    try:
        ncu_utils = importlib.import_module("gfaas_cli.assets.course_runner.ncu_utils")
        if getattr(ncu_utils, "ncu_report", None) is None:
            raise CliError(
                "ncu_report.py is not available. Install Nsight Compute or set "
                "PYTHONPATH to its extras/python directory."
            )
        profile_mod = importlib.import_module("gfaas_cli.assets.course_runner.profile")
    except CliError:
        raise
    except Exception as exc:
        raise CliError(f"failed to import packaged report parser: {exc}") from exc

    try:
        metrics, device = profile_mod.extract_curated_metrics(str(report_path))
        aggregate = profile_mod.aggregate_profiling_info(metrics)
    except Exception as exc:
        raise CliError(f"failed to summarize report: {exc}") from exc

    print("Report summary")
    print(f"Report: {report_path}")
    print(f"Device: {device}")
    print(f"Kernels: {aggregate.get('num_kernels', len(metrics))}")
    _print_summary_metric("Duration", aggregate.get("duration"), formatter=_format_duration_ns)
    _print_summary_metric("Cycles", aggregate.get("cycles"), formatter=_format_count)
    _print_summary_metric("Instructions", aggregate.get("instructions"), formatter=_format_count)
    _print_summary_metric("DRAM read", aggregate.get("dram_read_bytes"), formatter=_format_bytes)
    _print_summary_metric("DRAM write", aggregate.get("dram_write_bytes"), formatter=_format_bytes)
    _print_summary_metric("DRAM throughput", aggregate.get("dram_throughput"), suffix="%")
    _print_summary_metric("SM throughput", aggregate.get("sm_throughput"), suffix="%")
    _print_summary_metric("Occupancy", aggregate.get("occupancy"), suffix="%")
    _print_summary_metric("Global loads", aggregate.get("loads"), formatter=_format_count)
    _print_summary_metric("Global stores", aggregate.get("stores"), formatter=_format_count)
    _print_summary_metric("LDGSTS", aggregate.get("ldgsts"), formatter=_format_count)

    if args.per_kernel:
        print("\nPer-kernel metrics")
        for kernel_name, kernel_metrics in metrics.items():
            print(f"- {kernel_name}")
            _print_summary_metric(
                "  Duration", kernel_metrics.get("duration"), formatter=_format_duration_ns
            )
            _print_summary_metric(
                "  Instructions", kernel_metrics.get("instructions"), formatter=_format_count
            )
            _print_summary_metric(
                "  DRAM read", kernel_metrics.get("dram_read_bytes"), formatter=_format_bytes
            )
            _print_summary_metric(
                "  DRAM write", kernel_metrics.get("dram_write_bytes"), formatter=_format_bytes
            )
            _print_summary_metric(
                "  DRAM throughput", kernel_metrics.get("dram_throughput"), suffix="%"
            )

    if args.json_path:
        out = {
            "mode": "report-summary",
            "report": str(report_path),
            "device": device,
            "aggregate": _json_safe_dict(aggregate),
            "kernels": _json_safe_dict(metrics),
        }
        _write_json(Path(args.json_path), out)
    return RC_OK


def _cmd_report_feedback(args: argparse.Namespace) -> int:
    report_path = _validate_ncu_report_path(args.report)
    if not args.trust_course_code:
        raise CliError(
            "report feedback imports and executes the course exercise run.py locally; "
            "inspect the course and pass --trust-course-code to continue"
        )
    if not args.course_dir:
        raise CliError("report feedback needs --course-dir (or set CUDA_COURSE_DIR)")
    course_dir = Path(args.course_dir).expanduser().resolve()
    if not course_dir.is_dir():
        raise CliError(f"course directory not found: {course_dir}")
    runner_dir = course_dir / "runner"
    if not runner_dir.is_dir():
        raise CliError(f"course directory does not contain runner/: {course_dir}")

    benchmark_path = _resolve_course_exercise_path(
        course_dir=course_dir,
        exercise_id=args.exercise,
        rel_or_abs=args.benchmark,
    )
    if not benchmark_path.is_file():
        raise CliError(f"benchmark config not found: {benchmark_path}")

    exercise_run_py = course_dir / "exercises" / args.exercise / "run.py"
    if not exercise_run_py.is_file():
        raise CliError(f"exercise run.py not found: {exercise_run_py}")

    _prepend_sys_path(course_dir)
    try:
        ncu_utils = importlib.import_module("runner.ncu_utils")
        if getattr(ncu_utils, "ncu_report", None) is None:
            raise CliError(
                "ncu_report.py is not available. Install Nsight Compute or set "
                "PYTHONPATH to its extras/python directory."
            )
        profile_mod = importlib.import_module("runner.profile")
        results_mod = importlib.import_module("runner.results")
        testing_mod = importlib.import_module("runner.testing")
    except CliError:
        raise
    except Exception as exc:
        raise CliError(f"failed to import CUDA course runner modules: {exc}") from exc

    try:
        spec = importlib.util.spec_from_file_location(
            f"gfaas_cli_{args.exercise.replace('-', '_')}_run",
            exercise_run_py,
        )
        if spec is None or spec.loader is None:
            raise CliError(f"failed to load exercise module: {exercise_run_py}")
        exercise_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exercise_mod)
        exercise = exercise_mod.exercise

        config = testing_mod.parse_test_file(benchmark_path)
        metrics, device = profile_mod.extract_curated_metrics(str(report_path))
        payload = results_mod.ProfileReport(status="passed", device=device, metrics=metrics)
        feedback = list(exercise.format_profiling(payload, config))
    except CliError:
        raise
    except Exception as exc:
        raise CliError(f"failed to generate feedback: {exc}") from exc

    print(f"Course feedback: {args.exercise}")
    print(f"Report: {report_path}")
    print(f"Benchmark: {benchmark_path}")
    print(f"Device: {device}")
    if not feedback:
        print("No feedback returned")
    for item in feedback:
        kind = getattr(item, "kind", "info")
        text = getattr(item, "text", str(item)).strip()
        print(f"[{kind}] {text}")

    if args.json_path:
        out = {
            "mode": "report-feedback",
            "report": str(report_path),
            "course_dir": str(course_dir),
            "exercise": args.exercise,
            "benchmark": str(benchmark_path),
            "device": device,
            "feedback": [
                {
                    "kind": getattr(item, "kind", "info"),
                    "text": getattr(item, "text", str(item)),
                }
                for item in feedback
            ],
        }
        _write_json(Path(args.json_path), out)
    return RC_OK


def _validate_ncu_report_path(path: str) -> Path:
    report_path = Path(path).expanduser().resolve()
    if not report_path.is_file():
        raise CliError(f"report file not found: {report_path}")
    if report_path.suffix != ".ncu-rep":
        raise CliError(f"expected a .ncu-rep report file, got: {report_path}")
    return report_path


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as output:
            json.dump(value, output, indent=2)
            output.write("\n")
    except FileExistsError as exc:
        raise CliError(f"refusing to replace existing result file: {path}") from exc
    print(f"Results written to {path}")


def _print_summary_metric(
    label: str,
    value: Any,
    *,
    formatter: Any | None = None,
    suffix: str = "",
) -> None:
    if value is None:
        return
    if formatter:
        rendered = formatter(value)
    elif isinstance(value, float):
        rendered = f"{value:.1f}"
    else:
        rendered = str(value)
    print(f"{label}: {rendered}{suffix}")


def _format_count(value: Any) -> str:
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return str(value)


def _format_bytes(value: Any) -> str:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(value_f) >= 1024 * 1024 * 1024:
        return f"{value_f / (1024 * 1024 * 1024):.1f} GiB"
    if abs(value_f) >= 1024 * 1024:
        return f"{value_f / (1024 * 1024):.1f} MiB"
    if abs(value_f) >= 1024:
        return f"{value_f / 1024:.1f} KiB"
    return f"{value_f:.0f} B"


def _format_duration_ns(value: Any) -> str:
    try:
        ns = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(ns) >= 1_000_000:
        return f"{ns / 1_000_000:.3f} ms"
    if abs(ns) >= 1_000:
        return f"{ns / 1_000:.1f} us"
    return f"{ns:.0f} ns"


def _json_safe_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe_dict(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe_dict(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _prepend_sys_path(path: Path) -> None:
    resolved = str(path)
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _resolve_course_exercise_path(course_dir: Path, exercise_id: str, rel_or_abs: str) -> Path:
    path = Path(rel_or_abs).expanduser()
    if path.is_absolute():
        return path
    normalized = path.as_posix()
    if normalized.startswith("exercises/"):
        return course_dir / path
    return course_dir / "exercises" / exercise_id / path
