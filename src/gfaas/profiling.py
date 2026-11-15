"""Profiling requests, typed outcomes, and the Nsight Compute run that produces them.

A profiled run has two results that must not be confused: what the program did and what the
profiler captured. A program can exit 0 while the profiler captures no kernel, is denied the GPU
counters, fails, or runs out of its own time budget. :func:`run_profiler` runs the program under
``ncu`` and classifies that second result; :class:`ProfilingOutcome` is its typed form.

This module uses only the standard library. It ships in every source bundle, so the same code runs
in a remote Call and on a local workstation.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .errors import GfaasError

#: The service capability a profiled Call asks for. The caller's API key needs the grant.
GPU_COUNTERS = "gpu_counters"

METRIC_SETS = ("basic", "full")
MAXIMUM_ARGUMENTS = 32
MAXIMUM_ARGUMENT_CHARACTERS = 256
MAXIMUM_TIMEOUT_SECONDS = 86_400
#: A CSV report up to this size is returned inside the Call result as well as published.
INLINE_REPORT_BYTES = 1024 * 1024
DEFAULT_MAX_REPORT_BYTES = 64 * 1024 * 1024
MAXIMUM_REPORT_BYTES = 1024 * 1024 * 1024
_MAXIMUM_MESSAGES = 50
_MAXIMUM_MESSAGE_CHARACTERS = 500

PROFILED = "profiled"
NO_KERNELS = "no_kernels"
PERMISSION_DENIED = "permission_denied"
PROFILER_FAILED = "profiler_failed"
PROFILER_TIMED_OUT = "profiler_timed_out"
PROFILER_UNAVAILABLE = "profiler_unavailable"
PROGRAM_FAILED = "program_failed"
REPORT_TOO_LARGE = "report_too_large"
STATUSES = (
    PROFILED,
    NO_KERNELS,
    PERMISSION_DENIED,
    PROFILER_FAILED,
    PROFILER_TIMED_OUT,
    PROFILER_UNAVAILABLE,
    PROGRAM_FAILED,
    REPORT_TOO_LARGE,
)

# The run decides where the report goes and in which format, so a request cannot.
_RESERVED_OPTIONS = ("--export", "-o", "--import", "-i", "--log-file", "--csv", "--force-overwrite")
_METRIC_OPTIONS = ("--set", "--section", "--metrics", "--config-file")
_MESSAGE = re.compile(r"^==(PROF|WARNING|ERROR)== ?(.*)$")
_PASSES = re.compile(r'^Profiling ".*" - \d+.*: .* - (\d+) pass(?:es)?$')
_APPLICATION_ERROR = re.compile(r"^The application returned an error code \((-?\d+)\)")
_PERMISSION = re.compile(r"ERR_NVGPUCTRPERM|does not have permission", re.IGNORECASE)


class ProfilingNotGrantedError(GfaasError):
    """The API key has no grant for the profiling capability the Call asks for."""


class ProfilingUnsupportedError(GfaasError):
    """The service does not offer the profiling capability the Call asks for."""


class ProfilingError(GfaasError):
    """A profiled run did not produce a profile. ``outcome`` says why."""

    def __init__(self, outcome: ProfilingOutcome) -> None:
        self.outcome = outcome
        super().__init__(f"profiling ended as {outcome.status}: {outcome.detail}")


@dataclass(frozen=True)
class ProfilingRequest:
    """What to collect from one Nsight Compute run, and its budgets.

    ``metric_set`` names a supported set. ``arguments`` adds a bounded list of ``ncu`` options,
    for example ``("--section", "SpeedOfLight")``; pass ``metric_set=None`` when they select the
    metrics themselves. The report location and format belong to the run and cannot be set here.

    ``timeout_s`` bounds the profiled run alone, apart from the Call's own timeout. ``launch_count``
    bounds how many kernel launches the profiler replays. ``baseline`` runs the program once without
    the profiler first, which gives the program's own result and the profiling slowdown.
    """

    metric_set: str | None = "full"
    arguments: tuple[str, ...] = ()
    timeout_s: int = 300
    launch_count: int | None = None
    max_report_bytes: int = DEFAULT_MAX_REPORT_BYTES
    baseline: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", tuple(self.arguments))
        if self.metric_set is not None and self.metric_set not in METRIC_SETS:
            raise GfaasError(f"metric_set must be one of {', '.join(METRIC_SETS)}, or None")
        if len(self.arguments) > MAXIMUM_ARGUMENTS:
            raise GfaasError(f"a profiling request takes at most {MAXIMUM_ARGUMENTS} arguments")
        for argument in self.arguments:
            if not isinstance(argument, str) or not argument or "\0" in argument:
                raise GfaasError("profiler arguments must be non-empty strings")
            if len(argument) > MAXIMUM_ARGUMENT_CHARACTERS:
                raise GfaasError(
                    f"a profiler argument is longer than {MAXIMUM_ARGUMENT_CHARACTERS} characters"
                )
            if argument.split("=", 1)[0] in _RESERVED_OPTIONS:
                raise GfaasError(f"the profiled run sets {argument.split('=', 1)[0]} itself")
        if self.metric_set is None and not self.selects_metrics(self.arguments):
            raise GfaasError("name a metric_set or select metrics in the arguments")
        _require_range("timeout_s", self.timeout_s, 1, MAXIMUM_TIMEOUT_SECONDS)
        _require_range("max_report_bytes", self.max_report_bytes, 1, MAXIMUM_REPORT_BYTES)
        if self.launch_count is not None:
            _require_range("launch_count", self.launch_count, 1, 1_000_000)
        if not isinstance(self.baseline, bool):
            raise GfaasError("baseline must be true or false")

    @staticmethod
    def selects_metrics(arguments: Sequence[str]) -> bool:
        return any(argument.split("=", 1)[0] in _METRIC_OPTIONS for argument in arguments)

    @classmethod
    def from_arguments(
        cls, arguments: Sequence[str], *, default_set: str = "full", **budgets: Any
    ) -> ProfilingRequest:
        """A request from raw ``ncu`` options. They replace the default set if they choose one."""
        metric_set = None if cls.selects_metrics(arguments) else default_set
        return cls(metric_set=metric_set, arguments=tuple(arguments), **budgets)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProfilingRequest:
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise GfaasError(f"unknown profiling request fields: {', '.join(sorted(unknown))}")
        return cls(**{**value, "arguments": tuple(value.get("arguments", ()))})

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "arguments": list(self.arguments)}

    def command_arguments(self) -> list[str]:
        """The ``ncu`` options this request stands for, without the report or the program."""
        arguments = ["--set", self.metric_set] if self.metric_set else []
        if self.launch_count is not None:
            arguments += ["--launch-count", str(self.launch_count)]
        return [*arguments, *self.arguments]


@dataclass(frozen=True)
class ProfilingReport:
    """One report file from a profiled run, identified by its size and digest.

    ``output_name`` is the Call output the file was published under, or ``None`` when it was not
    published. ``artifact_id`` is known once the Call's Artifacts have been listed.
    """

    format: str
    filename: str
    size_bytes: int
    sha256: str
    output_name: str | None = None
    artifact_id: str | None = None


@dataclass(frozen=True)
class ProfilingOutcome:
    """What a profiled run captured, apart from what the program did.

    ``status`` is one of :data:`STATUSES`. ``program_returncode`` is the program's own exit status
    when the run establishes it, and ``None`` when the program never ran or the profiler hid it.
    """

    status: str
    detail: str = ""
    program_returncode: int | None = None
    profiler_returncode: int | None = None
    profiler_ms: int = 0
    baseline_ms: int | None = None
    kernels: int = 0
    passes: int | None = None
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    reports: tuple[ProfilingReport, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    call_id: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == PROFILED

    @property
    def slowdown(self) -> float | None:
        """Profiled run time over the unprofiled run time, when a baseline was measured."""
        if not self.baseline_ms or not self.profiler_ms:
            return None
        return self.profiler_ms / self.baseline_ms

    def report(self, format: str) -> ProfilingReport | None:
        return next((report for report in self.reports if report.format == format), None)

    def raise_for_status(self) -> None:
        if not self.ok:
            raise ProfilingError(self)

    @classmethod
    def from_dict(cls, value: object, *, call_id: str | None = None) -> ProfilingOutcome:
        if not isinstance(value, Mapping) or value.get("status") not in STATUSES:
            raise GfaasError("the result carries no valid profiling outcome")
        fields = {name: value[name] for name in cls.__dataclass_fields__ if name in value}
        fields["warnings"] = tuple(value.get("warnings", ()))
        fields["errors"] = tuple(value.get("errors", ()))
        fields["reports"] = tuple(ProfilingReport(**report) for report in value.get("reports", ()))
        fields["call_id"] = call_id or value.get("call_id")
        return cls(**fields)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "provenance": dict(self.provenance)}


def outcome_from_result(result: object, *, call_id: str | None = None) -> ProfilingOutcome | None:
    """The profiling outcome a Call result carries, or ``None`` for an unprofiled run."""
    if not isinstance(result, Mapping) or result.get("profiling") is None:
        return None
    return ProfilingOutcome.from_dict(result["profiling"], call_id=call_id)


def download_reports(
    client: Any,
    outcome: ProfilingOutcome,
    destination: str | os.PathLike[str],
    *,
    max_bytes: int = DEFAULT_MAX_REPORT_BYTES,
) -> list[Path]:
    """Download an outcome's published reports and check each against its recorded digest.

    Refuses before any transfer when the reports together exceed ``max_bytes``, and never
    replaces an existing file.
    """
    published = [report for report in outcome.reports if report.artifact_id]
    if not published:
        return []
    total = sum(report.size_bytes for report in published)
    if total > max_bytes:
        raise GfaasError(f"the reports are {total} bytes; the transfer limit is {max_bytes}")
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    wanted = [target / report.filename for report in published]
    for path in wanted:
        if path.exists() or path.is_symlink():
            raise GfaasError(f"refusing to replace existing profile report: {path}")
    downloaded: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="vfunc-profiles-") as temporary:
        trees: dict[str, Path] = {}
        for report, path in zip(published, wanted, strict=True):
            assert report.artifact_id is not None
            if report.artifact_id not in trees:
                tree = Path(temporary, str(len(trees)))
                client.download_artifact_directory(report.artifact_id, tree)
                trees[report.artifact_id] = tree
            source = trees[report.artifact_id] / report.filename
            if not source.is_file() or _report_record(source, report.format)["sha256"] != (
                report.sha256
            ):
                raise GfaasError(f"downloaded report {report.filename} does not match its digest")
            shutil.copyfile(source, path)
            downloaded.append(path)
    return downloaded


def run_profiler(
    program: Sequence[str],
    request: ProfilingRequest,
    *,
    ncu: str | None,
    cwd: str | os.PathLike[str],
    env: Mapping[str, str],
    report_directory: str | os.PathLike[str],
    report_name: str = "profile",
    publish_directory: str | os.PathLike[str] | None = None,
    output_name: str | None = None,
    remaining_s: float | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run ``program`` under Nsight Compute and classify what the profiler captured.

    Returns ``{"process", "outcome", "csv"}``. ``process`` describes the profiled run with the
    profiler's own messages removed from its output. ``outcome`` is the plain form of
    :class:`ProfilingOutcome`. ``csv`` is the CSV report when it fits :data:`INLINE_REPORT_BYTES`.

    Reports within ``request.max_report_bytes`` are copied to ``publish_directory`` with a
    provenance record beside them. ``remaining_s`` is the Call time still available; the profiler
    gets the smaller of that and its own budget.
    """
    record: dict[str, Any] = {
        "request": request.to_dict(),
        "arguments": request.command_arguments(),
        **dict(provenance or {}),
    }
    outcome: dict[str, Any] = {
        "status": PROFILER_UNAVAILABLE,
        "detail": "Nsight Compute (ncu) is not installed in this environment",
        "program_returncode": None,
        "profiler_returncode": None,
        "profiler_ms": 0,
        "baseline_ms": None,
        "kernels": 0,
        "passes": None,
        "warnings": [],
        "errors": [],
        "reports": [],
        "provenance": record,
    }
    process: dict[str, Any] = {
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "ms": 0,
        "timed_out": False,
    }
    result: dict[str, Any] = {"process": process, "outcome": outcome, "csv": None}
    if ncu is None:
        return result
    record["profiler"] = {"name": "ncu", "version": _profiler_version(ncu, env)}
    record["gpus"] = _gpus(env)

    started = time.monotonic()
    if request.baseline:
        baseline = _run(list(program), cwd, env, remaining_s)
        process.update(baseline)
        outcome["baseline_ms"] = baseline["ms"]
        if baseline["timed_out"] or baseline["returncode"] != 0:
            outcome["program_returncode"] = baseline["returncode"]
            outcome["status"] = PROGRAM_FAILED
            outcome["detail"] = "the program failed before it was profiled"
            return result
    if remaining_s is not None:
        remaining_s -= time.monotonic() - started
    budget = float(request.timeout_s)
    hit_call_deadline = remaining_s is not None and remaining_s < budget
    if hit_call_deadline:
        assert remaining_s is not None
        budget = max(remaining_s, 0.001)

    report_root = Path(report_directory)
    report = report_root / f"{report_name}.ncu-rep"
    report.unlink(missing_ok=True)
    command = [
        ncu,
        *request.command_arguments(),
        "--force-overwrite",
        "--export",
        str(report_root / report_name),
        *program,
    ]
    profiled = _run(command, cwd, env, budget)
    program_output, messages = _split_messages(profiled["stdout"])
    warnings = [text for level, text in messages if level == "WARNING"]
    errors = [text for level, text in messages if level == "ERROR"]
    passes = [int(match.group(1)) for _, text in messages if (match := _PASSES.match(text))]
    application = next((m for text in errors if (m := _APPLICATION_ERROR.match(text))), None)
    profiler_errors = [text for text in errors if not _APPLICATION_ERROR.match(text)]

    # With a baseline, the first run already is the program's result.
    if not request.baseline:
        process.update(profiled)
        process["stdout"] = program_output
    outcome.update(
        {
            "profiler_returncode": profiled["returncode"],
            "profiler_ms": profiled["ms"],
            "kernels": len(passes),
            "passes": sum(passes) if passes else None,
            "warnings": _bounded(warnings),
            "errors": _bounded(errors),
        }
    )
    if request.baseline:
        outcome["program_returncode"] = 0
    elif application is not None:
        outcome["program_returncode"] = int(application.group(1))
    elif profiled["returncode"] == 0:
        outcome["program_returncode"] = 0

    def conclude(status: str, detail: str) -> dict[str, Any]:
        outcome["status"], outcome["detail"] = status, detail
        return result

    if profiled["timed_out"]:
        if hit_call_deadline:
            return conclude(
                PROFILER_TIMED_OUT, "the Call's time ran out while the profiler was running"
            )
        return conclude(
            PROFILER_TIMED_OUT, f"the profiler exceeded its {request.timeout_s} s budget"
        )
    if any(_PERMISSION.search(text) for text in errors):
        return conclude(PERMISSION_DENIED, next(t for t in errors if _PERMISSION.search(t)))
    if profiler_errors or (profiled["returncode"] != 0 and application is None):
        detail = profiler_errors[0] if profiler_errors else profiled["stderr"].strip()[-500:]
        return conclude(
            PROFILER_FAILED, detail or f"ncu exited with status {profiled['returncode']}"
        )
    if not report.is_file():
        if application is not None:
            return conclude(PROGRAM_FAILED, errors[0])
        return conclude(NO_KERNELS, warnings[0] if warnings else "the profiler wrote no report")

    size = report.stat().st_size
    reports = [_report_record(report, "ncu-rep")]
    if size > request.max_report_bytes:
        outcome["reports"] = reports
        return conclude(
            REPORT_TOO_LARGE,
            f"the report is {size} bytes; the request allows {request.max_report_bytes}",
        )
    csv_path = report_root / f"{report_name}.csv"
    csv_text = _export_csv(ncu, report, csv_path, cwd, env)
    if csv_text is not None:
        reports.append(_report_record(csv_path, "csv"))
        if len(csv_text.encode()) <= INLINE_REPORT_BYTES:
            result["csv"] = csv_text
        if not outcome["kernels"]:
            outcome["kernels"] = _csv_kernels(csv_text)
    if publish_directory is not None:
        published = Path(publish_directory)
        published.mkdir(parents=True, exist_ok=True)
        for item in reports:
            destination = published / item["filename"]
            if destination.exists():
                raise FileExistsError(f"duplicate profile report name: {item['filename']}")
            shutil.copyfile(report_root / item["filename"], destination)
            item["output_name"] = output_name
    outcome["reports"] = reports
    conclude(PROFILED, f"{outcome['kernels']} kernel launches profiled")
    if publish_directory is not None:
        # The record travels with the report, so a downloaded report still says what made it.
        durable = {key: value for key, value in outcome.items() if key != "provenance"}
        text = json.dumps({**durable, "provenance": record}, indent=2, sort_keys=True)
        Path(publish_directory, f"{report_name}.provenance.json").write_text(
            text + "\n", encoding="utf-8"
        )
    return result


def _require_range(name: str, value: object, low: int, high: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
        raise GfaasError(f"{name} must be an integer from {low} to {high}")


def _run(
    command: list[str],
    cwd: str | os.PathLike[str],
    env: Mapping[str, str],
    timeout_s: float | None,
) -> dict[str, Any]:
    """Run one command in its own process group, so a timeout also ends the profiled program."""
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=True,
        )
    except OSError as error:
        return {"returncode": 127, "stdout": "", "stderr": str(error), "ms": 0, "timed_out": False}
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
    return {
        "returncode": None if timed_out else process.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "ms": int((time.monotonic() - started) * 1000),
        "timed_out": timed_out,
    }


def _split_messages(stdout: str) -> tuple[str, list[tuple[str, str]]]:
    """Separate the profiler's ``==PROF==``, ``==WARNING==`` and ``==ERROR==`` lines."""
    program: list[str] = []
    messages: list[tuple[str, str]] = []
    for line in stdout.splitlines(keepends=True):
        match = _MESSAGE.match(line.rstrip("\r\n"))
        if match:
            messages.append((match.group(1), match.group(2)))
        else:
            program.append(line)
    return "".join(program), messages


def _bounded(messages: list[str]) -> list[str]:
    return [text[:_MAXIMUM_MESSAGE_CHARACTERS] for text in messages[:_MAXIMUM_MESSAGES]]


def _report_record(path: Path, format: str) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "format": format,
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "output_name": None,
        "artifact_id": None,
    }


def _export_csv(
    ncu: str,
    report: Path,
    destination: Path,
    cwd: str | os.PathLike[str],
    env: Mapping[str, str],
) -> str | None:
    """The report's metrics as CSV. Reading a finished report runs nothing on the GPU."""
    exported = _run([ncu, "--import", str(report), "--csv"], cwd, env, 120.0)
    if exported["timed_out"] or exported["returncode"] != 0 or not exported["stdout"]:
        return None
    text, _ = _split_messages(exported["stdout"])
    destination.write_text(text, encoding="utf-8")
    return text


def _csv_kernels(csv_text: str) -> int:
    identifiers = {
        line.split(",", 1)[0] for line in csv_text.splitlines()[1:] if line.startswith('"')
    }
    return len(identifiers)


def _profiler_version(ncu: str, env: Mapping[str, str]) -> str | None:
    result = _run([ncu, "--version"], os.getcwd(), env, 20.0)
    match = re.search(r"^Version (.+)$", result["stdout"], re.MULTILINE)
    return match.group(1).strip() if match else None


def _gpus(env: Mapping[str, str]) -> list[dict[str, str]]:
    smi = shutil.which("nvidia-smi", path=env.get("PATH"))
    if smi is None:
        return []
    query = "--query-gpu=name,uuid,driver_version"
    result = _run([smi, query, "--format=csv,noheader"], os.getcwd(), env, 20.0)
    if result["timed_out"] or result["returncode"] != 0:
        return []
    gpus = []
    for line in result["stdout"].splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) == 3:
            gpus.append(dict(zip(("name", "uuid", "driver_version"), fields, strict=True)))
    return gpus
