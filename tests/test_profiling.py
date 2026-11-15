"""A profiled run reports what the profiler captured apart from what the program did.

The stand-in profiler below prints the messages Nsight Compute 2025.3.1 prints for each case, as
captured from real runs, and runs the profiled program for real. One test at the end uses the
real toolchain when this machine has one.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

import gfaas
from gfaas import cuda, profiling
from gfaas.errors import CudaProcessError, GfaasError
from gfaas.profiling import (
    ProfilingError,
    ProfilingNotGrantedError,
    ProfilingOutcome,
    ProfilingReport,
    ProfilingRequest,
    ProfilingUnsupportedError,
    download_reports,
    run_profiler,
)

FAKE_NCU = textwrap.dedent(
    """\
    #!{python}
    import os, subprocess, sys, time
    args = sys.argv[1:]
    mode = os.environ.get("FAKE_NCU", "profiled")
    if args == ["--version"]:
        print("NVIDIA (R) Nsight Compute Command Line Profiler")
        print("Version 2025.3.1.0 (build 36398880) (public-release)")
        sys.exit(0)
    if "--import" in args:
        print('"ID","Kernel Name","Metric Name","Metric Value"')
        for kernel in range(int(os.environ.get("FAKE_NCU_KERNELS", "1"))):
            print(f'"{{kernel}}","add(const float *)","Duration","1"')
        sys.exit(0)
    if mode == "bad_option":
        print("==ERROR== unrecognised option '--no-such-option'. Use --help for further details.")
        sys.exit(1)
    export = args[args.index("--export") + 1]
    program = args[args.index("--export") + 2:]
    with open(os.environ["FAKE_NCU_LOG"], "a") as log:
        log.write(" ".join(args) + "\\n")
    if mode == "denied":
        print("==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA "
              "GPU Performance Counters on the target device 0. For instructions on enabling "
              "permissions and to get more information see "
              "https://developer.nvidia.com/ERR_NVGPUCTRPERM")
        sys.exit(1)
    print(f"==PROF== Connected to process 4242 ({{program[0]}})", flush=True)
    kernels = 0 if mode == "no_kernels" else int(os.environ.get("FAKE_NCU_KERNELS", "1"))
    code = subprocess.run(program).returncode
    if mode == "hang":
        time.sleep(600)
    for kernel in range(kernels):
        print(f'==PROF== Profiling "add" - {{kernel}}: 0%....50%....100% - 9 passes')
    print("==PROF== Disconnected from process 4242")
    if code:
        print(f"==ERROR== The application returned an error code ({{code}}).")
    if kernels:
        with open(export + ".ncu-rep", "wb") as report:
            report.write(b"r" * int(os.environ.get("FAKE_NCU_REPORT_BYTES", "4096")))
        print(f"==PROF== Report: {{export}}.ncu-rep")
    elif not code:
        print("==WARNING== No kernels were profiled.")
    if mode == "warning":
        print("==WARNING== Unable to collect some metrics for this device.")
    sys.exit(code)
    """
)
PROGRAM = textwrap.dedent(
    """\
    #!{python}
    import os, sys
    print("add kernel done", flush=True)
    with open(os.environ["FAKE_NCU_LOG"], "a") as log:
        log.write(f"program pid={{os.getpid()}}\\n")
    sys.exit(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    """
)


def executable(path: Path, text: str) -> str:
    path.write_text(text.format(python=sys.executable), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


class Bench:
    """A scratch directory with the stand-in profiler, a program, and a publish directory."""

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = root
        self.ncu = executable(root / "ncu", FAKE_NCU)
        self.program = executable(root / "kernel", PROGRAM)
        self.log = root / "invocations.log"
        self.log.touch()
        self.published = root / "outputs" / "profiles"
        self.work = root / "work"
        self.work.mkdir()
        monkeypatch.setenv("FAKE_NCU_LOG", str(self.log))
        self.monkeypatch = monkeypatch

    def run(
        self, *program_args: str, mode: str = "profiled", ncu: bool = True, **request: Any
    ) -> tuple[ProfilingOutcome, dict[str, Any]]:
        self.monkeypatch.setenv("FAKE_NCU", mode)
        result = run_profiler(
            [self.program, *program_args],
            ProfilingRequest(**request),
            ncu=self.ncu if ncu else None,
            cwd=self.work,
            env=dict(os.environ),
            report_directory=self.work,
            publish_directory=self.published,
            output_name="profiles",
            provenance={"compiler": "nvcc 13.0"},
        )
        return ProfilingOutcome.from_dict(result["outcome"], call_id="call_1"), result


@pytest.fixture
def bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Bench:
    return Bench(tmp_path, monkeypatch)


def test_a_profiled_kernel_returns_reports_with_their_provenance(
    bench: Bench, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_NCU_KERNELS", "3")
    outcome, result = bench.run(metric_set="basic", launch_count=3)

    assert outcome.ok and outcome.status == "profiled"
    assert (outcome.program_returncode, outcome.profiler_returncode) == (0, 0)
    assert (outcome.kernels, outcome.passes) == (3, 27)
    # The profiler's own lines are not the program's output.
    assert result["process"]["stdout"] == "add kernel done\n"
    assert result["csv"].startswith('"ID","Kernel Name"')

    report = outcome.report("ncu-rep")
    assert report is not None and report.output_name == "profiles"
    published = bench.published / report.filename
    assert report.size_bytes == published.stat().st_size == 4096
    assert report.sha256 == hashlib.sha256(published.read_bytes()).hexdigest()
    assert outcome.report("csv") is not None

    assert outcome.call_id == "call_1"
    assert outcome.provenance["profiler"] == {
        "name": "ncu",
        "version": "2025.3.1.0 (build 36398880) (public-release)",
    }
    assert outcome.provenance["arguments"] == ["--set", "basic", "--launch-count", "3"]
    assert outcome.provenance["compiler"] == "nvcc 13.0"
    # The record published beside the report says the same, so the report explains itself.
    durable = json.loads((bench.published / "profile.provenance.json").read_text())
    assert durable["status"] == "profiled"
    assert durable["reports"][0]["sha256"] == report.sha256
    assert durable["provenance"]["request"]["metric_set"] == "basic"


def test_a_program_without_kernels_is_a_no_kernel_outcome_not_a_failure(bench: Bench) -> None:
    outcome, result = bench.run(mode="no_kernels")

    assert outcome.status == "no_kernels"
    assert outcome.detail == "No kernels were profiled."
    assert outcome.program_returncode == 0
    assert outcome.reports == ()
    assert not bench.published.exists()
    assert result["csv"] is None
    with pytest.raises(ProfilingError, match="no_kernels") as error:
        outcome.raise_for_status()
    assert error.value.outcome is outcome


def test_a_failing_program_is_told_apart_from_a_failing_profiler(bench: Bench) -> None:
    # The program failed after its kernel was profiled: there is a profile and a failure.
    outcome, _ = bench.run("7")
    assert (outcome.status, outcome.program_returncode) == ("profiled", 7)

    # The program failed and nothing was captured.
    outcome, _ = bench.run("3", mode="no_kernels")
    assert (outcome.status, outcome.program_returncode) == ("program_failed", 3)
    assert outcome.detail == "The application returned an error code (3)."

    # The profiler rejected its arguments; the program never ran.
    bench.log.write_text("")
    outcome, _ = bench.run(mode="bad_option")
    assert (outcome.status, outcome.program_returncode) == ("profiler_failed", None)
    assert outcome.profiler_returncode == 1
    assert "unrecognised option" in outcome.detail
    assert "program pid" not in bench.log.read_text()


def test_denied_gpu_counters_are_a_permission_outcome(bench: Bench) -> None:
    outcome, _ = bench.run(mode="denied")

    assert outcome.status == "permission_denied"
    assert "ERR_NVGPUCTRPERM" in outcome.detail
    assert outcome.program_returncode is None


def test_profiler_warnings_are_kept_with_a_successful_profile(bench: Bench) -> None:
    outcome, _ = bench.run(mode="warning")

    assert outcome.ok
    assert outcome.warnings == ("Unable to collect some metrics for this device.",)


def test_the_profiler_deadline_ends_the_profiled_program_and_is_its_own_outcome(
    bench: Bench,
) -> None:
    started = time.monotonic()
    outcome, _ = bench.run(mode="hang", timeout_s=1)

    assert time.monotonic() - started < 30
    assert outcome.status == "profiler_timed_out"
    assert outcome.detail == "the profiler exceeded its 1 s budget"
    assert outcome.program_returncode is None
    # The whole process group is gone, not only the profiler.
    survivors = subprocess.run(["pgrep", "-f", bench.ncu], capture_output=True, text=True).stdout
    assert survivors == ""


def test_the_calls_remaining_time_bounds_the_profiler_and_is_named(bench: Bench) -> None:
    bench.monkeypatch.setenv("FAKE_NCU", "hang")
    result = run_profiler(
        [bench.program],
        ProfilingRequest(timeout_s=300),
        ncu=bench.ncu,
        cwd=bench.work,
        env=dict(os.environ),
        report_directory=bench.work,
        remaining_s=1.0,
    )

    assert result["outcome"]["status"] == "profiler_timed_out"
    assert "Call's time ran out" in result["outcome"]["detail"]


def test_an_oversized_report_is_recorded_but_not_published(
    bench: Bench, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_NCU_REPORT_BYTES", "5000")
    outcome, result = bench.run(max_report_bytes=4999)

    assert outcome.status == "report_too_large"
    assert outcome.detail == "the report is 5000 bytes; the request allows 4999"
    report = outcome.report("ncu-rep")
    assert report is not None and (report.size_bytes, report.output_name) == (5000, None)
    assert not bench.published.exists()
    assert result["csv"] is None


def test_a_missing_profiler_is_an_outcome_and_the_program_is_not_run(bench: Bench) -> None:
    outcome, result = bench.run(ncu=False)

    assert outcome.status == "profiler_unavailable"
    assert result["process"]["returncode"] is None
    assert bench.log.read_text() == ""


def test_a_baseline_run_gives_the_slowdown_and_stops_a_failing_program_early(
    bench: Bench,
) -> None:
    outcome, result = bench.run(baseline=True)
    assert outcome.ok and outcome.program_returncode == 0
    assert outcome.baseline_ms is not None and outcome.slowdown is not None
    assert result["process"]["stdout"] == "add kernel done\n"
    assert bench.log.read_text().count("program pid") == 2

    bench.log.write_text("")
    outcome, _ = bench.run("5", baseline=True)
    assert (outcome.status, outcome.program_returncode) == ("program_failed", 5)
    assert outcome.slowdown is None
    # The profiler never started.
    assert "--export" not in bench.log.read_text()


def test_a_request_is_bounded_and_cannot_redirect_the_report() -> None:
    assert ProfilingRequest().command_arguments() == ["--set", "full"]
    selected = ProfilingRequest.from_arguments(["--section", "SpeedOfLight"], timeout_s=60)
    assert selected.metric_set is None
    assert selected.command_arguments() == ["--section", "SpeedOfLight"]
    assert ProfilingRequest.from_dict(selected.to_dict()) == selected
    kept = ProfilingRequest.from_arguments(["--target-processes", "all"], default_set="basic")
    assert kept.command_arguments() == ["--set", "basic", "--target-processes", "all"]

    for invalid in (
        {"metric_set": "everything"},
        {"metric_set": None},
        {"arguments": ("--export", "/etc/passwd")},
        {"arguments": ("--log-file=/tmp/x",)},
        {"arguments": ("",)},
        {"arguments": tuple("--x" for _ in range(33))},
        {"arguments": ("x" * 257,)},
        {"timeout_s": 0},
        {"timeout_s": True},
        {"launch_count": 0},
        {"max_report_bytes": 0},
        {"baseline": 1},
    ):
        with pytest.raises(GfaasError):
            ProfilingRequest(**invalid)  # type: ignore[arg-type]
    with pytest.raises(GfaasError, match="unknown profiling request fields"):
        ProfilingRequest.from_dict({"output": "x"})


def test_gpu_identity_comes_from_the_driver_when_the_image_has_no_nvidia_smi(
    tmp_path: Path,
) -> None:
    # As a GB300 workload sees it: the driver lists the host's GPUs, one device node is present.
    driver = tmp_path / "proc/driver/nvidia"
    for minor, bus in enumerate(("0008:06:00.0", "0009:06:00.0")):
        (driver / "gpus" / bus).mkdir(parents=True)
        (driver / "gpus" / bus / "information").write_text(
            f"Model: \t\t NVIDIA GB300\nIRQ:   \t\t 862\nGPU UUID: \t GPU-{minor}\n"
            f"Bus Location: \t {bus}\nDevice Minor: \t {minor}\nGPU Excluded:\t No\n"
        )
    (driver / "version").write_text(
        "NVRM version: NVIDIA UNIX Open Kernel Module for aarch64  610.57.04  Release Build  "
        "(dvs-builder@U22-I3-G07-01-3)  Wed Jul 29 03:02:27 UTC 2026\n"
        "GCC version:  gcc version 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1)\n"
    )
    (tmp_path / "dev").mkdir()
    (tmp_path / "dev/nvidia1").touch()

    assert profiling._driver_gpus(tmp_path) == [
        {"name": "NVIDIA GB300", "uuid": "GPU-1", "driver_version": "610.57.04"}
    ]
    assert profiling._driver_gpus(tmp_path / "missing") == []


class ReportStore:
    """A client that serves one published profile directory."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.downloads: list[str] = []

    def download_artifact_directory(self, artifact_id: str, destination: Path) -> Path:
        self.downloads.append(artifact_id)
        destination.mkdir(parents=True)
        for name, data in self.files.items():
            (destination / name).write_bytes(data)
        return destination


def outcome_with(data: bytes) -> ProfilingOutcome:
    digest = hashlib.sha256(data).hexdigest()
    reports = tuple(
        ProfilingReport(format, name, len(data), digest, "profiles", "artifact_1")
        for format, name in (("ncu-rep", "profile.ncu-rep"), ("csv", "profile.csv"))
    )
    return ProfilingOutcome(status="profiled", reports=reports)


def test_reports_download_within_a_limit_and_only_if_they_match_their_digest(
    tmp_path: Path,
) -> None:
    data = b"report"
    store = ReportStore({"profile.ncu-rep": data, "profile.csv": data})

    with pytest.raises(GfaasError, match="12 bytes; the transfer limit is 11"):
        download_reports(store, outcome_with(data), tmp_path / "limited", max_bytes=11)
    assert store.downloads == []

    saved = download_reports(store, outcome_with(data), tmp_path / "reports")
    assert [path.name for path in saved] == ["profile.ncu-rep", "profile.csv"]
    assert saved[0].read_bytes() == data
    # Both reports are in one Artifact, which is fetched once.
    assert store.downloads == ["artifact_1"]

    with pytest.raises(GfaasError, match="refusing to replace"):
        download_reports(store, outcome_with(data), tmp_path / "reports")
    tampered = ReportStore({"profile.ncu-rep": b"other!", "profile.csv": data})
    with pytest.raises(GfaasError, match="does not match its digest"):
        download_reports(tampered, outcome_with(data), tmp_path / "tampered")
    assert not (tmp_path / "tampered" / "profile.ncu-rep").exists()


def client_for(handle: Any) -> gfaas.Client:
    client = gfaas.Client(
        gfaas.ClientConfig(
            api_base="https://example.test/api",
            api_key="test",
            poll_interval_s=0.01,
            request_timeout_s=1,
        )
    )
    client._http.close()
    client._http = httpx.Client(base_url=client.cfg.api_base, transport=httpx.MockTransport(handle))
    return client


@pytest.mark.parametrize(
    ("offered", "error"),
    [
        ({"capabilities": ["gpu_counters"], "granted": False}, ProfilingNotGrantedError),
        ({"capabilities": [], "granted": True}, ProfilingUnsupportedError),
        (None, ProfilingUnsupportedError),
    ],
)
def test_profiling_the_service_would_refuse_is_refused_before_anything_is_sent(
    offered: Any, error: type[GfaasError]
) -> None:
    requests: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(f"{request.method} {request.url.path}")
        return httpx.Response(200, json={"profiling": offered})

    with client_for(handle) as client, pytest.raises(error):
        client.submit(image="cuda", function=("work", "run"), profiling="gpu_counters")
    assert requests == ["GET /api/v1/capabilities"]


def test_a_granted_profiling_request_reaches_the_call_and_a_late_denial_is_typed() -> None:
    bodies: dict[str, Any] = {}
    deny = False

    def handle(request: httpx.Request) -> httpx.Response:
        resource = request.url.path.rsplit("/", 1)[-1]
        if resource == "capabilities":
            return httpx.Response(
                200, json={"profiling": {"capabilities": ["gpu_counters"], "granted": True}}
            )
        if resource == "calls" and deny:
            return httpx.Response(
                403, json={"code": "profiling_not_granted", "message": "no grant"}
            )
        if request.headers.get("content-type") == "application/json":
            bodies[resource] = json.loads(request.content)
        return httpx.Response(201, json={"id": f"{resource}_1", "state": "available"})

    with client_for(handle) as client:
        client.submit(image="cuda", function=("work", "run"), profiling="gpu_counters")
        assert bodies["calls"]["profiling"] == {"capability": "gpu_counters"}
        client.submit(image="cuda", function=("work", "run"))
        assert "profiling" not in bodies["calls"]
        # The grant was withdrawn after the capability check.
        deny = True
        with pytest.raises(ProfilingNotGrantedError):
            client.submit(image="cuda", function=("work", "run"), profiling="gpu_counters")


class Remote:
    call_id = "call_9"

    def __init__(self, report: dict[str, Any]) -> None:
        self.report = report

    def wait(self, *, timeout_s: float | None = None) -> dict[str, Any]:
        return self.report

    def artifacts(self) -> dict[str, Any]:
        return {"items": [{"name": "profiles", "artifact": {"id": "artifact_7"}}]}


class Submitter:
    def __init__(self, report: dict[str, Any]) -> None:
        self.report = report
        self.calls: list[dict[str, Any]] = []

    def submit(self, **kwargs: Any) -> Remote:
        self.calls.append(kwargs)
        return Remote(self.report)


def cuda_report(outcome: dict[str, Any], returncode: int) -> dict[str, Any]:
    return {
        "phase": "run",
        "stdout": "",
        "stderr": "",
        "returncode": returncode,
        "compile_ms": 1,
        "run_ms": 1,
        "ncu_csv": None,
        "profiling": outcome,
    }


def test_a_profiled_cuda_run_asks_for_gpu_counters_and_locates_its_reports() -> None:
    report = {
        "format": "ncu-rep",
        "filename": "profile.ncu-rep",
        "size_bytes": 10,
        "sha256": "0" * 64,
        "output_name": "profiles",
        "artifact_id": None,
    }
    submitter = Submitter(cuda_report({"status": "profiled", "reports": [report]}, 0))
    request = ProfilingRequest(metric_set="basic", timeout_s=120, launch_count=2)

    result = gfaas.compile_and_run("int main() {}", profile=request, client=submitter)

    call = submitter.calls[0]
    assert call["profiling"] == "gpu_counters"
    assert call["kwargs"]["profiling"] == request.to_dict()
    assert [output.name for output in call["stages"][1].outputs] == ["profiles"]
    assert call["stages"][1].outputs[0].request()["kind"] == "profile"
    outcome = cuda.profiling_outcome(result)
    assert outcome is not None and outcome.call_id == "call_9"
    assert outcome.reports[0].artifact_id == "artifact_7"

    # An unprofiled run asks for nothing and publishes no profile output.
    plain = Submitter({**cuda_report({}, 0), "profiling": None})
    assert cuda.profiling_outcome(gfaas.compile_and_run("int main() {}", client=plain)) is None
    assert plain.calls[0]["profiling"] is None
    assert plain.calls[0]["kwargs"]["profiling"] is None
    assert plain.calls[0]["stages"][1].outputs == ()


def test_only_a_failing_program_raises_from_a_profiled_cuda_run() -> None:
    lost = {"status": "permission_denied", "program_returncode": None, "detail": "denied"}
    result = gfaas.compile_and_run(
        "int main() {}", profile=True, client=Submitter(cuda_report(lost, 1))
    )
    outcome = cuda.profiling_outcome(result)
    assert outcome is not None and outcome.status == "permission_denied"

    failed = {"status": "profiled", "program_returncode": 7}
    with pytest.raises(CudaProcessError) as error:
        gfaas.compile_and_run(
            "int main() {}", profile=True, client=Submitter(cuda_report(failed, 7))
        )
    assert error.value.returncode == 7

    with pytest.raises(GfaasError, match="ncu_args require profile=True"):
        gfaas.compile_and_run("int main() {}", ncu_args=["--set", "full"], client=Submitter({}))
    with pytest.raises(GfaasError, match="no valid profiling outcome"):
        cuda.profiling_outcome({"profiling": {"status": "finished"}})


@pytest.mark.skipif(
    not (shutil.which("nvcc") and shutil.which("ncu") and shutil.which("nvidia-smi")),
    reason="needs a local CUDA toolchain, Nsight Compute and a GPU",
)
def test_real_nsight_compute_profiles_a_kernel_and_finds_none_without_one(
    tmp_path: Path,
) -> None:
    kernel = """
        #include <cuda_runtime.h>
        __global__ void add(float* out, int n) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) out[i] += 1.0f;
        }
        int main() {
            float* out; cudaMalloc(&out, 1024 * sizeof(float));
            add<<<4, 256>>>(out, 1024);
            return cudaDeviceSynchronize() == cudaSuccess ? 0 : 1;
        }
    """
    for name, source in (("kernel", kernel), ("empty", "int main() { return 0; }")):
        (tmp_path / f"{name}.cu").write_text(source)
        subprocess.run(
            ["nvcc", str(tmp_path / f"{name}.cu"), "-o", str(tmp_path / name)], check=True
        )

    def profile(name: str) -> ProfilingOutcome:
        result = run_profiler(
            [str(tmp_path / name)],
            ProfilingRequest(metric_set="basic"),
            ncu=shutil.which("ncu"),
            cwd=tmp_path,
            env=dict(os.environ),
            report_directory=tmp_path,
            report_name=name,
            publish_directory=tmp_path / "published",
        )
        return ProfilingOutcome.from_dict(result["outcome"])

    profiled = profile("kernel")
    assert profiled.ok and profiled.kernels == 1 and profiled.passes
    assert profiled.report("ncu-rep") is not None and profiled.report("csv") is not None
    assert profiled.provenance["profiler"]["version"]
    assert profiled.provenance["gpus"][0]["driver_version"]
    assert profile("empty").status == profiling.NO_KERNELS
