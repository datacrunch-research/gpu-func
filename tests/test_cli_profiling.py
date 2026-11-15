"""The CLI reports a profiled run's program and profiler results separately."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

import gfaas_cli.main as cli
from gfaas import ArtifactOutput, local_cuda
from gfaas_cli import worker_job
from gfaas_cli.client import GfaasClient
from gfaas_cli.constants import RC_CRASH, RC_OK, RC_PROFILE
from gfaas_cli.output import _print_custom_result

REPORT = b"report"


def outcome(status: str = "profiled", **fields: Any) -> dict[str, Any]:
    reports = []
    if status == "profiled":
        reports = [
            {
                "format": "ncu-rep",
                "filename": "profile.ncu-rep",
                "size_bytes": len(REPORT),
                "sha256": hashlib.sha256(REPORT).hexdigest(),
                "output_name": "profiles",
                "artifact_id": None,
            }
        ]
    return {
        "status": status,
        "detail": "No kernels were profiled." if status == "no_kernels" else "",
        "program_returncode": 0,
        "profiler_returncode": 0,
        "kernels": 2 if status == "profiled" else 0,
        "passes": 18 if status == "profiled" else None,
        "warnings": ["Unable to collect some metrics."],
        "reports": reports,
        **fields,
    }


def cuda_result(profiling: dict[str, Any] | None, returncode: int = 0) -> dict[str, Any]:
    return {
        "phase": "run",
        "stdout": "kernel done\n",
        "stderr": "",
        "returncode": returncode,
        "compile_ms": 3,
        "run_ms": 9,
        "ncu_csv": None,
        "profiling": profiling,
    }


class Remote:
    call_id = "call_test"

    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result

    def iter_events(self, **_kwargs: Any) -> Any:
        return iter(())

    def wait(self) -> dict[str, Any]:
        return self.result

    def artifacts(self) -> dict[str, Any]:
        return {"items": [{"name": "profiles", "artifact": {"id": "artifact_7"}}]}


class Client:
    def __init__(self, result: dict[str, Any]) -> None:
        self.remote = Remote(result)
        self.submissions: list[dict[str, Any]] = []
        self.downloads: list[str] = []

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def submit(self, **kwargs: Any) -> Remote:
        self.submissions.append(kwargs)
        return self.remote

    def download_artifact_directory(self, artifact_id: str, destination: Path) -> Path:
        self.downloads.append(artifact_id)
        destination.mkdir(parents=True)
        (destination / "profile.ncu-rep").write_bytes(REPORT)
        return destination


def run_cli(client: Client, source: Path, *options: str) -> int:
    return cli.main(["run", str(source), *options], client_factory=lambda: client)


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "kernel.cu"
    path.write_text("int main() { return 0; }\n")
    return path


def test_a_profiled_run_requests_counters_and_saves_the_located_reports(
    source: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    client = Client(cuda_result(outcome()))
    reports = tmp_path / "reports"

    status = run_cli(
        client,
        source,
        "--profile",
        "--profile-set=basic",
        "--profile-timeout=120",
        "--profile-launch-count=4",
        "--ncu-arg=--target-processes",
        "--ncu-arg=all",
        f"--profile-dir={reports}",
    )

    assert status == 0
    submission = client.submissions[0]
    assert submission["profiling"] == "gpu_counters"
    assert submission["kwargs"]["profiling"] == {
        "metric_set": "basic",
        "arguments": ["--target-processes", "all"],
        "timeout_s": 120,
        "launch_count": 4,
        "max_report_bytes": 64 * 1024 * 1024,
        "baseline": False,
    }
    assert [output.name for output in submission["stages"][1].outputs] == ["profiles"]
    assert client.downloads == ["artifact_7"]
    assert (reports / "profile.ncu-rep").read_bytes() == REPORT
    captured = capsys.readouterr()
    assert captured.out == "kernel done\n"
    assert "[vfunc] profiling=profiled kernels=2 passes=18" in captured.err
    assert "profiler warning: Unable to collect some metrics." in captured.err
    assert f"sha256={hashlib.sha256(REPORT).hexdigest()} artifact=artifact_7" in captured.err
    assert f"profile report -> {reports / 'profile.ncu-rep'}" in captured.err


def test_a_run_that_only_lost_its_profile_exits_with_its_own_status(
    source: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_cli(Client(cuda_result(outcome("no_kernels"))), source, "--profile") == RC_PROFILE
    captured = capsys.readouterr()
    assert captured.out == "kernel done\n"
    assert "[vfunc] profiling=no_kernels: No kernels were profiled." in captured.err

    # The program's own failure comes first, whatever the profiler captured.
    failed = cuda_result(outcome(program_returncode=9), returncode=9)
    assert run_cli(Client(failed), source, "--profile") == 9
    capsys.readouterr()

    # JSON output carries the outcome with its Call and Artifact identity.
    assert run_cli(Client(cuda_result(outcome())), source, "--profile", "--json") == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    profiling = lines[-1]["value"]["profiling"]
    assert profiling["call_id"] == "call_test"
    assert profiling["reports"][0]["artifact_id"] == "artifact_7"


def test_profiler_options_need_profile_and_a_cuda_source(
    source: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    client = Client(cuda_result(None))
    assert run_cli(client, source, "--profile-timeout=5") == 1
    assert "profiler options require --profile" in capsys.readouterr().err
    assert run_cli(client, source, "--profile", "--ncu-arg=--export", "--ncu-arg=x") == 1
    assert "sets --export itself" in capsys.readouterr().err
    script = tmp_path / "work.py"
    script.write_text("print('x')\n")
    assert run_cli(client, script, "--profile") == 1
    assert client.submissions == []

    # An unprofiled CUDA run asks for nothing.
    assert run_cli(client, source) == 0
    assert client.submissions[0]["profiling"] is None
    assert client.submissions[0]["stages"][1].outputs == ()


def test_a_local_profiled_run_keeps_reports_and_needs_no_service(
    source: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    gpu = local_cuda.LocalGpu(0, "GPU-local", "NVIDIA GB10", "12.1", "sm_121", None, "580.42")
    toolchain = local_cuda.LocalCudaToolchain(
        nvcc="nvcc",
        nvcc_version="13.0.88",
        cuda_root="/usr/local/cuda",
        nvidia_smi="nvidia-smi",
        ncu=None,
        host_compiler=None,
        supported_architectures=("sm_121",),
        gpus=(gpu,),
        selected_gpu=gpu,
        cuda_visible_devices=None,
    )
    calls: list[dict[str, Any]] = []

    def local_run(_source: Path, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return cuda_result(outcome("profiler_unavailable", program_returncode=None))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(local_cuda, "discover", lambda **_kwargs: toolchain)
    monkeypatch.setattr(local_cuda, "run", local_run)

    def no_client() -> Client:
        raise AssertionError("a local run must not create a client")

    status = cli.main(
        ["local", "run", str(source), "--profile", "--profile-dir", "reports"],
        client_factory=no_client,
    )

    assert status == RC_PROFILE
    assert calls[0]["profiling"].metric_set == "full"
    assert calls[0]["profile_directory"] == Path("reports")
    assert "profiling=profiler_unavailable" in capsys.readouterr().err


def custom_job(profiling: dict[str, Any]) -> dict[str, Any]:
    return {
        "remote": {"timeout_s": 60, "image": "cuda-nvcc"},
        "custom": {
            "command": "profile",
            "output": "kernel",
            "program_args": ["4"],
            "report_name": "saxpy",
            "profiling": profiling,
        },
    }


@pytest.mark.parametrize(
    ("profiled", "status", "exit_code"),
    [
        (outcome(), "passed", RC_OK),
        (outcome("no_kernels"), "profile_failed", RC_PROFILE),
        (outcome("profiler_timed_out", program_returncode=None), "profile_failed", RC_PROFILE),
        (outcome(program_returncode=2), "error", RC_CRASH),
    ],
)
def test_a_custom_profile_job_separates_the_profilers_result_from_the_programs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    profiled: dict[str, Any],
    status: str,
    exit_code: int,
) -> None:
    request = {"metric_set": "basic", "arguments": ["--nvtx"], "timeout_s": 30}
    output = ArtifactOutput.directory("profiles", "profiles", kind="profile")
    calls: list[dict[str, Any]] = []

    def run_profiler(program: list[str], request: Any, **options: Any) -> dict[str, Any]:
        calls.append({"program": program, "request": request, **options})
        process = {"returncode": 0, "stdout": "ran\n", "stderr": "", "ms": 5}
        return {"process": process, "outcome": profiled, "csv": None}

    with (
        mock.patch.dict("os.environ", {"GFAAS_OUTPUT_ROOT": str(tmp_path / "outputs")}),
        mock.patch.object(worker_job, "run_profiler", run_profiler),
        mock.patch.object(worker_job, "_find", lambda _command: "/usr/bin/ncu"),
    ):
        result = worker_job._run_compiled_custom_job(
            tmp_path,
            custom_job(request),
            {},
            time.monotonic() + 60,
            output,
            {"returncode": 0, "args": ["nvcc"], "stdout": "", "stderr": ""},
        )

    assert result["status"] == status
    assert result["profiling"] == profiled
    assert result["run"]["timed_out"] is False
    assert result["run"]["returncode"] == profiled["program_returncode"]
    call = calls[0]
    assert call["program"] == ["./kernel", "4"]
    assert (call["request"].metric_set, call["request"].timeout_s) == ("basic", 30)
    assert call["report_name"] == "saxpy"
    assert call["publish_directory"] == tmp_path / "outputs" / "profiles"
    assert 0 < call["remaining_s"] <= 60

    args = argparse.Namespace(verbose=False, custom_command="profile")
    assert _print_custom_result(result, args) == exit_code
    captured = capsys.readouterr()
    if status == "profile_failed":
        assert f"profiling ended as {profiled['status']}" in captured.err
    if status != "error":
        assert "profiler warning" in captured.err


def test_course_reports_are_published_with_their_digest_within_the_size_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GFAAS_OUTPUT_ROOT", str(tmp_path / "outputs"))
    output = ArtifactOutput.directory("profiles", "profiles", kind="profile")
    report = tmp_path / "exercise.ncu-rep"
    report.write_bytes(REPORT)

    published = worker_job._publish_profile(report, output)
    assert published == {
        "filename": "exercise.ncu-rep",
        "output_name": "profiles",
        "size_bytes": len(REPORT),
        "sha256": hashlib.sha256(REPORT).hexdigest(),
    }

    monkeypatch.setattr(worker_job, "DEFAULT_MAX_REPORT_BYTES", 3)
    large = tmp_path / "large.ncu-rep"
    large.write_bytes(REPORT)
    skipped = worker_job._publish_profile(large, output)
    assert skipped["size_bytes"] == len(REPORT) and "skipped" in skipped
    assert not (tmp_path / "outputs" / "profiles" / "large.ncu-rep").exists()


class Sdk:
    def __init__(self) -> None:
        self.submission: dict[str, Any] = {}

    def upload_artifact_directory(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"id": "art_workspace", "child_artifact_ids": []}

    def submit(self, **kwargs: Any) -> object:
        self.submission = kwargs
        return object()


@pytest.mark.parametrize(
    ("mode", "requested"),
    [("custom-profile", "gpu_counters"), ("profile", "gpu_counters"), ("custom-run", None)],
)
def test_only_profile_jobs_ask_the_service_for_gpu_counters(
    tmp_path: Path, mode: str, requested: str | None
) -> None:
    sdk = Sdk()
    args = argparse.Namespace(
        image="cuda-nvcc",
        compatibility=None,
        gpu_count=1,
        gpu_type="gb300",
        timeout=600,
        capacity_wait=None,
        cpu_millicores=None,
        memory=None,
        storage=None,
        shared_memory=None,
        max_log=None,
        max_output=None,
        env=[],
        idempotency_key=None,
    )
    GfaasClient(sdk, poll_interval=0).submit_job(  # type: ignore[arg-type]
        job={"target": {"kind": "custom"}, "command": {"mode": mode}},
        workspace=tmp_path,
        args=args,
        app_name="vfunc-custom",
    )
    assert sdk.submission["profiling"] == requested
