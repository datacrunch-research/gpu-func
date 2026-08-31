from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import gfaas
from gfaas import cuda_runner
from gfaas.cuda import spawn

ADD_CU_SOURCE = r"""
#include <cuda_runtime.h>
#include <cstdio>

__global__ void add(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

int main() {
    const int n = 1 << 10;
    size_t bytes = n * sizeof(float);
    float *da, *db, *dout;
    cudaMalloc(&da, bytes);
    cudaMalloc(&db, bytes);
    cudaMalloc(&dout, bytes);
    cudaMemset(da, 0, bytes);
    cudaMemset(db, 0, bytes);

    int block = 256;
    int grid = (n + block - 1) / block;
    add<<<grid, block>>>(da, db, dout, n);
    cudaDeviceSynchronize();
    printf("add kernel done\n");

    cudaFree(da);
    cudaFree(db);
    cudaFree(dout);
    return 0;
}
"""


@dataclass
class FakeRemoteResult:
    payload: dict[str, Any]

    def wait(self, *, timeout_s: float | None = None) -> dict[str, Any]:
        return self.payload


class FakeClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def submit(self, **kwargs: Any) -> FakeRemoteResult:
        self.calls.append(kwargs)
        return FakeRemoteResult(self.payload)


def test_compile_and_run_profiled_cuda_submits_expected_request():
    fake_result = {
        "phase": "run",
        "stdout": "kernel output\n",
        "stderr": "",
        "returncode": 0,
        "compile_ms": 12,
        "run_ms": 4,
        "ncu_csv": "metric,value\nsm__throughput.avg.pct_of_peak_sustained_elapsed,12.3\n",
    }
    client = FakeClient(fake_result)

    result = gfaas.compile_and_run(
        ADD_CU_SOURCE,
        gpu="B200",
        gpu_type="b200",
        profile=True,
        ncu_args=["--set", "full", "--target-processes", "all"],
        nvcc_flags=["-lineinfo"],
        program_args=["--iters", "5"],
        timeout_s=900,
        client=client,
    )

    assert result == fake_result
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call == {
        "image": gfaas.Image("cuda-nvcc"),
        "function": cuda_runner.run,
        "kwargs": {
            "source": ADD_CU_SOURCE,
            "profile": True,
            "ncu_args": ["--set", "full", "--target-processes", "all"],
            "nvcc_flags": ["-lineinfo"],
            "program_args": ["--iters", "5"],
        },
        "gpu": "B200",
        "gpu_count": None,
        "gpu_type": "b200",
        "timeout_s": 900,
        "app_name": "cuda-nvcc",
    }
    assert "sm__throughput" in result["ncu_csv"]
    assert "__global__ void add" in call["kwargs"]["source"]


def test_compile_and_run_can_request_an_explicit_gpu_count():
    fake_result = {
        "phase": "run",
        "stdout": "",
        "stderr": "",
        "returncode": 0,
        "compile_ms": 1,
        "run_ms": 1,
    }
    client = FakeClient(fake_result)

    assert (
        gfaas.compile_and_run(
            ADD_CU_SOURCE,
            gpu_count=4,
            gpu_type="gb300",
            client=client,
        )
        == fake_result
    )

    assert client.calls[0]["gpu"] is None
    assert client.calls[0]["gpu_count"] == 4
    assert client.calls[0]["gpu_type"] == "gb300"


def test_cuda_source_profile_builder_sets_ncu_args():
    fake_result = {
        "phase": "run",
        "stdout": "",
        "stderr": "",
        "returncode": 0,
        "compile_ms": 1,
        "run_ms": 1,
        "ncu_csv": "metric,value\nx,1\n",
    }
    client = FakeClient(fake_result)

    result = (
        gfaas.CudaSource("int main() { return 0; }\n")
        .gpu("B200", gpu_type="b200")
        .profile("--set", "full")
        .run(client=client)
    )

    assert result == fake_result
    assert client.calls[0]["kwargs"]["profile"] is True
    assert client.calls[0]["kwargs"]["ncu_args"] == ["--set", "full"]


def test_cuda_runner_uses_ccbin_and_writable_ncu_config(monkeypatch, tmp_path: Path):
    calls: list[dict[str, Any]] = []
    io_root = tmp_path / "io"
    io_root.mkdir()
    output_root = tmp_path / "outputs"
    output_root.mkdir()

    monkeypatch.setenv("HOME", "/root")
    monkeypatch.setenv("FC_IO_ROOT", str(io_root))
    monkeypatch.setenv("GFAAS_OUTPUT_ROOT", str(output_root))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    mkdtemp_calls: list[dict[str, str | None]] = []

    def fake_mkdtemp(prefix: str, dir: str | None = None) -> str:
        mkdtemp_calls.append({"prefix": prefix, "dir": dir})
        return str(tmp_path)

    monkeypatch.setattr(cuda_runner.tempfile, "mkdtemp", fake_mkdtemp)
    monkeypatch.setattr(
        cuda_runner,
        "_which",
        lambda cmd: {
            "nvcc": "/usr/local/cuda/bin/nvcc",
            "ncu": "/usr/local/cuda/bin/ncu",
        }[cmd],
    )
    monkeypatch.setattr(cuda_runner, "_host_cxx_flags", lambda: ["-ccbin", "/usr/bin/g++"])

    real_access = cuda_runner.os.access
    real_exists = cuda_runner.os.path.exists

    def fake_access(path: str, mode: int) -> bool:
        if path == "/root":
            return False
        return real_access(path, mode)

    def fake_exists(path: str) -> bool:
        if path == str(tmp_path / "report.csv"):
            return True
        return real_exists(path)

    def fake_run(  # type: ignore[no-untyped-def]
        cmd, capture_output, text, check, env, cwd=None
    ):
        calls.append({"cmd": cmd, "env": env.copy(), "cwd": cwd})
        if cmd[0].endswith("ncu"):
            (tmp_path / "report.csv").write_text(
                'Kernel Name,Metric Name,Metric Value\n"add_kernel","Duration","1"\n',
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(cuda_runner.os, "access", fake_access)
    monkeypatch.setattr(cuda_runner.os.path, "exists", fake_exists)
    monkeypatch.setattr(cuda_runner.subprocess, "run", fake_run)

    result = cuda_runner.run(
        source="int main() { return 0; }\n",
        profile=True,
        ncu_args=["--set", "full"],
    )

    compile_call = calls[0]
    run_call = calls[1]
    assert compile_call["cmd"][:3] == [
        "/usr/local/cuda/bin/nvcc",
        "-ccbin",
        "/usr/bin/g++",
    ]
    assert mkdtemp_calls == [{"prefix": "gfaas-cuda-", "dir": str(io_root)}]
    assert run_call["env"]["HOME"] == str(tmp_path)
    assert run_call["env"]["XDG_CONFIG_HOME"] == str(tmp_path / ".config")
    assert run_call["cwd"] == str(output_root)
    assert "add_kernel" in result["ncu_csv"]
    assert result["phase"] == "run"


def test_cuda_runner_marks_compiler_failure(monkeypatch, tmp_path: Path):
    calls: list[list[str]] = []

    monkeypatch.setattr(cuda_runner, "_which", lambda _cmd: "/usr/local/cuda/bin/nvcc")
    monkeypatch.setattr(cuda_runner, "_host_cxx_flags", list)
    monkeypatch.setattr(cuda_runner, "_workdir_root", lambda: None)
    monkeypatch.setattr(
        cuda_runner.tempfile,
        "mkdtemp",
        lambda **_kwargs: str(tmp_path),
    )

    def fake_run(cmd, capture_output, text, check, env):  # type: ignore[no-untyped-def]
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 2, stdout="", stderr="invalid source\n")

    monkeypatch.setattr(cuda_runner.subprocess, "run", fake_run)

    result = cuda_runner.run(source="not valid CUDA")

    assert len(calls) == 1
    assert result == {
        "phase": "compile",
        "stdout": "",
        "stderr": "invalid source\n",
        "returncode": 2,
        "compile_ms": result["compile_ms"],
        "run_ms": 0,
        "ncu_csv": None,
    }


def test_compile_and_run_raises_typed_compilation_error():
    report = {
        "phase": "compile",
        "stdout": "",
        "stderr": "kernel.cu: error: expected a declaration\n",
        "returncode": 2,
        "compile_ms": 17,
        "run_ms": 0,
        "ncu_csv": None,
    }

    with pytest.raises(gfaas.CudaCompilationError, match="exited with status 2") as raised:
        gfaas.compile_and_run("invalid CUDA", client=FakeClient(report))

    assert raised.value.report == report
    assert raised.value.phase == "compile"
    assert raised.value.returncode == 2
    assert raised.value.stderr == report["stderr"]
    assert raised.value.compile_ms == 17
    assert raised.value.run_ms == 0


def test_compile_and_run_raises_typed_process_error():
    report = {
        "phase": "run",
        "stdout": "before failure\n",
        "stderr": "application error\n",
        "returncode": 23,
        "compile_ms": 10,
        "run_ms": 3,
        "ncu_csv": None,
    }

    with pytest.raises(gfaas.CudaProcessError, match="exited with status 23") as raised:
        gfaas.compile_and_run("int main() { return 23; }", client=FakeClient(report))

    assert raised.value.report == report
    assert raised.value.phase == "run"
    assert raised.value.stdout == "before failure\n"
    assert raised.value.stderr == "application error\n"


def test_compile_and_run_reports_process_signal():
    report = {
        "phase": "run",
        "stdout": "",
        "stderr": "",
        "returncode": -11,
        "compile_ms": 10,
        "run_ms": 1,
        "ncu_csv": None,
    }

    with pytest.raises(gfaas.CudaProcessError, match="terminated by signal 11"):
        gfaas.compile_and_run("int main() {}", client=FakeClient(report))


def test_spawn_preserves_raw_failed_cuda_report():
    report = {
        "phase": "compile",
        "stdout": "",
        "stderr": "invalid source\n",
        "returncode": 2,
        "compile_ms": 1,
        "run_ms": 0,
        "ncu_csv": None,
    }

    remote = spawn("invalid CUDA", client=FakeClient(report))

    assert remote.wait() == report


def test_cuda_runner_which_checks_common_cuda_locations(monkeypatch):
    monkeypatch.setattr(cuda_runner.shutil, "which", lambda _cmd: None)

    real_exists = cuda_runner.os.path.exists
    real_access = cuda_runner.os.access

    def fake_exists(path: str) -> bool:
        if path == "/usr/local/cuda/bin/nvcc":
            return True
        return real_exists(path)

    def fake_access(path: str, mode: int) -> bool:
        if path == "/usr/local/cuda/bin/nvcc":
            return True
        return real_access(path, mode)

    monkeypatch.setattr(cuda_runner.os.path, "exists", fake_exists)
    monkeypatch.setattr(cuda_runner.os, "access", fake_access)

    assert cuda_runner._which("nvcc") == "/usr/local/cuda/bin/nvcc"
