from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from gfaas import local_cuda


def _toolchain() -> local_cuda.LocalCudaToolchain:
    gpu = local_cuda.LocalGpu(
        index=0,
        uuid="GPU-local",
        name="NVIDIA GB10",
        compute_capability="12.1",
        architecture="sm_121",
        memory_total_mib=None,
        driver_version="580.42",
    )
    return local_cuda.LocalCudaToolchain(
        nvcc="/opt/cuda/bin/nvcc",
        nvcc_version="13.0.88",
        cuda_root="/opt/cuda",
        nvidia_smi="/usr/bin/nvidia-smi",
        ncu="/opt/cuda/bin/ncu",
        host_compiler="/usr/bin/g++",
        supported_architectures=("sm_120", "sm_121"),
        gpus=(gpu,),
        selected_gpu=gpu,
        cuda_visible_devices=None,
    )


def test_discover_uses_cuda_home_and_selects_visible_gpu(monkeypatch) -> None:
    resolved = {
        "/opt/cuda/bin/nvcc": "/opt/cuda/bin/nvcc",
        "/opt/cuda/bin/ncu": "/opt/cuda/bin/ncu",
        "nvidia-smi": "/usr/bin/nvidia-smi",
        "g++": "/usr/bin/g++",
    }
    monkeypatch.setattr(local_cuda, "_resolve_candidate", resolved.get)

    def command_output(command, _environment):  # type: ignore[no-untyped-def]
        if command[0].endswith("nvcc") and command[1] == "--version":
            return "Cuda compilation tools, release 13.0, V13.0.88\n"
        if command[0].endswith("nvcc"):
            return "sm_120\nsm_121\n"
        return (
            "0, GPU-zero, NVIDIA GB10, 12.1, [N/A], 580.42\n"
            "1, GPU-one, NVIDIA RTX, 12.0, 32768, 580.42\n"
        )

    monkeypatch.setattr(local_cuda, "_command_output", command_output)

    toolchain = local_cuda.discover(
        environment={"CUDA_HOME": "/opt/cuda", "CUDA_VISIBLE_DEVICES": "GPU-one"}
    )

    assert toolchain.nvcc == "/opt/cuda/bin/nvcc"
    assert toolchain.ncu == "/opt/cuda/bin/ncu"
    assert toolchain.nvcc_version == "13.0.88"
    assert toolchain.selected_gpu.index == 1
    assert toolchain.selected_gpu.architecture == "sm_120"
    assert toolchain.gpus[0].memory_total_mib is None
    assert toolchain.gpus[1].memory_total_mib == 32768


def test_discover_rejects_a_configured_compiler_that_does_not_exist(monkeypatch) -> None:
    monkeypatch.setattr(local_cuda, "_resolve_candidate", lambda _value: None)

    with pytest.raises(local_cuda.LocalCudaError, match="configured nvcc executable"):
        local_cuda.discover(environment={"CUDACXX": "/missing/nvcc"})


def test_architecture_flags_detect_native_and_reject_unsupported() -> None:
    toolchain = _toolchain()

    flags, architecture = local_cuda.architecture_flags(
        ["-O3"],
        requested=None,
        environment={},
        toolchain=toolchain,
    )

    assert flags == ["-arch=sm_121", "-O3"]
    assert architecture == "sm_121"
    with pytest.raises(local_cuda.LocalCudaError, match="does not support"):
        local_cuda.architecture_flags(
            [],
            requested="sm_103",
            environment={},
            toolchain=toolchain,
        )


def test_architecture_flags_preserve_explicit_nvcc_code() -> None:
    flags, architecture = local_cuda.architecture_flags(
        ["--generate-code=arch=compute_121,code=sm_121"],
        requested=None,
        environment={},
        toolchain=_toolchain(),
    )

    assert flags == ["--generate-code=arch=compute_121,code=sm_121"]
    assert architecture == "nvcc-flags"


def test_run_uses_local_compilers_environment_and_workdir(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "kernel.cu"
    source.write_text("int main() { return 0; }\n")
    workdir = tmp_path / "work"
    workdir.mkdir()
    calls: list[dict[str, object]] = []

    def run_process(command, **kwargs):  # type: ignore[no-untyped-def]
        calls.append({"command": list(command), **kwargs})
        return subprocess.CompletedProcess(command, 0, stdout="kernel output\n", stderr="")

    monkeypatch.setattr(local_cuda, "_run_process", run_process)

    result = local_cuda.run(
        source,
        toolchain=_toolchain(),
        nvcc_flags=["-arch=sm_121", "-O3"],
        program_args=["--size", "4096"],
        environment={"MODE": "benchmark"},
        workdir=workdir,
        timeout_seconds=30,
    )

    compile_command = calls[0]["command"]
    assert isinstance(compile_command, list)
    assert compile_command[:5] == [
        "/opt/cuda/bin/nvcc",
        "-ccbin",
        "/usr/bin/g++",
        "-arch=sm_121",
        "-O3",
    ]
    assert calls[1]["command"][-2:] == ["--size", "4096"]  # type: ignore[index]
    assert calls[1]["cwd"] == workdir
    assert calls[1]["environment"] == {"MODE": "benchmark"}
    assert result["returncode"] == 0
    assert result["stdout"] == "kernel output\nkernel output\n"


def test_run_process_reports_a_bounded_timeout(monkeypatch, tmp_path: Path) -> None:
    def timeout(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise subprocess.TimeoutExpired("nvcc", 1)

    monkeypatch.setattr(local_cuda.subprocess, "run", timeout)

    with pytest.raises(local_cuda.LocalCudaError, match="compilation exceeded"):
        local_cuda._run_process(
            ["nvcc"],
            phase="compilation",
            environment={},
            cwd=tmp_path,
            deadline=local_cuda.time.monotonic() + 1,
        )
