"""Discover and use a CUDA toolchain on the local host."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .errors import GfaasError

_DISCOVERY_TIMEOUT_SECONDS = 10
_NVCC_VERSION = re.compile(r"\bV(?P<version>[0-9]+(?:\.[0-9]+)+)\b")
_CUDA_ARCHITECTURE = re.compile(r"^sm_[0-9]+$")
_NVCC_ARCH_FLAGS = ("-arch", "--gpu-architecture", "-gencode", "--generate-code")


class LocalCudaError(GfaasError):
    """The local CUDA toolchain or execution environment is unusable."""


@dataclass(frozen=True)
class LocalGpu:
    index: int
    uuid: str
    name: str
    compute_capability: str
    architecture: str
    memory_total_mib: int | None
    driver_version: str


@dataclass(frozen=True)
class LocalCudaToolchain:
    nvcc: str
    nvcc_version: str
    cuda_root: str
    nvidia_smi: str
    ncu: str | None
    host_compiler: str | None
    supported_architectures: tuple[str, ...]
    gpus: tuple[LocalGpu, ...]
    selected_gpu: LocalGpu
    cuda_visible_devices: str | None

    def report(self) -> dict[str, Any]:
        return {
            "nvcc": self.nvcc,
            "nvcc_version": self.nvcc_version,
            "cuda_root": self.cuda_root,
            "nvidia_smi": self.nvidia_smi,
            "ncu": self.ncu,
            "host_compiler": self.host_compiler,
            "supported_architectures": list(self.supported_architectures),
            "gpus": [asdict(gpu) for gpu in self.gpus],
            "selected_gpu": asdict(self.selected_gpu),
            "cuda_visible_devices": self.cuda_visible_devices,
        }


def _resolve_candidate(value: str) -> str | None:
    expanded = os.path.expandvars(os.path.expanduser(value))
    if os.sep not in expanded:
        return shutil.which(expanded)
    path = Path(expanded)
    if path.is_file() and os.access(path, os.X_OK):
        return str(path.resolve())
    return None


def _configured_executable(label: str, value: str | None) -> str | None:
    if not value:
        return None
    resolved = _resolve_candidate(value)
    if resolved is None:
        raise LocalCudaError(f"configured {label} executable was not found: {value}")
    return resolved


def _find_executable(
    label: str,
    *,
    configured: str | None,
    candidates: Sequence[str],
    required: bool,
) -> str | None:
    resolved = _configured_executable(label, configured)
    if resolved is not None:
        return resolved
    for candidate in candidates:
        resolved = _resolve_candidate(candidate)
        if resolved is not None:
            return resolved
    if required:
        raise LocalCudaError(f"{label} executable was not found")
    return None


def _command_output(command: Sequence[str], environment: Mapping[str, str]) -> str:
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            check=False,
            env=dict(environment),
            timeout=_DISCOVERY_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise LocalCudaError(f"local discovery command timed out: {command[0]}") from error
    except OSError as error:
        raise LocalCudaError(f"local discovery command failed: {command[0]}: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic"
        raise LocalCudaError(f"local discovery command failed: {command[0]}: {detail}")
    return completed.stdout


def _parse_nvcc_version(output: str) -> str:
    match = _NVCC_VERSION.search(output)
    if match is not None:
        return match.group("version")
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[-1] if lines else "unknown"


def _parse_memory_mib(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _parse_gpus(output: str) -> tuple[LocalGpu, ...]:
    gpus: list[LocalGpu] = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",", maxsplit=5)]
        if len(fields) != 6:
            raise LocalCudaError(f"nvidia-smi returned an invalid GPU record: {line!r}")
        index, uuid, name, compute_capability, memory_total, driver_version = fields
        try:
            parsed_index = int(index)
        except ValueError as error:
            raise LocalCudaError(f"nvidia-smi returned an invalid GPU index: {index!r}") from error
        digits = compute_capability.replace(".", "")
        if not digits.isdigit():
            raise LocalCudaError(
                f"nvidia-smi returned an invalid compute capability: {compute_capability!r}"
            )
        gpus.append(
            LocalGpu(
                index=parsed_index,
                uuid=uuid,
                name=name,
                compute_capability=compute_capability,
                architecture=f"sm_{digits}",
                memory_total_mib=_parse_memory_mib(memory_total),
                driver_version=driver_version,
            )
        )
    if not gpus:
        raise LocalCudaError("nvidia-smi did not report a CUDA GPU")
    return tuple(gpus)


def _select_gpu(
    gpus: tuple[LocalGpu, ...],
    *,
    device: str | None,
    cuda_visible_devices: str | None,
) -> LocalGpu:
    selector = device
    if selector is None and cuda_visible_devices is not None:
        selectors = [value.strip() for value in cuda_visible_devices.split(",") if value.strip()]
        if not selectors or selectors[0] == "-1":
            raise LocalCudaError("CUDA_VISIBLE_DEVICES does not expose a local GPU")
        selector = selectors[0]
    if selector is None:
        return gpus[0]
    for gpu in gpus:
        if selector == str(gpu.index) or gpu.uuid == selector or gpu.uuid.startswith(selector):
            return gpu
    raise LocalCudaError(f"local CUDA device was not found: {selector}")


def discover(
    *,
    nvcc: str | None = None,
    ncu: str | None = None,
    host_compiler: str | None = None,
    device: str | None = None,
    environment: Mapping[str, str] | None = None,
    require_profiler: bool = False,
) -> LocalCudaToolchain:
    env = dict(os.environ if environment is None else environment)
    cuda_home = env.get("CUDA_HOME")
    cuda_bin = str(Path(cuda_home) / "bin") if cuda_home else None

    nvcc_path = _find_executable(
        "nvcc",
        configured=nvcc or env.get("CUDACXX"),
        candidates=tuple(
            candidate
            for candidate in (
                f"{cuda_bin}/nvcc" if cuda_bin else None,
                "nvcc",
                "/usr/local/cuda/bin/nvcc",
            )
            if candidate is not None
        ),
        required=True,
    )
    assert nvcc_path is not None
    nvidia_smi_path = _find_executable(
        "nvidia-smi",
        configured=env.get("GFAAS_NVIDIA_SMI"),
        candidates=("nvidia-smi", "/usr/bin/nvidia-smi"),
        required=True,
    )
    assert nvidia_smi_path is not None
    ncu_path = _find_executable(
        "ncu",
        configured=ncu or env.get("GFAAS_NCU"),
        candidates=tuple(
            candidate
            for candidate in (
                f"{cuda_bin}/ncu" if cuda_bin else None,
                "ncu",
                "/usr/local/cuda/bin/ncu",
            )
            if candidate is not None
        ),
        required=require_profiler,
    )
    compiler_path = _find_executable(
        "host compiler",
        configured=host_compiler or env.get("GFAAS_NVCC_CCBIN") or env.get("CXX"),
        candidates=("g++", "c++"),
        required=False,
    )

    version_output = _command_output((nvcc_path, "--version"), env)
    supported_output = _command_output((nvcc_path, "--list-gpu-code"), env)
    supported = tuple(
        line.strip() for line in supported_output.splitlines() if line.strip().startswith("sm_")
    )
    if not supported:
        raise LocalCudaError("nvcc did not report a supported GPU architecture")
    gpu_output = _command_output(
        (
            nvidia_smi_path,
            "--query-gpu=index,uuid,name,compute_cap,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ),
        env,
    )
    gpus = _parse_gpus(gpu_output)
    visible_devices = device if device is not None else env.get("CUDA_VISIBLE_DEVICES")
    selected_gpu = _select_gpu(
        gpus,
        device=device,
        cuda_visible_devices=env.get("CUDA_VISIBLE_DEVICES"),
    )
    return LocalCudaToolchain(
        nvcc=nvcc_path,
        nvcc_version=_parse_nvcc_version(version_output),
        cuda_root=str(Path(nvcc_path).parent.parent),
        nvidia_smi=nvidia_smi_path,
        ncu=ncu_path,
        host_compiler=compiler_path,
        supported_architectures=supported,
        gpus=gpus,
        selected_gpu=selected_gpu,
        cuda_visible_devices=visible_devices,
    )


def _has_architecture_flag(flags: Sequence[str]) -> bool:
    for flag in flags:
        if flag in _NVCC_ARCH_FLAGS or any(
            flag.startswith(f"{name}=") for name in _NVCC_ARCH_FLAGS
        ):
            return True
    return False


def architecture_flags(
    flags: Sequence[str],
    *,
    requested: str | None,
    environment: Mapping[str, str],
    toolchain: LocalCudaToolchain,
) -> tuple[list[str], str]:
    values = list(flags)
    if _has_architecture_flag(values):
        if requested is not None:
            raise LocalCudaError("use either --arch or an nvcc architecture flag, not both")
        return values, "nvcc-flags"

    architecture = requested or environment.get("GFAAS_CUDA_ARCH") or "native"
    if architecture == "native":
        architecture = toolchain.selected_gpu.architecture
    if not _CUDA_ARCHITECTURE.fullmatch(architecture):
        raise LocalCudaError("CUDA architecture must use native or sm_NNN")
    if architecture not in toolchain.supported_architectures:
        raise LocalCudaError(
            f"nvcc {toolchain.nvcc_version} does not support local architecture {architecture}"
        )
    return [f"-arch={architecture}", *values], architecture


def _remaining_timeout(deadline: float, phase: str) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise LocalCudaError(f"local CUDA {phase} exceeded the execution timeout")
    return remaining


def _run_process(
    command: Sequence[str],
    *,
    phase: str,
    environment: Mapping[str, str],
    cwd: Path | None,
    deadline: float,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            check=False,
            env=dict(environment),
            cwd=cwd,
            timeout=_remaining_timeout(deadline, phase),
        )
    except subprocess.TimeoutExpired as error:
        raise LocalCudaError(f"local CUDA {phase} exceeded the execution timeout") from error
    except OSError as error:
        raise LocalCudaError(f"local CUDA {phase} failed to start: {error}") from error


def run(
    source: Path,
    *,
    toolchain: LocalCudaToolchain,
    nvcc_flags: Sequence[str],
    program_args: Sequence[str],
    environment: Mapping[str, str],
    workdir: Path,
    timeout_seconds: int,
    profile: bool = False,
    ncu_args: Sequence[str] = (),
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    with tempfile.TemporaryDirectory(prefix="gfaas-local-cuda-") as temporary_directory:
        temporary = Path(temporary_directory)
        binary = temporary / "kernel"
        compile_command = [toolchain.nvcc]
        if toolchain.host_compiler is not None:
            compile_command.extend(("-ccbin", toolchain.host_compiler))
        compile_command.extend((*nvcc_flags, str(source.resolve()), "-o", str(binary)))
        compile_started = time.monotonic()
        compile_process = _run_process(
            compile_command,
            phase="compilation",
            environment=environment,
            cwd=None,
            deadline=deadline,
        )
        compile_ms = int((time.monotonic() - compile_started) * 1000)
        if compile_process.returncode != 0:
            return {
                "phase": "compile",
                "stdout": compile_process.stdout,
                "stderr": compile_process.stderr,
                "returncode": compile_process.returncode,
                "compile_ms": compile_ms,
                "run_ms": 0,
                "ncu_csv": None,
            }

        ncu_csv: str | None = None
        if profile:
            if toolchain.ncu is None:
                raise LocalCudaError("ncu is required for local profiling")
            report_path = temporary / "report.csv"
            run_command = [
                toolchain.ncu,
                "--csv",
                "--log-file",
                str(report_path),
                *(ncu_args or ("--set", "full")),
                str(binary),
                *program_args,
            ]
        else:
            report_path = None
            run_command = [str(binary), *program_args]

        run_started = time.monotonic()
        run_process = _run_process(
            run_command,
            phase="execution",
            environment=environment,
            cwd=workdir,
            deadline=deadline,
        )
        run_ms = int((time.monotonic() - run_started) * 1000)
        if report_path is not None and report_path.is_file():
            ncu_csv = report_path.read_text(encoding="utf-8")
        return {
            "phase": "run",
            "stdout": (compile_process.stdout or "") + run_process.stdout,
            "stderr": (compile_process.stderr or "") + run_process.stderr,
            "returncode": run_process.returncode,
            "compile_ms": compile_ms,
            "run_ms": run_ms,
            "ncu_csv": ncu_csv,
        }
