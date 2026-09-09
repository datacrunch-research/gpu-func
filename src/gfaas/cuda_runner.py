"""Harness shipped to the worker on every ``gfaas.compile_and_run`` call.

This file is intentionally part of the SDK, **not** baked into an image:
the SDK packages it as a one-file bundle on each submit, fc-worker
extracts it under ``/workspace/app``, imports it, and calls ``run(...)``.
That way the "cuda-nvcc" image is just any CUDA-devel rootfs with
``python3 + cloudpickle + nvcc + ncu`` available — no separate Dockerfile,
no separate publish step.

Returned dict (cloudpickle-encoded by fc-worker's wrapper)::

    {
        "phase":      "compile" | "run",
        "stdout":     str,
        "stderr":     str,
        "returncode": int,
        "compile_ms": int,
        "run_ms":     int,
        "ncu_csv":    str | None,
    }
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

_COMPILED_ARTIFACT_ENV = "GFAAS_COMPILED_ARTIFACT_ID"
_COMPILED_OUTPUT_DIRECTORY = "compiled-cuda"


def _which(cmd: str) -> str:
    found = shutil.which(cmd)
    if found:
        return found

    candidates = [f"/usr/local/cuda/bin/{cmd}"]
    if cmd == "ncu":
        candidates.extend(sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"), reverse=True))

    for candidate in candidates:
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return candidate

    raise RuntimeError(f"required binary not found in PATH: {cmd}")


def _host_cxx_flags() -> list[str]:
    override = os.environ.get("GFAAS_NVCC_CCBIN") or os.environ.get("CXX")
    candidates = [
        override,
        "/usr/bin/g++",
        shutil.which("g++"),
        shutil.which("c++"),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return ["-ccbin", candidate]
    return []


def _subprocess_env(workdir: str) -> dict[str, str]:
    env = os.environ.copy()
    base_path = "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    current_path = env.get("PATH", "")
    env["PATH"] = f"{base_path}:{current_path}" if current_path else base_path

    home = env.get("HOME", "")
    if not home or not os.path.isdir(home) or not os.access(home, os.W_OK):
        env["HOME"] = workdir

    xdg = env.get("XDG_CONFIG_HOME", "")
    if not xdg or not os.path.isdir(xdg) or not os.access(xdg, os.W_OK):
        env["XDG_CONFIG_HOME"] = os.path.join(workdir, ".config")

    os.makedirs(env["HOME"], exist_ok=True)
    os.makedirs(env["XDG_CONFIG_HOME"], exist_ok=True)
    return env


def _workdir_root() -> str | None:
    # fc-worker binds a writable host directory at FC_IO_ROOT. Prefer it over
    # /tmp because the runtime mounts /tmp with noexec, which breaks running
    # nvcc output binaries from the default tempfile location.
    root = os.environ.get("FC_IO_ROOT")
    if root and os.path.isdir(root) and os.access(root, os.W_OK | os.X_OK):
        return root
    return None


def _program_workdir(fallback: str) -> str:
    output_root = os.environ.get("GFAAS_OUTPUT_ROOT")
    if not output_root:
        return fallback
    if not os.path.isdir(output_root) or not os.access(output_root, os.W_OK | os.X_OK):
        raise RuntimeError("CUDA program output directory is unavailable")
    return output_root


def run(
    *,
    source: str,
    nvcc_flags: list[str] | None = None,
    program_args: list[str] | None = None,
    profile: bool = False,
    ncu_args: list[str] | None = None,
) -> dict[str, Any]:
    """Compatibility path for coordinators that do not support Call stages."""
    workdir, bin_path, compile_report, tool_env = _compile_program(source, nvcc_flags)
    if compile_report["returncode"] != 0:
        return compile_report
    return _execute_program(
        bin_path,
        workdir,
        tool_env,
        compile_report,
        program_args=program_args,
        profile=profile,
        ncu_args=ncu_args,
    )


def compile_stage(
    *,
    source: str,
    nvcc_flags: list[str] | None = None,
    program_args: list[str] | None = None,
    profile: bool = False,
    ncu_args: list[str] | None = None,
) -> dict[str, Any]:
    """Compile CUDA without a GPU lease and publish an executable tree."""
    del program_args, profile, ncu_args
    workdir, bin_path, report, tool_env = _compile_program(source, nvcc_flags)
    if report["stdout"]:
        print(report["stdout"], end="")
    if report["stderr"]:
        print(report["stderr"], end="", file=sys.stderr)
    if report["returncode"] != 0:
        raise RuntimeError(
            f"nvcc failed with status {report['returncode']}: {report['stderr'].strip()}"
        )

    output_root = os.environ.get("GFAAS_OUTPUT_ROOT")
    if not output_root:
        raise RuntimeError("CUDA compilation output is only available inside a remote Call")
    output_dir = os.path.join(output_root, _COMPILED_OUTPUT_DIRECTORY)
    os.makedirs(output_dir, exist_ok=False)
    published_binary = os.path.join(output_dir, "kernel")
    shutil.copy2(bin_path, published_binary)
    os.chmod(published_binary, 0o755)
    manifest = {
        "schema": "vfunc.cuda-compiled/v1",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "nvcc_flags": list(nvcc_flags or []),
        "compiler": _compiler_identity(_which("nvcc"), tool_env),
        "target_gpu_pool": os.environ.get("GFAAS_TARGET_GPU_POOL"),
        "build_image": {
            "name": os.environ.get("GFAAS_BUILD_IMAGE_NAME"),
            "digest": os.environ.get("GFAAS_BUILD_IMAGE_DIGEST"),
        },
        "platform": {
            "machine": platform.machine(),
            "system": platform.system(),
            "libc": list(platform.libc_ver()),
        },
        "compile": report,
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, sort_keys=True)
        manifest_file.flush()
        os.fsync(manifest_file.fileno())
    return report


def execute_stage(
    *,
    source: str,
    nvcc_flags: list[str] | None = None,
    program_args: list[str] | None = None,
    profile: bool = False,
    ncu_args: list[str] | None = None,
) -> dict[str, Any]:
    """Run a compiled CUDA Artifact after the coordinator acquires a GPU."""
    artifact_id = os.environ.get(_COMPILED_ARTIFACT_ENV)
    artifact_root = os.environ.get("GFAAS_ARTIFACT_ROOT")
    if not artifact_id or not artifact_root:
        raise RuntimeError("compiled CUDA Artifact was not staged for execution")
    compiled_root = os.path.join(artifact_root, artifact_id)
    manifest_path = os.path.join(compiled_root, "manifest.json")
    with open(manifest_path, encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if manifest.get("schema") != "vfunc.cuda-compiled/v1":
        raise RuntimeError("compiled CUDA Artifact has an unsupported manifest")
    if manifest.get("source_sha256") != hashlib.sha256(source.encode()).hexdigest():
        raise RuntimeError("compiled CUDA Artifact does not match the submitted source")
    if manifest.get("nvcc_flags") != list(nvcc_flags or []):
        raise RuntimeError("compiled CUDA Artifact does not match the compiler flags")
    expected_image = {
        "name": os.environ.get("GFAAS_BUILD_IMAGE_NAME"),
        "digest": os.environ.get("GFAAS_BUILD_IMAGE_DIGEST"),
    }
    if manifest.get("build_image") != expected_image:
        raise RuntimeError("compiled CUDA Artifact does not match the execution image")
    if manifest.get("target_gpu_pool") != os.environ.get("GFAAS_TARGET_GPU_POOL"):
        raise RuntimeError("compiled CUDA Artifact does not match the target GPU pool")
    expected_platform = {
        "machine": platform.machine(),
        "system": platform.system(),
        "libc": list(platform.libc_ver()),
    }
    if manifest.get("platform") != expected_platform:
        raise RuntimeError("compiled CUDA Artifact does not match the execution platform")
    compile_report = manifest.get("compile")
    if not isinstance(compile_report, dict) or compile_report.get("returncode") != 0:
        raise RuntimeError("compiled CUDA Artifact does not contain a successful compilation")
    workdir = tempfile.mkdtemp(prefix="gfaas-cuda-run-", dir=_workdir_root())
    return _execute_program(
        os.path.join(compiled_root, "kernel"),
        workdir,
        _subprocess_env(workdir),
        compile_report,
        program_args=program_args,
        profile=profile,
        ncu_args=ncu_args,
    )


def _compile_program(
    source: str,
    nvcc_flags: list[str] | None,
) -> tuple[str, str, dict[str, Any], dict[str, str]]:
    nvcc = _which("nvcc")
    workdir = tempfile.mkdtemp(prefix="gfaas-cuda-", dir=_workdir_root())
    src_path = os.path.join(workdir, "kernel.cu")
    bin_path = os.path.join(workdir, "kernel")
    tool_env = _subprocess_env(workdir)

    with open(src_path, "w") as f:
        f.write(source)

    compile_cmd = [nvcc, *_host_cxx_flags(), *(nvcc_flags or []), src_path, "-o", bin_path]
    t0 = time.monotonic()
    compile_proc = subprocess.run(
        compile_cmd,
        capture_output=True,
        text=True,
        check=False,
        env=tool_env,
    )
    compile_ms = int((time.monotonic() - t0) * 1000)

    report = {
        "phase": "compile",
        "stdout": compile_proc.stdout,
        "stderr": compile_proc.stderr,
        "returncode": compile_proc.returncode,
        "compile_ms": compile_ms,
        "run_ms": 0,
        "ncu_csv": None,
    }
    return workdir, bin_path, report, tool_env


def _execute_program(
    bin_path: str,
    workdir: str,
    tool_env: dict[str, str],
    compile_report: dict[str, Any],
    *,
    program_args: list[str] | None,
    profile: bool,
    ncu_args: list[str] | None,
) -> dict[str, Any]:

    program_args = list(program_args or [])
    ncu_csv: str | None = None
    csv_path = os.path.join(workdir, "report.csv")

    if profile:
        ncu = _which("ncu")
        run_cmd = [
            ncu,
            "--csv",
            "--log-file",
            csv_path,
            *(ncu_args or ["--set", "full"]),
            bin_path,
            *program_args,
        ]
    else:
        run_cmd = [bin_path, *program_args]

    t0 = time.monotonic()
    run_proc = subprocess.run(
        run_cmd,
        capture_output=True,
        text=True,
        check=False,
        env=tool_env,
        cwd=_program_workdir(workdir),
    )
    run_ms = int((time.monotonic() - t0) * 1000)

    if profile and os.path.exists(csv_path):
        with open(csv_path) as f:
            ncu_csv = f.read()

    return {
        "phase": "run",
        "stdout": (compile_report.get("stdout") or "") + run_proc.stdout,
        "stderr": (compile_report.get("stderr") or "") + run_proc.stderr,
        "returncode": run_proc.returncode,
        "compile_ms": compile_report.get("compile_ms", 0),
        "run_ms": run_ms,
        "ncu_csv": ncu_csv,
    }


def _compiler_identity(nvcc: str, environment: dict[str, str]) -> str:
    result = subprocess.run(
        [nvcc, "--version"],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    if result.returncode != 0:
        return nvcc
    return (result.stdout or result.stderr).strip()
