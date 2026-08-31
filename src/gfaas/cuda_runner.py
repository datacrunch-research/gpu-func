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
import os
import shutil
import subprocess
import tempfile
import time
from typing import Any


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

    if compile_proc.returncode != 0:
        return {
            "phase": "compile",
            "stdout": compile_proc.stdout,
            "stderr": compile_proc.stderr,
            "returncode": compile_proc.returncode,
            "compile_ms": compile_ms,
            "run_ms": 0,
            "ncu_csv": None,
        }

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
        "stdout": (compile_proc.stdout or "") + run_proc.stdout,
        "stderr": (compile_proc.stderr or "") + run_proc.stderr,
        "returncode": run_proc.returncode,
        "compile_ms": compile_ms,
        "run_ms": run_ms,
        "ncu_csv": ncu_csv,
    }
