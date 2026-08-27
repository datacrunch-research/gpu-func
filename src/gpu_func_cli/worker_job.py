"""Worker-side CUDA exercise runner submitted through the gfaas SDK."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

from gfaas import ArtifactOutput, ArtifactRef, scratch_path

PROFILE_OUTPUT = ArtifactOutput.directory(
    "profiles",
    "profiles",
    kind="profile",
    required=False,
    publish_on_failure=True,
)


def run(
    *,
    job: dict[str, Any],
    workspace: ArtifactRef,
    profile_output: ArtifactOutput = PROFILE_OUTPUT,
) -> dict[str, Any]:
    """Run one validated custom-kernel or CUDA-course job."""
    try:
        with tempfile.TemporaryDirectory(prefix="gpu-func-", dir=scratch_path()) as temporary:
            workdir = Path(temporary, "workspace")
            shutil.copytree(workspace.path, workdir, symlinks=False)
            _make_workspace_writable(workdir)
            file_hashes = _verify_workspace(workdir, job.get("hashes", {}))
            deadline = time.monotonic() + int(job.get("remote", {}).get("timeout_s", 600))
            if job.get("target", {}).get("kind") == "custom":
                return _run_custom_job(
                    workdir,
                    job,
                    file_hashes,
                    deadline,
                    profile_output,
                )
            if job.get("course_runner", {}).get("enabled"):
                return _run_course_runner(
                    workdir,
                    job,
                    file_hashes,
                    deadline,
                    profile_output,
                )
            raise ValueError("job has no supported execution target")
    except Exception as exc:
        return {
            "schema_version": 2,
            "status": "setup_error",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def _run_custom_job(
    workdir: Path,
    job: dict[str, Any],
    file_hashes: dict[str, str],
    deadline: float,
    profile_output: ArtifactOutput,
) -> dict[str, Any]:
    custom = job["custom"]
    flags = list(custom.get("flags", []))
    if not any(flag == "-arch" or flag.startswith("-arch=") for flag in flags):
        detected_arch = _detect_cuda_arch()
        if detected_arch:
            flags.append(f"-arch={detected_arch}")
    compile_args = [
        _which("nvcc"),
        *_host_cxx_flags(),
        *flags,
        *custom["sources"],
        "-o",
        custom["output"],
    ]
    compile_result = _run_process(
        compile_args,
        workdir,
        min(_remaining(deadline), 300.0),
    )
    if compile_result["timed_out"]:
        return _custom_result(
            "timeout",
            compile_result,
            None,
            [],
            file_hashes,
            error="nvcc exceeded the Call time budget",
        )
    if compile_result["returncode"] != 0:
        return _custom_result(
            "compile_failed",
            compile_result,
            None,
            [],
            file_hashes,
        )
    if custom["command"] == "compile":
        return _custom_result("passed", compile_result, None, [], file_hashes)

    program = ["./" + custom["output"], *custom.get("program_args", [])]
    export_base = Path(custom.get("report_name") or "custom_profile").name
    if custom["command"] == "profile":
        run_args = [_which("ncu"), *custom.get("ncu_args", ["--set", "basic"])]
        nvtx_range = custom.get("nvtx_range", "")
        if nvtx_range:
            suffix = "/" if not nvtx_range.endswith("/") else ""
            run_args += ["--nvtx", "--nvtx-include", nvtx_range + suffix]
        run_args += ["--force-overwrite", "--export", export_base, *program]
    else:
        run_args = program

    run_result = _run_process(run_args, workdir, _remaining(deadline))
    profiles: list[dict[str, str]] = []
    if custom["command"] == "profile":
        report = workdir / f"{export_base}.ncu-rep"
        if report.is_file():
            profiles.append(_publish_profile(report, profile_output))

    status = "passed"
    if run_result["timed_out"]:
        status = "timeout"
    elif run_result["returncode"] != 0:
        status = "error"
    return _custom_result(status, compile_result, run_result, profiles, file_hashes)


def _custom_result(
    status: str,
    compile_result: dict[str, Any],
    run_result: dict[str, Any] | None,
    profiles: list[dict[str, str]],
    file_hashes: dict[str, str],
    *,
    error: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 2,
        "status": status,
        "compile": compile_result,
        "run": run_result,
        "artifacts": {"profiles": profiles},
        "worker": _worker_info(),
        "file_hashes": file_hashes,
    }
    if error:
        result["error"] = error
    return result


def _run_course_runner(
    workdir: Path,
    job: dict[str, Any],
    file_hashes: dict[str, str],
    deadline: float,
    profile_output: ArtifactOutput,
) -> dict[str, Any]:
    spec = job["course_runner"]
    cwd = _safe_path(workdir, spec.get("cwd", "."))
    result = _run_process(
        list(spec["command"]),
        cwd,
        _remaining(deadline),
        env={"PYTHONPATH": str(workdir)},
    )

    report_json = None
    json_name = spec.get("json_out", "_gpu_func_cli.json")
    json_path = cwd / json_name
    if json_path.is_file():
        try:
            report_json = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            report_json = None

    profiles: list[dict[str, str]] = []
    seen: set[Path] = set()
    for pattern in spec.get("artifact_globs", []):
        if pattern == json_name:
            continue
        for raw_path in glob.glob(str(cwd / pattern)):
            path = Path(raw_path)
            if path in seen or not path.is_file() or path.suffix != ".ncu-rep":
                continue
            seen.add(path)
            profiles.append(_publish_profile(path, profile_output))

    status = "passed"
    if result["timed_out"]:
        status = "timeout"
    elif result["returncode"] != 0:
        status = "error"
    return {
        "schema_version": 2,
        "status": status,
        "course_runner": result,
        "report_json": report_json,
        "artifacts": {"profiles": profiles},
        "worker": _worker_info(),
        "file_hashes": file_hashes,
    }


def _publish_profile(path: Path, output: ArtifactOutput) -> dict[str, str]:
    output.path.mkdir(parents=True, exist_ok=True)
    destination = output.path / path.name
    if destination.exists():
        raise FileExistsError(f"duplicate profile report name: {path.name}")
    shutil.copyfile(path, destination)
    return {"filename": path.name, "output_name": output.name}


def _verify_workspace(root: Path, expected: Any) -> dict[str, str]:
    if not isinstance(expected, dict):
        raise ValueError("job workspace hashes are invalid")
    actual: dict[str, str] = {}
    for raw_name, raw_digest in expected.items():
        if not isinstance(raw_name, str) or not isinstance(raw_digest, str):
            raise ValueError("job workspace hash entry is invalid")
        path = _safe_path(root, raw_name)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"workspace file is unavailable: {raw_name}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != raw_digest:
            raise ValueError(f"workspace hash mismatch: {raw_name}")
        actual[raw_name] = digest
    return actual


def _make_workspace_writable(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_dir():
            path.chmod(0o755)
        elif path.is_file():
            path.chmod(0o644)
    root.chmod(0o755)


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Call time budget was exhausted before the next phase")
    return remaining


def _run_process(
    args: list[str],
    cwd: Path,
    timeout_s: float,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    start = time.monotonic()
    tool_env = _subprocess_env(cwd)
    if env:
        tool_env.update(env)
    try:
        process = subprocess.Popen(
            args,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            env=tool_env,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        return _process_result(args, 127, "", str(exc), start, timed_out=False)
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
        return _process_result(
            args,
            process.returncode,
            stdout,
            stderr,
            start,
            timed_out=False,
        )
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        return _process_result(
            args,
            None,
            stdout,
            stderr,
            start,
            timed_out=True,
        )


def _process_result(
    args: list[str],
    returncode: int | None,
    stdout: str,
    stderr: str,
    start: float,
    *,
    timed_out: bool,
) -> dict[str, Any]:
    return {
        "args": args,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "ms": int((time.monotonic() - start) * 1000),
        "timed_out": timed_out,
    }


def _safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"unsafe workspace path: {relative}")
    return path


def _which(command: str) -> str:
    found = shutil.which(command)
    if found:
        return found
    candidates = [f"/usr/local/cuda/bin/{command}"]
    if command == "ncu":
        candidates.extend(sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"), reverse=True))
    for candidate in candidates:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError(f"required binary not found in PATH: {command}")


def _host_cxx_flags() -> list[str]:
    override = os.environ.get("GFAAS_NVCC_CCBIN") or os.environ.get("CXX")
    candidates = [override, "/usr/bin/g++", shutil.which("g++"), shutil.which("c++")]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return ["-ccbin", candidate]
    return []


def _detect_cuda_arch() -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=compute_cap",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    lines = result.stdout.splitlines()
    first = lines[0].strip() if lines else ""
    if not first or any(part and not part.isdigit() for part in first.split(".")):
        return None
    return "sm_" + first.replace(".", "")


def _subprocess_env(workdir: Path) -> dict[str, str]:
    env = os.environ.copy()
    base_path = "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    current_path = env.get("PATH", "")
    env["PATH"] = base_path + (":" + current_path if current_path else "")
    home = env.get("HOME", "")
    if not home or not os.path.isdir(home) or not os.access(home, os.W_OK):
        env["HOME"] = str(workdir)
    xdg = env.get("XDG_CONFIG_HOME", "")
    if not xdg or not os.path.isdir(xdg) or not os.access(xdg, os.W_OK):
        env["XDG_CONFIG_HOME"] = str(workdir / ".config")
    Path(env["HOME"]).mkdir(parents=True, exist_ok=True)
    Path(env["XDG_CONFIG_HOME"]).mkdir(parents=True, exist_ok=True)
    return env


def _worker_info() -> dict[str, str]:
    return {
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
