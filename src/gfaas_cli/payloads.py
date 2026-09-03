"""Payload builders for checkout-exercise and custom CUDA remote jobs."""

from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import stat
from pathlib import Path
from typing import Any

from .constants import (
    _CHECKOUT_SKIP_DIRS,
    GPU_DEFAULTS,
    MAX_WORKSPACE_BYTES,
    MAX_WORKSPACE_FILES,
)
from .errors import CliError


def _resolve_course_root(args: argparse.Namespace) -> Path | None:
    """Locate a cuda-course checkout for the requested exercise, or ``None``.

    Tries ``--course-root`` (and its parents), then walks up from ``--file`` and
    the cwd. Returns ``None`` when no checkout is found -- exercises require one,
    so the caller raises a clear error in that case.
    """
    exercise_id = args.exercise_id

    def is_root(d: Path) -> bool:
        return (d / "runner" / "cli.py").is_file() and (
            d / "exercises" / exercise_id / "run.py"
        ).is_file()

    explicit = getattr(args, "course_root", None)
    if explicit:
        p = Path(explicit).expanduser()
        for d in [p, *p.resolve().parents]:
            if is_root(d):
                return d
        return None

    starts: list[Path] = []
    if args.source_file:
        starts.append(Path(args.source_file).expanduser())
    starts.append(Path.cwd())
    seen: set[str] = set()
    for start in starts:
        for d in [start, *start.resolve().parents]:
            key = str(d)
            if key in seen:
                continue
            seen.add(key)
            if is_root(d):
                return d

    return None


def _norm_spec(spec: str, exercise_id: str) -> str:
    """Normalise a spec arg to a checkout-relative path (drop the ``exercises/<id>/`` prefix)."""
    return spec.replace("\\", "/").removeprefix(f"exercises/{exercise_id}/")


def _walk_checkout(
    files: dict[str, bytes],
    hashes: dict[str, str],
    root: Path,
    base: Path,
    extra_skip: frozenset[str] | set[str] = frozenset(),
) -> None:
    """Collect regular files without following links.

    The selected files become an immutable gfaas tree Artifact. Reject links
    instead of resolving them so an untrusted exercise cannot include data from
    outside its directory. Binary inputs are preserved unchanged.
    """
    skip = _CHECKOUT_SKIP_DIRS | set(extra_skip)
    if root.is_symlink() or base.is_symlink():
        raise CliError(f"exercise workspace root cannot be a symbolic link: {base}")
    root = root.resolve()
    base = base.resolve()
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        kept_dirs: list[str] = []
        for name in sorted(dirnames):
            child = Path(dirpath, name)
            if name in skip:
                continue
            if child.is_symlink():
                raise CliError(f"exercise workspace contains a symbolic link: {child}")
            kept_dirs.append(name)
        dirnames[:] = kept_dirs
        for fn in sorted(filenames):
            full = Path(dirpath) / fn
            metadata = full.lstat()
            if full.is_symlink():
                raise CliError(f"exercise workspace contains a symbolic link: {full}")
            if not stat.S_ISREG(metadata.st_mode):
                raise CliError(f"exercise workspace entry is not a regular file: {full}")
            if metadata.st_nlink != 1:
                raise CliError(f"exercise workspace contains a hard-linked file: {full}")
            data = full.read_bytes()
            rel = _clean_payload_path(full.relative_to(root).as_posix())
            files[rel] = data
            hashes[rel] = hashlib.sha256(data).hexdigest()
    _check_workspace_limits(files)


def _build_checkout_payload(
    *,
    course_root: Path,
    exercise_id: str,
    mode: str,
    source_file: Path | None,
    specs: list[str],
    gpu: str,
    gpu_type: str,
    arch: str,
    image: str,
    timeout_s: int,
    verbose: bool,
) -> dict[str, Any]:
    """Build a job that ships the live ``runner/`` + the chosen exercise and runs
    the exercise's own ``run.py`` on the worker. The exercise's ``solutions/`` dir
    is never shipped."""
    ex_dir = course_root / "exercises" / exercise_id
    runner_dir = course_root / "runner"
    if not (ex_dir / "run.py").is_file():
        raise CliError(f"exercise {exercise_id!r} not found under {course_root}")
    if not (runner_dir / "cli.py").is_file():
        raise CliError(f"no course runner under {runner_dir}")

    files: dict[str, bytes] = {}
    hashes: dict[str, str] = {}
    _walk_checkout(files, hashes, course_root, runner_dir)
    _walk_checkout(files, hashes, course_root, ex_dir, extra_skip={"solutions"})

    inc = runner_dir / "include"
    if inc.is_dir():
        for header in sorted(inc.iterdir()):
            if header.is_symlink():
                raise CliError(f"exercise workspace contains a symbolic link: {header}")
            if header.is_file():
                data = header.read_bytes()
                rel = _clean_payload_path(f"exercises/{exercise_id}/runner/include/{header.name}")
                files[rel] = data
                hashes[rel] = hashlib.sha256(data).hexdigest()
        _check_workspace_limits(files)

    file_arg: str | None = None
    if source_file is not None:
        if not source_file.is_file():
            raise CliError(f"--file not found: {source_file}")
        if source_file.is_symlink():
            raise CliError(f"--file cannot be a symbolic link: {source_file}")
        data = source_file.read_bytes()
        abs_src = source_file.resolve()
        try:
            file_arg = abs_src.relative_to(ex_dir.resolve()).as_posix()
        except ValueError:
            file_arg = "__submitted__.cu"
        rel = _clean_payload_path(f"exercises/{exercise_id}/{file_arg}")
        files[rel] = data
        hashes[rel] = hashlib.sha256(data).hexdigest()
        _check_workspace_limits(files)

    json_out = "_gfaas_cli.json"
    command = ["python3", "run.py", "--json", json_out]
    if file_arg:
        command += ["--file", file_arg]
    if verbose:
        command.append("-v")
    if arch:
        command += ["--arch", arch]
    command.append(mode)
    command.extend(_norm_spec(s, exercise_id) for s in specs)

    return {
        "schema_version": 1,
        "asset_version": "checkout",
        "target": {"kind": "exercise", "exercise_id": exercise_id, "source": "checkout"},
        "remote": {
            "gpu": gpu,
            "gpu_type": gpu_type,
            "arch": arch,
            "image": image,
            "timeout_s": timeout_s,
        },
        "command": {"mode": mode},
        "course_runner": {
            "enabled": True,
            "cwd": f"exercises/{exercise_id}",
            "command": command,
            "json_out": json_out,
            "artifact_globs": [json_out, "*.ncu-rep"],
            "timeout_s": timeout_s,
        },
        "files": files,
        "hashes": hashes,
    }


def _build_flat_exercise_payload(
    *,
    exercise_dir: Path,
    exercise_id: str,
    mode: str,
    source_file: Path | None,
    specs: list[str],
    gpu: str,
    gpu_type: str,
    arch: str,
    image: str,
    timeout_s: int,
    verbose: bool,
) -> dict[str, Any]:
    """Build a course-runner job from a *flat* exercise dir (``run.py`` and
    ``runner/`` as siblings, e.g. an unzipped exercise) instead of a cuda-course
    checkout.

    ``runner/`` is shipped at the payload root so the worker's
    ``PYTHONPATH=workdir`` imports it; the whole exercise (its own ``runner/``
    included) is shipped under ``exercises/<id>/`` and run from there, so
    ``run.py``'s ``base_dir/runner/include`` include path resolves -- the same
    shape the checkout path produces, with no relocation of the source files."""
    runner_dir = exercise_dir / "runner"
    if not (exercise_dir / "run.py").is_file():
        raise CliError(f"no run.py under {exercise_dir}")
    if not (runner_dir / "cli.py").is_file():
        raise CliError(f"no runner/cli.py under {exercise_dir}")

    files: dict[str, bytes] = {}
    hashes: dict[str, str] = {}
    # runner/ at the payload root -> importable via the worker's PYTHONPATH=workdir.
    _walk_checkout(files, hashes, exercise_dir, runner_dir)

    # The whole exercise (its own runner/, incl. include/, comes along) under
    # exercises/<id>/, which becomes run.py's cwd and base_dir.
    sub = f"exercises/{exercise_id}"
    sub_files: dict[str, bytes] = {}
    sub_hashes: dict[str, str] = {}
    _walk_checkout(sub_files, sub_hashes, exercise_dir, exercise_dir, extra_skip={"solutions"})
    for key, data in sub_files.items():
        rel = _clean_payload_path(f"{sub}/{key}")
        files[rel] = data
        hashes[rel] = sub_hashes[key]
    _check_workspace_limits(files)

    file_arg: str | None = None
    if source_file is not None:
        if not source_file.is_file():
            raise CliError(f"--file not found: {source_file}")
        if source_file.is_symlink():
            raise CliError(f"--file cannot be a symbolic link: {source_file}")
        data = source_file.read_bytes()
        abs_src = source_file.resolve()
        try:
            file_arg = abs_src.relative_to(exercise_dir.resolve()).as_posix()
        except ValueError:
            file_arg = "__submitted__.cu"
        rel = _clean_payload_path(f"{sub}/{file_arg}")
        files[rel] = data
        hashes[rel] = hashlib.sha256(data).hexdigest()
        _check_workspace_limits(files)

    json_out = "_gfaas_cli.json"
    command = ["python3", "run.py", "--json", json_out]
    if file_arg:
        command += ["--file", file_arg]
    if verbose:
        command.append("-v")
    if arch:
        command += ["--arch", arch]
    command.append(mode)
    command.extend(s.replace("\\", "/") for s in specs)

    return {
        "schema_version": 1,
        "asset_version": "flat",
        "target": {"kind": "exercise", "exercise_id": exercise_id, "source": "flat"},
        "remote": {
            "gpu": gpu,
            "gpu_type": gpu_type,
            "arch": arch,
            "image": image,
            "timeout_s": timeout_s,
        },
        "command": {"mode": mode},
        "course_runner": {
            "enabled": True,
            "cwd": sub,
            "command": command,
            "json_out": json_out,
            "artifact_globs": [json_out, "*.ncu-rep"],
            "timeout_s": timeout_s,
        },
        "files": files,
        "hashes": hashes,
    }


def _resolve_gpu(
    gpu: str | None,
    gpu_type: str | None,
    arch: str | None,
) -> tuple[str, str]:
    """Resolve the compatibility GPU label and current pool options."""
    if gpu:
        default_type, default_arch = GPU_DEFAULTS.get(gpu.upper(), (gpu.lower(), ""))
    else:
        default_type = gpu_type or "any"
        _, default_arch = GPU_DEFAULTS.get(default_type.upper(), (default_type, ""))
    return gpu_type or default_type, arch if arch is not None else default_arch


def _build_custom_payload(args: argparse.Namespace, gpu_type: str, arch: str) -> dict[str, Any]:
    """Build a custom-kernel job: source -> ``kernel.cu`` (+ optional ``harness.cu``),
    compile flags, ncu args, and a sanitised ``report_name`` (default: source stem)."""
    source_path = Path(args.source)
    if not source_path.is_file():
        raise CliError(f"source file not found: {source_path}")
    if source_path.is_symlink():
        raise CliError(f"source file cannot be a symbolic link: {source_path}")
    raw_report_name = args.report_name or source_path.stem
    report_name = (
        "".join(c if c.isalnum() or c in "._-" else "_" for c in raw_report_name)
        or "custom_profile"
    )
    harness_path = Path(args.harness) if args.harness else None
    if harness_path and not harness_path.is_file():
        raise CliError(f"harness file not found: {harness_path}")
    if harness_path and harness_path.is_symlink():
        raise CliError(f"harness file cannot be a symbolic link: {harness_path}")
    output_path = Path(args.output)
    if (
        output_path.name != args.output
        or output_path.name in {"", ".", ".."}
        or "/" in args.output
        or "\\" in args.output
    ):
        raise CliError("--output must be a file name without a directory")

    files: dict[str, bytes] = {}
    hashes: dict[str, str] = {}

    def add_text(path: str, text: str) -> None:
        clean = _clean_payload_path(path)
        data = text.encode("utf-8")
        files[clean] = data
        hashes[clean] = hashlib.sha256(data).hexdigest()
        _check_workspace_limits(files)

    add_text("kernel.cu", source_path.read_text(encoding="utf-8"))
    sources = ["kernel.cu"]
    if harness_path:
        add_text("harness.cu", harness_path.read_text(encoding="utf-8"))
        sources.append("harness.cu")

    flags = shlex.split(args.nvcc_flags) if args.nvcc_flags else []
    if arch:
        flags.append(f"-arch={arch}")

    return {
        "schema_version": 1,
        "target": {
            "kind": "custom",
            "source": "kernel.cu",
            "harness": "harness.cu" if harness_path else None,
        },
        "remote": {
            "gpu": args.gpu,
            "gpu_type": gpu_type,
            "image": args.image,
            "timeout_s": args.timeout,
        },
        "command": {
            "mode": f"custom-{args.custom_command}",
        },
        "custom": {
            "command": args.custom_command,
            "sources": sources,
            "flags": flags,
            "output": args.output,
            "report_name": report_name,
            "program_args": list(args.arg),
            "ncu_args": shlex.split(args.ncu_args) if args.ncu_args else ["--set", "basic"],
            "nvtx_range": "" if args.no_nvtx_filter else args.nvtx_range,
            "timeout_s": args.timeout,
        },
        "files": files,
        "hashes": hashes,
    }


def _clean_payload_path(path: str) -> str:
    """Return a normalised relative payload path, rejecting absolute or ``..`` paths."""
    clean = Path(path)
    if clean == Path(".") or clean.is_absolute() or ".." in clean.parts:
        raise CliError(f"unsafe payload path {path!r}")
    return clean.as_posix()


def _materialize_workspace(payload: dict[str, Any], destination: Path) -> dict[str, Any]:
    """Write payload files into *destination* and return the remote job spec.

    File bytes travel through a gfaas tree Artifact rather than through the
    cloudpickle argument. The returned job retains the expected hashes so the
    worker verifies its staged copy before executing it.
    """
    raw_files = payload.get("files")
    if not isinstance(raw_files, dict):
        raise CliError("payload has no workspace files")
    destination.mkdir(parents=True, exist_ok=False)
    for raw_name, raw_data in sorted(raw_files.items()):
        if not isinstance(raw_name, str) or not isinstance(raw_data, bytes):
            raise CliError("payload workspace contains an invalid file")
        relative = _clean_payload_path(raw_name)
        path = destination.joinpath(*Path(relative).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw_data)
    return {key: value for key, value in payload.items() if key != "files"}


def _check_workspace_limits(files: dict[str, bytes]) -> None:
    if len(files) > MAX_WORKSPACE_FILES:
        raise CliError(f"exercise workspace contains more than {MAX_WORKSPACE_FILES} files")
    size_bytes = sum(len(data) for data in files.values())
    if size_bytes > MAX_WORKSPACE_BYTES:
        raise CliError(f"exercise workspace exceeds the {MAX_WORKSPACE_BYTES // 1024**2} MiB limit")
