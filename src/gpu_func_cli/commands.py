"""Command handlers for CUDA exercises and custom kernels."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from .client import GfaasClient
from .constants import RC_OK, RC_SETUP
from .errors import CliError
from .output import _print_course_runner_result, _print_custom_result
from .payloads import (
    _build_checkout_payload,
    _build_custom_payload,
    _build_flat_exercise_payload,
    _materialize_workspace,
    _resolve_course_root,
    _resolve_gpu,
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _clean_output_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            key: (
                _ANSI_RE.sub("", value).splitlines()
                if key in {"stdout", "stderr"} and isinstance(value, str)
                else _clean_output_for_json(value)
            )
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [_clean_output_for_json(value) for value in obj]
    return obj


def _write_result_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as output:
            json.dump(_clean_output_for_json(value), output, indent=2)
            output.write("\n")
    except FileExistsError as exc:
        raise CliError(f"refusing to replace existing result file: {path}") from exc
    print(f"Results written to {path}")


def _cmd_workers(client: GfaasClient) -> int:
    pools = client.capabilities().get("gpu_pools", [])
    if not pools:
        print("No configured GPU pools")
        return RC_OK
    print(f"{'NAME':<20} {'STATUS':<14} {'CONNECTED':>9} {'AVAILABLE':>9}")
    for pool in pools:
        print(
            f"{str(pool.get('name', '')):<20} "
            f"{str(pool.get('status', 'unknown')):<14} "
            f"{str(pool.get('connected_workers', '')):>9} "
            f"{str(pool.get('available_workers', '')):>9}"
        )
    return RC_OK


def _cmd_custom(args: argparse.Namespace) -> int:
    with GfaasClient.from_args(args) as client:
        gpu_type, arch = _resolve_gpu(args.gpu, args.gpu_type, args.arch)
        gpu_type = client.resolve_gpu_type(gpu_type)
        args.gpu_type = gpu_type
        payload = _build_custom_payload(args, gpu_type, arch)
        result, call_id, downloaded = _submit_payload(
            client,
            args,
            payload,
            app_name="gpu-func-custom",
            label=f"custom {args.custom_command}",
            arch=arch or "worker-detected",
        )
        if result is None:
            return RC_OK if args.detach else RC_SETUP
        exit_code = _print_custom_result(
            result,
            args,
            downloaded_profiles=downloaded,
        )
        if args.json_path:
            _write_result_json(
                Path(args.json_path),
                {
                    "mode": f"custom-{args.custom_command}",
                    "remote": {
                        "call_id": call_id,
                        "gpu_type": gpu_type,
                        "gpu_count": args.gpu_count,
                        "image": args.image,
                    },
                    "result": result,
                },
            )
        return exit_code


def _autodetect_exercise(args: argparse.Namespace) -> None:
    if getattr(args, "exercise_dir", None):
        exercise_dir = Path(args.exercise_dir).expanduser()
        if not getattr(args, "exercise_id", None):
            args.exercise_id = exercise_dir.name
        return

    starts: list[Path] = []
    if args.source_file:
        starts.append(Path(args.source_file).expanduser().resolve().parent)
    starts.append(Path.cwd())
    seen: set[Path] = set()
    for start in starts:
        for directory in [start, *start.resolve().parents]:
            resolved = directory.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if (resolved / "run.py").is_file() and (resolved / "runner" / "cli.py").is_file():
                args.exercise_dir = str(resolved)
                if not getattr(args, "exercise_id", None):
                    args.exercise_id = resolved.name
                return

    if not getattr(args, "exercise_id", None):
        parts = Path.cwd().resolve().parts
        derived = next(
            (
                parts[index + 1]
                for index in range(len(parts) - 2, -1, -1)
                if parts[index] == "exercises"
            ),
            None,
        )
        if derived is None:
            raise CliError(
                "could not auto-detect an exercise. Run from inside an unzipped "
                "exercise, pass --exercise-dir, or pass --exercise-id with a "
                "cuda-course checkout"
            )
        args.exercise_id = derived


def _cmd_exercise_mode(args: argparse.Namespace) -> int:
    _autodetect_exercise(args)
    return _cmd_exercise(args)


def _cmd_exercise(args: argparse.Namespace) -> int:
    with GfaasClient.from_args(args) as client:
        gpu_type, arch = _resolve_gpu(args.gpu, args.gpu_type, args.arch)
        gpu_type = client.resolve_gpu_type(gpu_type)
        args.gpu_type = gpu_type
        exercise_dir = getattr(args, "exercise_dir", None)
        if exercise_dir:
            path = Path(exercise_dir).expanduser()
            if not (path / "run.py").is_file() or not (path / "runner" / "cli.py").is_file():
                raise CliError(
                    f"--exercise-dir {path} is not a flat exercise "
                    "(expected run.py and runner/cli.py)"
                )
            payload = _build_flat_exercise_payload(
                exercise_dir=path,
                exercise_id=args.exercise_id,
                mode=args.exercise_command,
                source_file=Path(args.source_file) if args.source_file else None,
                specs=list(args.specs),
                gpu=args.gpu,
                gpu_type=gpu_type,
                arch=arch,
                image=args.image,
                timeout_s=args.timeout,
                verbose=args.verbose,
            )
        else:
            course_root = _resolve_course_root(args)
            if (
                course_root is None
                or not (course_root / "exercises" / args.exercise_id / "run.py").is_file()
            ):
                raise CliError(
                    f"no cuda-course checkout with exercises/{args.exercise_id}/run.py "
                    "was found; pass --course-root or --exercise-dir"
                )
            payload = _build_checkout_payload(
                course_root=course_root,
                exercise_id=args.exercise_id,
                mode=args.exercise_command,
                source_file=Path(args.source_file) if args.source_file else None,
                specs=list(args.specs),
                gpu=args.gpu,
                gpu_type=gpu_type,
                arch=arch,
                image=args.image,
                timeout_s=args.timeout,
                verbose=args.verbose,
            )
        return _run_exercise_payload(args, client, payload, arch)


def _run_exercise_payload(
    args: argparse.Namespace,
    client: GfaasClient,
    payload: dict[str, Any],
    arch: str,
) -> int:
    mode = args.exercise_command
    result, call_id, downloaded = _submit_payload(
        client,
        args,
        payload,
        app_name="gpu-func-course",
        label=mode,
        arch=arch or "worker-detected",
    )
    if result is None:
        return RC_OK if args.detach else RC_SETUP
    code = _print_course_runner_result(
        result,
        args,
        downloaded_profiles=downloaded,
    )
    if args.json_path:
        _write_result_json(
            Path(args.json_path),
            {
                "mode": mode,
                "exercise": args.exercise_id,
                "remote": {
                    "call_id": call_id,
                    "gpu_type": args.gpu_type,
                    "gpu_count": args.gpu_count,
                    "arch": payload["remote"].get("arch"),
                    "image": args.image,
                },
                "status": result.get("status"),
                "course_runner": result.get("course_runner"),
                "report_json": result.get("report_json"),
            },
        )
    return code


def _submit_payload(
    client: GfaasClient,
    args: argparse.Namespace,
    payload: dict[str, Any],
    *,
    app_name: str,
    label: str,
    arch: str,
) -> tuple[dict[str, Any] | None, str, list[Path]]:
    print(
        f"Remote {label} on {args.gpu_type} (gpus={args.gpu_count}), "
        f"image={args.image}, arch={arch}"
    )
    with tempfile.TemporaryDirectory(prefix="gpu-func-workspace-") as temporary:
        workspace = Path(temporary, "workspace")
        job = _materialize_workspace(payload, workspace)
        remote = client.submit_job(
            job=job,
            workspace=workspace,
            args=args,
            app_name=app_name,
        )
    print(f"call: {remote.call_id}")
    if args.detach:
        print(f"resume: gfaas call watch {remote.call_id}", file=sys.stderr)
        return None, remote.call_id, []
    result = client.wait_for_result(
        remote,
        timeout_s=args.wait_timeout,
        json_events=args.json_events,
    )
    downloaded: list[Path] = []
    if args.artifact_dir:
        downloaded = client.download_profiles(remote.call_id, Path(args.artifact_dir))
    else:
        for publication in client.profile_publications(remote.call_id):
            artifact = publication.get("artifact")
            artifact_id = artifact.get("id") if isinstance(artifact, dict) else "unknown"
            print(
                f"[gpu-func] profile artifact={artifact_id}; use --artifact-dir to download it",
                file=sys.stderr,
            )
    return result, remote.call_id, downloaded
