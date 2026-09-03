"""Command-line interface for file-based gfaas workloads and durable Calls."""

# PYTHON_ARGCOMPLETE_OK

from __future__ import annotations

import argparse
import json
import keyword
import os
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import argcomplete
from argcomplete.completers import FilesCompleter

from gfaas import cuda_runner, local_cuda, python_runner
from gfaas.artifacts import ArtifactOutput
from gfaas.client import Client, RemoteResult

from .errors import CliError
from .events import show_event

_SIZE = re.compile(r"^(?P<value>[0-9]+)(?P<unit>B|KiB|MiB|GiB|TiB)?$")
_SIZE_MULTIPLIERS = {
    None: 1,
    "B": 1,
    "KiB": 1024,
    "MiB": 1024**2,
    "GiB": 1024**3,
    "TiB": 1024**4,
}


def _size_bytes(value: str) -> int:
    match = _SIZE.fullmatch(value)
    if match is None:
        raise argparse.ArgumentTypeError("size must be bytes or use B, KiB, MiB, GiB, or TiB")
    amount = int(match.group("value"))
    if amount <= 0:
        raise argparse.ArgumentTypeError("size must be positive")
    return amount * _SIZE_MULTIPLIERS[match.group("unit")]


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _name_value(value: str) -> tuple[str, str]:
    name, separator, item_value = value.partition("=")
    if not separator or not name or not item_value:
        raise argparse.ArgumentTypeError("value must use NAME=VALUE")
    return name, item_value


def _add_output_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output",
        action="append",
        default=[],
        metavar="NAME=PATH",
        type=_name_value,
        help="publish one output file",
    )
    parser.add_argument(
        "--output-directory",
        action="append",
        default=[],
        metavar="NAME=PATH",
        type=_name_value,
        help="publish one output directory",
    )


def _add_json_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json",
        action="store_true",
        help="write machine-readable JSON or JSON Lines to standard output",
    )


def _add_local_toolchain_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--nvcc", help="path or command name for the CUDA compiler")
    parser.add_argument("--ncu", help="path or command name for Nsight Compute")
    parser.add_argument("--ccbin", help="path or command name for the host C++ compiler")
    parser.add_argument(
        "--device",
        help="physical GPU index or UUID; defaults to the first visible device",
    )
    parser.add_argument(
        "--env", action="append", default=[], type=_name_value, metavar="NAME=VALUE"
    )


def add_commands(commands: Any) -> None:
    """Add the general workload and resource commands to a root parser."""
    run = commands.add_parser("run", help="submit a Python or CUDA source file")
    target = run.add_argument("target", help="a .py file, file.py:callable, or .cu file")
    target.completer = FilesCompleter(("py", "cu"))
    run.add_argument("--runtime", choices=("python", "cuda"))
    run.add_argument("--image", help="registered image name")
    run.add_argument("--gpu-type", help="GPU pool name; defaults to GFAAS_GPU_TYPE or any")
    run.add_argument("--gpu-count", type=_positive_integer, default=1)
    run.add_argument("--timeout", type=_positive_integer, default=600, metavar="SECONDS")
    run.add_argument("--capacity-wait", type=_positive_integer, metavar="SECONDS")
    run.add_argument("--cpu-millicores", type=_positive_integer)
    run.add_argument("--memory", type=_size_bytes)
    run.add_argument("--storage", type=_size_bytes)
    run.add_argument("--shared-memory", type=_size_bytes)
    run.add_argument("--max-log", type=_size_bytes)
    run.add_argument("--max-output", type=_size_bytes)
    run.add_argument("--env", action="append", default=[], type=_name_value, metavar="NAME=VALUE")
    run.add_argument("--nvcc-flag", action="append", default=[])
    run.add_argument("--profile", action="store_true")
    run.add_argument("--ncu-arg", action="append", default=[])
    run.add_argument("--detach", action="store_true")
    _add_output_options(run)
    _add_json_option(run)

    local = commands.add_parser("local", help="use a CUDA GPU on this host")
    local_commands = local.add_subparsers(dest="local_command", required=True)

    local_info = local_commands.add_parser("info", help="show the local CUDA toolchain")
    _add_local_toolchain_options(local_info)
    _add_json_option(local_info)

    local_run = local_commands.add_parser("run", help="compile and run a local CUDA source file")
    local_target = local_run.add_argument("target", help="a self-contained .cu file")
    local_target.completer = FilesCompleter(("cu",))
    _add_local_toolchain_options(local_run)
    local_run.add_argument(
        "--arch",
        help="CUDA architecture; defaults to GFAAS_CUDA_ARCH or native detection",
    )
    local_run.add_argument("--timeout", type=_positive_integer, default=600, metavar="SECONDS")
    local_run.add_argument("--nvcc-flag", action="append", default=[])
    local_run.add_argument("--profile", action="store_true")
    local_run.add_argument("--ncu-arg", action="append", default=[])
    _add_output_options(local_run)
    _add_json_option(local_run)

    call = commands.add_parser("call", help="inspect and control durable Calls")
    call_commands = call.add_subparsers(dest="call_command", required=True)

    show = call_commands.add_parser("show", help="show the current Call state")
    show.add_argument("call_id")
    _add_json_option(show)

    watch = call_commands.add_parser("watch", help="follow retained and live Call events")
    watch.add_argument("call_id")
    watch.add_argument("--after", help="resume after an event cursor")
    _add_json_option(watch)

    logs = call_commands.add_parser("logs", help="read retained Call output")
    logs.add_argument("call_id")
    logs.add_argument("--follow", action="store_true")
    logs.add_argument("--after", help="resume after an event cursor when following")
    _add_json_option(logs)

    cancel = call_commands.add_parser("cancel", help="request Call cancellation")
    cancel.add_argument("call_id")
    cancel.add_argument("--reason")
    _add_json_option(cancel)

    artifacts = call_commands.add_parser("artifacts", help="list Artifacts from a Call")
    artifacts.add_argument("call_id")
    _add_json_option(artifacts)

    artifact = commands.add_parser("artifact", help="manage Artifacts")
    artifact_commands = artifact.add_subparsers(dest="artifact_command", required=True)
    download = artifact_commands.add_parser("download", help="download one Artifact")
    download.add_argument("artifact_id")
    download.add_argument("destination", nargs="?")
    _add_json_option(download)

    pool = commands.add_parser("pool", help="inspect configured GPU pools")
    pool_commands = pool.add_subparsers(dest="pool_command", required=True)
    pool_list = pool_commands.add_parser("list", help="list configured GPU pools")
    _add_json_option(pool_list)

    completion = commands.add_parser("completion", help="write shell completion setup")
    completion.add_argument(
        "shell",
        choices=("bash", "fish", "zsh", "powershell"),
        help="shell that will load the generated setup",
    )


def _split_program_args(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    values = list(argv)
    try:
        separator = values.index("--")
    except ValueError:
        return values, []
    return values[:separator], values[separator + 1 :]


def _target(value: str, runtime_override: str | None) -> tuple[Path, str, str | None]:
    callable_name: str | None = None
    path_value = value
    prefix, separator, suffix = value.rpartition(":")
    if separator and (prefix.endswith(".py") or runtime_override == "python"):
        path_value = prefix
        callable_name = suffix
        if not callable_name.isidentifier() or keyword.iskeyword(callable_name):
            raise CliError("Python callable must use file.py:name")

    path = Path(path_value)
    if not path.is_file():
        raise CliError(f"source file does not exist: {path}")
    inferred = {".py": "python", ".cu": "cuda"}.get(path.suffix.lower())
    runtime = runtime_override or inferred
    if runtime is None:
        raise CliError("cannot infer runtime; use --runtime python or --runtime cuda")
    if runtime == "cuda" and callable_name is not None:
        raise CliError("CUDA files do not support callable entrypoints")
    if callable_name is not None and (not path.stem.isidentifier() or keyword.iskeyword(path.stem)):
        raise CliError("Python callable source filename must be an importable module name")
    return path, runtime, callable_name


def _outputs(args: argparse.Namespace) -> tuple[ArtifactOutput, ...]:
    files = tuple(ArtifactOutput(name, path) for name, path in args.output)
    directories = tuple(
        ArtifactOutput.directory(name, path) for name, path in args.output_directory
    )
    return files + directories


def _submission_options(args: argparse.Namespace) -> dict[str, Any]:
    gpu_type = args.gpu_type or os.environ.get("GFAAS_GPU_TYPE") or "any"
    return {
        "gpu_count": args.gpu_count,
        "gpu_type": gpu_type,
        "timeout_s": args.timeout,
        "capacity_wait_s": args.capacity_wait,
        "cpu_millicores": args.cpu_millicores,
        "memory_bytes": args.memory,
        "ephemeral_storage_bytes": args.storage,
        "shared_memory_bytes": args.shared_memory,
        "max_log_bytes": args.max_log,
        "max_output_bytes": args.max_output,
        "env": dict(args.env),
        "outputs": _outputs(args),
    }


def _submit_run(
    client: Client,
    args: argparse.Namespace,
    program_args: list[str],
) -> tuple[RemoteResult, str]:
    path, runtime, callable_name = _target(args.target, args.runtime)
    options = _submission_options(args)
    if runtime == "cuda":
        image = args.image or "cuda-nvcc"
        remote = client.submit(
            image=image,
            function=cuda_runner.run,
            kwargs={
                "source": path.read_text(encoding="utf-8"),
                "profile": args.profile,
                "ncu_args": args.ncu_arg,
                "nvcc_flags": args.nvcc_flag,
                "program_args": program_args,
            },
            app_name="cuda-nvcc",
            **options,
        )
        return remote, runtime

    if args.profile or args.nvcc_flag or args.ncu_arg:
        raise CliError("CUDA compiler and profiler options require a CUDA source file")
    image = args.image or os.environ.get("GFAAS_IMAGE") or "pytorch-cu130"
    if callable_name is not None:
        remote = client.submit(
            image=image,
            function=(path.stem, callable_name),
            args=tuple(program_args),
            source_file=path,
            app_name=path.stem,
            **options,
        )
        return remote, runtime

    remote = client.submit(
        image=image,
        function=python_runner.run_script,
        kwargs={
            "source": path.read_text(encoding="utf-8"),
            "filename": path.name,
            "program_args": program_args,
        },
        app_name=path.stem,
        **options,
    )
    return remote, runtime


def _json_value(value: Any) -> Any:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return {"representation": repr(value)}
    return value


def _write_json(value: Any) -> None:
    print(json.dumps(value, separators=(",", ":"), sort_keys=True))


def _show_result(call_id: str, result: Any, *, runtime: str, json_output: bool) -> int:
    if json_output:
        _write_json({"type": "result", "call_id": call_id, "value": _json_value(result)})
    elif isinstance(result, dict) and isinstance(result.get("returncode"), int):
        stdout = result.get("stdout")
        stderr = result.get("stderr")
        if stdout:
            print(str(stdout), end="", file=sys.stdout)
        if stderr:
            print(str(stderr), end="", file=sys.stderr)
        ncu_csv = result.get("ncu_csv")
        if ncu_csv:
            print(str(ncu_csv), end="", file=sys.stdout)
        if runtime == "cuda":
            compile_ms = result.get("compile_ms")
            run_ms = result.get("run_ms")
            print(f"[vfunc] compile_ms={compile_ms} run_ms={run_ms}", file=sys.stderr)
    elif result is not None:
        if isinstance(result, (dict, list, str, int, float, bool)):
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(repr(result))

    if isinstance(result, dict):
        returncode = result.get("returncode")
        if isinstance(returncode, int) and not isinstance(returncode, bool) and returncode != 0:
            return min(returncode, 125) if returncode > 0 else 1
    return 0


def _local_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(dict(args.env))
    if args.device is not None:
        environment["CUDA_VISIBLE_DEVICES"] = args.device
    return environment


def _local_output_paths(
    args: argparse.Namespace,
    *,
    workdir: Path,
    require_outputs: bool,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for output in _outputs(args):
        path = workdir.joinpath(*Path(output.relative_path).parts)
        exists = path.is_dir() if output.layout == "tree" else path.is_file()
        if require_outputs and output.required and not exists:
            raise CliError(f"local CUDA program did not write required output: {path}")
        reports.append(
            {
                "name": output.name,
                "path": str(path),
                "layout": output.layout,
                "exists": exists,
            }
        )
    return reports


def _reserve_local_output_paths(args: argparse.Namespace, workdir: Path) -> None:
    for output in _outputs(args):
        path = workdir.joinpath(*Path(output.relative_path).parts)
        if path.exists() or path.is_symlink():
            raise CliError(f"local output path already exists: {path}")


def _show_local_info(toolchain: local_cuda.LocalCudaToolchain, *, json_output: bool) -> int:
    report = toolchain.report()
    if json_output:
        _write_json({"type": "local_cuda_info", **report})
        return 0
    print("LOCAL CUDA")
    print(f"  nvcc={toolchain.nvcc} version={toolchain.nvcc_version}")
    print(f"  cuda-root={toolchain.cuda_root}")
    print(f"  host-compiler={toolchain.host_compiler or 'not detected'}")
    print(f"  ncu={toolchain.ncu or 'not detected'}")
    print(f"  CUDA_VISIBLE_DEVICES={toolchain.cuda_visible_devices or 'unset'}")
    print()
    print(f"{'SELECTED':<9} {'INDEX':>5} {'ARCH':<8} {'MEMORY':>10}  NAME")
    for gpu in toolchain.gpus:
        memory = f"{gpu.memory_total_mib}MiB" if gpu.memory_total_mib is not None else "unknown"
        selected = "yes" if gpu == toolchain.selected_gpu else ""
        print(f"{selected:<9} {gpu.index:>5} {gpu.architecture:<8} {memory:>10}  {gpu.name}")
    return 0


def _local_command(args: argparse.Namespace, program_args: list[str]) -> int:
    environment = _local_environment(args)
    toolchain = local_cuda.discover(
        nvcc=args.nvcc,
        ncu=args.ncu,
        host_compiler=args.ccbin,
        device=args.device,
        environment=environment,
        require_profiler=getattr(args, "profile", False),
    )
    if args.local_command == "info":
        return _show_local_info(toolchain, json_output=args.json)
    if args.local_command != "run":
        raise AssertionError(f"unknown local command {args.local_command!r}")

    path, runtime, _callable_name = _target(args.target, "cuda")
    if path.suffix.lower() != ".cu":
        raise CliError("local CUDA source must use the .cu suffix")
    workdir = Path.cwd()
    _reserve_local_output_paths(args, workdir)
    nvcc_flags, architecture = local_cuda.architecture_flags(
        args.nvcc_flag,
        requested=args.arch,
        environment=environment,
        toolchain=toolchain,
    )
    result = local_cuda.run(
        path,
        toolchain=toolchain,
        nvcc_flags=nvcc_flags,
        program_args=program_args,
        environment=environment,
        workdir=workdir,
        timeout_seconds=args.timeout,
        profile=args.profile,
        ncu_args=args.ncu_arg,
    )
    result.update(
        {
            "execution": "local",
            "architecture": architecture,
            "device": asdict(toolchain.selected_gpu),
            "toolchain": {
                "nvcc": toolchain.nvcc,
                "nvcc_version": toolchain.nvcc_version,
                "ncu": toolchain.ncu,
                "host_compiler": toolchain.host_compiler,
            },
        }
    )
    result["outputs"] = _local_output_paths(
        args,
        workdir=workdir,
        require_outputs=result.get("returncode") == 0,
    )
    if not args.json:
        print(
            f"[vfunc] local device={toolchain.selected_gpu.name} "
            f"arch={architecture} nvcc={toolchain.nvcc_version}",
            file=sys.stderr,
        )
        for output in result["outputs"]:
            if output["exists"]:
                print(
                    f"[vfunc] local output name={output['name']} path={output['path']}",
                    file=sys.stderr,
                )
    return _show_result("local", result, runtime=runtime, json_output=args.json)


def _run_command(client: Client, args: argparse.Namespace, program_args: list[str]) -> int:
    remote, runtime = _submit_run(client, args, program_args)
    if args.json:
        _write_json({"type": "submitted", "call_id": remote.call_id})
    elif args.detach:
        print(remote.call_id)
    else:
        print(f"[vfunc] call={remote.call_id}", file=sys.stderr)
    if args.detach:
        return 0

    try:
        for event in remote.iter_events(follow=True):
            show_event(event, json_output=args.json)
        result = remote.wait()
    except KeyboardInterrupt:
        cancellation = remote.cancel(reason="local CLI interrupted")
        state = cancellation.get("state", "cancelling")
        if args.json:
            _write_json(
                {"type": "cancellation_requested", "call_id": remote.call_id, "state": state}
            )
        else:
            print(
                f"\n[vfunc] cancellation requested call={remote.call_id} state={state}",
                file=sys.stderr,
            )
        return 130
    return _show_result(remote.call_id, result, runtime=runtime, json_output=args.json)


def _call_command(client: Client, args: argparse.Namespace) -> int:
    if args.call_command == "show":
        value = client.get_call(args.call_id)
        _write_json(value) if args.json else print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    if args.call_command == "watch":
        for event in client.iter_events(args.call_id, after=args.after, follow=True):
            show_event(event, json_output=args.json)
        return 0
    if args.call_command == "logs":
        if args.follow:
            for stream_name, data in RemoteResult(
                client=client,
                job_id=args.call_id,
                gpu_type="any",
            ).iter_logs(after=args.after, follow=True):
                if args.json:
                    _write_json({"type": stream_name, "stream_data": data})
                else:
                    stream = sys.stdout if stream_name == "stdout" else sys.stderr
                    print(data, end="", file=stream, flush=True)
            return 0
        logs = client.get_call_logs(args.call_id)
        if args.json:
            _write_json(logs)
        else:
            print(logs.get("stdout", ""), end="", file=sys.stdout)
            print(logs.get("stderr", ""), end="", file=sys.stderr)
            if logs.get("truncated"):
                print("[vfunc] retained logs are truncated", file=sys.stderr)
        return 0
    if args.call_command == "cancel":
        value = client.cancel_call(args.call_id, reason=args.reason)
        _write_json(value) if args.json else print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    if args.call_command == "artifacts":
        value = client.list_call_artifacts(args.call_id)
        _write_json(value) if args.json else print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    raise AssertionError(f"unknown Call command {args.call_command!r}")


def _artifact_command(client: Client, args: argparse.Namespace) -> int:
    if args.artifact_command != "download":
        raise AssertionError(f"unknown Artifact command {args.artifact_command!r}")
    metadata = client.get_artifact(args.artifact_id)
    suggested = metadata.get("filename")
    default_name = Path(str(suggested)).name if suggested else args.artifact_id
    destination = Path(args.destination or default_name)
    if destination.exists() or destination.is_symlink():
        raise CliError(f"Artifact destination already exists: {destination}")
    if metadata.get("layout") == "tree":
        path = client.download_artifact_directory(args.artifact_id, destination)
        media_type = metadata.get("media_type")
    else:
        path, media_type = client.download_artifact_file(args.artifact_id, destination)
    value = {
        "artifact_id": args.artifact_id,
        "path": str(path),
        "media_type": media_type,
    }
    _write_json(value) if args.json else print(path)
    return 0


def _pool_command(client: Client, args: argparse.Namespace) -> int:
    if args.pool_command != "list":
        raise AssertionError(f"unknown pool command {args.pool_command!r}")
    capabilities = client.get_capabilities()
    if args.json:
        _write_json(capabilities)
        return 0
    pools = capabilities.get("gpu_pools", [])
    print(f"{'NAME':<20} {'STATUS':<14} {'CONNECTED':>9} {'AVAILABLE':>9}")
    for pool in pools:
        print(
            f"{str(pool.get('name', '')):<20} "
            f"{str(pool.get('status', 'unknown')):<14} "
            f"{str(pool.get('connected_workers', '')):>9} "
            f"{str(pool.get('available_workers', '')):>9}"
        )
    return 0


def _completion_command(args: argparse.Namespace) -> int:
    print(argcomplete.shellcode(["vfunc"], shell=args.shell), end="")
    return 0


GENERAL_COMMANDS = frozenset({"run", "local", "call", "artifact", "pool", "completion"})


def accepts_program_args(args: argparse.Namespace) -> bool:
    return args.command_name == "run" or (
        args.command_name == "local" and args.local_command == "run"
    )


def dispatch(
    args: argparse.Namespace,
    program_args: list[str],
    *,
    client_factory: Callable[[], Client],
) -> int:
    """Run one parsed general command."""
    if args.command_name == "completion":
        return _completion_command(args)
    if args.command_name == "local":
        return _local_command(args, program_args)
    with client_factory() as client:
        if args.command_name == "run":
            return _run_command(client, args, program_args)
        if args.command_name == "call":
            return _call_command(client, args)
        if args.command_name == "artifact":
            return _artifact_command(client, args)
        if args.command_name == "pool":
            return _pool_command(client, args)
    raise AssertionError(f"unknown command {args.command_name!r}")
