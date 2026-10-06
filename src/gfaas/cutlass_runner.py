"""CPU CUTLASS compilation and the native tensor-launch ABI."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

TENSOR_DTYPES = (
    "float32",
    "float16",
    "bfloat16",
    "float64",
    "int8",
    "uint8",
    "int16",
    "int32",
    "int64",
    "bool",
    "float8_e4m3fn",
    "float8_e5m2",
    "float8_e4m3fnuz",
    "float8_e5m2fnuz",
    "uint16",
    "uint32",
    "uint64",
    "complex64",
    "complex128",
    "complex32",
    "float8_e8m0fnu",
    "float4_e2m1fn_x2",
)


class _LaunchRejected(RuntimeError):
    """A launcher rejected this configuration or input specialization."""


ABI_HEADER = r"""
#include <cuda_runtime.h>
#include <stdint.h>
// Tensor strides are in elements; data points to the tensor's first element.
// kind: 0=None, 1=tensor, 2=int64, 3=float64, 4=bool.
// dtype: 1=float32, 2=float16, 3=bfloat16, 4=float64,
//        5=int8, 6=uint8, 7=int16, 8=int32, 9=int64, 10=bool.
struct VFuncArgument {
    int32_t kind;
    int32_t dtype;
    void* data;
    int64_t ndim;
    const int64_t* shape;
    const int64_t* strides;
    int64_t i64;
    double f64;
};
extern "C" int vfunc_launch(const VFuncArgument*, int64_t, cudaStream_t);
"""


class Argument(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_int32),
        ("dtype", ctypes.c_int32),
        ("data", ctypes.c_void_p),
        ("ndim", ctypes.c_int64),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("i64", ctypes.c_int64),
        ("f64", ctypes.c_double),
    ]


def cuda_toolchain() -> tuple[str, list[str]]:
    """Support conventional CUDA installs and NVIDIA's split Python wheels."""
    from gfaas.cuda_runner import _which

    try:
        nvcc = _which("nvcc")
    except RuntimeError:
        import importlib.metadata

        distribution = importlib.metadata.distribution("nvidia-cuda-nvcc")
        candidates = [
            Path(str(distribution.locate_file(f)))
            for f in distribution.files or []
            if Path(str(f)).name == "nvcc"
        ]
        nvcc = next((str(p) for p in candidates if p.is_file()), "")
        if not nvcc:
            raise RuntimeError("The image must contain an nvcc compiler") from None
    root = Path(nvcc).resolve().parent.parent
    libraries = [
        *root.glob("lib/libcudart.so*"),
        *root.glob("lib64/libcudart.so*"),
        *root.glob("targets/*/lib/libcudart.so*"),
    ]
    if not libraries:
        return nvcc, ["--cudart=shared"]
    runtime = sorted(libraries, key=lambda p: (len(str(p)), str(p)))[0]
    # nvcc treats a versioned .so positional argument as an unknown input.
    # Forward it to the linker explicitly, retaining its immutable image path.
    return nvcc, [
        "--cudart=none",
        "-Xlinker",
        str(runtime),
        "-Xlinker",
        "-rpath",
        "-Xlinker",
        str(runtime.parent),
    ]


def compile_batch(
    *,
    source: str,
    variants: str,
    target: dict[str, Any],
    architecture: str | None,
    nvcc_flags: list[str],
    include_dirs: list[str],
    headers: Any,
    workers: int,
) -> dict[str, Any]:
    from gfaas.cuda_runner import _host_cxx_flags, _subprocess_env

    output = Path(os.environ["GFAAS_OUTPUT_ROOT"]) / "compiled-cutlass"
    output.mkdir()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="vfunc-cutlass-") as folder:
        root = Path(folder)
        includes = list(include_dirs)
        if headers is not None:
            with tarfile.open(os.fspath(headers)) as archive:
                for member in archive.getmembers():
                    path = Path(member.name)
                    if (
                        path.is_absolute()
                        or ".." in path.parts
                        or not (member.isfile() or member.isdir())
                    ):
                        raise RuntimeError("Unsafe CUTLASS header archive")
                archive.extractall(root / "headers", filter="data")
            includes += [
                str(root / "headers" / "include"),
                str(root / "headers" / "tools" / "util" / "include"),
            ]
        unit = root / "kernel.cu"
        unit.write_text(ABI_HEADER + source)
        tool_env = _subprocess_env(folder)
        nvcc, link_flags = cuda_toolchain()
        compiler = subprocess.run(
            [nvcc, "--version"], capture_output=True, text=True, check=True, env=tool_env
        ).stdout

        def compile_one(variant: dict[str, Any]) -> dict[str, Any]:
            begin = time.perf_counter()
            library = output / (variant["id"] + ".so")
            command = [
                nvcc,
                *_host_cxx_flags(),
                "-std=c++17",
                "-O3",
                "--shared",
                "-Xcompiler=-fPIC",
                *link_flags,
                "-arch=" + (architecture or f"sm_{target['arch']}"),
                *nvcc_flags,
                *["-I" + path for path in includes],
                *[
                    f"-D{k}={int(v) if type(v) is bool else v}"
                    for k, v in variant["constants"].items()
                ],
                str(unit),
                "-o",
                str(library),
            ]
            result = subprocess.run(command, capture_output=True, text=True, env=tool_env)
            row = {
                "id": variant["id"],
                "status": "compiled" if result.returncode == 0 else "failed",
                "wall_seconds": time.perf_counter() - begin,
                "diagnostics": result.stderr,
                "stdout": result.stdout,
            }
            if result.returncode == 0:
                row.update(
                    sha256=hashlib.sha256(library.read_bytes()).hexdigest(), library=library.name
                )
            return row

        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(compile_one, json.loads(variants)))
    manifest = {
        "schema": "vfunc.cutlass-compilation/v1",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "target": target,
        "architecture": architecture,
        "compiler": compiler,
        "nvcc_flags": nvcc_flags,
        "variants": json.loads(variants),
        "results": results,
        "wall_seconds": time.perf_counter() - started,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    return manifest


def restore_libraries(
    artifacts: list[Any], destination: Path, source: str, target: dict[str, Any], selected: set[str]
) -> dict[str, dict[str, Any]]:
    records = {}
    for artifact in artifacts:
        root = Path(os.fspath(artifact))
        manifest = json.loads((root / "manifest.json").read_text())
        if (
            manifest.get("schema") != "vfunc.cutlass-compilation/v1"
            or manifest["target"] != target
            or manifest["source_sha256"] != hashlib.sha256(source.encode()).hexdigest()
        ):
            raise RuntimeError("CUTLASS artifact does not match the source or target")
        for row in manifest["results"]:
            if row["id"] not in selected:
                continue
            if row["id"] in records:
                raise RuntimeError("Duplicate compiled CUTLASS configuration")
            records[row["id"]] = row
            if row["status"] != "compiled":
                continue
            if row["library"] != row["id"] + ".so":
                raise RuntimeError("Invalid CUTLASS library path")
            data = (root / row["library"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise RuntimeError("CUTLASS library checksum mismatch")
            (destination / row["library"]).write_bytes(data)
    if set(records) != selected:
        raise RuntimeError("Missing CUTLASS configuration")
    return records


def candidate(library: Path, argument_names: list[str]) -> Any:
    import torch  # type: ignore[import-not-found]

    module = ctypes.CDLL(str(library))
    launch = module.vfunc_launch
    launch.argtypes = [ctypes.POINTER(Argument), ctypes.c_int64, ctypes.c_void_p]
    launch.restype = ctypes.c_int
    dtypes = {
        getattr(torch, name): code
        for code, name in enumerate(TENSOR_DTYPES, 1)
        if hasattr(torch, name)
    }

    def invoke(*args: Any, **kwargs: Any) -> None:
        if (
            len(args) > len(argument_names)
            or set(kwargs) - set(argument_names)
            or set(argument_names[: len(args)]) & kwargs.keys()
        ):
            raise TypeError("Invalid CUTLASS launch arguments")
        bound = dict(zip(argument_names, args, strict=False))
        bound.update(kwargs)
        if set(bound) != set(argument_names):
            raise TypeError("Missing CUTLASS launch arguments")
        storage: list[Any] = []
        native = (Argument * len(argument_names))()
        for index, name in enumerate(argument_names):
            value = bound[name]
            if isinstance(value, torch.Tensor):
                if (
                    not value.is_cuda
                    or value.device.index != torch.cuda.current_device()
                    or value.dtype not in dtypes
                ):
                    raise TypeError(
                        "CUTLASS arguments require supported tensors on the current GPU"
                    )
                shape = (ctypes.c_int64 * value.ndim)(*value.shape)
                strides = (ctypes.c_int64 * value.ndim)(*value.stride())
                storage.extend((value, shape, strides))
                native[index] = Argument(
                    1, dtypes[value.dtype], value.data_ptr(), value.ndim, shape, strides, 0, 0
                )
            elif value is None:
                native[index].kind = 0
            elif type(value) in (int, bool):
                if not -(2**63) <= value < 2**63:
                    raise OverflowError("CUTLASS integer arguments must fit int64")
                native[index].kind = 4 if type(value) is bool else 2
                native[index].i64 = value
            elif type(value) is float:
                native[index].kind, native[index].f64 = 3, value
            else:
                raise TypeError("Unsupported CUTLASS launch argument")
        status = launch(native, len(native), torch.cuda.current_stream().cuda_stream)
        if status:
            raise _LaunchRejected(f"CUTLASS launcher returned status {status}")

    return invoke


def benchmark_cycle(
    *,
    source: str,
    variants: str,
    artifacts: list[Any],
    metadata: dict[str, Any],
    callbacks: bytes,
    target: dict[str, Any],
    policy: dict[str, Any],
    inputs: dict[str, Any],
    argument_names: list[str],
    reset_arguments: list[str],
    restore_arguments: list[str],
    backend: str = "cutlass",
    final_only: bool = False,
    excluded_gpu_uuids: list[str] | None = None,
    estimates: dict[str, float] | None = None,
    prepared_cache: tuple[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    from contextlib import nullcontext

    import cloudpickle
    import torch

    from gfaas.triton_inputs import SnapshotInputs
    from gfaas.triton_quick_runner import (
        InputRing,
        l2_flush_buffer,
        measure_candidates,
        probe_target,
    )

    if backend != "cutlass" or probe_target(torch.cuda.current_device()) != target:
        raise RuntimeError("CUTLASS benchmark target differs from compilation")
    gpu_uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
    if gpu_uuid in (excluded_gpu_uuids or []):
        return {"status": "duplicate_gpu", "gpu_uuid": gpu_uuid, "results": []}
    _, evaluate = cloudpickle.loads(callbacks)
    factory = SnapshotInputs(inputs, reset_arguments, restore_arguments, argument_names)
    flush = l2_flush_buffer(torch)
    ring = InputRing(
        factory,
        factory.reset,
        metadata,
        torch,
        flush.numel() // 2,
        policy["max_input_sets"],
        policy["max_ring_bytes"],
    )
    decoded = json.loads(variants)
    context = (
        nullcontext(prepared_cache[0])
        if prepared_cache
        else tempfile.TemporaryDirectory(prefix="vfunc-cutlass-", dir=os.environ.get("FC_IO_ROOT"))
    )
    with context as folder:
        records = (
            prepared_cache[1]
            if prepared_cache
            else restore_libraries(
                artifacts, Path(folder), source, target, {v["id"] for v in decoded}
            )
        )
        rows, entries = [], []
        for variant in decoded:
            row = {"id": variant["id"]}
            rows.append(row)
            if records[row["id"]]["status"] != "compiled":
                row["status"] = "compile_failed"
                continue
            launch = candidate(Path(folder) / records[row["id"]]["library"], argument_names)
            args, kwargs = ring.next()
            factory.reset(*args, **kwargs)
            try:
                launch(*args, **kwargs)
                torch.cuda.synchronize()
            except _LaunchRejected as error:
                # Confirm the context is healthy before skipping this variant.
                # CUDA faults still escape and terminate this process.
                torch.cuda.synchronize()
                row.update(status="benchmark_failed", diagnostics=str(error))
                continue
            entries.append((row, launch))
        return measure_candidates(
            entries, rows, ring, torch, flush, evaluate, policy, gpu_uuid, final_only, estimates
        )


def benchmark_replicas(*, device_count: int, **kwargs: Any) -> dict[str, Any]:
    import torch

    if torch.cuda.device_count() != device_count:
        raise RuntimeError("CUTLASS Call did not receive its requested GPU count")
    with tempfile.TemporaryDirectory(
        prefix="vfunc-cutlass-", dir=os.environ.get("FC_IO_ROOT")
    ) as folder:
        selected = {v["id"] for v in json.loads(kwargs["variants"])}
        records = restore_libraries(
            kwargs["artifacts"], Path(folder), kwargs["source"], kwargs["target"], selected
        )
        reports = []
        for device in range(device_count):
            with torch.cuda.device(device):
                reports.append(benchmark_cycle(**kwargs, prepared_cache=(folder, records)))
        return {
            "status": "replica_bundle",
            "replica_reports": reports,
            "prepared_configurations": len(records),
        }


def benchmark_selected_replicas(**kwargs: Any) -> dict[str, Any]:
    kwargs["final_only"] = True
    return benchmark_replicas(**kwargs)


def execute_winner(
    *,
    source: str,
    variant: dict[str, Any],
    artifacts: list[Any],
    target: dict[str, Any],
    inputs: dict[str, Any],
    argument_names: list[str],
) -> dict[str, Any]:
    import torch

    from gfaas.triton_inputs import SnapshotInputs, snapshot_inputs
    from gfaas.triton_quick_runner import probe_target

    if probe_target() != target:
        raise RuntimeError("CUTLASS execution target differs from compilation")
    with tempfile.TemporaryDirectory(
        prefix="vfunc-cutlass-", dir=os.environ.get("FC_IO_ROOT")
    ) as folder:
        row = restore_libraries(artifacts, Path(folder), source, target, {variant["id"]})[
            variant["id"]
        ]
        args, kwargs = SnapshotInputs(inputs, [], [], argument_names)(inputs["metadata"])
        candidate(Path(folder) / row["library"], argument_names)(*args, **kwargs)
        torch.cuda.synchronize()
        return snapshot_inputs(args, kwargs)
