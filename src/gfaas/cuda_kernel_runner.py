"""CPU nvcc compilation and CUDA Driver API launches for managed kernels."""

from __future__ import annotations

import ctypes
import hashlib
import importlib.metadata
import inspect
import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

SCALARS = {
    "int32": ctypes.c_int32,
    "uint32": ctypes.c_uint32,
    "int64": ctypes.c_int64,
    "uint64": ctypes.c_uint64,
    "float32": ctypes.c_float,
    "float64": ctypes.c_double,
}


def compile_batch(
    *,
    source: str,
    kernel_name: str,
    variants: str,
    target: dict[str, Any],
    workers: int,
    nvcc_flags: list[str],
) -> dict[str, Any]:
    from gfaas.cuda_runner import _host_cxx_flags, _subprocess_env, _which

    decoded = json.loads(variants)
    root = Path(os.environ["GFAAS_OUTPUT_ROOT"]) / "compiled-cuda-kernel"
    root.mkdir(parents=True)
    path = root / "kernel.cu"
    path.write_text(source)
    try:
        nvcc = _which("nvcc")
    except RuntimeError:
        # NVIDIA wheels install the toolkit under site-packages, outside PATH.
        nvcc = None
        for distribution in importlib.metadata.distributions():
            if "cuda-nvcc" not in distribution.metadata["Name"].lower():
                continue
            for entry in distribution.files or ():
                if str(entry).endswith("/bin/nvcc"):
                    candidate = Path(str(distribution.locate_file(entry)))
                    if candidate.is_file() and os.access(candidate, os.X_OK):
                        nvcc = str(candidate)
                        break
            if nvcc:
                break
        if not nvcc:
            raise RuntimeError(
                "nvcc was not found in PATH, /usr/local/cuda, or NVIDIA toolkit wheels"
            ) from None
    nvcc = str(nvcc)
    env = _subprocess_env(str(root))

    def compile_one(variant: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        output = root / (variant["id"] + ".cubin")
        flags = [f"-D{k}={v}" for k, v in variant["defines"].items()]
        command = [
            nvcc,
            *_host_cxx_flags(),
            "-O3",
            "--cubin",
            f"-arch=sm_{target['arch']}",
            *nvcc_flags,
            *flags,
            str(path),
            "-o",
            str(output),
        ]
        result = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
        row: dict[str, Any] = {
            "id": variant["id"],
            "status": "compiled" if result.returncode == 0 else "failed",
            "wall_seconds": time.perf_counter() - started,
        }
        if result.returncode:
            row["diagnostics"] = result.stderr
        else:
            row["sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
        return row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(compile_one, decoded))
    manifest = {
        "schema": "vfunc.cuda-kernel/v1",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "kernel_name": kernel_name,
        "target": target,
        "results": results,
        "nvcc_flags": nvcc_flags,
        "compiler": subprocess.check_output([nvcc, "--version"], text=True),
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    path.unlink()
    return manifest


def load_binaries(
    artifacts: list[Any], source: str, kernel_name: str, target: dict[str, Any], selected: set[str]
) -> dict[str, dict[str, Any]]:
    records = {}
    for artifact in artifacts:
        root = Path(os.fspath(artifact))
        manifest = json.loads((root / "manifest.json").read_text())
        if (
            manifest.get("schema") != "vfunc.cuda-kernel/v1"
            or manifest["source_sha256"] != hashlib.sha256(source.encode()).hexdigest()
            or manifest["kernel_name"] != kernel_name
            or manifest["target"] != target
        ):
            raise RuntimeError("CUDA kernel artifact does not match the requested source/target")
        for row in manifest["results"]:
            if row["id"] not in selected:
                continue
            if row["id"] in records:
                raise RuntimeError("Duplicate CUDA configuration")
            record = dict(row)
            if row["status"] == "compiled":
                data = (root / (row["id"] + ".cubin")).read_bytes()
                if hashlib.sha256(data).hexdigest() != row["sha256"]:
                    raise RuntimeError("CUDA binary checksum mismatch")
                record["binary"] = data
            records[row["id"]] = record
    if set(records) != selected:
        raise RuntimeError("Missing CUDA configuration artifact")
    return records


def check_cuda(status: int) -> None:
    if status:
        raise RuntimeError(f"CUDA Driver API failed with status {status}")


def scalar_value(kind: str, value: Any) -> Any:
    if kind not in SCALARS:
        raise ValueError(f"Unsupported CUDA scalar type: {kind}")
    if kind.startswith(("int", "uint")):
        bits = 32 if kind.endswith("32") else 64
        unsigned = kind.startswith("uint")
        low, high = (0, 2**bits - 1) if unsigned else (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"Argument is outside {kind} range")
    elif type(value) not in (int, float):
        raise TypeError(f"Expected numeric {kind} argument")
    return SCALARS[kind](value)


def load_candidate(binary: bytes, kernel_name: str, variant: dict[str, Any], grid: Any) -> Any:
    import torch  # type: ignore[import-not-found]

    cuda = ctypes.CDLL("libcuda.so.1")
    context = ctypes.c_void_p()
    check_cuda(cuda.cuCtxGetCurrent(ctypes.byref(context)))
    if not context.value:
        # PyTorch lazy initialization alone need not activate a primary context.
        # Allocating through PyTorch establishes the context used by its streams.
        torch.empty(1, device="cuda")
    module, function = ctypes.c_void_p(), ctypes.c_void_p()
    image = ctypes.create_string_buffer(binary)
    check_cuda(cuda.cuModuleLoadData(ctypes.byref(module), image))
    check_cuda(cuda.cuModuleGetFunction(ctypes.byref(function), module, kernel_name.encode()))
    cuda.cuLaunchKernel.argtypes = [
        ctypes.c_void_p,
        *([ctypes.c_uint] * 7),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    if variant["shared_memory"]:
        # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES opts in above the
        # default shared-memory allowance; the driver validates the GPU limit.
        check_cuda(cuda.cuFuncSetAttribute(function, 8, variant["shared_memory"]))
    signature = inspect.Signature(
        [
            inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for n in variant["signature"]
        ]
    )

    def candidate(*args: Any, **kwargs: Any) -> None:
        bound = signature.bind(*args, **kwargs)
        values = []
        for name, kind in variant["signature"].items():
            value = bound.arguments[name]
            if kind == "pointer":
                if value is not None and (not hasattr(value, "data_ptr") or not value.is_cuda):
                    raise TypeError("CUDA pointers must be GPU tensors or None")
                values.append(ctypes.c_void_p(value.data_ptr() if value is not None else 0))
            else:
                values.append(scalar_value(kind, value))
        params = (ctypes.c_void_p * len(values))(*(ctypes.addressof(v) for v in values))
        meta = {**bound.arguments, **variant["defines"], "block": tuple(variant["block"])}
        dimensions = grid(meta) if callable(grid) else grid
        from gfaas.cuda_kernel import dimensions3

        launch = dimensions3(dimensions)
        block = tuple(variant["block"])
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        check_cuda(
            cuda.cuLaunchKernel(
                function, *launch, *block, variant["shared_memory"], stream, params, None
            )
        )

    # Keep module and its image alive for the lifetime of the candidate/graphs.
    candidate._cuda_module = (cuda, module, image)  # type: ignore[attr-defined]
    return candidate


def prepare_entries(
    records: dict[str, dict[str, Any]],
    variants: list[dict[str, Any]],
    kernel_name: str,
    grid: Any,
    ring: Any,
    reset: Any,
) -> list[Any]:
    import torch

    entries = []
    for variant in variants:
        row = {"id": variant["id"]}
        if records[row["id"]]["status"] != "compiled":
            row["status"] = "compile_failed"
            entries.append((row, None))
            continue
        candidate = load_candidate(records[row["id"]]["binary"], kernel_name, variant, grid)
        args, kwargs = ring.next()
        reset(*args, **kwargs)
        candidate(*args, **kwargs)
        torch.cuda.synchronize()
        row["prepared_files_unchanged"] = True
        entries.append((row, candidate))
    return entries


def benchmark_replicas(*, device_count: int, **kwargs: Any) -> dict[str, Any]:
    import torch

    from gfaas.triton_quick_runner import benchmark_cycle

    records = load_binaries(
        kwargs["artifacts"],
        kwargs["source"],
        kwargs["kernel_name"],
        kwargs["target"],
        {v["id"] for v in json.loads(kwargs["variants"])},
    )
    if torch.cuda.device_count() != device_count:
        raise RuntimeError("CUDA benchmark did not receive requested GPUs")
    reports = []
    with tempfile.TemporaryDirectory() as folder:
        for device in range(device_count):
            with torch.cuda.device(device):
                reports.append(benchmark_cycle(**kwargs, prepared_cache=(folder, records)))
    return {"status": "replica_bundle", "replica_reports": reports}


def execute(
    *,
    source: str,
    kernel_name: str,
    variant: dict[str, Any],
    artifacts: list[Any],
    target: dict[str, Any],
    callbacks: bytes,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    import cloudpickle
    import torch

    from gfaas.triton_inputs import SnapshotInputs, snapshot_inputs
    from gfaas.triton_quick_runner import probe_target

    if probe_target(torch.cuda.current_device()) != target:
        raise RuntimeError("CUDA execution target changed")
    records = load_binaries(artifacts, source, kernel_name, target, {variant["id"]})
    grid, _ = cloudpickle.loads(callbacks)
    candidate = load_candidate(records[variant["id"]]["binary"], kernel_name, variant, grid)
    args, kwargs = SnapshotInputs(inputs, [], [], list(variant["signature"]))(inputs["metadata"])
    candidate(*args, **kwargs)
    torch.cuda.synchronize()
    return snapshot_inputs(args, kwargs)
