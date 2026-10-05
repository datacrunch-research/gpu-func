"""CPU nvcc compilation and CUDA Driver API launches for managed kernels."""

from __future__ import annotations

import ctypes
import hashlib
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
    headers: Any,
    headers_sha256: str,
) -> dict[str, Any]:
    from gfaas.cuda_runner import _host_cxx_flags, _subprocess_env

    decoded = json.loads(variants)
    if not decoded or not 1 <= workers <= 32:
        raise ValueError("Invalid CUDA compiler batch or worker count")
    root = Path(os.environ["GFAAS_OUTPUT_ROOT"]) / "compiled-cuda-kernel"
    root.mkdir(parents=True)
    path = root / "kernel.cu"
    path.write_text(
        """#include <cuda_runtime.h>
#define VFUNC_LAUNCH_ARGS unsigned int grid_x, unsigned int grid_y, unsigned int grid_z, unsigned int block_x, unsigned int block_y, unsigned int block_z, unsigned int shared_bytes, cudaStream_t stream
#define VFUNC_GRID dim3(grid_x, grid_y, grid_z)
#define VFUNC_BLOCK dim3(block_x, block_y, block_z)
#define VFUNC_SHARED_BYTES shared_bytes
#define VFUNC_STREAM stream
"""
        + source
        + entrypoint_checks(kernel_name, decoded[0]["signature"], decoded[0]["entrypoint_kind"])
    )
    nvcc = find_nvcc()
    env = _subprocess_env(str(root))
    header_root = root / "headers"
    header_root.mkdir()
    import io
    import tarfile

    data = headers.read_bytes()
    if hashlib.sha256(data).hexdigest() != headers_sha256:
        raise RuntimeError("ThunderKittens header checksum mismatch")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar:
            name = Path(member.name)
            if name.is_absolute() or ".." in name.parts or not member.isfile():
                raise RuntimeError("Unsafe ThunderKittens header archive")
            handle = tar.extractfile(member)
            assert handle is not None
            output = header_root / name
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(handle.read())
    master = (header_root / "kittens.cuh").read_text()
    arch = target["arch"]
    macro = f"KITTENS_SM{arch}"
    if macro not in master:
        macro = (
            "KITTENS_BLACKWELL"
            if 100 <= arch < 120
            else "KITTENS_HOPPER"
            if arch == 90
            else "KITTENS_AMPERE"
        )
    nvcc_flags = [
        "-std=c++20",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-I",
        str(header_root),
        f"-D{macro}",
        *nvcc_flags,
    ]

    def compile_one(variant: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        extension = ".so" if variant["entrypoint_kind"] == "launcher" else ".cubin"
        output = root / (variant["id"] + extension)
        flags = [f"-D{k}={v}" for k, v in variant["defines"].items()]
        command = [
            nvcc,
            *_host_cxx_flags(),
            "-O3",
            *(
                ["--shared", "-Xcompiler=-fPIC", "--cudart=none", "--cudadevrt=none"]
                if extension == ".so"
                else ["--cubin"]
            ),
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
            row["extension"] = extension
        return row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(compile_one, decoded))
    manifest = {
        "schema": "vfunc.cuda-kernel/v1",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "headers_sha256": headers_sha256,
        "kernel_name": kernel_name,
        "target": target,
        "results": results,
        "nvcc_flags": nvcc_flags,
        "compiler": subprocess.check_output([nvcc, "--version"], text=True),
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    path.unlink()
    import shutil

    shutil.rmtree(header_root)
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
                data = (root / (row["id"] + row["extension"])).read_bytes()
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

    if variant["entrypoint_kind"] == "launcher":
        return load_host_launcher(binary, kernel_name, variant, grid)
    # CUDA 12+ cudaSetDevice initializes the selected primary context, including
    # for scalar-only kernels that did not allocate any input tensors.
    torch.cuda.set_device(torch.cuda.current_device())
    cuda = ctypes.CDLL("libcuda.so.1")
    module, function = ctypes.c_void_p(), ctypes.c_void_p()
    image = ctypes.create_string_buffer(binary)
    check_cuda(cuda.cuModuleLoadData(ctypes.byref(module), image))
    check_cuda(cuda.cuModuleGetFunction(ctypes.byref(function), module, kernel_name.encode()))
    if variant["shared_memory"] > 48 * 1024:
        check_cuda(cuda.cuFuncSetAttribute(function, 8, variant["shared_memory"]))
    cuda.cuLaunchKernel.argtypes = [
        ctypes.c_void_p,
        *([ctypes.c_uint] * 7),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
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
        from gfaas.thunderkittens_kernel import dimensions3

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
    # Tensor allocation establishes the current CUDA context before the Driver
    # API loads a raw cubin. Device discovery alone need not create that context.
    args, kwargs = SnapshotInputs(inputs, [], [], list(variant["signature"]))(inputs["metadata"])
    candidate = load_candidate(records[variant["id"]]["binary"], kernel_name, variant, grid)
    candidate(*args, **kwargs)
    torch.cuda.synchronize()
    return snapshot_inputs(args, kwargs)


def load_host_launcher(binary: bytes, kernel_name: str, variant: dict[str, Any], grid: Any) -> Any:
    """Load a CUDA host bridge; preserve native TK layout/descriptor construction."""
    import torch

    from gfaas.cuda_runner import _workdir_root
    from gfaas.thunderkittens_kernel import dimensions3

    # The toolkit's SONAME is resolved at runtime, never passed to nvcc as an input.
    ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
    major = torch.version.cuda.split(".")[0]
    runtime = ctypes.CDLL(f"libcudart.so.{major}", mode=ctypes.RTLD_GLOBAL)
    folder = tempfile.TemporaryDirectory(prefix="vfunc-tk-launch-", dir=_workdir_root())
    path = Path(folder.name) / "kernel.so"
    path.write_bytes(binary)
    library = ctypes.CDLL(str(path))
    function = getattr(library, kernel_name)
    function.restype = ctypes.c_int
    function.argtypes = (
        [ctypes.c_void_p if k == "pointer" else SCALARS[k] for k in variant["signature"].values()]
        + [ctypes.c_uint] * 7
        + [ctypes.c_void_p]
    )
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
        meta = {**bound.arguments, **variant["defines"], "block": tuple(variant["block"])}
        launch = dimensions3(grid(meta) if callable(grid) else grid)
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        check_cuda(function(*values, *launch, *variant["block"], variant["shared_memory"], stream))

    candidate._cuda_library = (runtime, library, folder)  # type: ignore[attr-defined]
    return candidate


def entrypoint_checks(name: str, signature: dict[str, str], kind: str) -> str:
    """Compile-time ABI checks prevent malformed launches from reaching GPUs."""
    count = len(signature) + (8 if kind == "launcher" else 0)
    lines = [
        "#include <tuple>",
        "#include <type_traits>",
        "template<class> struct __vfunc_fn;",
        "template<class R, class... A> struct __vfunc_fn<R(*)(A...)> { using result=R; using args=std::tuple<A...>; };",
        f"using __vfunc_entry = __vfunc_fn<decltype(&{name})>;",
        f'static_assert(std::tuple_size_v<__vfunc_entry::args> == {count}, "vFunc argument count mismatch");',
    ]
    result = (
        "(std::is_same_v<__vfunc_entry::result, int> || std::is_same_v<__vfunc_entry::result, cudaError_t>)"
        if kind == "launcher"
        else "std::is_same_v<__vfunc_entry::result, void>"
    )
    lines.append(f'static_assert({result}, "vFunc entry-point return type mismatch");')
    for index, arg in enumerate(signature.values()):
        typ = f"std::tuple_element_t<{index}, __vfunc_entry::args>"
        if arg == "pointer":
            check = f"std::is_pointer_v<{typ}> && !std::is_function_v<std::remove_pointer_t<{typ}>>"
        elif arg.startswith(("int", "uint")):
            size = 4 if arg.endswith("32") else 8
            signed = "is_unsigned_v" if arg.startswith("uint") else "is_signed_v"
            check = f"std::is_integral_v<{typ}> && std::{signed}<{typ}> && sizeof({typ}) == {size}"
        else:
            check = f"std::is_same_v<{typ}, {'float' if arg == 'float32' else 'double'}>"
        lines.append(f'static_assert({check}, "vFunc argument {index} ABI mismatch");')
    if kind == "launcher":
        for index in range(len(signature), count - 1):
            lines.append(
                f'static_assert(std::is_same_v<std::tuple_element_t<{index}, __vfunc_entry::args>, unsigned int>, "Use VFUNC_LAUNCH_ARGS for launcher dimensions");'
            )
        lines.append(
            f'static_assert(std::is_same_v<std::tuple_element_t<{count - 1}, __vfunc_entry::args>, cudaStream_t>, "Use VFUNC_LAUNCH_ARGS for launcher stream");'
        )
    return "\n" + "\n".join(lines) + "\n"


def find_nvcc() -> str:
    """Honor the image toolchain, including NVIDIA's pip-distributed compiler."""
    import importlib.metadata

    from gfaas.cuda_runner import _which

    override = os.environ.get("CUDACXX")
    if override:
        if not Path(override).is_file() or not os.access(override, os.X_OK):
            raise RuntimeError("CUDACXX does not identify an executable compiler")
        return override
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        candidate = Path(cuda_home) / "bin" / "nvcc"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    try:
        return _which("nvcc")
    except RuntimeError:
        try:
            distribution = importlib.metadata.distribution("nvidia-cuda-nvcc")
        except importlib.metadata.PackageNotFoundError:
            raise RuntimeError("The selected image must include the CUDA nvcc compiler") from None
        for entry in distribution.files or []:
            if entry.name == "nvcc" and entry.parent.name == "bin":
                candidate = Path(str(distribution.locate_file(entry)))
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    return str(candidate)
        raise RuntimeError("NVIDIA's compiler package contains no executable nvcc") from None
