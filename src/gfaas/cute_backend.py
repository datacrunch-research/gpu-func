"""CuTe metadata compilation and verified AOT loading, using public library APIs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any


def version() -> str:
    return importlib.metadata.version("nvidia-cutlass-dsl")


def load_source(source: str, root: Path, name: str) -> Any:
    path = root / "cute_source.py"
    path.write_text(source)
    module_name = "vfunc_cute_" + hashlib.sha256(source.encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load CuTe source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, name)


def fake_arguments(metadata: dict[str, Any], names: list[str]) -> dict[str, Any]:
    import cutlass  # type: ignore[import-not-found]
    from cutlass.cute.runtime import make_fake_tensor  # type: ignore[import-not-found]

    types = {
        "float16": "Float16",
        "bfloat16": "BFloat16",
        "float32": "Float32",
        "float64": "Float64",
        "int8": "Int8",
        "int16": "Int16",
        "int32": "Int32",
        "int64": "Int64",
        "uint8": "Uint8",
        "bool": "Boolean",
    }

    def fake(spec: dict[str, Any]) -> Any:
        if spec["kind"] == "value":
            return spec["value"]
        if spec["dtype"] not in types:
            raise ValueError(f"Unsupported CuTe tensor dtype: {spec['dtype']}")
        return make_fake_tensor(
            getattr(cutlass, types[spec["dtype"]]),
            tuple(spec["shape"]),
            stride=tuple(spec["stride"]),
            assumed_align=1,
        )

    return {
        **{n: fake(s) for n, s in zip(names, metadata["args"], strict=False)},
        **{n: fake(s) for n, s in metadata["kwargs"].items()},
    }


def compile_variant(
    source: str,
    kernel_name: str,
    variant: dict[str, Any],
    metadata: dict[str, Any],
    argument_names: list[str],
    target: dict[str, Any],
    options: str,
    folder: str,
) -> dict[str, Any]:
    # AOT export needs CuTe runtime symbols even on a CPU-only compiler worker.
    import ctypes

    import cuda.bindings.driver as cuda  # type: ignore[import-not-found]
    import cutlass.cute as cute  # type: ignore[import-not-found]

    libraries = cute.runtime.find_runtime_libraries
    flags = (
        {"enable_tvm_ffi": False}
        if "enable_tvm_ffi" in inspect.signature(libraries).parameters
        else {}
    )
    runtime_handles = [ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL) for path in libraries(**flags)]
    if not runtime_handles:
        raise RuntimeError("CuTe AOT runtime libraries are unavailable")
    root = Path(folder)
    root.mkdir(parents=True, exist_ok=True)
    function = load_source(source, root, kernel_name)
    values = fake_arguments(metadata, argument_names)
    from gfaas.cute_compat import decode_constants

    values.update(decode_constants(variant["constants"]))
    for name, kind in variant["signature"].items():
        if kind == "stream":
            values[name] = cuda.CUstream(0)
    signature = inspect.signature(function)
    bound = signature.bind(**values)
    bound.apply_defaults()
    compiled = cute.compile(
        function,
        *bound.args,
        **bound.kwargs,
        options=f"--gpu-arch sm_{target['arch']}a {options}".strip(),
    )
    export = compiled.export_to_c
    parameters = inspect.signature(export).parameters
    if "file_name" in parameters:
        export(str(root), "kernel", function_prefix="vfunc_kernel")
    elif "function_name" in parameters:
        export(str(root / "kernel.o"), function_name="vfunc_kernel")
    else:
        raise RuntimeError("Unsupported CuTe export_to_c interface")
    object_file = root / "kernel.o"
    if not object_file.is_file():
        raise RuntimeError("CuTe exporter returned no object file")
    return {
        "id": variant["id"],
        "status": "compiled",
        "object": object_file.name,
        "sha256": hashlib.sha256(object_file.read_bytes()).hexdigest(),
    }


def load_artifacts(
    artifacts: list[Any],
    source: str,
    target: dict[str, Any],
    requested_version: str,
    variants: list[dict[str, Any]],
) -> dict[str, Path]:
    if version() != requested_version:
        raise RuntimeError("CuTe DSL runtime version differs from compilation")
    requested = {v["id"] for v in variants}
    objects: dict[str, Path] = {}
    for artifact in artifacts:
        root = Path(os.fspath(artifact))
        manifest = json.loads((root / "manifest.json").read_text())
        if (
            manifest.get("schema") != "vfunc.cute-compilation/v1"
            or manifest["source_sha256"] != hashlib.sha256(source.encode()).hexdigest()
            or manifest["target"] != target
            or manifest["cute_version"] != requested_version
        ):
            raise RuntimeError("CuTe artifact does not match source, target or library version")
        for row in manifest["results"]:
            if row["id"] not in requested or row["status"] != "compiled":
                continue
            if (
                row["id"] in objects
                or len(row["id"]) != 64
                or any(c not in "0123456789abcdef" for c in row["id"])
            ):
                raise RuntimeError("Invalid or duplicate CuTe object identity")
            path = root / row["id"] / "kernel.o"
            if (
                path.is_symlink()
                or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]
            ):
                raise RuntimeError("CuTe object checksum mismatch")
            objects[row["id"]] = path
    return objects


def candidate(object_file: Path, variant: dict[str, Any], names: list[str]) -> Any:
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    import torch  # type: ignore[import-not-found]
    from cutlass.cute.runtime import from_dlpack

    module = cute.runtime.load_module(str(object_file))
    function = module.vfunc_kernel

    def launch(*args: Any, **kwargs: Any) -> None:
        values = {**variant.get("defaults", {}), **dict(zip(names, args, strict=False)), **kwargs}
        converted = []
        for name, kind in variant["signature"].items():
            if kind == "constexpr":
                continue
            if kind == "stream":
                value = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            else:
                value = values[name]
                if hasattr(value, "untyped_storage"):
                    value = from_dlpack(value, assumed_align=1)
            converted.append(value)
        # The exported host ABI may return a runtime status even when the
        # Python host entry has no return value. Results travel through tensors.
        function(*converted)

    # Keep the loaded module alive, including its CUDA module and host execution engine.
    launch.module = module  # type: ignore[attr-defined]
    launch.supports_cuda_graph = "stream" in variant["signature"].values()  # type: ignore[attr-defined]
    return launch
