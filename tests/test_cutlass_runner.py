import ctypes
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gfaas.cutlass_runner import Argument, candidate, restore_libraries


def test_selected_library_restore_and_tamper_detection(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    data = b"library"
    (artifact / "winner.so").write_bytes(data)
    (artifact / "other.so").write_bytes(b"unused")
    target = {"arch": 103}
    manifest = {
        "schema": "vfunc.cutlass-compilation/v1",
        "target": target,
        "source_sha256": hashlib.sha256(b"source").hexdigest(),
        "results": [
            {
                "id": "winner",
                "status": "compiled",
                "library": "winner.so",
                "sha256": hashlib.sha256(data).hexdigest(),
            },
            {"id": "other", "status": "compiled", "library": "other.so", "sha256": "bad"},
        ],
    }
    (artifact / "manifest.json").write_text(json.dumps(manifest))
    destination = tmp_path / "cache"
    destination.mkdir()
    result = restore_libraries([artifact], destination, "source", target, {"winner"})
    assert set(result) == {"winner"}
    assert list(destination.iterdir()) == [destination / "winner.so"]
    (artifact / "winner.so").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="checksum"):
        restore_libraries([artifact], destination, "source", target, {"winner"})


def test_native_abi_binding_preserves_tensor_metadata_scalars_and_stream(monkeypatch):
    import sys

    captured = []

    class Tensor:
        is_cuda = True
        device = SimpleNamespace(index=2)
        dtype = "float32"
        ndim = 2
        shape = (3, 4)

        def stride(self):
            return (8, 2)

        def data_ptr(self):
            return 1234

    class Launch:
        def __call__(self, arguments, size, stream):
            assert size == 5 and stream == 5678
            assert arguments[0].kind == 1 and arguments[0].data == 1234
            assert list(arguments[0].shape[:2]) == [3, 4]
            assert list(arguments[0].strides[:2]) == [8, 2]
            assert arguments[1].i64 == 7 and arguments[1].kind == 2
            assert arguments[2].f64 == 1.5 and arguments[2].kind == 3
            assert arguments[3].kind == 4 and arguments[3].i64 == 1
            assert arguments[4].kind == 0
            captured.append(True)
            return 0

    torch = SimpleNamespace(
        Tensor=Tensor,
        cuda=SimpleNamespace(
            current_device=lambda: 2, current_stream=lambda: SimpleNamespace(cuda_stream=5678)
        ),
    )
    for name in (
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
    ):
        setattr(torch, name, name)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(ctypes, "CDLL", lambda _: SimpleNamespace(vfunc_launch=Launch()))
    fn = candidate(Path("/kernel.so"), ["a", "n", "alpha", "flag", "optional"])
    fn(Tensor(), 7, alpha=1.5, flag=True, optional=None)
    assert captured
    with pytest.raises(TypeError):
        fn(Tensor(), 7, alpha=1.5, flag=True)
    assert ctypes.sizeof(Argument) == 56


def test_wheel_nvcc_and_versioned_cudart_are_passed_to_linker(tmp_path, monkeypatch):
    import importlib.metadata

    from gfaas.cutlass_runner import cuda_toolchain

    root = tmp_path / "nvidia" / "cu13"
    (root / "bin").mkdir(parents=True)
    (root / "lib").mkdir()
    nvcc = root / "bin" / "nvcc"
    runtime = root / "lib" / "libcudart.so.13"
    nvcc.write_text("compiler")
    runtime.write_bytes(b"runtime")

    def missing(_):
        raise RuntimeError("not on PATH")

    monkeypatch.setattr("gfaas.cuda_runner._which", missing)
    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        lambda _: SimpleNamespace(files=["nvidia/cu13/bin/nvcc"], locate_file=lambda _: nvcc),
    )
    compiler, flags = cuda_toolchain()
    assert compiler == str(nvcc)
    assert flags == [
        "--cudart=none",
        "-Xlinker",
        str(runtime),
        "-Xlinker",
        "-rpath",
        "-Xlinker",
        str(runtime.parent),
    ]
