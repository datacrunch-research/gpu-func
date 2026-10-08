from __future__ import annotations

import hashlib
import json
import os
import sys
import tarfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from gfaas import triton_compiler_runner as runner


@pytest.mark.parametrize("modern", [False, True])
@pytest.mark.parametrize("serialized", [False, True])
def test_old_and_new_astsource_interfaces_account_for_success_and_failure(
    modern, serialized, monkeypatch, tmp_path
):
    received = []
    if modern:

        class ASTSource:
            def __init__(self, fn, signature, constexprs=None, attrs=None):
                received.append((signature, constexprs))
                self.constants = constexprs
    else:

        class ASTSource:
            def __init__(self, fn, signature, constants=None, attrs=None):
                received.append((signature, constants))
                self.constants = constants

    class Target:
        def __init__(self, **kwargs):
            assert kwargs == {"backend": "cuda", "arch": 103, "warp_size": 32}

    def compile(source, **kwargs):
        values = source.constants.values()
        if 128 in values:
            raise RuntimeError("bad configuration")
        cache_file = Path(os.environ["TRITON_CACHE_DIR"]) / "kernel.cubin"
        cache_file.write_bytes(b"compiled binary")
        return SimpleNamespace(hash="compiledhash")

    modules = {
        name: ModuleType(name) for name in ("triton", "triton.compiler", "triton.backends.compiler")
    }
    modules["triton"].__version__ = "3.8.0"
    modules["triton"].compile = compile
    modules["triton.compiler"].ASTSource = ASTSource
    modules["triton.backends.compiler"].GPUTarget = Target
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("GFAAS_OUTPUT_ROOT", str(tmp_path))
    variants = [
        {
            "id": str(block),
            "signature": {"X": "*fp32", "BLOCK": "constexpr"},
            "constants": {"BLOCK": block},
            "options": {},
        }
        for block in [64, 128]
    ]
    report = runner.compile_batch(
        source="from types import SimpleNamespace\nkernel=SimpleNamespace(arg_names=['X','BLOCK'])",
        kernel_name="kernel",
        variants=json.dumps(variants) if serialized else variants,
        triton_version="3.8.0",
        target={"backend": "cuda", "arch": 103, "warp_size": 32},
        workers=2,
    )
    assert [r["status"] for r in report["results"]] == ["compiled", "failed"]
    expected = {"X": "*fp32", "BLOCK": "constexpr"} if modern else {0: "*fp32"}
    assert all(signature == expected for signature, _ in received)
    manifest = json.loads((tmp_path / "compiled-triton/manifest.json").read_text())
    assert manifest["results"] == report["results"]
    root = tmp_path / "compiled-triton"
    assert not (root / "cache").exists()
    archive = root / report["cache_archive"]["path"]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == report["cache_archive"]["sha256"]
    with tarfile.open(archive) as packed:
        binary = packed.extractfile("cache/kernel.cubin").read()
    assert binary == b"compiled binary"
    assert hashlib.sha256(binary).hexdigest() == report["files"]["cache/kernel.cubin"]
    assert report["cache_file_count"] == 1
    assert report["cache_bytes"] == len(binary)
    assert report["cache_archive_bytes"] == archive.stat().st_size
    assert all(value >= 0 for value in report["phase_seconds"].values())
    assert sum(report["phase_seconds"].values()) <= report["wall_seconds"]
