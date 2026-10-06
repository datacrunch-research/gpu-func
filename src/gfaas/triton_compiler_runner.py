"""CPU-only batch compiler. Self-contained source bundle; no GPU initialization."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def compile_batch(
    *,
    source: str,
    kernel_name: str,
    variants: list[dict[str, Any]] | str,
    triton_version: str,
    target: dict[str, Any],
    workers: int,
    compression_level: int = 9,
) -> dict[str, Any]:
    """Compile every variant, recording individual failures and publishing one tree."""
    started = time.perf_counter()
    started_unix = time.time()
    cpu_started = os.times()
    import triton  # type: ignore[import-not-found]
    from triton.backends.compiler import GPUTarget  # type: ignore[import-not-found]

    from gfaas.triton_compat import ast_source_type

    imports_finished = time.perf_counter()
    if isinstance(variants, str):
        variants = json.loads(variants)
    decoded = time.perf_counter()
    if triton.__version__ != triton_version:
        raise RuntimeError(
            f"Compiler environment has Triton {triton.__version__}; requested {triton_version}"
        )
    if not 1 <= workers <= 32 or not variants or not 0 <= compression_level <= 9:
        raise ValueError("Invalid compilation workers or empty variant batch")
    root = Path(os.environ["GFAAS_OUTPUT_ROOT"]) / "compiled-triton"
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache)
    # Recent releases cache settings in knobs; older releases read the environment.
    if hasattr(triton, "knobs") and hasattr(triton.knobs, "cache"):
        triton.knobs.cache.dir = str(cache)
    gpu_target = GPUTarget(**target)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "compile_source.py"
        path.write_text(source)
        name = "vfunc_compile_" + hashlib.sha256(source.encode()).hexdigest()[:16]
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot import kernel source")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        kernel = getattr(module, kernel_name)
        ASTSource = ast_source_type(kernel)
        api = inspect.signature(ASTSource).parameters
        if "constexprs" in api:
            modern = True
        elif "constants" in api:
            modern = False
        else:
            raise RuntimeError("Unsupported ASTSource interface: missing constants parameter")
        loaded = time.perf_counter()

        def compile_one(variant: dict[str, Any]) -> dict[str, Any]:
            begin = time.perf_counter()
            record: dict[str, Any] = {"id": variant["id"]}
            try:
                signature, constants = variant["signature"], variant["constants"]
                if modern:
                    ast_source = ASTSource(kernel, signature, constexprs=constants)
                else:
                    signature = {
                        kernel.arg_names.index(k): v
                        for k, v in signature.items()
                        if v != "constexpr"
                    }
                    constants = {kernel.arg_names.index(k): v for k, v in constants.items()}
                    ast_source = ASTSource(kernel, signature, constants=constants)
                binary = triton.compile(ast_source, target=gpu_target, options=variant["options"])
                record.update(status="compiled", cache_hash=binary.hash)
            except Exception as error:
                record.update(status="failed", diagnostics=f"{type(error).__name__}: {error}")
            record["wall_seconds"] = time.perf_counter() - begin
            return record

        with ThreadPoolExecutor(max_workers=min(workers, len(variants))) as pool:
            results = list(pool.map(compile_one, variants))
        compiled = time.perf_counter()
    files = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(cache.rglob("*"))
        if p.is_file()
    }
    cache_bytes = sum(p.stat().st_size for p in cache.rglob("*") if p.is_file())
    hashed = time.perf_counter()
    # Publish a cache as one archive, avoiding an Artifact/API round trip per file.
    archive = root / "cache.tar.gz"

    def normalized(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = info.gid = info.mtime = 0
        info.uname = info.gname = ""
        return info

    with tarfile.open(archive, "w:gz", compresslevel=compression_level) as tar:
        for path in sorted(cache.rglob("*")):
            if path.is_symlink():
                raise RuntimeError("Compiler cache contains a symbolic link")
            tar.add(
                path, arcname=path.relative_to(root).as_posix(), recursive=False, filter=normalized
            )
    archive_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archived = time.perf_counter()
    shutil.rmtree(cache)
    cleaned = time.perf_counter()
    cpu_finished = os.times()
    manifest = {
        "schema": "vfunc.triton-compilation/v1",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "triton_version": triton_version,
        "target": target,
        "variants": variants,
        "results": results,
        "files": files,
        "cache_archive": {"path": "cache.tar.gz", "sha256": archive_digest},
        "environment_digest": os.environ.get("GFAAS_BUILD_IMAGE_DIGEST"),
        "compiler_import_seconds": imports_finished - started,
        "source_load_seconds": loaded - decoded,
        "wall_seconds": time.perf_counter() - started,
        "started_unix": started_unix,
        "finished_unix": time.time(),
        "phase_seconds": {
            "imports": imports_finished - started,
            "metadata_decode": decoded - imports_finished,
            "source_setup": loaded - decoded,
            "compilation": compiled - loaded,
            "cache_hashing": hashed - compiled,
            "cache_archive": archived - hashed,
            "cache_cleanup": cleaned - archived,
        },
        "cpu_seconds": {
            field: getattr(cpu_finished, field) - getattr(cpu_started, field)
            for field in ("user", "system", "children_user", "children_system")
        },
        "cache_file_count": len(files),
        "cache_bytes": cache_bytes,
        "cache_archive_bytes": archive.stat().st_size,
        "cache_compression_level": compression_level,
        "specialization": "conservative-types-no-runtime-value-or-alignment-specialization",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    return manifest
