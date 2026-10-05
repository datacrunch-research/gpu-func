"""CPU CuTe AOT compilation in isolated parallel processes."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any


def _compile_one(request: dict[str, Any]) -> dict[str, Any]:
    from gfaas.cute_backend import compile_variant

    start = time.perf_counter()
    try:
        row = compile_variant(**request)
    except Exception as error:
        row = {
            "id": request["variant"]["id"],
            "status": "failed",
            "diagnostics": f"{type(error).__name__}: {error}",
        }
    row["wall_seconds"] = time.perf_counter() - start
    return row


def compile_batch(
    *,
    source: str,
    kernel_name: str,
    variants: str,
    cute_version: str,
    metadata: dict[str, Any],
    argument_names: list[str],
    target: dict[str, Any],
    workers: int,
    compile_options: str = "",
) -> dict[str, Any]:
    from gfaas.cute_backend import version

    start = time.perf_counter()
    if version() != cute_version:
        raise RuntimeError("CuTe compiler library version differs from client")
    decoded = json.loads(variants)
    if not decoded or not 1 <= workers <= 32:
        raise ValueError("Empty configurations or invalid compiler worker count")
    root = Path(os.environ["GFAAS_OUTPUT_ROOT"]) / "compiled-cute"
    root.mkdir(parents=True, exist_ok=True)
    requests = [
        dict(
            source=source,
            kernel_name=kernel_name,
            variant=v,
            metadata=metadata,
            argument_names=argument_names,
            target=target,
            options=compile_options,
            folder=str(root / v["id"]),
        )
        for v in decoded
    ]
    if workers == 1:
        results = [_compile_one(r) for r in requests]
    else:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(decoded)), mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            results = list(pool.map(_compile_one, requests))
    # Store standalone objects, so loading one winner never decompresses other configurations.
    manifest = {
        "schema": "vfunc.cute-compilation/v1",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "cute_version": cute_version,
        "target": target,
        "variants": decoded,
        "results": results,
        "wall_seconds": time.perf_counter() - start,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    return manifest
