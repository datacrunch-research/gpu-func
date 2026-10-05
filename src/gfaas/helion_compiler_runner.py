"""Parallel CPU compilation of the device launches emitted by Helion."""

from __future__ import annotations

import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any


def _compile_launch(request: dict[str, Any]) -> dict[str, Any]:
    from gfaas.triton_compiler_runner import compile_batch

    os.environ["GFAAS_OUTPUT_ROOT"] = request.pop("output_root")
    return compile_batch(**request)


def compile_batch(
    *,
    variants: list[dict[str, Any]],
    target: dict[str, Any],
    triton_version: str,
    workers: int,
    compression_level: int,
) -> dict[str, Any]:
    if not variants or not 1 <= workers <= 32:
        raise ValueError("Invalid Helion compiler batch or worker count")
    root = Path(os.environ["GFAAS_OUTPUT_ROOT"]) / "compiled-helion"
    root.mkdir(parents=True, exist_ok=True)
    requests = []
    positions = []
    for variant in variants:
        if variant["status"] != "prepared":
            raise ValueError("Compiler requires prepared Helion configurations")
        folder = root / variant["id"]
        folder.mkdir()
        (folder / "variant.json").write_text(json.dumps(variant, sort_keys=True))
        for index, launch in enumerate(variant["launches"]):
            requests.append(
                {
                    "output_root": str(folder / str(index)),
                    "source": variant["source"],
                    "kernel_name": launch["kernel_name"],
                    "variants": [{k: v for k, v in launch.items() if k != "kernel_name"}],
                    "target": target,
                    "triton_version": triton_version,
                    "workers": 1,
                    "compression_level": compression_level,
                }
            )
            positions.append(variant["id"])
    reports: dict[str, list[dict[str, Any]]] = {v["id"]: [] for v in variants}
    # Separate processes keep Triton cache globals independent. No GPU context
    # is initialized in these CPU jobs or their compiler children.
    with ProcessPoolExecutor(
        max_workers=min(workers, len(requests)), mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for identity, report in zip(positions, pool.map(_compile_launch, requests), strict=True):
            reports[identity].append(report)
    rows = []
    for variant in variants:
        units = reports[variant["id"]]
        rows.append(
            {
                "id": variant["id"],
                "configuration": variant["configuration"],
                "status": "compiled"
                if all(u["results"][0]["status"] == "compiled" for u in units)
                else "failed",
                "units": [u["results"][0] for u in units],
            }
        )
    result = {
        "schema": "vfunc.helion-compilation/v1",
        "results": rows,
        "target": target,
        "triton_version": triton_version,
    }
    (root / "manifest.json").write_text(json.dumps(result, sort_keys=True))
    return result
