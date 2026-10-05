"""Execute Helion's generated host functions using CPU-precompiled GPU binaries."""

from __future__ import annotations

import inspect
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .helion_compat import installed_version


def restore_variants(
    variants: list[dict[str, Any]],
    artifacts: list[Any],
    folder: str,
    target: dict[str, Any],
    triton_version: str,
) -> list[dict[str, Any]]:
    from torch._inductor.codecache import PyCodeCache  # type: ignore[import-not-found]

    from gfaas.triton_quick_runner import restore_caches

    owners = {}
    for artifact in artifacts:
        root = Path(os.fspath(artifact))
        manifest = json.loads((root / "manifest.json").read_text())
        for row in manifest["results"]:
            if row["status"] == "compiled":
                if row["id"] in owners:
                    raise RuntimeError("Duplicate Helion compiler artifact")
                owners[row["id"]] = root
    prepared = []
    for variant in variants:
        identity = variant["id"]
        if identity not in owners:
            raise RuntimeError("Missing Helion compiler artifact")
        root = owners[identity] / identity
        record = json.loads((root / "variant.json").read_text())
        if record["id"] != identity or record["configuration"] != variant["configuration"]:
            raise RuntimeError("Helion configuration differs from compiled artifact")
        module = PyCodeCache.load(record["source"])
        units = []
        for index, unit in enumerate(record["launches"]):
            cache = Path(folder) / identity / str(index)
            cache.mkdir(parents=True)
            records = restore_caches(
                [root / str(index) / "compiled-triton"],
                cache,
                record["source"],
                target,
                triton_version,
            )
            units.append({"cache": str(cache), "launch": unit, "compiled": records[unit["id"]]})
        prepared.append({"record": record, "module": module, "units": units})
    return prepared


def load_candidate(prepared: dict[str, Any], target: dict[str, Any]) -> Any:
    import triton  # type: ignore[import-not-found]
    from triton.backends.compiler import GPUTarget  # type: ignore[import-not-found]
    from triton.compiler import ASTSource  # type: ignore[import-not-found]

    from gfaas.triton_compat import argument_type
    from gfaas.triton_quick_runner import bound_candidate

    module = prepared["module"]
    launches = []
    for unit in prepared["units"]:
        cache = Path(unit["cache"])
        os.environ["TRITON_CACHE_DIR"] = str(cache)
        if hasattr(triton, "knobs") and hasattr(triton.knobs, "cache"):
            triton.knobs.cache.dir = str(cache)
        spec = unit["launch"]
        jit = getattr(module, spec["kernel_name"])
        if "constexprs" in inspect.signature(ASTSource).parameters:
            ast = ASTSource(jit, spec["signature"], constexprs=spec["constants"])
        else:
            ast = ASTSource(
                jit,
                {
                    jit.arg_names.index(k): v
                    for k, v in spec["signature"].items()
                    if v != "constexpr"
                },
                constants={jit.arg_names.index(k): v for k, v in spec["constants"].items()},
            )
        before = {
            str(p): (p.stat().st_mtime_ns, p.stat().st_size)
            for p in cache.rglob("*")
            if p.is_file()
        }
        binary = triton.compile(ast, target=GPUTarget(**target), options=spec["options"])
        after = {
            str(p): (p.stat().st_mtime_ns, p.stat().st_size)
            for p in cache.rglob("*")
            if p.is_file()
        }
        if binary.hash != unit["compiled"]["cache_hash"] or before != after:
            raise RuntimeError("Helion GPU preparation recompiled a CPU-prepared launch")
        launches.append((spec, jit, binary))

    def launch(kernel: Any, grid: Any, *args: Any, **kwargs: Any) -> None:
        signature = inspect.signature(kernel.fn)
        values = {k: v for k, v in kwargs.items() if k in signature.parameters}
        binding = signature.bind(*args, **values)
        binding.apply_defaults()
        parameters = {p.name: p for p in kernel.params}
        actual_signature = {
            name: argument_type(value, parameters[name])
            for name, value in binding.arguments.items()
        }
        options = {k: v for k, v in kwargs.items() if k not in signature.parameters}
        matches = [
            (spec, jit, binary)
            for spec, jit, binary in launches
            if spec["kernel_name"] == kernel.fn.__name__
            and spec["signature"] == actual_signature
            and spec["options"] == options
            and all(binding.arguments[k] == v for k, v in spec["constants"].items())
        ]
        if len(matches) != 1:
            raise RuntimeError("Helion emitted an unprepared GPU launch")
        spec, _, binary = matches[0]
        candidate = bound_candidate(
            binary, spec["constants"], spec["options"], signature, lambda _: grid
        )
        candidate(*args, **values)

    run = getattr(module, prepared["record"]["kernel_name"])

    def candidate(*args: Any, **kwargs: Any) -> Any:
        return run(*args, _launcher=launch, **kwargs)

    return candidate


def benchmark_cycle(
    *,
    variants: str,
    artifacts: list[Any],
    target: dict[str, Any],
    triton_version: str,
    helion_version: str,
    inputs: dict[str, Any],
    argument_names: list[str],
    callbacks: bytes,
    policy: dict[str, Any],
    final_only: bool = False,
    excluded_gpu_uuids: list[str] | None = None,
    estimates: dict[str, float] | None = None,
    prepared: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    import cloudpickle
    import torch  # type: ignore[import-not-found]
    import triton

    from gfaas.kernel_benchmark_runner import measure_entries
    from gfaas.triton_inputs import SnapshotInputs, _storage_tensor
    from gfaas.triton_quick_runner import InputRing, l2_flush_buffer, probe_target

    if (
        probe_target(torch.cuda.current_device()) != target
        or triton.__version__ != triton_version
        or installed_version() != helion_version
    ):
        raise RuntimeError("Helion benchmark environment differs from compilation")
    gpu_uuid = str(
        getattr(torch.cuda.get_device_properties(torch.cuda.current_device()), "uuid", "")
    )
    if not gpu_uuid:
        raise RuntimeError("GPU UUID is required for replication")
    if gpu_uuid in (excluded_gpu_uuids or []):
        return {"status": "duplicate_gpu", "gpu_uuid": gpu_uuid, "results": []}
    factory = SnapshotInputs(inputs, [], [], argument_names)
    flush = l2_flush_buffer(torch)
    ring = InputRing(
        factory,
        factory.reset,
        inputs["metadata"],
        torch,
        flush.numel() // 2,
        policy["max_input_sets"],
        policy["max_ring_bytes"],
    )
    evaluate = cloudpickle.loads(callbacks)
    with tempfile.TemporaryDirectory(prefix="vfunc-helion-") as folder:
        records = (
            prepared
            if prepared is not None
            else restore_variants(json.loads(variants), artifacts, folder, target, triton_version)
        )
        rows, entries = [], []
        mutated = set()
        for record in records:
            row = {"id": record["record"]["id"], "prepared_files_unchanged": True}
            candidate = load_candidate(record, target)
            args, kwargs = ring.fresh()
            candidate(*args, **kwargs)
            torch.cuda.synchronize()
            for name, value, spec in zip(
                argument_names, args, inputs["metadata"]["args"], strict=True
            ):
                if spec["kind"] == "tensor" and not torch.equal(
                    _storage_tensor(value, torch).cpu(), factory.host[spec["storage_group"]]
                ):
                    mutated.add(name)
            rows.append(row)
            entries.append((row, candidate))
        factory.restore_names = sorted(mutated)
        measure_entries(
            entries,
            rows,
            ring,
            torch,
            flush,
            policy,
            evaluate,
            final_only=final_only,
            estimates=estimates,
        )
        valid = [r for r in rows if r["status"] == "measured"]
        best = min(valid, key=lambda r: r["runtime_us"]) if valid else None
        return {
            "schema": "vfunc.helion-benchmark/v1",
            "status": "passed" if valid else "failed",
            "gpu_uuid": gpu_uuid,
            "results": rows,
            "best_id": best["id"] if best else None,
            "best_runtime_us": best["runtime_us"] if best else None,
            "ring_sets": len(ring.sets),
            "ring_allocated_bytes": ring.allocated_bytes,
        }


def benchmark_replicas(*, device_count: int, **kwargs: Any) -> dict[str, Any]:
    import torch

    if torch.cuda.device_count() != device_count:
        raise RuntimeError("Helion benchmark did not receive its requested GPU count")
    with tempfile.TemporaryDirectory(prefix="vfunc-helion-replicas-") as folder:
        prepared = restore_variants(
            json.loads(kwargs["variants"]),
            kwargs["artifacts"],
            folder,
            kwargs["target"],
            kwargs["triton_version"],
        )
        reports = []
        for index in range(device_count):
            with torch.cuda.device(index):
                reports.append(benchmark_cycle(**kwargs, prepared=prepared))
        return {"status": "replica_bundle", "replica_reports": reports}


def execute_winner(
    *,
    variant: dict[str, Any],
    artifacts: list[Any],
    target: dict[str, Any],
    triton_version: str,
    helion_version: str,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    import torch
    import triton

    from gfaas.kernel_results import snapshot_result
    from gfaas.triton_inputs import SnapshotInputs
    from gfaas.triton_quick_runner import probe_target

    if (
        probe_target() != target
        or installed_version() != helion_version
        or triton.__version__ != triton_version
    ):
        raise RuntimeError("Helion execution environment differs from compilation")
    with tempfile.TemporaryDirectory(prefix="vfunc-helion-execute-") as folder:
        prepared = restore_variants([variant], artifacts, folder, target, triton_version)[0]
        candidate = load_candidate(prepared, target)
        args, _ = SnapshotInputs(inputs, [], [], [])(inputs["metadata"])
        value = candidate(*args)
        torch.cuda.synchronize()
        return snapshot_result(value, args)
