"""Self-contained GPU target probe and quick benchmark workload."""

from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import inspect
import io
import json
import math
import os
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any


def probe_target() -> dict[str, Any]:
    cuda = ctypes.CDLL("libcuda.so.1")
    device, major, minor = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    for name, args in (
        ("cuInit", (0,)),
        ("cuDeviceGet", (ctypes.byref(device), 0)),
        ("cuDeviceComputeCapability", (ctypes.byref(major), ctypes.byref(minor), device)),
    ):
        status = getattr(cuda, name)(*args)
        if status:
            raise RuntimeError(f"{name} failed with CUDA status {status}")
    return {"backend": "cuda", "arch": major.value * 10 + minor.value, "warp_size": 32}


def select_rows(
    rows: list[dict[str, Any]], best_us: float | None, delta: float, minimum_us: float
) -> list[str]:
    if best_us is None:
        return []
    return [
        r["id"]
        for r in rows
        if r.get("status") == "measured"
        and (r["runtime_us"] < minimum_us or r["runtime_us"] <= best_us * (1 + delta))
    ]


def restore_caches(
    artifacts: list[Any], destination: Path, source: str, target: dict[str, Any], version: str
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        root = Path(os.fspath(artifact))
        manifest = json.loads((root / "manifest.json").read_text())
        if (
            manifest["source_sha256"] != hashlib.sha256(source.encode()).hexdigest()
            or manifest["target"] != target
            or manifest["triton_version"] != version
        ):
            raise RuntimeError("Compiled artifact does not match source, target or Triton version")
        archive = root / "cache.tar.gz"
        if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest["cache_archive"]["sha256"]:
            raise RuntimeError("Compiler archive checksum mismatch")
        found = {}
        with tarfile.open(archive) as tar:
            for member in tar:
                path = Path(member.name)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or not path.parts
                    or path.parts[0] != "cache"
                ):
                    raise RuntimeError("Unsafe compiler cache path")
                if member.isdir():
                    continue
                if not member.isfile() or member.name in found:
                    raise RuntimeError("Unsafe or duplicate compiler cache entry")
                handle = tar.extractfile(member)
                assert handle is not None
                data = handle.read()
                digest = hashlib.sha256(data).hexdigest()
                if manifest["files"].get(member.name) != digest:
                    raise RuntimeError("Compiler file checksum mismatch")
                found[member.name] = digest
                output = destination.joinpath(*path.parts[1:])
                output.parent.mkdir(parents=True, exist_ok=True)
                if output.exists() and output.read_bytes() != data:
                    raise RuntimeError("Conflicting compiler cache files")
                output.write_bytes(data)
        if found != manifest["files"]:
            raise RuntimeError("Incomplete compiler archive")
        for result in manifest["results"]:
            if result["id"] in records:
                raise RuntimeError("Duplicate compiled variant")
            records[result["id"]] = result
    # Triton cache group manifests contain absolute paths from the CPU worker.
    for group in destination.rglob("__grp__*.json"):
        data = json.loads(group.read_text())
        children = data.get("child_paths")
        if not isinstance(children, dict) or any(Path(n).name != n for n in children):
            raise RuntimeError("Invalid compiler cache group")
        data["child_paths"] = {n: str(group.parent / n) for n in children}
        group.write_text(json.dumps(data))
    return records


def bound_candidate(
    compiled: Any, constants: dict[str, Any], options: dict[str, Any], signature: Any, grid: Any
) -> Any:
    def candidate(*args: Any, **kwargs: Any) -> None:
        supplied = signature.bind_partial(*args, **kwargs)
        merged = dict(constants)
        merged.update(supplied.arguments)
        bound = signature.bind(**merged)
        bound.apply_defaults()
        if any(bound.arguments[n] != v for n, v in constants.items()):
            raise ValueError("Evaluation changed compiled constexpr values")
        meta = {**bound.arguments, **options}
        dimensions = tuple(grid(meta) if callable(grid) else grid)
        if not 1 <= len(dimensions) <= 3 or any(type(d) is not int or d < 1 for d in dimensions):
            raise ValueError("Grid must have one to three positive dimensions")
        dimensions += (1,) * (3 - len(dimensions))
        compiled[dimensions](*bound.arguments.values())

    return candidate


def l2_flush_buffer(torch: Any) -> Any:
    cuda = ctypes.CDLL("libcuda.so.1")
    size = ctypes.c_int()
    # CUDA's CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE. The job exposes one device.
    status = cuda.cuDeviceGetAttribute(ctypes.byref(size), 38, 0)
    if status or size.value <= 0:
        raise RuntimeError("Cannot determine GPU L2 cache size")
    return torch.empty(2 * size.value, dtype=torch.uint8, device="cuda:0")


def measure_quick(
    func: Any, args: tuple[Any, ...], kwargs: dict[str, Any], flush: Any, torch: Any
) -> dict[str, Any]:
    torch.cuda.synchronize()
    func(*args, **kwargs)
    torch.cuda.synchronize()

    def trials(count: int) -> list[float]:
        for _ in range(100):
            flush.zero_()
        events = []
        for _ in range(count):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            func(*args, **kwargs)
            end.record()
            flush.zero_()
            events.append((start, end))
        torch.cuda.synchronize()
        return [start.elapsed_time(end) * 1000 for start, end in events]

    pilot = trials(5)
    fastest = min(pilot)
    if not math.isfinite(fastest) or fastest <= 0:
        raise RuntimeError("Quick benchmark returned an invalid pilot runtime")
    iterations = math.ceil(10_000 / fastest)
    refined = iterations >= 10
    final = trials(iterations) if refined else pilot
    return {
        "runtime_us": min(final) if refined else fastest,
        "pilot_trial_us": pilot,
        "trial_us": final,
        "refinement_iterations": iterations if refined else 0,
    }


def measure_group(
    candidates: list[Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    flush: Any,
    torch: Any,
    counts: list[int],
) -> list[list[float]]:
    for _ in range(100):
        flush.zero_()
    events: list[list[Any]] = [[] for _ in candidates]
    for iteration in range(max(counts)):
        for index, candidate in enumerate(candidates):
            if iteration >= counts[index]:
                continue
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            candidate(*args, **kwargs)
            end.record()
            flush.zero_()
            events[index].append((start, end))
    torch.cuda.synchronize()
    return [[start.elapsed_time(end) * 1000 for start, end in pairs] for pairs in events]


def benchmark_candidates(
    entries: list[tuple[dict[str, Any], Any]],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    fresh_inputs: Any,
    evaluate: Any,
    flush: Any,
    torch: Any,
    group_size: int,
    minimum_us: float,
) -> tuple[float | None, str | None]:
    best_us = None
    best_id = None

    def accept(row: dict[str, Any], candidate: Any) -> None:
        nonlocal best_us, best_id
        runtime = row["runtime_us"]
        if not math.isfinite(runtime) or runtime <= 0:
            raise RuntimeError("Quick benchmark returned an invalid runtime")
        if best_us is None or runtime < best_us:
            if evaluate is not None and row.get("evaluation") != "passed":
                eval_args, eval_kwargs = fresh_inputs()
                verdict = evaluate(candidate, *eval_args, **eval_kwargs)
                torch.cuda.synchronize()
                if type(verdict) is not bool:
                    raise TypeError("evaluate must return a bool")
                row["evaluation"] = "passed" if verdict else "failed"
                if not verdict:
                    row["status"] = "invalid"
                    return
            best_us, best_id = runtime, row["id"]

    ready = []
    for row, candidate in entries:
        try:
            torch.cuda.synchronize()
            candidate(*args, **kwargs)
            torch.cuda.synchronize()
            ready.append((row, candidate))
        except Exception as error:
            row.update(status="benchmark_failed", diagnostics=str(error))
    for offset in range(0, len(ready), group_size):
        group = ready[offset : offset + group_size]
        trials = measure_group([c for _, c in group], args, kwargs, flush, torch, [5] * len(group))
        for (row, candidate), values in zip(group, trials, strict=True):
            row.update(
                status="measured",
                runtime_us=min(values),
                pilot_trial_us=values,
                trial_us=values,
                refinement_iterations=0,
                evaluation="not_needed" if evaluate else "skipped",
            )
            accept(row, candidate)
    survivors = []
    for row, candidate in ready:
        if row["status"] != "measured":
            continue
        if (
            best_us is not None
            and row["runtime_us"] >= minimum_us
            and row["runtime_us"] >= 2 * best_us
        ):
            row["status"] = "pilot_pruned"
            continue
        count = math.ceil(10_000 / row["runtime_us"])
        if count >= 10:
            survivors.append((row, candidate, count))
    # Re-establish the best in the second-stage timing domain. Unrefined long
    # kernels retain their pilot time; candidates already verified stay verified.
    best_us, best_id = None, None
    for row, candidate in ready:
        if row["status"] == "measured" and math.ceil(10_000 / row["runtime_us"]) < 10:
            accept(row, candidate)
    for offset in range(0, len(survivors), group_size):
        refined_group = survivors[offset : offset + group_size]
        trials = measure_group(
            [c for _, c, _ in refined_group],
            args,
            kwargs,
            flush,
            torch,
            [count for _, _, count in refined_group],
        )
        for (row, candidate, count), values in zip(refined_group, trials, strict=True):
            row.update(runtime_us=min(values), trial_us=values, refinement_iterations=count)
            accept(row, candidate)
    return best_us, best_id


def quick_benchmark(
    *,
    source: str,
    kernel_name: str,
    variants: str,
    artifacts: list[Any],
    inputs: bytes,
    callbacks: bytes,
    target: dict[str, Any],
    triton_version: str,
    delta: float,
    minimum_us: float,
    group_size: int = 8,
) -> dict[str, Any]:
    import cloudpickle
    import torch  # type: ignore[import-not-found]
    import triton  # type: ignore[import-not-found]
    from triton.backends.compiler import GPUTarget  # type: ignore[import-not-found]
    from triton.compiler import ASTSource  # type: ignore[import-not-found]

    if probe_target() != target or triton.__version__ != triton_version:
        raise RuntimeError("Benchmark GPU target or Triton version differs from compilation")
    grid, evaluate = cloudpickle.loads(callbacks)
    variants_list = json.loads(variants)
    rows: list[dict[str, Any]] = []
    best_us: float | None = None
    best_id = None
    with tempfile.TemporaryDirectory(prefix="vfunc-quick-") as folder:
        root = Path(folder)
        cache = root / "cache"
        cache.mkdir()
        prepared = restore_caches(artifacts, cache, source, target, triton_version)
        if not {v["id"] for v in variants_list} <= set(prepared):
            raise RuntimeError("Compiler artifacts have an incomplete variant set")
        os.environ["TRITON_CACHE_DIR"] = str(cache)
        if hasattr(triton, "knobs") and hasattr(triton.knobs, "cache"):
            triton.knobs.cache.dir = str(cache)
        path = root / "compile_source.py"
        path.write_text(source)
        name = "vfunc_compile_" + hashlib.sha256(source.encode()).hexdigest()[:16]
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        jit = getattr(module, kernel_name)
        signature = inspect.signature(jit.fn)
        modern = "constexprs" in inspect.signature(ASTSource).parameters

        def fresh_inputs() -> tuple[Any, Any]:
            # Shared storage and strides survive; device ordinals are remapped.
            return torch.load(io.BytesIO(inputs), map_location="cuda:0", weights_only=False)

        benchmark_args, benchmark_kwargs = fresh_inputs()
        flush = l2_flush_buffer(torch)
        entries = []
        for variant in variants_list:
            row: dict[str, Any] = {"id": variant["id"]}
            if prepared[row["id"]]["status"] != "compiled":
                rows.append({**row, "status": "compile_failed"})
                continue
            constants = variant["constants"]
            kinds = variant["signature"]
            if modern:
                ast = ASTSource(jit, kinds, constexprs=constants)
            else:
                ast = ASTSource(
                    jit,
                    {jit.arg_names.index(n): k for n, k in kinds.items() if k != "constexpr"},
                    constants={jit.arg_names.index(n): v for n, v in constants.items()},
                )
            before = {
                p: (p.stat().st_mtime_ns, p.stat().st_size)
                for p in cache.rglob("*")
                if p.is_file() and p.suffix in {".cubin", ".ptx", ".llir", ".ttir", ".ttgir"}
            }
            compiled = triton.compile(ast, target=GPUTarget(**target), options=variant["options"])
            after = {
                p: (p.stat().st_mtime_ns, p.stat().st_size)
                for p in cache.rglob("*")
                if p.is_file() and p.suffix in {".cubin", ".ptx", ".llir", ".ttir", ".ttgir"}
            }
            if before != after:
                raise RuntimeError("GPU preparation unexpectedly recompiled a prepared variant")
            row["prepared_files_unchanged"] = True
            if compiled.hash != prepared[row["id"]]["cache_hash"]:
                raise RuntimeError("Compiled cache identity differs from CPU preparation")

            candidate = bound_candidate(compiled, constants, variant["options"], signature, grid)

            rows.append(row)
            entries.append((row, candidate))
        best_us, best_id = benchmark_candidates(
            entries,
            benchmark_args,
            benchmark_kwargs,
            fresh_inputs,
            evaluate,
            flush,
            torch,
            group_size,
            minimum_us,
        )
    return {
        "schema": "vfunc.triton-quick/v1",
        "status": "passed" if best_id else "failed",
        "results": rows,
        "best_id": best_id,
        "best_runtime_us": best_us,
        "retained_ids": select_rows(rows, best_us, delta, minimum_us),
        "quick_benchmark_delta": delta,
        "pruning_min_runtime_us": minimum_us,
        "evaluation_enabled": evaluate is not None,
        "timing_method": "Interleaved CUDA events: 5-trial pilot, 2x pruning, regrouped 10 ms minimum; 100 preflushes per group",
        "benchmark_input_loads": 1,
        "group_size": group_size,
        "pilot_pruning_factor": 2,
        "l2_flush_bytes": flush.numel() * flush.element_size(),
    }
