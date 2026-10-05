"""Self-contained GPU target probe and quick benchmark workload."""

from __future__ import annotations

import base64
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
from contextlib import nullcontext
from pathlib import Path
from typing import Any


def probe_target(device_index: int = 0) -> dict[str, Any]:
    cuda = ctypes.CDLL("libcuda.so.1")
    device, major, minor = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    for name, args in (
        ("cuInit", (0,)),
        ("cuDeviceGet", (ctypes.byref(device), device_index)),
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
    artifacts: list[Any],
    destination: Path,
    source: str,
    target: dict[str, Any],
    version: str,
    selected_ids: set[str] | None = None,
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
        selected_files = manifest["files"]
        selected_results = manifest["results"]
        if selected_ids is not None:
            selected_results = [r for r in selected_results if r["id"] in selected_ids]
            directories: set[str] = set()
            for result in selected_results:
                # Triton uses hexadecimal cache keys in older releases and
                # base32 keys in newer ones. Reject unknown layouts explicitly.
                digest = result["cache_hash"]
                directories.update(
                    (digest, base64.b32encode(bytes.fromhex(digest)).decode().rstrip("="))
                )
            selected_files = {
                name: digest
                for name, digest in manifest["files"].items()
                if len(Path(name).parts) > 2 and Path(name).parts[1] in directories
            }
            if selected_results and not selected_files:
                raise RuntimeError("Unsupported Triton cache layout for selected configuration")
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
                if member.name not in selected_files:
                    continue
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
        if found != selected_files:
            raise RuntimeError("Incomplete compiler archive")
        for result in selected_results:
            if result["id"] in records:
                raise RuntimeError("Duplicate compiled variant")
            records[result["id"]] = result
    if selected_ids is not None and set(records) != selected_ids:
        raise RuntimeError("Missing compiled configuration in selected artifacts")
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
    status = cuda.cuDeviceGetAttribute(ctypes.byref(size), 38, torch.cuda.current_device())
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


def pruning_limit(best: float, policy: dict[str, float] | None) -> float:
    if policy is None:
        return math.inf
    return best + max(best * policy["relative_delta"], policy["absolute_us"])


def prune_timings(
    rows: list[dict[str, Any]], field: str, policy: dict[str, float] | None
) -> list[str]:
    valid = [r for r in rows if r.get("status") == "measured"]
    if not valid:
        return []
    limit = pruning_limit(min(r[field] for r in valid), policy)
    return [r["id"] for r in valid if r[field] <= limit]


class InputRing:
    """Independent CUDA storage; factory metadata and aliasing are checked per set."""

    def __init__(
        self,
        factory: Any,
        reset: Any,
        metadata: dict[str, Any],
        torch: Any,
        l2_bytes: int,
        max_sets: int,
        max_bytes: int,
    ) -> None:
        self.factory, self.reset, self.metadata, self.torch = factory, reset, metadata, torch
        self.max_sets, self.max_bytes = max_sets, max_bytes
        self.sets: list[Any] = []
        self.pointers: set[int] = set()
        self.allocated_bytes = 0
        self.cursor = 0
        self.footprint_bytes: int = 0
        self.append()
        count = l2_bytes // self.footprint_bytes + 1
        self.ensure(count)

    def append(self) -> None:
        if len(self.sets) >= self.max_sets:
            raise ValueError(
                "Input ring exceeds max_input_sets; use larger input sets or raise the limit"
            )
        import copy

        result = self.factory(copy.deepcopy(self.metadata))
        if (
            not isinstance(result, tuple)
            or len(result) != 2
            or not isinstance(result[0], tuple)
            or not isinstance(result[1], dict)
        ):
            raise TypeError("make_inputs must return (args_tuple, kwargs_dict)")
        args, kwargs = result
        if (
            len(args) != len(self.metadata["args"])
            or kwargs.keys() != self.metadata["kwargs"].keys()
        ):
            raise ValueError("Generated arguments do not match captured input metadata")
        groups: dict[int, int] = {}
        allocations: dict[int, int] = {}
        logical: dict[int, int] = {}
        values = list(zip(args, self.metadata["args"], strict=True))
        values += [(kwargs[k], spec) for k, spec in self.metadata["kwargs"].items()]
        for value, spec in values:
            if spec["kind"] == "value":
                if type(value) is not type(spec["value"]) or value != spec["value"]:
                    raise ValueError("Generated scalar differs from captured metadata")
                continue
            if not isinstance(value, self.torch.Tensor) or value.device.type != "cuda":
                raise ValueError("make_inputs must generate CUDA tensors on the assigned device")
            if value.device.index != self.torch.cuda.current_device():
                raise ValueError("Generated tensor is on another GPU")
            if (
                list(value.shape) != spec["shape"]
                or list(value.stride()) != spec["stride"]
                or str(value.dtype).removeprefix("torch.") != spec["dtype"]
                or value.storage_offset() != spec["storage_offset"]
                or value.requires_grad
            ):
                raise ValueError(
                    "Generated tensor shape, stride, dtype, offset or gradient mode differs"
                )
            pointer = value.untyped_storage().data_ptr()
            group = spec["storage_group"]
            if group in groups and groups[group] != pointer:
                raise ValueError("Generated tensors do not preserve input storage aliasing")
            if group not in groups and pointer in groups.values():
                raise ValueError("Generated tensors introduce storage aliasing")
            groups[group] = pointer
            allocations[pointer] = value.untyped_storage().nbytes()
            logical[pointer] = max(logical.get(pointer, 0), value.numel() * value.element_size())
        footprint = sum(logical.values())
        if footprint <= 0 or self.pointers.intersection(groups.values()):
            raise ValueError("Input sets must contain nonempty, independent CUDA storage")
        if self.sets and footprint != self.footprint_bytes:
            raise ValueError("Input set footprints differ")
        if self.allocated_bytes + sum(allocations.values()) > self.max_bytes:
            raise ValueError("Input ring exceeds max_ring_bytes")
        self.footprint_bytes = footprint
        self.allocated_bytes += sum(allocations.values())
        self.pointers.update(groups.values())
        self.sets.append(result)

    def ensure(self, count: int) -> None:
        if count > self.max_sets:
            raise ValueError("Input ring exceeds max_input_sets")
        while len(self.sets) < count:
            self.append()

    def next(self) -> Any:
        result = self.sets[self.cursor % len(self.sets)]
        self.cursor += 1
        return result

    def fresh(self) -> Any:
        fresh = InputRing(self.factory, self.reset, self.metadata, self.torch, 0, 1, self.max_bytes)
        if self.pointers.intersection(fresh.pointers):
            raise ValueError("Evaluation inputs share benchmark-ring storage")
        args, kwargs = fresh.sets[0]
        self.reset(*args, **kwargs)
        return args, kwargs


def ring_trials(
    candidates: list[Any],
    counts: list[int],
    ring: InputRing,
    torch: Any,
    flush: Any,
    flush_iterations: int = 100,
) -> list[list[float]]:
    for _ in range(flush_iterations):
        flush.zero_()
    events: list[list[Any]] = [[] for _ in candidates]
    for iteration in range(max(counts)):
        for index, candidate in enumerate(candidates):
            if iteration >= counts[index]:
                continue
            args, kwargs = ring.next()
            ring.reset(*args, **kwargs)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            candidate(*args, **kwargs)
            end.record()
            events[index].append((start, end))
    torch.cuda.synchronize()
    return [[s.elapsed_time(e) * 1000 for s, e in pairs] for pairs in events]


def final_benchmark(
    candidate: Any,
    estimate_us: float,
    ring: InputRing,
    torch: Any,
    flush: Any,
    benchmark: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = benchmark or {}
    flush_iterations = settings.get("l2_flush_iterations", 100)
    final_us = settings.get("final_duration_ms", 25.0) * 1000
    maximum = settings.get("max_final_trials", 1000)
    z = max(
        1,
        min(
            settings.get("max_calls_per_graph", 100),
            math.floor(settings.get("graph_duration_ms", 1.0) * 1000 / estimate_us),
        ),
    )
    padded_sets = math.ceil(len(ring.sets) / z) * z if ring is not None else z
    graph_fits = padded_sets <= getattr(ring, "max_sets", math.inf) and padded_sets * (
        getattr(ring, "allocated_bytes", 0) / max(1, len(ring.sets)) if ring is not None else 0
    ) <= getattr(ring, "max_bytes", math.inf)
    if z < settings.get("min_calls_per_graph", 10) or not graph_fits:
        count = min(
            maximum, max(settings.get("min_final_trials", 25), math.ceil(final_us / estimate_us))
        )
        trials = ring_trials([candidate], [count], ring, torch, flush, flush_iterations)[0]
        return {
            "runtime_us": min(trials),
            "trial_us": trials,
            "method": "events-ring",
            "calls_per_graph": 0,
            "iterations": count,
        }
    # Graph pointers cannot advance at replay time. Pad the ring to whole graphs,
    # then cycle distinct graphs so no replay continually reuses a small subset.
    ring.ensure(math.ceil(len(ring.sets) / z) * z)
    graphs: list[Any] = []
    torch.cuda.synchronize()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for args, kwargs in ring.sets:
            ring.reset(*args, **kwargs)
            candidate(*args, **kwargs)
    capture_stream.synchronize()
    pool = torch.cuda.graph_pool_handle()
    for offset in range(0, len(ring.sets), z):
        sets = ring.sets[offset : offset + z]
        # Warmup may mutate inputs. Restore before capture, outside event timing.
        with torch.cuda.stream(capture_stream):
            for args, kwargs in sets:
                ring.reset(*args, **kwargs)
        capture_stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream, pool=pool):
            for args, kwargs in sets:
                candidate(*args, **kwargs)
        graphs.append((graph, sets))
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    count = min(maximum, max(1, math.ceil(final_us / (z * estimate_us))))
    for _ in range(flush_iterations):
        flush.zero_()
    events = []
    for index in range(count):
        graph, sets = graphs[index % len(graphs)]
        for args, kwargs in sets:
            ring.reset(*args, **kwargs)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        events.append((start, end))
    torch.cuda.synchronize()
    trials = [s.elapsed_time(e) * 1000 / z for s, e in events]
    return {
        "runtime_us": min(trials),
        "trial_us": trials,
        "method": "cudagraphs-ring",
        "calls_per_graph": z,
        "graph_count": len(graphs),
        "iterations": count,
    }


def benchmark_cycle(
    *,
    source: str,
    kernel_name: str,
    variants: str,
    artifacts: list[Any],
    metadata: dict[str, Any],
    callbacks: bytes,
    target: dict[str, Any],
    triton_version: str,
    policy: dict[str, Any],
    final_only: bool = False,
    excluded_gpu_uuids: list[str] | None = None,
    estimates: dict[str, float] | None = None,
    inputs: dict[str, Any] | None = None,
    reset_arguments: list[str] | None = None,
    restore_arguments: list[str] | None = None,
    argument_names: list[str] | None = None,
    prepared_cache: tuple[str, dict[str, dict[str, Any]]] | None = None,
    backend: str = "triton",
) -> dict[str, Any]:
    import cloudpickle
    import torch

    if probe_target(torch.cuda.current_device()) != target:
        raise RuntimeError("Benchmark GPU target differs from compilation")
    if backend == "triton":
        import triton
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource

        if triton.__version__ != triton_version:
            raise RuntimeError("Benchmark Triton version differs from compilation")
    elif backend != "cuda":
        raise ValueError("Unsupported kernel backend")
    torch.cuda.init()
    gpu_uuid = str(
        getattr(torch.cuda.get_device_properties(torch.cuda.current_device()), "uuid", "") or ""
    )
    if not gpu_uuid:
        raise RuntimeError("GPU UUID is required to verify independent replication")
    if gpu_uuid in (excluded_gpu_uuids or []):
        return {"status": "duplicate_gpu", "gpu_uuid": gpu_uuid, "results": []}
    if inputs is None:
        grid, evaluate, factory, reset = cloudpickle.loads(callbacks)
    else:
        from gfaas.triton_inputs import SnapshotInputs

        grid, evaluate = cloudpickle.loads(callbacks)
        factory = SnapshotInputs(
            inputs, reset_arguments or [], restore_arguments or [], argument_names or []
        )
        reset = factory.reset
    variants_list = json.loads(variants)
    rows: list[dict[str, Any]] = []
    flush = l2_flush_buffer(torch)
    ring = InputRing(
        factory,
        reset,
        metadata,
        torch,
        flush.numel() // 2,
        policy["max_input_sets"],
        policy["max_ring_bytes"],
    )
    context = (
        nullcontext(prepared_cache[0])
        if prepared_cache
        else tempfile.TemporaryDirectory(prefix="vfunc-ring-")
    )
    with context as folder:
        if backend == "cuda":
            from gfaas.cuda_kernel_runner import load_binaries, prepare_entries

            prepared = (
                prepared_cache[1]
                if prepared_cache
                else load_binaries(
                    artifacts, source, kernel_name, target, {v["id"] for v in variants_list}
                )
            )
            pairs = prepare_entries(prepared, variants_list, kernel_name, grid, ring, reset)
            rows.extend(row for row, _ in pairs)
            entries = [(row, candidate) for row, candidate in pairs if candidate is not None]
        else:
            cache = Path(folder) / "cache"
            cache.mkdir(exist_ok=True)
            prepared = (
                prepared_cache[1]
                if prepared_cache
                else restore_caches(artifacts, cache, source, target, triton_version)
            )
            os.environ["TRITON_CACHE_DIR"] = str(cache)
            if hasattr(triton, "knobs") and hasattr(triton.knobs, "cache"):
                triton.knobs.cache.dir = str(cache)
            path = Path(folder) / "compile_source.py"
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
            entries = []
            for variant in variants_list:
                row = {"id": variant["id"]}
                rows.append(row)
                if prepared.get(row["id"], {}).get("status") != "compiled":
                    row["status"] = "compile_failed"
                    continue
                kinds, constants = variant["signature"], variant["constants"]
                ast = (
                    ASTSource(jit, kinds, constexprs=constants)
                    if modern
                    else ASTSource(
                        jit,
                        {jit.arg_names.index(n): k for n, k in kinds.items() if k != "constexpr"},
                        constants={jit.arg_names.index(n): v for n, v in constants.items()},
                    )
                )

                def snapshot() -> dict[str, Any]:
                    return {
                        str(p): (p.stat().st_mtime_ns, p.stat().st_size)
                        for p in cache.rglob("*")
                        if p.is_file()
                        and p.suffix in {".cubin", ".ptx", ".llir", ".ttir", ".ttgir", ".glir"}
                    }

                before = snapshot()
                compiled = triton.compile(
                    ast, target=GPUTarget(**target), options=variant["options"]
                )
                if before != snapshot() or compiled.hash != prepared[row["id"]]["cache_hash"]:
                    raise RuntimeError(
                        "GPU preparation recompiled or changed a CPU-prepared variant"
                    )
                row["prepared_files_unchanged"] = True
                candidate = bound_candidate(
                    compiled, constants, variant["options"], signature, grid
                )
                args, kwargs = ring.next()
                reset(*args, **kwargs)
                candidate(*args, **kwargs)
                torch.cuda.synchronize()
                entries.append((row, candidate))

        def check(row: dict[str, Any], candidate: Any) -> bool:
            if evaluate is None:
                row["evaluation"] = "skipped"
                return True
            args, kwargs = ring.fresh()
            verdict = evaluate(candidate, *args, **kwargs)
            torch.cuda.synchronize()
            if type(verdict) is not bool:
                raise TypeError("evaluate must return a bool")
            row["evaluation"] = "passed" if verdict else "failed"
            if not verdict:
                row["status"] = "invalid"
            return verdict

        settings = policy.get("benchmark", {})
        if not final_only:
            best = None
            for offset in range(0, len(entries), policy["group_size"]):
                group = entries[offset : offset + policy["group_size"]]
                values = ring_trials(
                    [c for _, c in group],
                    [settings.get("pilot_trials", 3)] * len(group),
                    ring,
                    torch,
                    flush,
                    settings.get("l2_flush_iterations", 100),
                )
                for (row, candidate), trials in zip(group, values, strict=True):
                    runtime = min(trials)
                    if not math.isfinite(runtime) or runtime <= 0:
                        raise RuntimeError("Invalid pilot timing")
                    row.update(status="measured", pilot_us=runtime, pilot_trial_us=trials)
                    if (best is None or runtime < best) and check(row, candidate):
                        best = runtime
            keep = set(prune_timings(rows, "pilot_us", policy["pilot_pruning"]))
            survivors = [(r, c) for r, c in entries if r["id"] in keep]
            for row, _ in entries:
                if row["status"] == "measured" and row["id"] not in keep:
                    row["status"] = "pilot_pruned"
            for offset in range(0, len(survivors), policy["group_size"]):
                group = survivors[offset : offset + policy["group_size"]]
                counts = [
                    min(
                        settings.get("max_refinement_trials", 250),
                        math.ceil(
                            settings.get("refinement_duration_ms", 1.0) * 1000 / r["pilot_us"]
                        ),
                    )
                    for r, _ in group
                ]
                active = [
                    (r, c, n)
                    for (r, c), n in zip(group, counts, strict=True)
                    if n >= settings.get("min_refinement_trials", 10)
                ]
                for (row, _), n in zip(group, counts, strict=True):
                    if n < settings.get("min_refinement_trials", 10):
                        row.update(
                            refined_us=row["pilot_us"], refined_trial_us=row["pilot_trial_us"]
                        )
                if active:
                    values = ring_trials(
                        [c for _, c, _ in active],
                        [n for _, _, n in active],
                        ring,
                        torch,
                        flush,
                        settings.get("l2_flush_iterations", 100),
                    )
                    for (row, _, n), trials in zip(active, values, strict=True):
                        row.update(
                            refined_us=min(trials), refined_trial_us=trials, refinement_iterations=n
                        )
            # Validate the proposed refined best before it can prune any peers.
            refined_evaluated = set()
            for row, candidate in sorted(survivors, key=lambda pair: pair[0]["refined_us"]):
                refined_evaluated.add(row["id"])
                if check(row, candidate):
                    break
            keep = set(prune_timings(rows, "refined_us", policy["refined_pruning"]))
            for row, candidate in survivors:
                if row["status"] != "measured":
                    continue
                if row["id"] not in keep:
                    row["status"] = "refined_pruned"
                elif row["id"] not in refined_evaluated and not check(row, candidate):
                    continue
        for row, candidate in entries:
            if final_only:
                row["status"] = "measured"
                if not check(row, candidate):
                    continue
                if estimates is None:
                    samples = ring_trials(
                        [candidate],
                        [settings.get("pilot_trials", 3)],
                        ring,
                        torch,
                        flush,
                        settings.get("l2_flush_iterations", 100),
                    )[0]
                    estimate = min(samples)
                    row.update(refined_us=estimate, estimate_trial_us=samples)
                else:
                    estimate = estimates[row["id"]]
                    row["refined_us"] = estimate
            elif row.get("status") == "measured":
                estimate = row["refined_us"]
            else:
                continue
            if not math.isfinite(estimate) or estimate <= 0:
                raise RuntimeError("Invalid refined runtime")
            final = final_benchmark(candidate, estimate, ring, torch, flush, settings)
            if not math.isfinite(final["runtime_us"]) or final["runtime_us"] <= 0:
                raise RuntimeError("Invalid final runtime")
            row.update(runtime_us=final["runtime_us"], final=final)
        valid = [r for r in rows if r["status"] == "measured"]
        best_row = min(valid, key=lambda r: r["runtime_us"]) if valid else None
        return {
            "schema": "vfunc.triton-benchmark/v1",
            "status": "passed" if valid else "failed",
            "gpu_uuid": gpu_uuid,
            "results": rows,
            "best_id": best_row["id"] if best_row else None,
            "best_runtime_us": best_row["runtime_us"] if best_row else None,
            "ring_sets": len(ring.sets),
            "ring_footprint_bytes": ring.footprint_bytes,
            "ring_allocated_bytes": ring.allocated_bytes,
            "l2_bytes": flush.numel() // 2,
            "cache_assumption": "Generated storage footprint exceeds L2; eviction depends on actual kernel accesses and reset traffic.",
        }


def benchmark_replicas(*, device_count: int, **kwargs: Any) -> dict[str, Any]:
    import torch

    if torch.cuda.device_count() < device_count:
        raise RuntimeError("Replica Call did not receive its requested GPU count")
    reports = []
    for device in range(device_count):
        with torch.cuda.device(device):
            reports.append(benchmark_cycle(**kwargs))
    return {"status": "replica_bundle", "replica_reports": reports}


def execute_winner(
    *,
    source: str,
    kernel_name: str,
    variant: dict[str, Any],
    artifacts: list[Any],
    inputs: dict[str, Any],
    callbacks: bytes,
    target: dict[str, Any],
    triton_version: str,
) -> dict[str, Any]:
    """Run once on the invocation snapshot and return its resulting storage bytes."""
    import cloudpickle
    import torch
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    from gfaas.triton_inputs import SnapshotInputs, snapshot_inputs

    if probe_target() != target or triton.__version__ != triton_version:
        raise RuntimeError("Execution GPU target or Triton version differs from compilation")
    grid, _ = cloudpickle.loads(callbacks)
    with tempfile.TemporaryDirectory(prefix="vfunc-execute-") as folder:
        cache = Path(folder) / "cache"
        cache.mkdir()
        prepared = restore_caches(artifacts, cache, source, target, triton_version)
        os.environ["TRITON_CACHE_DIR"] = str(cache)
        if hasattr(triton, "knobs") and hasattr(triton.knobs, "cache"):
            triton.knobs.cache.dir = str(cache)
        path = Path(folder) / "compile_source.py"
        path.write_text(source)
        name = "vfunc_execute_" + hashlib.sha256(source.encode()).hexdigest()[:16]
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        jit = getattr(module, kernel_name)
        kinds, constants = variant["signature"], variant["constants"]
        if "constexprs" in inspect.signature(ASTSource).parameters:
            ast = ASTSource(jit, kinds, constexprs=constants)
        else:
            ast = ASTSource(
                jit,
                {jit.arg_names.index(n): k for n, k in kinds.items() if k != "constexpr"},
                constants={jit.arg_names.index(n): v for n, v in constants.items()},
            )
        compiled = triton.compile(ast, target=GPUTarget(**target), options=variant["options"])
        if compiled.hash != prepared[variant["id"]]["cache_hash"]:
            raise RuntimeError("Execution did not load the CPU-prepared variant")
        candidate = bound_candidate(
            compiled, constants, variant["options"], inspect.signature(jit.fn), grid
        )
        factory = SnapshotInputs(inputs, [], [], jit.arg_names)
        args, kwargs = factory(inputs["metadata"])
        # Reset/restore declarations govern benchmarking only. The real launch
        # observes the exact values supplied by the caller and preserves writes.
        candidate(*args, **kwargs)
        torch.cuda.synchronize()
        return snapshot_inputs(args, kwargs)


def benchmark_selected(**kwargs: Any) -> dict[str, Any]:
    """Measure only the selected variant, with a fresh launch-duration estimate."""
    kwargs["final_only"] = True
    return benchmark_cycle(**kwargs)


def benchmark_selected_replicas(*, device_count: int, **kwargs: Any) -> dict[str, Any]:
    """Load only selected binaries once, then measure each assigned GPU."""
    import torch

    if torch.cuda.device_count() != device_count:
        raise RuntimeError("Benchmark Call did not receive its requested GPU count")
    selected = {v["id"] for v in json.loads(kwargs["variants"])}
    with tempfile.TemporaryDirectory(prefix="vfunc-selected-") as folder:
        cache = Path(folder) / "cache"
        cache.mkdir()
        prepared = restore_caches(
            kwargs["artifacts"],
            cache,
            kwargs["source"],
            kwargs["target"],
            kwargs["triton_version"],
            selected,
        )
        reports = []
        for device in range(device_count):
            with torch.cuda.device(device):
                reports.append(benchmark_cycle(**kwargs, prepared_cache=(folder, prepared)))
        return {
            "status": "replica_bundle",
            "replica_reports": reports,
            "prepared_configurations": len(prepared),
            "prepared_cache_files": sum(p.is_file() for p in cache.rglob("*")),
        }
