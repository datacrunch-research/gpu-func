"""Single-file workload bundle for staged Triton candidate tuning."""

from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

_SCHEMA = "vfunc.triton-tuning/v1"
_OUTPUT = "compiled-triton"
_ARTIFACT_ENV = "GFAAS_COMPILED_TRITON_ARTIFACT_ID"
_MAX_CACHE_BYTES = 128 * 1024 * 1024


def run(**_kwargs: Any) -> dict[str, Any]:
    raise RuntimeError("Triton tuning requires staged Call support")


def _validate_limits(
    source: str,
    candidates: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    compile_workers: int,
    warmup: int,
    trials: int,
    max_adaptive_candidates: int,
) -> None:
    if len(source.encode()) > 1024 * 1024 or not 1 <= len(candidates) <= 64:
        raise ValueError("Triton source or candidate count is out of range")
    if not 1 <= len(cases) <= 16 or len(candidates) * len(cases) > 128:
        raise ValueError("Triton tuning case or variant count is out of range")
    if not 1 <= compile_workers <= 8 or not 0 <= warmup <= 20 or not 1 <= trials <= 50:
        raise ValueError("Triton tuning worker or trial count is out of range")
    if not 0 <= max_adaptive_candidates <= 16:
        raise ValueError("Triton adaptive candidate count is out of range")


def _triton(cache: Path, version: str) -> Any:
    os.environ["TRITON_CACHE_DIR"] = str(cache)
    import triton  # type: ignore[import-not-found]

    if triton.__version__ != version:
        raise RuntimeError(f"Triton {version} is required; found {triton.__version__}")
    triton.knobs.cache.dir = str(cache)
    return triton


def _source_module(source: str, directory: Path) -> Any:
    path = directory / "tuning_workload.py"
    path.write_text(source, encoding="utf-8")
    name = "gfaas_triton_workload_" + hashlib.sha256(source.encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Triton workload source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _source(kernel: Any, signature: dict[str, str], constants: dict[str, Any]) -> Any:
    from triton.compiler import ASTSource  # type: ignore[import-not-found]

    if set(signature) != set(kernel.arg_names):
        raise ValueError("signature does not match the Triton kernel arguments")
    if set(constants) != {name for name, kind in signature.items() if kind == "constexpr"}:
        raise ValueError("case and candidate constants do not match constexpr arguments")
    return ASTSource(fn=kernel, signature=signature, constexprs=constants)


def _constants(case: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    constants = dict(case["constexprs"])
    if constants.keys() & candidate["constexprs"].keys():
        raise ValueError("case and candidate constexpr names overlap")
    constants.update(candidate["constexprs"])
    return constants


def _compile(
    triton: Any,
    kernel: Any,
    signature: dict[str, str],
    case: dict[str, Any],
    candidate: dict[str, Any],
    target_arch: int,
) -> Any:
    from triton.backends.compiler import GPUTarget  # type: ignore[import-not-found]

    return triton.compile(
        _source(kernel, signature, _constants(case, candidate)),
        target=GPUTarget("cuda", target_arch, 32),
        options={"num_warps": candidate["num_warps"], "num_stages": candidate["num_stages"]},
    )


def _compile_record(
    triton: Any,
    kernel: Any,
    signature: dict[str, str],
    case: dict[str, Any],
    candidate: dict[str, Any],
    target_arch: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    record: dict[str, Any] = {"key": case["key"], "candidate": candidate["name"]}
    try:
        compiled = _compile(triton, kernel, signature, case, candidate, target_arch)
        record["cache_hash"] = compiled.hash
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"[:2048]
    record["compile_ms"] = (time.perf_counter() - start) * 1000
    return record


def _cache_files(cache: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    total = 0
    for path in sorted(cache.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("Triton cache contains a symbolic link")
        if not path.is_file():
            continue
        total += path.stat().st_size
        if total > _MAX_CACHE_BYTES:
            raise RuntimeError("compiled Triton cache exceeds 128 MiB")
        files[path.relative_to(cache).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def compile_stage(
    *,
    source: str,
    kernel_name: str,
    signature: dict[str, str],
    candidates: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    target_arch: int,
    triton_version: str,
    compile_workers: int,
    warmup: int,
    trials: int,
    max_adaptive_candidates: int,
) -> dict[str, Any]:
    """Compile all initial variants with bounded CPU parallelism and no GPU query."""
    _validate_limits(
        source, candidates, cases, compile_workers, warmup, trials, max_adaptive_candidates
    )
    output_root = os.environ.get("GFAAS_OUTPUT_ROOT")
    if not output_root:
        raise RuntimeError("Triton compile output requires a remote Call")
    output = Path(output_root) / _OUTPUT
    output.mkdir(parents=True, exist_ok=False)
    cache = output / "cache"
    cache.mkdir()
    triton = _triton(cache, triton_version)
    with tempfile.TemporaryDirectory(prefix="triton-source-") as source_dir:
        module = _source_module(source, Path(source_dir))
        kernel = getattr(module, kernel_name)
        jobs = [(case, candidate) for case in cases for candidate in candidates]
        with ThreadPoolExecutor(max_workers=compile_workers) as executor:
            records = list(
                executor.map(
                    lambda job: _compile_record(
                        triton, kernel, signature, job[0], job[1], target_arch
                    ),
                    jobs,
                )
            )
    manifest = {
        "schema": _SCHEMA,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "triton_version": triton_version,
        "target_arch": target_arch,
        "target_gpu_pool": os.environ.get("GFAAS_TARGET_GPU_POOL"),
        "build_image": {
            "name": os.environ.get("GFAAS_BUILD_IMAGE_NAME"),
            "digest": os.environ.get("GFAAS_BUILD_IMAGE_DIGEST"),
        },
        "signature": signature,
        "candidates": candidates,
        "cases": cases,
        "records": records,
        "cache_files": _cache_files(cache),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return {
        "phase": "compile",
        "target_arch": target_arch,
        "variants": len(jobs),
        "compiled": sum("error" not in record for record in records),
        "errors": sum("error" in record for record in records),
    }


def _load_artifact(
    source: str,
    signature: dict[str, str],
    candidates: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    target_arch: int,
    triton_version: str,
    cache: Path,
) -> dict[str, Any]:
    artifact_id = os.environ.get(_ARTIFACT_ENV)
    artifact_root = os.environ.get("GFAAS_ARTIFACT_ROOT")
    if not artifact_id or not artifact_root:
        raise RuntimeError("compiled Triton Artifact was not staged")
    artifact = Path(artifact_root) / artifact_id
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    expected = {
        "schema": _SCHEMA,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "triton_version": triton_version,
        "target_arch": target_arch,
        "target_gpu_pool": os.environ.get("GFAAS_TARGET_GPU_POOL"),
        "build_image": {
            "name": os.environ.get("GFAAS_BUILD_IMAGE_NAME"),
            "digest": os.environ.get("GFAAS_BUILD_IMAGE_DIGEST"),
        },
        "signature": signature,
        "candidates": candidates,
        "cases": cases,
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise RuntimeError(f"compiled Triton Artifact has mismatched {name}")
    source_cache = artifact / "cache"
    if _cache_files(source_cache) != manifest.get("cache_files"):
        raise RuntimeError("compiled Triton cache failed its SHA-256 checks")
    shutil.copytree(source_cache, cache)
    for group in cache.rglob("__grp__*.json"):
        data = json.loads(group.read_text(encoding="utf-8"))
        children = data.get("child_paths")
        if not isinstance(children, dict) or any(Path(name).name != name for name in children):
            raise RuntimeError("compiled Triton cache has an invalid group")
        data["child_paths"] = {name: str(group.parent / name) for name in children}
        group.write_text(json.dumps(data), encoding="utf-8")
    return manifest


def _driver_version() -> int:
    driver = ctypes.CDLL("libcuda.so.1")
    version = ctypes.c_int()
    status = driver.cuDriverGetVersion(ctypes.byref(version))
    if status != 0:
        raise RuntimeError(f"cuDriverGetVersion failed with status {status}")
    return version.value


def _benchmark(
    compiled: Any,
    module: Any,
    case: dict[str, Any],
    candidate: dict[str, Any],
    kernel: Any,
    inputs: dict[str, Any],
    warmup: int,
    trials: int,
    torch: Any,
) -> list[float]:
    reset = getattr(module, "reset_inputs", None)
    restore = getattr(module, "restore_inputs", None)
    validate = getattr(module, "validate", None)
    if not callable(validate):
        raise RuntimeError("Triton workload must define validate(case, inputs)")
    grid = tuple(module.grid(case, candidate))
    if not 1 <= len(grid) <= 3 or any(not isinstance(v, int) or v < 1 for v in grid):
        raise ValueError("grid must contain 1 to 3 positive dimensions")
    grid = (*grid, *((1,) * (3 - len(grid))))
    constants = _constants(case, candidate)

    def arguments() -> list[Any]:
        return [constants[name] if name in constants else inputs[name] for name in kernel.arg_names]

    def launch() -> None:
        if callable(reset):
            reset(case, inputs)
        compiled[grid](*arguments())

    try:
        launch()
        torch.cuda.synchronize()
        if not validate(case, inputs):
            raise RuntimeError("candidate failed correctness validation")
        for _ in range(warmup):
            launch()
        torch.cuda.synchronize()
        timings = []
        for _ in range(trials):
            if callable(reset):
                reset(case, inputs)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            compiled[grid](*arguments())
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end))
        return timings
    finally:
        if callable(restore):
            restore(case, inputs)


def tune_stage(
    *,
    source: str,
    kernel_name: str,
    signature: dict[str, str],
    candidates: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    target_arch: int,
    triton_version: str,
    compile_workers: int,
    warmup: int,
    trials: int,
    max_adaptive_candidates: int,
) -> dict[str, Any]:
    """Load prepared variants, validate them, and time them under one GPU lease."""
    _validate_limits(
        source, candidates, cases, compile_workers, warmup, trials, max_adaptive_candidates
    )
    import torch  # type: ignore[import-not-found]

    if not torch.cuda.is_available():
        raise RuntimeError("Triton tuning requires one visible CUDA GPU")
    capability = torch.cuda.get_device_capability(0)
    actual_arch = capability[0] * 10 + capability[1]
    if actual_arch != target_arch:
        raise RuntimeError(f"CUDA target sm_{target_arch} does not match leased sm_{actual_arch}")
    gpu_uuid = getattr(torch.cuda.get_device_properties(0), "uuid", None)
    if not gpu_uuid:
        raise RuntimeError("CUDA device identity is unavailable")
    with tempfile.TemporaryDirectory(
        prefix="triton-tune-", dir=os.environ.get("FC_IO_ROOT")
    ) as root:
        workdir = Path(root)
        cache = workdir / "cache"
        manifest = _load_artifact(
            source, signature, candidates, cases, target_arch, triton_version, cache
        )
        triton = _triton(cache, triton_version)
        module = _source_module(source, workdir)
        kernel = getattr(module, kernel_name)
        prepared = {(record["key"], record["candidate"]): record for record in manifest["records"]}
        expected_variants = {
            (case["key"], candidate["name"]) for case in cases for candidate in candidates
        }
        if set(prepared) != expected_variants:
            raise RuntimeError("compiled Triton Artifact has an incomplete variant set")
        case_reports = []
        gpu_compilations = 0
        for case in cases:
            inputs = module.make_inputs(case)
            if not isinstance(inputs, dict):
                raise RuntimeError("make_inputs(case) must return a dictionary")
            results: list[dict[str, Any]] = []
            remaining = list(candidates)
            adaptive_added = False
            while remaining:
                candidate = remaining.pop(0)
                prepared_record = prepared.get((case["key"], candidate["name"]))
                adaptive = prepared_record is None
                result: dict[str, Any] = {
                    "name": candidate["name"],
                    "candidate_origin": "adaptive" if adaptive else "initial",
                    "compile_location": "gpu" if adaptive else "cpu",
                    "compile_ms": 0.0 if prepared_record is None else prepared_record["compile_ms"],
                }
                if prepared_record and "error" in prepared_record:
                    result["error"] = prepared_record["error"]
                else:
                    cache_hits: list[bool] = []
                    previous_listener = triton.knobs.compilation.listener
                    triton.knobs.compilation.listener = lambda cache_hits=cache_hits, **data: (
                        cache_hits.append(data["cache_hit"])
                    )
                    started = time.perf_counter()
                    try:
                        compiled = _compile(triton, kernel, signature, case, candidate, target_arch)
                        gpu_prepare_ms = (time.perf_counter() - started) * 1000
                        result["gpu_prepare_ms"] = gpu_prepare_ms
                        result["cache_hit"] = bool(cache_hits) and all(cache_hits)
                        if not result["cache_hit"]:
                            gpu_compilations += 1
                            result["gpu_compile_ms"] = gpu_prepare_ms
                        elif adaptive:
                            result["compile_location"] = "cache"
                        if not adaptive and not result["cache_hit"]:
                            raise RuntimeError(
                                "prepared Triton variant missed the transferred cache"
                            )
                        times = _benchmark(
                            compiled, module, case, candidate, kernel, inputs, warmup, trials, torch
                        )
                        result["trial_ms"] = times
                        result["median_ms"] = statistics.median(times)
                    except Exception as error:
                        result["error"] = f"{type(error).__name__}: {error}"[:2048]
                    finally:
                        triton.knobs.compilation.listener = previous_listener
                results.append(result)
                if not remaining and not adaptive_added and max_adaptive_candidates:
                    suggest = getattr(module, "suggest_candidates", None)
                    if callable(suggest):
                        from itertools import islice

                        proposed = list(islice(suggest(case, results), max_adaptive_candidates + 1))
                        if len(proposed) > max_adaptive_candidates:
                            raise RuntimeError("adaptive candidate limit exceeded")
                        existing = {item["name"] for item in candidates}
                        for item in proposed:
                            if not isinstance(item, dict) or item.get("name") in existing:
                                raise RuntimeError(
                                    "adaptive candidate has an invalid or duplicate name"
                                )
                            existing.add(item["name"])
                        remaining.extend(proposed)
                    adaptive_added = True
            valid = [item for item in results if "median_ms" in item]
            winner = min(valid, key=lambda item: item["median_ms"])["name"] if valid else None
            case_reports.append({"key": case["key"], "winner": winner, "candidates": results})
        return {
            "schema": _SCHEMA,
            "status": "succeeded" if all(item["winner"] for item in case_reports) else "failed",
            "triton_version": triton_version,
            "target_arch": target_arch,
            "driver_version": _driver_version(),
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_uuid": str(gpu_uuid),
            "gpu_compilations": gpu_compilations,
            "cases": case_reports,
        }
