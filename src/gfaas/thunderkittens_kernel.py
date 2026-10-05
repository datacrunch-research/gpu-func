"""Managed ThunderKittens CUDA kernels with replicated vFunc tuning."""

from __future__ import annotations

import copy
import hashlib
import inspect
import io
import json
import re
import tarfile
import tempfile
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Any

from . import cuda_kernel_runner, triton_quick_runner
from .app import Function, active_function_scope
from .artifacts import ArtifactOutput, ArtifactRef
from .errors import GfaasError
from .kernel import Kernel, KernelBenchmark
from .triton_inputs import apply_writes, snapshot_inputs
from .triton_kernel import _freeze, portable_callable
from .triton_policy import TritonTuning
from .triton_replication import benchmark_shards

IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def dimensions3(value: Any) -> tuple[int, int, int]:
    """Normalize CUDA launch dimensions without coercing invalid values."""
    if type(value) is int:
        value = (value,)
    if not isinstance(value, (tuple, list)) or not 1 <= len(value) <= 3:
        raise ValueError("CUDA dimensions must contain one to three positive integers")
    if any(type(v) is not int or not 1 <= v <= 2**31 - 1 for v in value):
        raise ValueError("CUDA dimensions must contain one to three positive integers")
    return tuple([*value, *([1] * (3 - len(value)))])


@dataclass(frozen=True)
class ThunderKittensConfig:
    """Compile-time numeric macros plus CUDA block and dynamic shared memory."""

    defines: Mapping[str, int | float] = field(default_factory=dict)
    block: tuple[int, ...] = (128,)
    shared_memory: int = 0

    def __post_init__(self) -> None:
        defines = dict(self.defines)
        if any(
            not IDENTIFIER.fullmatch(k) or k.startswith(("KITTENS_", "VFUNC_", "__vfunc_"))
            for k in defines
        ):
            raise ValueError("Configuration macros must be identifiers, excluding KITTENS_ macros")
        if any(type(v) not in (int, float) for v in defines.values()):
            raise TypeError("Configuration macro values must be numeric")
        json.dumps(defines, allow_nan=False)
        block = dimensions3(self.block)
        if block[0] * block[1] * block[2] > 1024 or block[2] > 64:
            raise ValueError("CUDA blocks may have at most 1024 threads")
        if type(self.shared_memory) is not int or self.shared_memory < 0:
            raise ValueError("Dynamic shared memory must be a nonnegative byte count")
        object.__setattr__(self, "defines", MappingProxyType(defines))
        object.__setattr__(self, "block", block)


class ThunderKittensCompilationError(GfaasError):
    def __init__(self, report: dict[str, Any]) -> None:
        self.report = report
        self.call_ids = report["call_ids"]
        super().__init__("ThunderKittens compilation could not produce usable configurations")


class ThunderKittensKernel(Kernel):
    """Compile CUDA entry points using a content-addressed ThunderKittens header snapshot.

    signature maps entry-point argument names to ``pointer`` or fixed-width scalar
    types. CUDA source normally exposes an extern "C" int host launcher accepting those
    arguments followed by VFUNC_LAUNCH_ARGS. Construct TK layouts/TMA descriptors
    there and launch the native kernel on VFUNC_STREAM. A raw __global__ entry
    point is also supported with entrypoint_kind="kernel". Header paths are
    local include directories; the SDK snapshots and transports their contents.
    """

    compiler: Function
    gpu_function: Function
    callbacks: bytes

    def __init__(
        self,
        source: str,
        entrypoint: str,
        *,
        signature: Mapping[str, str],
        entrypoint_kind: str = "launcher",
        headers: str | Path,
        configs: Sequence[ThunderKittensConfig] = (ThunderKittensConfig(),),
        tuning: TritonTuning | None = None,
        reset_to_zero: Sequence[str] = (),
        restore_value: Sequence[str] = (),
        nvcc_flags: Sequence[str] = (),
        variants_per_job: int = 16,
        max_concurrent_jobs: int = 8,
    ) -> None:
        if (
            not isinstance(source, str)
            or not source.strip()
            or not IDENTIFIER.fullmatch(entrypoint)
        ):
            raise ValueError("Supply CUDA source and an unmangled entry-point identifier")
        if entrypoint_kind not in ("launcher", "kernel"):
            raise ValueError("entrypoint_kind must be launcher or kernel")
        self.entrypoint_kind = entrypoint_kind
        types = dict(signature)
        if not types or "block" in types or any(not IDENTIFIER.fullmatch(name) for name in types):
            raise ValueError("Supply a nonempty argument signature with valid names")
        if any(kind not in {"pointer", *cuda_kernel_runner.SCALARS} for kind in types.values()):
            raise ValueError(
                "Signature supports pointer, int32/uint32/int64/uint64/float32/float64"
            )
        if not configs or any(not isinstance(c, ThunderKittensConfig) for c in configs):
            raise TypeError("configs must contain ThunderKittensConfig objects")
        if type(variants_per_job) is not int or variants_per_job < 1:
            raise ValueError("variants_per_job must be positive")
        if type(max_concurrent_jobs) is not int or not 1 <= max_concurrent_jobs <= 32:
            raise ValueError("max_concurrent_jobs must be between 1 and 32")
        if tuning is not None and not isinstance(tuning, TritonTuning):
            raise TypeError("tuning must be TritonTuning")
        if any(not isinstance(f, str) or not f for f in nvcc_flags):
            raise ValueError("nvcc_flags must contain nonempty strings")
        if any(
            f.startswith(("-o", "--output", "-arch", "--gpu-architecture", "-DKITTENS_"))
            or f in ("--cubin", "-cubin", "--ptx", "-ptx", "-c", "--compile")
            for f in nvcc_flags
        ):
            raise ValueError(
                "Output mode, target architecture and KITTENS macros are managed by vFunc"
            )
        reset, restore = tuple(reset_to_zero), tuple(restore_value)
        if any(name not in types or types[name] != "pointer" for name in (*reset, *restore)):
            raise ValueError("Reset/restore names must refer to pointer arguments")
        self.source, self.entrypoint = source, entrypoint
        self.signature = MappingProxyType(types)
        self._signature = inspect.Signature(
            [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in types]
        )
        if any(set(c.defines) & types.keys() for c in configs):
            raise ValueError("Compile-time macro names must not overlap runtime argument names")
        self.configs = tuple(configs)
        self.tuning = (
            tuning if tuning is not None else (TritonTuning() if len(configs) > 1 else None)
        )
        self.reset_to_zero, self.restore_value = reset, restore
        self.nvcc_flags = tuple(nvcc_flags)
        self.variants_per_job, self.max_concurrent_jobs = variants_per_job, max_concurrent_jobs
        root = Path(headers)
        if not (root / "kittens.cuh").is_file():
            raise ValueError(
                "headers must be a ThunderKittens include directory containing kittens.cuh"
            )
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w:gz", compresslevel=1) as tar:
            for path in sorted(root.rglob("*")):
                if path.is_symlink():
                    raise ValueError("Header snapshots cannot contain symlinks")
                if not path.is_file():
                    continue
                data = path.read_bytes()
                member = tarfile.TarInfo(path.relative_to(root).as_posix())
                member.size = len(data)
                tar.addfile(member, io.BytesIO(data))
        # gzip embeds its creation time; identity is based on deterministic tar content.
        import gzip

        self._headers = gzip.compress(gzip.decompress(archive.getvalue()), compresslevel=1, mtime=0)
        self.headers_sha256 = hashlib.sha256(self._headers).hexdigest()
        self._header_artifacts: dict[int, ArtifactRef] = {}
        self._cache: dict[tuple[int, str], Any] = {}
        self._results: dict[str, Any] = {}
        self._lock = RLock()

    @property
    def tuning_results(self) -> Any:
        with self._lock:
            return MappingProxyType(dict(self._results))

    def _variants(self) -> list[dict[str, Any]]:
        unique = {}
        for config in self.configs:
            variant = {
                "defines": dict(config.defines),
                "block": list(config.block),
                "shared_memory": config.shared_memory,
                "signature": dict(self.signature),
                "entrypoint_kind": self.entrypoint_kind,
            }
            variant["id"] = hashlib.sha256(
                json.dumps(variant, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
            unique[variant["id"]] = variant
        return list(unique.values())

    def _invoke(
        self,
        grid: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        options: KernelBenchmark | None,
    ) -> Any:
        import cloudpickle

        bound = self._signature.bind(*args, **kwargs)
        for name, kind in self.signature.items():
            value = bound.arguments[name]
            if kind == "pointer":
                if value is not None and not hasattr(value, "untyped_storage"):
                    raise TypeError(f"{name} must be a tensor or None")
            else:
                cuda_kernel_runner.scalar_value(kind, value)
        if not callable(grid):
            dimensions3(grid)
        inputs = snapshot_inputs(args, kwargs)
        callbacks = cloudpickle.dumps(
            (
                portable_callable(grid),
                portable_callable(self.tuning.evaluate if self.tuning else None),
            )
        )
        invocation = copy.copy(self)
        scope = active_function_scope()
        invocation.compiler = scope.bind(cuda_kernel_runner.compile_batch)
        invocation.gpu_function = scope.bind(triton_quick_runner.probe_target)
        image = invocation.compiler._resolve_image()
        gpu = invocation.gpu_function
        if (gpu.gpu_count is not None and gpu.gpu_count != 1) or (
            gpu.gpu_count is None and gpu.gpu is None
        ):
            raise ValueError("Select exactly one GPU target in app.function")
        if gpu.gpu and "," in gpu.gpu:
            raise ValueError("Select a single GPU target")
        pool = gpu.gpu_type
        if pool == "any" and gpu.gpu and not gpu.gpu.isdigit() and gpu.gpu != "any":
            pool = gpu.gpu
        invocation.gpu_function = replace(
            gpu, image=image, gpu=None, gpu_count=1, gpu_type=pool, outputs=()
        )
        invocation.compiler = replace(
            invocation.compiler,
            image=image,
            gpu=None,
            gpu_count=0,
            gpu_type=pool,
            cpu_millicores=invocation.compiler.cpu_millicores or 16000,
            memory_bytes=invocation.compiler.memory_bytes or 8 * 1024**3,
            outputs=(ArtifactOutput.directory("compiled-cuda-kernel", "compiled-cuda-kernel"),),
        )
        if not 1000 <= (invocation.compiler.cpu_millicores or 0) <= 32000:
            raise ValueError("Compiler CPU budget must be between 1000 and 32000 millicores")
        invocation.callbacks = callbacks
        variants = self._variants()
        identity = {
            "source": self.source,
            "entrypoint": self.entrypoint,
            "variants": variants,
            "headers_sha256": self.headers_sha256,
            "nvcc_flags": self.nvcc_flags,
            "tuning_policy": self.tuning.request() if self.tuning else None,
            "inputs": inputs["metadata"],
            "image": asdict(image),
            "gpu_type": pool,
            "callbacks_sha256": hashlib.sha256(callbacks).hexdigest(),
            "reset_to_zero": self.reset_to_zero,
            "restore_value": self.restore_value,
        }
        key = hashlib.sha256(
            json.dumps(identity, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        client = invocation.compiler.app.client
        cache_key = (id(client), key)
        with self._lock:
            reused = cache_key in self._cache
            if reused:
                cached = self._cache[cache_key]
            else:
                cached = invocation._prepare(inputs, variants, key, client)
                self._cache[cache_key] = cached
            if options is not None:
                policy = options.tuning_policy()
                request = invocation._request(cached, inputs, policy)
                result = benchmark_shards(
                    replace(invocation.gpu_function, handler=triton_quick_runner.benchmark_cycle),
                    [cached["variant"]],
                    request,
                    policy,
                    [],
                    single_job=True,
                )
                winner = next(r for r in result["results"] if r["id"] == result["best_id"])
                result.update(
                    specialization=key,
                    reused_specialization=reused,
                    autotuned=not reused and self.tuning is not None,
                    runtime_us=result["best_runtime_us"],
                    configuration=cached["variant"],
                    replicas=winner["replicas"],
                    input_metadata=inputs["metadata"],
                )
                return _freeze(result)
            executor = replace(invocation.gpu_function, handler=cuda_kernel_runner.execute)
            result = executor.spawn(
                source=self.source,
                kernel_name=self.entrypoint,
                variant=cached["variant"],
                artifacts=cached["artifacts"],
                target=cached["target"],
                callbacks=callbacks,
                inputs=inputs,
            ).wait()
            apply_writes(result, args, kwargs)
            return None

    def _request(
        self, cached: dict[str, Any], inputs: dict[str, Any], policy: TritonTuning
    ) -> dict[str, Any]:
        return {
            "source": self.source,
            "kernel_name": self.entrypoint,
            "artifacts": cached["artifacts"],
            "metadata": inputs["metadata"],
            "callbacks": self.callbacks,
            "inputs": inputs,
            "reset_arguments": list(self.reset_to_zero),
            "restore_arguments": list(self.restore_value),
            "argument_names": list(self.signature),
            "target": cached["target"],
            "triton_version": "",
            "backend": "cuda",
            "policy": policy.request(),
        }

    def _prepare(
        self, inputs: dict[str, Any], variants: list[dict[str, Any]], key: str, client: Any
    ) -> dict[str, Any]:
        started = time.perf_counter()
        probe = self.gpu_function.spawn()
        target = probe.wait()
        if not isinstance(target, dict) or target.get("backend") != "cuda":
            raise GfaasError("GPU target discovery returned an invalid target")
        if id(client) not in self._header_artifacts:
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "headers.tar.gz"
                path.write_bytes(self._headers)
                uploaded = client.upload_artifact_file(
                    path, filename="thunderkittens-headers.tar.gz", kind="input"
                )
            self._header_artifacts[id(client)] = ArtifactRef(uploaded["id"])
        header = self._header_artifacts[id(client)]
        ordered = sorted(variants, key=lambda v: v["id"])
        job_count = (len(variants) + self.variants_per_job - 1) // self.variants_per_job
        chunks = [ordered[i::job_count] for i in range(job_count)]

        def compile_chunk(chunk: list[dict[str, Any]]) -> Any:
            worker = replace(
                self.compiler,
                cpu_millicores=min(self.compiler.cpu_millicores or 1000, len(chunk) * 1000),
            )
            job = worker.spawn(
                source=self.source,
                kernel_name=self.entrypoint,
                variants=json.dumps(chunk),
                target=target,
                workers=min(4, len(chunk), (worker.cpu_millicores or 1000) // 1000),
                nvcc_flags=list(self.nvcc_flags),
                headers=header,
                headers_sha256=self.headers_sha256,
            )
            report = job.wait()
            outputs = client.get_call_result(job.call_id)["artifacts"]
            artifact = ArtifactRef(
                next(a["artifact_id"] for a in outputs if a["name"] == "compiled-cuda-kernel")
            )
            return {"call_id": job.call_id, "report": report}, artifact

        with ThreadPoolExecutor(max_workers=self.max_concurrent_jobs) as pool:
            outcomes = list(pool.map(compile_chunk, chunks))
        shards = [o[0] for o in outcomes]
        artifacts = [o[1] for o in outcomes]
        by_id = {r["id"]: r for s in shards for r in s["report"]["results"]}
        report = {
            "schema": "vfunc.thunderkittens-compilation/v1",
            "variants": variants,
            "results": [by_id[v["id"]] for v in variants],
            "shards": shards,
            "call_ids": [probe.call_id, *[s["call_id"] for s in shards]],
            "target_probe_call_id": probe.call_id,
            "headers_artifact_id": header.artifact_id,
            "nvcc_flags": list(self.nvcc_flags),
            "headers_sha256": self.headers_sha256,
            "input_metadata": inputs["metadata"],
            "target": target,
            "specialization": key,
        }
        if not any(r["status"] == "compiled" for r in report["results"]) or (
            self.tuning is None and any(r["status"] != "compiled" for r in report["results"])
        ):
            raise ThunderKittensCompilationError(report)
        cached = {"artifacts": artifacts, "target": target}
        if self.tuning is not None:
            report["benchmark"] = benchmark_shards(
                replace(self.gpu_function, handler=triton_quick_runner.benchmark_cycle),
                variants,
                self._request(cached, inputs, self.tuning),
                self.tuning,
                report["call_ids"],
            )
            winner = report["benchmark"]["best_configuration"]
        else:
            winner = variants[0]
        # Keep only the winning shard; binary loader reads only the winner's cubin.
        cached["artifacts"] = [
            a for s, a in outcomes if any(r["id"] == winner["id"] for r in s["report"]["results"])
        ]
        cached["variant"] = winner
        report["batch_wall_seconds"] = time.perf_counter() - started
        self._results[key] = _freeze(report)
        return cached
