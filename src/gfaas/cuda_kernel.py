"""Managed CUDA C++ kernels: compile, tune, execute and benchmark."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from threading import RLock
from types import MappingProxyType
from typing import Any

from . import cuda_kernel_runner, triton_quick_runner
from .app import active_function_scope
from .artifacts import ArtifactOutput, ArtifactRef
from .errors import GfaasError
from .kernel import Kernel, KernelBenchmark
from .triton_inputs import apply_writes, snapshot_inputs
from .triton_kernel import _freeze
from .triton_policy import TritonTuning, portable_callable
from .triton_replication import benchmark_shards

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def dimensions3(value: Any) -> tuple[int, int, int]:
    if type(value) is int:
        value = (value,)
    if not isinstance(value, (tuple, list)) or not 1 <= len(value) <= 3:
        raise ValueError("Launch dimensions must contain one to three positive integers")
    if any(type(v) is not int or not 1 <= v <= 2**31 - 1 for v in value):
        raise ValueError("Launch dimensions must contain positive integers")
    return tuple(value) + (1,) * (3 - len(value))


@dataclass(frozen=True)
class CUDAConfig:
    """Compiler defines, threads per block and dynamic shared-memory bytes."""

    block: tuple[int, ...] = (256,)
    defines: dict[str, int | float] = field(default_factory=dict)
    shared_memory: int = 0

    def __post_init__(self) -> None:
        block = dimensions3(self.block)
        if math.prod(block) > 1024 or block[2] > 64:
            raise ValueError("A CUDA block may contain at most 1024 threads")
        if type(self.shared_memory) is not int or not 0 <= self.shared_memory <= 2**32 - 1:
            raise ValueError("shared_memory must be a nonnegative byte count")
        for name, value in self.defines.items():
            if (
                not _NAME.fullmatch(name)
                or type(value) not in (int, float)
                or not math.isfinite(value)
            ):
                raise ValueError("CUDA defines require identifier names and finite numeric values")
        object.__setattr__(self, "block", block)
        object.__setattr__(self, "defines", MappingProxyType(dict(self.defines)))

    def request(self) -> dict[str, Any]:
        return {
            "block": list(self.block),
            "defines": dict(self.defines),
            "shared_memory": self.shared_memory,
        }


class CUDAKernel(Kernel):
    """CUDA C++ entry point, with explicit ordered ABI types and optional tuning.

    Source must declare an extern "C" __global__ function. Signature order and
    scalar widths must match that function exactly. Pointer args are tensors/None.
    Each configuration compiles with nvcc on CPU and launches with the CUDA driver.
    """

    def __init__(
        self,
        source: str,
        *,
        name: str,
        signature: dict[str, str],
        configs: list[CUDAConfig] | None = None,
        tuning: TritonTuning | None = None,
        reset_to_zero: tuple[str, ...] = (),
        restore_value: tuple[str, ...] = (),
        nvcc_flags: tuple[str, ...] = (),
        variants_per_job: int = 128,
        max_concurrent_jobs: int = 8,
    ):
        if not isinstance(source, str) or not source.strip() or not _NAME.fullmatch(name):
            raise ValueError("CUDAKernel requires source and a valid entry-point name")
        if not signature or any(
            not _NAME.fullmatch(n) or t not in {"pointer", *cuda_kernel_runner.SCALARS}
            for n, t in signature.items()
        ):
            raise ValueError("Signature requires ordered parameter names and supported ABI types")
        if (
            type(variants_per_job) is not int
            or variants_per_job < 1
            or type(max_concurrent_jobs) is not int
            or not 1 <= max_concurrent_jobs <= 32
        ):
            raise ValueError("Invalid compiler grouping/concurrency")
        if tuning is not None and not isinstance(tuning, TritonTuning):
            raise TypeError("tuning must be KernelTuning")
        for name_ in (*reset_to_zero, *restore_value):
            if signature.get(name_) != "pointer":
                raise ValueError("reset/restore names must identify pointer arguments")
        # Compilation phase, target and output are managed by the service.
        # Other nvcc options (including include paths and C++ dialect) pass through.
        managed = {
            "-o",
            "--output-file",
            "-arch",
            "--gpu-architecture",
            "-gencode",
            "--generate-code",
            "-code",
            "--gpu-code",
            "--cubin",
            "-cubin",
            "--ptx",
            "-ptx",
            "--fatbin",
            "-fatbin",
            "-c",
            "--compile",
            "-dc",
            "--device-c",
            "-dlink",
            "--device-link",
            "--run",
            "-run",
            "--lib",
            "-lib",
            "-cuda",
            "--cuda",
        }
        if isinstance(nvcc_flags, str):
            raise TypeError("nvcc_flags must be a sequence of argument strings")
        for flag in nvcc_flags:
            if (
                not isinstance(flag, str)
                or not flag
                or "\0" in flag
                or flag.split("=", 1)[0] in managed
            ):
                raise ValueError(f"Invalid or managed nvcc flag: {flag}")
        self.source, self.name, self.signature = source, name, dict(signature)
        self.configs = tuple(configs if configs is not None else [CUDAConfig()])
        if not self.configs or any(not isinstance(c, CUDAConfig) for c in self.configs):
            raise ValueError("configs must contain CUDAConfig values")
        self.tuning = (
            tuning if tuning is not None else (TritonTuning() if len(self.configs) > 1 else None)
        )
        self.reset_to_zero, self.restore_value = list(reset_to_zero), list(restore_value)
        self.nvcc_flags = tuple(nvcc_flags)
        self.variants_per_job, self.max_concurrent_jobs = variants_per_job, max_concurrent_jobs
        self._cache: dict[Any, Any] = {}
        self._results: dict[str, Any] = {}
        self._lock = RLock()
        self._signature = inspect.Signature(
            [inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD) for n in signature]
        )

    @property
    def tuning_results(self) -> Any:
        return MappingProxyType(self._results)

    def compile(self) -> dict[str, Any]:
        """Compile configurations without inputs; retain artifacts for isolated execution.

        CUDA compilation depends on source, flags and target architecture, rather
        than tensor values. This entry point lets an orchestrator compile before
        submitting an untrusted solution to a credential-free GPU evaluator.
        It does not autotune or populate the execution specialization cache.
        """
        scope = active_function_scope()
        probe = scope.bind(triton_quick_runner.probe_target)
        if probe.gpu_count not in (None, 1) or probe.gpu is None and probe.gpu_count is None:
            raise ValueError("Select exactly one GPU target in app.function for CUDAKernel")
        image = probe._resolve_image()
        pool = probe.gpu_type
        if probe.gpu and not probe.gpu.isdigit() and probe.gpu != "any":
            pool = probe.gpu
        probe = replace(probe, image=image, gpu=None, gpu_count=1, gpu_type=pool, outputs=())
        target = probe.spawn().wait()
        compiler = replace(
            scope.bind(cuda_kernel_runner.compile_batch),
            image=image,
            gpu=None,
            gpu_count=0,
            gpu_type=pool,
            outputs=(ArtifactOutput.directory("compiled-cuda-kernel", "compiled-cuda-kernel"),),
            cpu_millicores=probe.cpu_millicores or 16000,
            memory_bytes=probe.memory_bytes or 4 * 1024**3,
        )
        variants: list[dict[str, Any]] = []
        for config in self.configs:
            variant = {**config.request(), "signature": self.signature}
            variant["id"] = hashlib.sha256(json.dumps(variant, sort_keys=True).encode()).hexdigest()
            if all(v["id"] != variant["id"] for v in variants):
                variants.append(variant)

        def submit(chunk: list[Any]) -> tuple[Any, Any]:
            job = compiler.spawn(
                source=self.source,
                kernel_name=self.name,
                variants=json.dumps(chunk),
                target=target,
                workers=max(1, min(32, (compiler.cpu_millicores or 1000) // 1000)),
                nvcc_flags=list(self.nvcc_flags),
            )
            report = job.wait()
            outputs = compiler.app.client.get_call_result(job.call_id)["artifacts"]
            artifact = ArtifactRef(
                next(a["artifact_id"] for a in outputs if a["name"] == "compiled-cuda-kernel")
            )
            return artifact, {"call_id": job.call_id, "report": report}

        chunks = [
            variants[i : i + self.variants_per_job]
            for i in range(0, len(variants), self.variants_per_job)
        ]
        with ThreadPoolExecutor(max_workers=self.max_concurrent_jobs) as executor:
            compiled = list(executor.map(submit, chunks))
        return {
            "source": self.source,
            "kernel_name": self.name,
            "target": target,
            "variants": variants,
            "artifacts": [a for a, _ in compiled],
            "shards": [s for _, s in compiled],
            "results": [r for _, s in compiled for r in s["report"]["results"]],
        }

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
                    raise TypeError("CUDA pointer arguments must be tensors or None")
            else:
                cuda_kernel_runner.scalar_value(kind, value)
        if not callable(grid):
            grid = dimensions3(grid)
        scope = active_function_scope()
        gpu = scope.bind(triton_quick_runner.probe_target)
        image = gpu._resolve_image()
        if gpu.gpu_count not in (None, 1) or gpu.gpu is None and gpu.gpu_count is None:
            raise ValueError("Select exactly one GPU target in app.function for CUDAKernel")
        pool = gpu.gpu_type
        if gpu.gpu and not gpu.gpu.isdigit() and gpu.gpu != "any":
            pool = gpu.gpu
        gpu = replace(gpu, image=image, gpu=None, gpu_count=1, gpu_type=pool, outputs=())
        compiler = replace(
            scope.bind(cuda_kernel_runner.compile_batch),
            image=image,
            gpu=None,
            gpu_count=0,
            gpu_type=pool,
            outputs=(ArtifactOutput.directory("compiled-cuda-kernel", "compiled-cuda-kernel"),),
            cpu_millicores=gpu.cpu_millicores or 16000,
            memory_bytes=gpu.memory_bytes or 4 * 1024**3,
        )
        inputs = snapshot_inputs(args, kwargs)
        callbacks = cloudpickle.dumps(
            (
                portable_callable(grid),
                portable_callable(self.tuning.evaluate if self.tuning else None),
            )
        )
        variants: list[dict[str, Any]] = []
        for config in self.configs:
            variant = {**config.request(), "signature": self.signature}
            variant["id"] = hashlib.sha256(json.dumps(variant, sort_keys=True).encode()).hexdigest()
            if all(v["id"] != variant["id"] for v in variants):
                variants.append(variant)
        identity = {
            "source": self.source,
            "name": self.name,
            "signature": self.signature,
            "variants": variants,
            "metadata": inputs["metadata"],
            "image": asdict(image),
            "pool": pool,
            "env": gpu.env,
            "flags": self.nvcc_flags,
            "reset": self.reset_to_zero,
            "restore": self.restore_value,
        }
        key = hashlib.sha256(
            json.dumps(identity, sort_keys=True, allow_nan=False).encode() + callbacks
        ).hexdigest()
        cache_key = (id(gpu.app.client), key)
        with self._lock:
            reused = cache_key in self._cache
            if not reused:
                target = gpu.spawn().wait()
                chunks = [
                    variants[i : i + self.variants_per_job]
                    for i in range(0, len(variants), self.variants_per_job)
                ]

                def submit(chunk: list[Any]) -> tuple[Any, Any]:
                    job = compiler.spawn(
                        source=self.source,
                        kernel_name=self.name,
                        variants=json.dumps(chunk),
                        target=target,
                        workers=max(1, min(32, (compiler.cpu_millicores or 1000) // 1000)),
                        nvcc_flags=list(self.nvcc_flags),
                    )
                    report = job.wait()
                    outputs = compiler.app.client.get_call_result(job.call_id)["artifacts"]
                    artifact = ArtifactRef(
                        next(
                            a["artifact_id"] for a in outputs if a["name"] == "compiled-cuda-kernel"
                        )
                    )
                    return artifact, {"call_id": job.call_id, "report": report}

                with ThreadPoolExecutor(max_workers=self.max_concurrent_jobs) as executor:
                    compiled = list(executor.map(submit, chunks))
                report = {
                    "schema": "vfunc.cuda-kernel-compilation/v1",
                    "variants": variants,
                    "results": [r for _, s in compiled for r in s["report"]["results"]],
                    "shards": [s for _, s in compiled],
                    "specialization": key,
                    "input_metadata": inputs["metadata"],
                }
                artifacts = [a for a, _ in compiled]
                valid = {r["id"] for r in report["results"] if r["status"] == "compiled"}
                if not valid:
                    raise GfaasError(
                        f"All CUDA configurations failed compilation: {report['results']}"
                    )
                viable = [v for v in variants if v["id"] in valid]
                request = self._request(inputs, callbacks, target, artifacts)
                if self.tuning:
                    report["benchmark"] = benchmark_shards(
                        replace(gpu, handler=triton_quick_runner.benchmark_cycle),
                        viable,
                        request,
                        self.tuning,
                        [],
                    )
                    winner = report["benchmark"]["best_configuration"]
                else:
                    winner = viable[0]
                selected_artifacts = [
                    a
                    for a, s in compiled
                    if any(r["id"] == winner["id"] for r in s["report"]["results"])
                ]
                cached = {
                    "source": self.source,
                    "kernel_name": self.name,
                    "variant": winner,
                    "artifacts": selected_artifacts,
                    "target": target,
                }
                self._cache[cache_key] = cached
                self._results[key] = _freeze(report)
            cached = self._cache[cache_key]
            if options is None:
                result = (
                    replace(gpu, handler=cuda_kernel_runner.execute)
                    .spawn(**cached, callbacks=callbacks, inputs=inputs)
                    .wait()
                )
                apply_writes(result, args, kwargs)
                return None
            policy = options.tuning_policy()
            measured = benchmark_shards(
                replace(gpu, handler=triton_quick_runner.benchmark_cycle),
                [cached["variant"]],
                self._request(inputs, callbacks, cached["target"], cached["artifacts"], policy),
                policy,
                [],
                single_job=True,
            )
            winner_row = next(r for r in measured["results"] if r["id"] == measured["best_id"])
            measured.update(
                specialization=key,
                reused_specialization=reused,
                autotuned=not reused and self.tuning is not None,
                runtime_us=measured["best_runtime_us"],
                configuration=cached["variant"],
                replicas=winner_row["replicas"],
            )
            return _freeze(measured)

    def _request(
        self,
        inputs: Any,
        callbacks: bytes,
        target: Any,
        artifacts: Any,
        policy: TritonTuning | None = None,
    ) -> dict[str, Any]:
        return {
            "source": self.source,
            "kernel_name": self.name,
            "artifacts": artifacts,
            "metadata": inputs["metadata"],
            "callbacks": callbacks,
            "inputs": inputs,
            "reset_arguments": self.reset_to_zero,
            "restore_arguments": self.restore_value,
            "argument_names": list(self.signature),
            "target": target,
            "triton_version": "",
            "backend": "cuda",
            "policy": (policy or self.tuning or TritonTuning()).request(),
        }
