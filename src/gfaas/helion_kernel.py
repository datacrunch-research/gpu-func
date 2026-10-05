"""Helion calls backed by vFunc compilation, tuning, execution and benchmarking."""

from __future__ import annotations

import hashlib
import inspect
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from threading import RLock
from types import MappingProxyType
from typing import Any

from . import helion_compiler_runner, helion_prepare, helion_runner
from .app import active_function_scope
from .artifacts import ArtifactOutput, ArtifactRef
from .errors import GfaasError
from .helion_compat import configuration_dicts, installed_version, source_bundle, validate_kernel
from .kernel import Kernel, KernelBenchmark
from .kernel_results import apply_result
from .triton_inputs import snapshot_inputs
from .triton_kernel import _freeze
from .triton_policy import TritonTuning, portable_callable
from .triton_replication import benchmark_shards


class HelionCompilationError(GfaasError):
    def __init__(self, report: dict[str, Any]):
        self.report = report
        self.call_ids = report["call_ids"]
        super().__init__("No Helion configuration compiled successfully")


class HelionKernel(Kernel):
    """Preserve normal Helion calls and return values using app.function settings.

    Native decorator configs are used unless configs is supplied. With neither,
    the bound kernel's default config is compiled. Tuning reports are immutable
    and indexed by the exact input specialization. No local GPU is required.
    """

    def __init__(
        self,
        kernel: Any,
        *,
        configs: Any = None,
        tuning: TritonTuning | None = None,
        variants_per_job: int | None = None,
        max_concurrent_jobs: int = 8,
        cache_compression_level: int = 1,
    ) -> None:
        validate_kernel(kernel)
        if variants_per_job is not None and (
            type(variants_per_job) is not int or variants_per_job < 1
        ):
            raise ValueError("variants_per_job must be positive")
        if type(max_concurrent_jobs) is not int or not 1 <= max_concurrent_jobs <= 32:
            raise ValueError("max_concurrent_jobs must be between 1 and 32")
        if type(cache_compression_level) is not int or not 0 <= cache_compression_level <= 9:
            raise ValueError("cache_compression_level must be between 0 and 9")
        if tuning is not None and not isinstance(tuning, TritonTuning):
            raise TypeError("tuning must be KernelTuning")
        self.kernel = kernel
        self.configurations = configuration_dicts(kernel.configs if configs is None else configs)
        if configs is not None and not self.configurations:
            raise ValueError("configs must contain at least one configuration")
        self.tuning = tuning or TritonTuning()
        self.variants_per_job = variants_per_job
        self.max_concurrent_jobs = max_concurrent_jobs
        self.cache_compression_level = cache_compression_level
        self._lock = RLock()
        self._cache: dict[tuple[int, str], dict[str, Any]] = {}
        self._results: dict[str, Any] = {}

    @property
    def tuning_results(self) -> Any:
        with self._lock:
            return MappingProxyType(dict(self._results))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._invoke(None, args, kwargs, None)

    def _invoke(
        self,
        grid: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        options: KernelBenchmark | None,
    ) -> Any:
        if grid is not None:
            raise ValueError("Helion kernels determine their own launch grids")
        import cloudpickle

        validate_kernel(self.kernel)
        binding = inspect.signature(self.kernel.fn).bind(*args, **kwargs)
        binding.apply_defaults()
        arguments = tuple(binding.arguments.values())
        names = list(binding.arguments)
        inputs = snapshot_inputs(arguments, {})
        source = source_bundle(self.kernel)
        settings = cloudpickle.dumps(self.kernel.settings.to_dict())
        callbacks = cloudpickle.dumps(portable_callable(self.tuning.evaluate))
        scope = active_function_scope()
        gpu = scope.bind(helion_prepare.prepare)
        image = gpu._resolve_image()
        if (gpu.gpu_count is None and gpu.gpu is None) or (
            gpu.gpu_count is not None and gpu.gpu_count != 1
        ):
            raise ValueError("Select exactly one GPU target in app.function for HelionKernel")
        if gpu.gpu and "," in gpu.gpu:
            raise ValueError("HelionKernel requires a single GPU target")
        pool = gpu.gpu_type
        if pool == "any" and gpu.gpu and not gpu.gpu.isdigit() and gpu.gpu != "any":
            pool = gpu.gpu
        gpu = replace(gpu, image=image, gpu=None, gpu_count=1, gpu_type=pool, outputs=())
        compiler = replace(
            scope.bind(helion_compiler_runner.compile_batch),
            image=image,
            gpu=None,
            gpu_count=0,
            gpu_type=pool,
            cpu_millicores=gpu.cpu_millicores or 16000,
            memory_bytes=gpu.memory_bytes or 8 * 1024**3,
            outputs=(ArtifactOutput.directory("compiled-helion", "compiled-helion"),),
        )
        key = hashlib.sha256(
            json.dumps(
                {
                    "source": source,
                    "metadata": inputs["metadata"],
                    "configs": self.configurations,
                    "image": asdict(image),
                    "gpu_pool": pool,
                    "helion_version": installed_version(),
                    "settings": hashlib.sha256(settings).hexdigest(),
                },
                sort_keys=True,
                allow_nan=False,
            ).encode()
            + callbacks
        ).hexdigest()
        cache_key = (id(compiler.app.client), key)
        with self._lock:
            reused = cache_key in self._cache
            if not reused:
                report = self._prepare_compile_tune(
                    gpu, compiler, source, settings, inputs, callbacks, names, installed_version()
                )
                report["specialization"] = key
                report["input_metadata"] = inputs["metadata"]
                self._cache[cache_key] = report
                self._results[key] = _freeze(report)
            cached = self._cache[cache_key]
            winner = cached["benchmark"]["best_configuration"]
            owner = next(
                s["artifact"]
                for s in cached["shards"]
                if any(r["id"] == winner["id"] for r in s["report"]["results"])
            )
            artifacts = [ArtifactRef(owner)]
            common = {
                "target": cached["target"],
                "triton_version": cached["triton_version"],
                "helion_version": installed_version(),
                "inputs": inputs,
                "artifacts": artifacts,
            }
            if options is not None:
                policy = options.tuning_policy()
                report = benchmark_shards(
                    replace(gpu, handler=helion_runner.benchmark_cycle),
                    [winner],
                    {
                        **common,
                        "callbacks": callbacks,
                        "argument_names": names,
                        "policy": policy.request(),
                    },
                    policy,
                    [],
                    replica_handler=helion_runner.benchmark_replicas,
                    single_job=True,
                )
                row = report["results"][0]
                report.update(
                    specialization=key,
                    reused_specialization=reused,
                    autotuned=not reused,
                    runtime_us=report["best_runtime_us"],
                    configuration=winner,
                    replicas=row["replicas"],
                    input_metadata=inputs["metadata"],
                )
                return _freeze(report)
            job = replace(gpu, handler=helion_runner.execute_winner).spawn(**common, variant=winner)
            return apply_result(job.wait(), arguments)

    def _prepare_compile_tune(
        self,
        gpu: Any,
        compiler: Any,
        source: str,
        settings: bytes,
        inputs: dict[str, Any],
        callbacks: bytes,
        names: list[str],
        helion_version: str,
    ) -> dict[str, Any]:
        job = gpu.spawn(
            source=source,
            kernel_name=self.kernel.fn.__name__,
            settings=settings,
            configurations=self.configurations,
            inputs=inputs,
            helion_version=helion_version,
        )
        prepared = job.wait()
        variants = [v for v in prepared["variants"] if v["status"] == "prepared"]
        report = {
            "schema": "vfunc.helion-kernel/v1",
            "preparation": prepared,
            "call_ids": [job.call_id],
            "shards": [],
            "results": [],
            "target": prepared["target"],
            "triton_version": prepared["triton_version"],
        }
        if not variants:
            raise HelionCompilationError(report)
        size = self.variants_per_job or max(
            1, min(128, (len(variants) + self.max_concurrent_jobs - 1) // self.max_concurrent_jobs)
        )

        def compile_chunk(chunk: list[dict[str, Any]]) -> dict[str, Any]:
            worker = compiler.spawn(
                variants=chunk,
                target=prepared["target"],
                triton_version=prepared["triton_version"],
                workers=max(1, min(32, (compiler.cpu_millicores or 1000) // 1000)),
                compression_level=self.cache_compression_level,
            )
            result = worker.wait()
            artifacts = compiler.app.client.get_call_result(worker.call_id)["artifacts"]
            artifact = next(a["artifact_id"] for a in artifacts if a["name"] == "compiled-helion")
            return {"call_id": worker.call_id, "artifact": artifact, "report": result}

        with ThreadPoolExecutor(max_workers=self.max_concurrent_jobs) as pool:
            shards = list(
                pool.map(
                    compile_chunk, [variants[i : i + size] for i in range(0, len(variants), size)]
                )
            )
        report["shards"] = shards
        report["call_ids"].extend(s["call_id"] for s in shards)
        report["results"] = [r for s in shards for r in s["report"]["results"]]
        compiled = {r["id"] for r in report["results"] if r["status"] == "compiled"}
        choices = [
            {"id": v["id"], "configuration": v["configuration"]}
            for v in variants
            if v["id"] in compiled
        ]
        if not choices:
            raise HelionCompilationError(report)
        report["benchmark"] = benchmark_shards(
            replace(gpu, handler=helion_runner.benchmark_cycle),
            choices,
            {
                "artifacts": [ArtifactRef(s["artifact"]) for s in shards],
                "target": prepared["target"],
                "triton_version": prepared["triton_version"],
                "helion_version": helion_version,
                "inputs": inputs,
                "argument_names": names,
                "callbacks": callbacks,
                "policy": self.tuning.request(),
            },
            self.tuning,
            report["call_ids"],
            replica_handler=helion_runner.benchmark_replicas,
        )
        return report
