"""Managed CuTe host entry points with CPU compilation, tuning and tensor writes."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import shlex
from dataclasses import asdict, replace
from threading import RLock
from types import MappingProxyType
from typing import Any

from . import cute_compiler_runner, cute_runner
from .app import Function, active_function_scope
from .artifacts import ArtifactOutput, ArtifactRef
from .cute_compat import (
    UnsupportedCuteDSLKernelError,
    decode_constants,
    encode_constant,
    kernel_function,
    source_bundle,
)
from .errors import GfaasError
from .kernel import Kernel, KernelBenchmark
from .kernel_compilation import compile_shards
from .triton_inputs import apply_writes, snapshot_inputs
from .triton_kernel import _freeze
from .triton_policy import TritonTuning, portable_callable
from .triton_replication import benchmark_shards


class CuteDSLCompilationError(GfaasError):
    def __init__(self, report: dict[str, Any]):
        self.report = report
        self.call_ids = report.get("call_ids", [])
        super().__init__("CuTe DSL compilation could not produce a runnable configuration")


class CuteDSLKernel(Kernel):
    """Wrap a @cute.jit host function or callable class; launches write supplied tensors.

    Configurations contain host entry-point constexpr keyword arguments. CuTe's host
    entry controls the launch grid. A CUstream parameter named stream is injected.
    """

    compiler: Function
    gpu_function: Function

    def __init__(
        self,
        kernel: Any,
        *,
        configurations: list[dict[str, Any]] | None = None,
        tuning: TritonTuning | None = None,
        variants_per_job: int | None = None,
        max_concurrent_jobs: int = 8,
        compile_options: str = "",
        reset_to_zero: tuple[str, ...] = (),
        restore_value: tuple[str, ...] = (),
    ):
        self.function = kernel_function(kernel)
        self.signature = inspect.signature(self.function)
        self.kernel = kernel
        configs = configurations if configurations is not None else [{}]
        if not configs or any(not isinstance(c, dict) for c in configs):
            raise ValueError("configurations must be a nonempty list of mappings")
        for config in configs:
            for name, value in config.items():
                parameter = self.signature.parameters.get(name)
                if parameter is None or "Constexpr" not in str(parameter.annotation):
                    raise UnsupportedCuteDSLKernelError(
                        f"Configuration {name} must name a cutlass.Constexpr argument"
                    )
                encode_constant(value)
        self.configurations = copy.deepcopy(configs)
        if variants_per_job is not None and (
            type(variants_per_job) is not int or variants_per_job < 1
        ):
            raise ValueError("Invalid variants_per_job")
        if type(max_concurrent_jobs) is not int or not 1 <= max_concurrent_jobs <= 32:
            raise ValueError("Invalid max_concurrent_jobs")
        if not isinstance(compile_options, str):
            raise TypeError("compile_options must be an option string")
        forbidden = {"--gpu-arch", "--host-target", "--enable-tvm-ffi"}
        if any(token.split("=")[0] in forbidden for token in shlex.split(compile_options)):
            raise ValueError("GPU/host target and AOT ABI are managed by vFunc")
        for field, names in (("reset_to_zero", reset_to_zero), ("restore_value", restore_value)):
            if not isinstance(names, (tuple, list)) or any(
                n not in self.signature.parameters for n in names
            ):
                raise ValueError(f"{field} must name host entry-point arguments")
        if "stream" in self.signature.parameters and "CUstream" not in str(
            self.signature.parameters["stream"].annotation
        ):
            raise UnsupportedCuteDSLKernelError(
                "The reserved stream argument must have CUstream typing"
            )
        self.tuning = tuning if tuning is not None else TritonTuning() if len(configs) > 1 else None
        if self.tuning is not None and not isinstance(self.tuning, TritonTuning):
            raise TypeError("tuning must be KernelTuning")
        self.variants_per_job, self.max_concurrent_jobs = variants_per_job, max_concurrent_jobs
        self.compile_options = compile_options
        self.reset_to_zero, self.restore_value = list(reset_to_zero), list(restore_value)
        self._lock = RLock()
        self._cache: dict[tuple[int, str], Any] = {}
        self._results: dict[str, Any] = {}

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return self._invoke(None, args, kwargs, None)

    @property
    def tuning_results(self) -> Any:
        with self._lock:
            return MappingProxyType(dict(self._results))

    def _job_count(self, count: int) -> int:
        if self.variants_per_job is not None:
            return (count + self.variants_per_job - 1) // self.variants_per_job
        return min(count, self.max_concurrent_jobs)

    @property
    def _workers(self) -> int:
        return max(1, min(32, (self.compiler.cpu_millicores or 1000) // 1000))

    def _invoke(
        self,
        grid: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        options: KernelBenchmark | None,
    ) -> Any:
        if grid is not None:
            raise ValueError("CuTe host entries define their own grid; call kernel(...) directly")
        invocation = copy.copy(self)
        scope = active_function_scope()
        invocation.compiler = scope.bind(cute_compiler_runner.compile_batch)
        invocation.gpu_function = scope.bind(cute_runner.probe_environment)
        gpu, count = invocation.gpu_function.gpu, invocation.gpu_function.gpu_count
        if (
            (count is None and gpu is None)
            or (count is not None and count != 1)
            or (gpu and "," in gpu)
        ):
            raise ValueError("Select one GPU target in app.function for CuteDSLKernel")
        pool = invocation.gpu_function.gpu_type
        if pool == "any" and gpu and not gpu.isdigit() and gpu != "any":
            pool = gpu
        image = invocation.compiler._resolve_image()
        invocation.gpu_function = replace(
            invocation.gpu_function, image=image, gpu=None, gpu_count=1, gpu_type=pool, outputs=()
        )
        invocation.compiler = replace(
            invocation.compiler,
            image=image,
            gpu=None,
            gpu_count=0,
            gpu_type=pool,
            cpu_millicores=invocation.compiler.cpu_millicores or 4000,
            memory_bytes=invocation.compiler.memory_bytes or 8 * 1024**3,
            outputs=(
                ArtifactOutput.directory("compiled-cute", "compiled-cute", publish_on_failure=True),
            ),
        )
        with self._lock:
            cached, inputs, key, reused = invocation._prepare(args, kwargs)
            if options is None:
                executor = replace(invocation.gpu_function, handler=cute_runner.execute_winner)
                result = executor.spawn(**cached, inputs=inputs).wait()
                apply_writes(result, args, kwargs)
                return None
            policy = options.tuning_policy()
            function = replace(invocation.gpu_function, handler=cute_runner.benchmark_selected)
            report = benchmark_shards(
                function,
                [cached["variant"]],
                {
                    **{k: v for k, v in cached.items() if k != "variant"},
                    "inputs": inputs,
                    "metadata": inputs["metadata"],
                    "reset_arguments": self.reset_to_zero,
                    "restore_arguments": self.restore_value,
                    "policy": policy.request(),
                },
                policy,
                [],
            )
            winner = next(r for r in report["results"] if r["id"] == report["best_id"])
            report.update(
                specialization=key,
                reused_specialization=reused,
                autotuned=not reused and self.tuning is not None,
                runtime_us=report["best_runtime_us"],
                configuration=cached["variant"],
                replicas=winner["replicas"],
                input_metadata=inputs["metadata"],
            )
            return _freeze(decode_constants(report))

    def _prepare(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        import cloudpickle

        from .cute_backend import version

        bound = self.signature.bind_partial(*args, **kwargs)
        if "stream" in bound.arguments:
            raise ValueError("vFunc manages the CUDA stream")
        bound.arguments.pop("stream", None)
        names = [name for name in self.signature.parameters if name != "stream"]
        # Retain exactly the supplied argument layout for storage/writeback and callbacks.
        for name in set(self.reset_to_zero + self.restore_value):
            value = bound.arguments.get(name)
            if value is not None and not hasattr(value, "untyped_storage"):
                raise ValueError("Reset/restore declarations must name tensor inputs")
        inputs = snapshot_inputs(args, kwargs)
        self.metadata, self.argument_names = inputs["metadata"], names
        source, kernel_name = source_bundle(self.kernel)
        variants = []
        seen = set()
        for config in self.configurations:
            if bound.arguments.keys() & config.keys():
                raise ValueError("Configuration constants conflict with invocation arguments")
            values = {**bound.arguments, **config}
            if "stream" in self.signature.parameters:
                values["stream"] = None
            complete = self.signature.bind(**values)
            complete.apply_defaults()
            signature, constants = {}, {}
            for name, parameter in self.signature.parameters.items():
                if name == "stream":
                    signature[name] = "stream"
                elif "Constexpr" in str(parameter.annotation):
                    signature[name] = "constexpr"
                    constants[name] = encode_constant(complete.arguments[name])
                else:
                    signature[name] = "runtime"
            variant = {
                "signature": signature,
                "constants": constants,
                "options": {},
                "defaults": {
                    n: complete.arguments[n]
                    for n, kind in signature.items()
                    if kind == "runtime" and n not in bound.arguments
                },
            }
            identity = hashlib.sha256(
                json.dumps(variant, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
            if identity not in seen:
                variants.append({**variant, "id": identity})
                seen.add(identity)
        callbacks = cloudpickle.dumps(
            (None, portable_callable(self.tuning.evaluate if self.tuning else None))
        )
        library_version = version()
        specification = {
            "source": source,
            "variants": variants,
            "metadata": self.metadata,
            "cute_version": library_version,
            "image": asdict(self.compiler._resolve_image()),
            "gpu_type": self.gpu_function.gpu_type,
            "env": self.gpu_function.env,
            "compile_options": self.compile_options,
            "callbacks": hashlib.sha256(callbacks).hexdigest(),
            "reset": self.reset_to_zero,
            "restore": self.restore_value,
            "tuning": self.tuning.request() if self.tuning else None,
            "replication": self.tuning.replication_factor if self.tuning else 1,
        }
        key = hashlib.sha256(
            json.dumps(specification, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        cache_key = (id(self.compiler.app.client), key)
        if cache_key in self._cache:
            return self._cache[cache_key], inputs, key, True
        probe = self.gpu_function.spawn()
        environment = probe.wait()
        if environment["cute_version"] != library_version:
            raise ValueError("Client and execution image must use the same CuTe DSL version")
        self.target = environment["target"]
        report, call_id = compile_shards(
            self, source, kernel_name, library_version, variants, backend="cute"
        )
        report["call_ids"] = [probe.call_id, *report.get("call_ids", [call_id])]
        if (
            any(s["report"].get("job_failed") for s in report.get("shards", []))
            or not any(r["status"] == "compiled" for r in report["results"])
            or (self.tuning is None and any(r["status"] != "compiled" for r in report["results"]))
        ):
            raise CuteDSLCompilationError(report)
        artifacts, mapping = [], {}
        for identity in report["call_ids"][1:]:
            result = self.compiler.app.client.get_call_result(identity)
            output = next(a for a in result["artifacts"] if a.get("name") == "compiled-cute")
            artifact = ArtifactRef(output["artifact_id"])
            artifacts.append(artifact)
            shard = next(
                (s["report"] for s in report.get("shards", []) if s["call_id"] == identity), report
            )
            for row in shard["results"]:
                mapping[row["id"]] = artifact
        request = {
            "source": source,
            "kernel_name": kernel_name,
            "artifacts": artifacts,
            "callbacks": callbacks,
            "target": self.target,
            "cute_version": library_version,
            "argument_names": names,
        }
        if self.tuning is not None:
            benchmark = replace(self.gpu_function, handler=cute_runner.benchmark_cycle)
            report["benchmark"] = benchmark_shards(
                benchmark,
                variants,
                {
                    **request,
                    "inputs": inputs,
                    "metadata": self.metadata,
                    "reset_arguments": self.reset_to_zero,
                    "restore_arguments": self.restore_value,
                    "policy": self.tuning.request(),
                },
                self.tuning,
                report["call_ids"],
            )
            winner = report["benchmark"]["best_configuration"]
        else:
            winner = variants[0]
        cached = {**request, "variant": winner, "artifacts": [mapping[winner["id"]]]}
        report.update(specialization=key, input_metadata=self.metadata, target=self.target)
        self._cache[cache_key] = cached
        self._results[key] = _freeze(decode_constants(report))
        return cached, inputs, key, False

    def _submit_shard(
        self, source: str, kernel_name: str, library_version: str, variants: list[dict[str, Any]]
    ) -> Any:
        compiler = replace(
            self.compiler,
            cpu_millicores=min(self.compiler.cpu_millicores or 1000, len(variants) * 1000),
        )
        job = compiler.spawn(
            source=source,
            kernel_name=kernel_name,
            variants=json.dumps(variants),
            cute_version=library_version,
            metadata=self.metadata,
            argument_names=self.argument_names,
            target=self.target,
            workers=min(self._workers, len(variants)),
            compile_options=self.compile_options,
        )
        try:
            report = job.wait()
        except Exception as error:
            report = {
                "job_failed": True,
                "results": [
                    {"id": v["id"], "status": "failed", "diagnostics": str(error)} for v in variants
                ],
            }
        rows = report.get("results", [])
        if (
            len(rows) != len(variants)
            or {r.get("id") for r in rows} != {v["id"] for v in variants}
            or any(r.get("status") not in ("compiled", "failed") for r in rows)
        ):
            raise GfaasError("Incomplete CuTe compiler report")
        return job.call_id, report
