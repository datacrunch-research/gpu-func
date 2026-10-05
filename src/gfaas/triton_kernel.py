"""Normal Triton launch syntax backed by batch compilation and replicated tuning."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
from dataclasses import asdict, replace
from threading import RLock
from types import MappingProxyType
from typing import Any

from . import triton_compiler_runner, triton_quick_runner
from .app import Function, active_function_scope
from .artifacts import ArtifactOutput, ArtifactRef
from .errors import GfaasError
from .kernel import Kernel, KernelBenchmark
from .triton_compat import (
    UnsupportedTritonKernelError,
    argument_type,
    reset_arguments,
    source_bundle,
    validate_kernel,
)
from .triton_inputs import apply_writes, snapshot_inputs
from .triton_policy import TritonTuning, portable_callable
from .triton_replication import benchmark_shards


class TritonCompilationError(GfaasError):
    """One or more variants failed; successful artifacts remain available."""

    def __init__(self, report: dict[str, Any], call_id: str | None) -> None:
        self.report, self.call_id = report, call_id
        self.call_ids = report.get("call_ids", [call_id] if call_id else [])
        super().__init__(
            f"Triton compilation failed for one or more variants; {len(self.call_ids)} compiler Calls"
        )


class TritonExecutionNotImplementedError(GfaasError):
    """Current phases completed; the remaining benchmark/execution phases are deferred."""

    def __init__(self, report: dict[str, Any], call_id: str | None) -> None:
        self.report, self.call_id = report, call_id
        self.call_ids = report.get("call_ids", [call_id] if call_id else [])
        super().__init__(
            f"Compiled {len(report['results'])} variants across {len(self.call_ids)} compiler Calls; "
            "the remaining benchmark/execution pipeline is not implemented"
        )


class TritonQuickBenchmarkError(GfaasError):
    def __init__(self, report: dict[str, Any], call_id: str) -> None:
        self.report, self.call_id = report, call_id
        self.call_ids = report["call_ids"]
        super().__init__(
            f"Triton quick benchmark could not find a valid best configuration; Call {call_id}"
        )


class _IncompleteCompilerReportError(GfaasError):
    def __init__(self, call_id: str) -> None:
        self.call_id = call_id
        super().__init__(f"Compiler returned an incomplete variant report; Call {call_id}")


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


class TritonKernel(Kernel):
    """Compile, tune, and execute using supplied tensors; cache results by specialization.

    Launches return None and copy remote tensor writes into the original tensors.
    Read immutable reports through tuning_results. Calls capture app.function settings.
    """

    compiler: Function
    gpu_function: Function
    grid: Any

    def __init__(
        self,
        kernel: Any,
        *,
        tuning: TritonTuning | None = None,
        variants_per_job: int | None = None,
        max_concurrent_jobs: int = 8,
        cache_compression_level: int = 1,
    ) -> None:
        _, configurations = validate_kernel(kernel)
        if variants_per_job is not None and variants_per_job < 1:
            raise ValueError("Invalid variants per compiler job")
        if not 1 <= max_concurrent_jobs <= 32:
            raise ValueError("Invalid concurrent compiler jobs")
        if not 0 <= cache_compression_level <= 9:
            raise ValueError("Invalid cache compression level")
        self.kernel = kernel
        self.tuning = (
            tuning
            if tuning is not None
            else (
                TritonTuning() if configurations is not None and len(configurations) > 1 else None
            )
        )
        self.variants_per_job = variants_per_job
        self.max_concurrent_jobs = max_concurrent_jobs
        self.cache_compression_level = cache_compression_level
        self._cache: dict[tuple[int, str], Any] = {}
        self._results: dict[str, Any] = {}
        self._lock = RLock()

    @property
    def tuning_results(self) -> Any:
        """Read-only reports indexed by an opaque specialization digest."""
        with self._lock:
            return MappingProxyType(dict(self._results))

    def _job_count(self, variant_count: int) -> int:
        if self.variants_per_job is not None:
            return (variant_count + self.variants_per_job - 1) // self.variants_per_job
        # Aim for 128 variants/job, fill a dispatch window before another wave,
        # and cap chunks at 256 for larger batches.
        return max(
            (variant_count + 255) // 256,
            min((variant_count + 127) // 128, self.max_concurrent_jobs),
        )

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
        scope = active_function_scope()
        invocation = copy.copy(self)
        invocation.compiler = scope.bind(triton_compiler_runner.compile_batch)
        invocation.gpu_function = scope.bind(triton_quick_runner.probe_target)
        image = invocation.compiler._resolve_image()
        invocation.compiler = replace(invocation.compiler, image=image)
        invocation.gpu_function = replace(invocation.gpu_function, image=image)
        gpu = invocation.gpu_function.gpu
        count = invocation.gpu_function.gpu_count
        if count is None and gpu is None or count is not None and count != 1:
            raise ValueError("Select exactly one GPU target in app.function for TritonKernel")
        if gpu and "," in gpu:
            raise ValueError("TritonKernel currently supports a single GPU target")
        pool = invocation.gpu_function.gpu_type
        if pool == "any" and gpu and not gpu.isdigit() and gpu != "any":
            pool = gpu
        invocation.gpu_function = replace(
            invocation.gpu_function, gpu=None, gpu_count=1, gpu_type=pool, outputs=()
        )
        invocation.compiler = replace(
            invocation.compiler,
            gpu=None,
            gpu_count=0,
            gpu_type=pool,
            cpu_millicores=(
                16000
                if invocation.compiler.cpu_millicores is None
                else invocation.compiler.cpu_millicores
            ),
            memory_bytes=(
                4 * 1024**3
                if invocation.compiler.memory_bytes is None
                else invocation.compiler.memory_bytes
            ),
            outputs=(
                ArtifactOutput.directory(
                    "compiled-triton", "compiled-triton", publish_on_failure=True
                ),
            ),
        )
        if not 1000 <= (invocation.compiler.cpu_millicores or 0) <= 32000:
            raise ValueError("Compiler CPU budget must be between 1000 and 32000 millicores")
        invocation.grid = grid
        # Serialize duplicate specialization requests so they do not tune twice.
        with self._lock:
            if options is None:
                return invocation._compile(args, kwargs)
            return invocation._benchmark(args, kwargs, options)

    def _compile(
        self, args: tuple[Any, ...], kwargs: dict[str, Any], *, execute: bool = True
    ) -> Any:
        import triton  # type: ignore[import-not-found]

        jit, configs = validate_kernel(self.kernel)  # Revalidate mutable wrappers at invocation.
        if not hasattr(jit, "params") or not callable(getattr(jit, "fn", None)):
            raise UnsupportedTritonKernelError("Unrecognized JIT signature interface")
        parameters = {p.name: p for p in jit.params}
        signature = inspect.signature(jit.fn)
        argument_kwargs = {k: v for k, v in kwargs.items() if k in parameters}
        options = {k: v for k, v in kwargs.items() if k not in parameters}
        forbidden = {
            "stream",
            "warmup",
            "launch_metadata",
            "ir_override",
            "pre_hook",
            "post_hook",
            "prune_configs_by",
            "reset_to_zero",
            "restore_value",
            "do_bench",
            "cache_results",
            "rep",
            "use_cuda_graph",
        } & options.keys()
        if forbidden:
            raise UnsupportedTritonKernelError(f"Unsupported launch modifiers: {sorted(forbidden)}")
        variants: list[dict[str, Any]] = []
        variant_ids: set[str] = set()
        for config in configs if configs is not None else [{}]:
            config_arguments = {k: v for k, v in config.items() if k in parameters}
            # Bind once before merging to catch positional as well as keyword conflicts.
            supplied = signature.bind_partial(*args, **argument_kwargs)
            overlap = supplied.arguments.keys() & config_arguments.keys()
            if overlap:
                raise ValueError(
                    f"Configuration conflicts with supplied arguments: {sorted(overlap)}"
                )
            bound = signature.bind(*args, **argument_kwargs, **config_arguments)
            bound.apply_defaults()
            types, constants = {}, {}
            for name, value in bound.arguments.items():
                types[name] = argument_type(value, parameters[name])
                if types[name] == "constexpr":
                    if value is not None and type(value) not in (bool, int, float, str):
                        raise UnsupportedTritonKernelError(f"Unsupported constexpr value: {name}")
                    constants[name] = value
            compiler_options = {k: v for k, v in config.items() if k not in parameters}
            overlap = compiler_options.keys() & options.keys()
            if overlap:
                raise ValueError(
                    f"Configuration conflicts with compiler options: {sorted(overlap)}"
                )
            compiler_options.update(options)
            variant: dict[str, Any] = {
                "signature": types,
                "constants": constants,
                "options": compiler_options,
            }
            try:
                identity = json.dumps(variant, sort_keys=True, allow_nan=False)
            except (TypeError, ValueError) as error:
                raise UnsupportedTritonKernelError(
                    "Compiler options must be finite JSON values"
                ) from error
            variant["id"] = hashlib.sha256(identity.encode()).hexdigest()
            if variant["id"] not in variant_ids:
                variants.append(variant)
                variant_ids.add(variant["id"])
        source = source_bundle(jit)
        import cloudpickle

        inputs = snapshot_inputs(args, argument_kwargs)
        metadata = inputs["metadata"]
        callbacks = cloudpickle.dumps(
            (
                portable_callable(self.grid),
                portable_callable(self.tuning.evaluate if self.tuning else None),
            )
        )
        reset, restore = reset_arguments(self.kernel)
        for name in set(reset + restore):
            value = bound.arguments.get(name)
            if value is not None and not hasattr(value, "untyped_storage"):
                raise ValueError(
                    "Reset/restore declarations must refer to tensor arguments or None"
                )
        specialization = {
            "source": source,
            "variants": variants,
            "metadata": metadata,
            "image": asdict(self.compiler._resolve_image()),
            "gpu_type": self.gpu_function.gpu_type,
            "env": self.gpu_function.env,
            "triton_version": triton.__version__,
            "reset": reset,
            "restore": restore,
            "callbacks": hashlib.sha256(callbacks).hexdigest(),
            "tuning": self.tuning.request() if self.tuning else None,
            "replication_factor": self.tuning.replication_factor if self.tuning else 1,
        }
        key = hashlib.sha256(
            json.dumps(specialization, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        # Artifacts belong to a particular client/account; never reuse across clients.
        cache_key = (id(self.compiler.app.client), key)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if execute:
                return self._execute(cached, inputs, args, argument_kwargs)
            return cached, inputs, key, True
        probe = self.gpu_function.spawn()
        target = probe.wait()
        if not isinstance(target, dict) or target.get("backend") != "cuda":
            raise GfaasError("GPU target discovery returned an invalid target")
        self.target_arch = target["arch"]
        try:
            report, call_id = self._dispatch(source, jit.fn.__name__, triton.__version__, variants)
        except (TritonCompilationError, TritonExecutionNotImplementedError) as error:
            error.report["target_probe_call_id"] = probe.call_id
            error.report["call_ids"] = [probe.call_id, *error.call_ids]
            error.call_ids = error.report["call_ids"]
            raise
        compile_ids = report.get("call_ids", [call_id])
        report["call_ids"] = [probe.call_id, *compile_ids]
        report["target_probe_call_id"] = probe.call_id
        report["next_phase"] = "execution"
        artifacts = []
        artifact_variants = {}
        for identity in compile_ids:
            result = self.compiler.app.client.get_call_result(identity)
            output = next(a for a in result["artifacts"] if a.get("name") == "compiled-triton")
            artifact = ArtifactRef(output["artifact_id"])
            artifacts.append(artifact)
            shard = next((s for s in report.get("shards", []) if s["call_id"] == identity), None)
            if shard:
                for row in shard["report"]["results"]:
                    artifact_variants[row["id"]] = artifact
        if self.tuning is not None:
            benchmark = replace(self.gpu_function, handler=triton_quick_runner.benchmark_cycle)
            report["benchmark"] = benchmark_shards(
                benchmark,
                variants,
                {
                    "source": source,
                    "kernel_name": jit.fn.__name__,
                    "artifacts": artifacts,
                    "metadata": metadata,
                    "callbacks": callbacks,
                    "inputs": inputs,
                    "reset_arguments": reset,
                    "restore_arguments": restore,
                    "argument_names": jit.arg_names,
                    "target": target,
                    "triton_version": triton.__version__,
                    "policy": self.tuning.request(),
                },
                self.tuning,
                report["call_ids"],
            )
            winner = report["benchmark"]["best_configuration"]
        elif len(variants) == 1:
            winner = variants[0]
        else:
            raise ValueError("Multiple configurations require TritonTuning")
        cached = {
            "source": source,
            "kernel_name": jit.fn.__name__,
            "variant": winner,
            "artifacts": [artifact_variants[winner["id"]]]
            if winner["id"] in artifact_variants
            else artifacts,
            "callbacks": callbacks,
            "target": target,
            "triton_version": triton.__version__,
        }
        report.pop("next_phase", None)
        report["specialization"] = key
        report["input_metadata"] = metadata
        report["argument_names"] = jit.arg_names
        report["target"] = target
        self._cache[cache_key] = cached
        self._results[key] = _freeze(report)
        if execute:
            return self._execute(cached, inputs, args, argument_kwargs)
        return cached, inputs, key, False

    def _benchmark(
        self, args: tuple[Any, ...], kwargs: dict[str, Any], options: KernelBenchmark
    ) -> Any:
        cached, inputs, key, reused = self._compile(args, kwargs, execute=False)
        import cloudpickle

        grid, evaluate = cloudpickle.loads(cached["callbacks"])
        reset, restore = reset_arguments(self.kernel)
        jit, _ = validate_kernel(self.kernel)
        policy = options.tuning_policy()
        function = replace(self.gpu_function, handler=triton_quick_runner.benchmark_selected)
        calls: list[str] = []
        report = benchmark_shards(
            function,
            [cached["variant"]],
            {
                "source": cached["source"],
                "kernel_name": cached["kernel_name"],
                "artifacts": cached["artifacts"],
                "metadata": inputs["metadata"],
                "callbacks": cloudpickle.dumps((grid, evaluate)),
                "inputs": inputs,
                "reset_arguments": reset,
                "restore_arguments": restore,
                "argument_names": jit.arg_names,
                "target": cached["target"],
                "triton_version": cached["triton_version"],
                "policy": policy.request(),
            },
            policy,
            calls,
            single_job=True,
        )
        winner = next(row for row in report["results"] if row["id"] == report["best_id"])
        report.update(
            specialization=key,
            reused_specialization=reused,
            autotuned=not reused and self.tuning is not None,
            runtime_us=report["best_runtime_us"],
            configuration=cached["variant"],
            replicas=winner["replicas"],
            input_metadata=inputs["metadata"],
        )
        return _freeze(report)

    def _execute(
        self,
        cached: dict[str, Any],
        inputs: dict[str, Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        executor = replace(self.gpu_function, handler=triton_quick_runner.execute_winner)
        result = executor.spawn(**cached, inputs=inputs).wait()
        apply_writes(result, args, kwargs)
        return None

    def _dispatch(
        self,
        source: str,
        kernel_name: str,
        triton_version: str,
        variants: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], str | None]:
        from .kernel_compilation import compile_shards

        report, call_id = compile_shards(
            self, source, kernel_name, triton_version, variants, backend="triton"
        )
        failed = any(row["status"] != "compiled" for row in report["results"])
        quick_enabled = self.tuning is not None
        structural = report.get("job_failed") or any(
            s["report"].get("job_failed") for s in report.get("shards", [])
        )
        if failed and (
            not quick_enabled
            or structural
            or not any(r["status"] == "compiled" for r in report["results"])
        ):
            raise TritonCompilationError(report, call_id)
        return report, call_id

    def _submit_shard(
        self,
        source: str,
        kernel_name: str,
        triton_version: str,
        variants: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any]]:
        # Clamp small chunks without mutating the shared Function configuration.
        compiler = replace(
            self.compiler,
            cpu_millicores=min(self.compiler.cpu_millicores or 1000, len(variants) * 1000),
        )
        result = compiler.spawn(
            source=source,
            kernel_name=kernel_name,
            # JSON keeps large sets opaque to the bounded ArtifactRef scanner.
            variants=json.dumps(variants, separators=(",", ":"), allow_nan=False),
            triton_version=triton_version,
            target={"backend": "cuda", "arch": self.target_arch, "warp_size": 32},
            workers=min(self._workers, len(variants)),
            compression_level=self.cache_compression_level,
        )
        try:
            report = result.wait()
        except Exception as error:
            report = {
                "variants": variants,
                "results": [
                    {
                        "id": v["id"],
                        "status": "failed",
                        "diagnostics": f"Compiler job failed: {type(error).__name__}: {error}",
                    }
                    for v in variants
                ],
                "job_failed": True,
            }
        rows = report.get("results") if isinstance(report, dict) else None
        if (
            not isinstance(rows, list)
            or len(rows) != len(variants)
            or any(not isinstance(row, dict) for row in rows)
            or {row.get("id") for row in rows} != {v["id"] for v in variants}
            or any(row.get("status") not in ("compiled", "failed") for row in rows)
        ):
            raise _IncompleteCompilerReportError(result.call_id)
        return result.call_id, report
