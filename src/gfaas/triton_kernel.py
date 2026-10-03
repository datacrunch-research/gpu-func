"""Normal Triton launch syntax backed by CPU-only batch compilation."""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import replace
from typing import Any

from . import triton_compiler_runner
from .app import App
from .artifacts import ArtifactOutput
from .errors import GfaasError
from .image import Image
from .triton_compat import (
    UnsupportedTritonKernelError,
    argument_type,
    source_bundle,
    validate_kernel,
)


class TritonCompilationError(GfaasError):
    """One or more variants failed; successful artifacts remain available."""

    def __init__(self, report: dict[str, Any], call_id: str | None) -> None:
        self.report, self.call_id = report, call_id
        self.call_ids = report.get("call_ids", [call_id] if call_id else [])
        super().__init__(
            f"Triton compilation failed for one or more variants; {len(self.call_ids)} compiler Calls"
        )


class TritonExecutionNotImplementedError(GfaasError):
    """Compilation completed successfully; this API deliberately never launches."""

    def __init__(self, report: dict[str, Any], call_id: str | None) -> None:
        self.report, self.call_id = report, call_id
        self.call_ids = report.get("call_ids", [call_id] if call_id else [])
        super().__init__(
            f"Compiled {len(report['results'])} variants across {len(self.call_ids)} compiler Calls; "
            "execution is not implemented"
        )


class _IncompleteCompilerReportError(GfaasError):
    def __init__(self, call_id: str) -> None:
        self.call_id = call_id
        super().__init__(f"Compiler returned an incomplete variant report; Call {call_id}")


class TritonKernel:
    """Wrap plain jit or vanilla autotune(jit), compile on call, and never execute.

    Each call blocks until all configurations have finished compilation. Errors
    expose ``report``, ``call_id``, and ``call_ids``; each Call publishes a named
    ``compiled-triton`` Artifact. Neither the grid callable nor tensor contents
    are evaluated or uploaded. Runtime scalar and pointer alignment specialization
    are deliberately conservative in this first implementation.
    """

    def __init__(
        self,
        kernel: Any,
        *,
        app: App,
        target_arch: int,
        image: Image | None = None,
        gpu_type: str = "gb300",
        cpu_millicores: int = 16000,
        memory_bytes: int = 4 * 1024**3,
        timeout: int = 300,
        capacity_wait: int | None = None,
        ephemeral_storage_bytes: int | None = None,
        shared_memory_bytes: int | None = None,
        max_log_bytes: int | None = None,
        max_output_bytes: int | None = None,
        env: dict[str, str] | None = None,
        variants_per_job: int | None = None,
        max_concurrent_jobs: int = 8,
        cache_compression_level: int = 1,
    ) -> None:
        validate_kernel(kernel)
        if not 1000 <= cpu_millicores <= 32000 or not 70 <= target_arch <= 999:
            raise ValueError("Invalid compilation CPU resources or CUDA target architecture")
        if memory_bytes < 1024**3 or not 1 <= timeout <= 86400:
            raise ValueError("Invalid compiler memory or execution timeout")
        if not 0 <= cache_compression_level <= 9:
            raise ValueError("Invalid cache compression level")
        if variants_per_job is not None and variants_per_job < 1:
            raise ValueError("Invalid variants per compiler job")
        if not 1 <= max_concurrent_jobs <= 32:
            raise ValueError("Invalid concurrent compiler jobs")
        self.kernel, self.app, self.target_arch = kernel, app, target_arch
        self.variants_per_job = variants_per_job
        self.max_concurrent_jobs = max_concurrent_jobs
        self.cache_compression_level = cache_compression_level
        # Use the same environment resolution and submission path as ordinary
        # @app.function jobs. The App owns its client and authentication settings.
        self.compiler = app.function(
            image=image,
            gpu_count=0,
            gpu_type=gpu_type,
            cpu_millicores=cpu_millicores,
            memory_bytes=memory_bytes,
            timeout=timeout,
            capacity_wait=capacity_wait,
            ephemeral_storage_bytes=ephemeral_storage_bytes,
            shared_memory_bytes=shared_memory_bytes,
            max_log_bytes=max_log_bytes,
            max_output_bytes=max_output_bytes,
            env=env,
            outputs=(
                ArtifactOutput.directory(
                    "compiled-triton", "compiled-triton", publish_on_failure=True
                ),
            ),
        )(triton_compiler_runner.compile_batch)
        self.compiler._resolve_image()

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

    def __getitem__(self, grid: Any) -> Any:
        # Accept normal launch syntax, but do not call a potentially effectful grid.
        def launch(*args: Any, **kwargs: Any) -> None:
            self._compile(args, kwargs)

        return launch

    def _compile(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
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
        # Resolve once before fan-out; concurrent jobs share the App's client.
        _ = self.app.client
        self._dispatch(source, jit.fn.__name__, triton.__version__, variants)

    def _dispatch(
        self,
        source: str,
        kernel_name: str,
        triton_version: str,
        variants: list[dict[str, Any]],
    ) -> None:
        started = time.perf_counter()
        call_id: str | None
        job_count = self._job_count(len(variants))
        if job_count == 1:
            call_id, report = self._submit_shard(source, kernel_name, triton_version, variants)
        else:
            # Hash order spreads neighboring tile/configuration families across
            # jobs. Final results still follow the user's original order.
            ordered = sorted(variants, key=lambda v: v["id"])
            chunks = [ordered[i::job_count] for i in range(job_count)]

            def submit(chunk: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
                return self._submit_shard(source, kernel_name, triton_version, chunk)

            outcomes: list[tuple[str | None, dict[str, Any]]] = []
            completed: dict[int, tuple[str | None, dict[str, Any]]] = {}
            limit = min(self.max_concurrent_jobs, job_count)
            next_index = 0
            stopped = False
            with ThreadPoolExecutor(max_workers=limit) as pool:
                pending = {}
                while next_index < limit:
                    pending[pool.submit(submit, chunks[next_index])] = next_index
                    next_index += 1
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        index = pending.pop(future)
                        outcome: tuple[str | None, dict[str, Any]]
                        try:
                            outcome = future.result()
                        except Exception as error:
                            outcome = (
                                getattr(error, "call_id", None),
                                {
                                    "job_failed": True,
                                    "results": [
                                        {
                                            "id": v["id"],
                                            "status": "failed",
                                            "diagnostics": f"Compiler job could not complete: {type(error).__name__}: {error}",
                                        }
                                        for v in chunks[index]
                                    ],
                                },
                            )
                        completed[index] = outcome
                        stopped = stopped or bool(outcome[1].get("job_failed"))
                    # Structural job failures stop new submissions; already
                    # running jobs are awaited and their results remain visible.
                    while not stopped and next_index < job_count and len(pending) < limit:
                        pending[pool.submit(submit, chunks[next_index])] = next_index
                        next_index += 1
            for index in range(job_count):
                if index in completed:
                    outcomes.append(completed[index])
                else:
                    outcomes.append(
                        (
                            None,
                            {
                                "not_submitted": True,
                                "results": [
                                    {
                                        "id": v["id"],
                                        "status": "failed",
                                        "diagnostics": "Not submitted after another compiler job failed",
                                    }
                                    for v in chunks[index]
                                ],
                            },
                        )
                    )
            by_id = {row["id"]: row for _, shard in outcomes for row in shard["results"]}
            call_id = next((identity for identity, _ in outcomes if identity), None)
            report = {
                "schema": "vfunc.triton-compilation-batch/v1",
                "variants": variants,
                "results": [by_id[v["id"]] for v in variants],
                "call_ids": [identity for identity, _ in outcomes if identity],
                "shards": [
                    {"call_id": identity, "output_name": "compiled-triton", "report": shard}
                    for identity, shard in outcomes
                ],
                "batch_wall_seconds": time.perf_counter() - started,
                "max_concurrent_jobs": self.max_concurrent_jobs,
                "workers_per_job": self._workers,
            }
        if any(row["status"] != "compiled" for row in report["results"]):
            raise TritonCompilationError(report, call_id)
        raise TritonExecutionNotImplementedError(report, call_id)

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
