"""Managed C++ CUTLASS kernels with configuration search and replicated timing."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from typing import Any

from . import cutlass_runner, triton_quick_runner
from .app import Function, active_function_scope
from .artifacts import ArtifactOutput, ArtifactRef
from .errors import GfaasError
from .kernel import Kernel, KernelBenchmark
from .triton_inputs import apply_writes, snapshot_inputs
from .triton_kernel import _freeze
from .triton_policy import TritonTuning, portable_callable
from .triton_replication import benchmark_shards


@dataclass(frozen=True)
class CutlassTuning(TritonTuning):
    """The shared pruning, ring, replication and benchmark policy for CUTLASS."""


class CutlassCompilationError(GfaasError):
    def __init__(self, report: dict[str, Any]):
        self.report = report
        self.call_ids = report["call_ids"]
        super().__init__("No CUTLASS configuration compiled successfully")


class CutlassKernel(Kernel):
    """Wrap C++ source exporting the documented vfunc_launch tensor ABI.

    Each configuration supplies preprocessor definitions. CPU jobs compile
    shared libraries; GPU jobs tune and execute them. The active app.function
    supplies image, target and resources. Source includes and CUTLASS version
    are supplied by that image or a header archive ArtifactRef.
    """

    compiler: Function
    gpu_function: Function

    def __init__(
        self,
        source: str,
        *,
        argument_names: Sequence[str],
        configurations: Sequence[Mapping[str, bool | int | float]] | None = None,
        tuning: CutlassTuning | None = None,
        headers: ArtifactRef | None = None,
        include_dirs: Sequence[str] = (),
        nvcc_flags: Sequence[str] = (),
        architecture: str | None = None,
        reset_to_zero: Sequence[str] = (),
        restore_value: Sequence[str] = (),
        variants_per_job: int | None = None,
        max_concurrent_jobs: int = 8,
    ) -> None:
        if not isinstance(source, str) or not source.strip():
            raise ValueError("CUTLASS source must be nonempty C++ source")
        if (
            not argument_names
            or len(set(argument_names)) != len(argument_names)
            or any(not isinstance(n, str) or not n.isidentifier() for n in argument_names)
        ):
            raise ValueError("argument_names must be unique identifiers")
        if tuning is not None and not isinstance(tuning, CutlassTuning):
            raise TypeError("tuning must be CutlassTuning")
        if headers is not None and not isinstance(headers, ArtifactRef):
            raise TypeError("headers must be ArtifactRef or None")
        if variants_per_job is not None and (
            type(variants_per_job) is not int or variants_per_job < 1
        ):
            raise ValueError("variants_per_job must be a positive integer")
        if type(max_concurrent_jobs) is not int or not 1 <= max_concurrent_jobs <= 32:
            raise ValueError("max_concurrent_jobs must be between 1 and 32")
        if architecture is not None and not re.fullmatch(r"sm_[0-9]+[af]?", architecture):
            raise ValueError("architecture must be a CUDA sm_ target")
        if any(
            not isinstance(s, str) or not s or "\x00" in s for s in (*include_dirs, *nvcc_flags)
        ):
            raise ValueError("Include directories and compiler flags must be nonempty strings")
        if any(n not in argument_names for n in (*reset_to_zero, *restore_value)) or set(
            reset_to_zero
        ) & set(restore_value):
            raise ValueError("Reset/restore declarations must name distinct supplied arguments")
        configurations = [{}] if configurations is None else configurations
        if not configurations:
            raise ValueError("Supply at least one CUTLASS configuration")
        variants: list[dict[str, Any]] = []
        for config in configurations:
            constants = dict(config)
            for name, value in constants.items():
                if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                    raise ValueError("Configuration keys must be C++ identifiers")
                if type(value) not in (bool, int, float) or not math.isfinite(value):
                    raise ValueError("Configuration values must be finite bool/int/float scalars")
            identity = hashlib.sha256(json.dumps(constants, sort_keys=True).encode()).hexdigest()
            if not any(v["id"] == identity for v in variants):
                variants.append({"id": identity, "constants": constants})
        self.source = source
        self.argument_names = tuple(argument_names)
        self.variants = variants
        self.tuning = tuning or (CutlassTuning() if len(variants) > 1 else None)
        self.headers, self.include_dirs, self.nvcc_flags = (
            headers,
            tuple(include_dirs),
            tuple(nvcc_flags),
        )
        self.architecture = architecture
        self.reset_to_zero, self.restore_value = tuple(reset_to_zero), tuple(restore_value)
        self.variants_per_job, self.max_concurrent_jobs = variants_per_job, max_concurrent_jobs
        self._cache: dict[tuple[int, str], Any] = {}
        self._results: dict[str, Any] = {}
        self._lock = threading.Lock()

    @property
    def tuning_results(self) -> Mapping[str, Any]:
        return _freeze(self._results)

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return self._invoke(None, args, kwargs, None)

    def _invoke(
        self,
        grid: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        options: KernelBenchmark | None,
    ) -> Any:
        if grid is not None:
            raise ValueError("CUTLASS launch geometry belongs in the C++ launcher; use kernel(...)")
        if (
            len(args) > len(self.argument_names)
            or set(kwargs) - set(self.argument_names)
            or set(self.argument_names[: len(args)]) & kwargs.keys()
        ):
            raise TypeError("Invalid CUTLASS launch arguments")
        if set(self.argument_names[: len(args)]) | kwargs.keys() != set(self.argument_names):
            raise TypeError("Missing CUTLASS launch arguments")
        scope = active_function_scope()
        invocation = copy.copy(self)
        function = scope.bind(cutlass_runner.benchmark_cycle)
        image = function._resolve_image()
        gpu, count = function.gpu, function.gpu_count
        if (
            (count is None and gpu is None)
            or (count is not None and count != 1)
            or (gpu and "," in gpu)
        ):
            raise ValueError("Select one GPU target in app.function for CutlassKernel")
        pool = (
            gpu
            if function.gpu_type == "any" and gpu and not gpu.isdigit() and gpu != "any"
            else function.gpu_type
        )
        invocation.gpu_function = replace(
            function, image=image, gpu=None, gpu_count=1, gpu_type=pool, outputs=()
        )
        invocation.compiler = replace(
            scope.bind(cutlass_runner.compile_batch),
            image=image,
            gpu=None,
            gpu_count=0,
            gpu_type=pool,
            cpu_millicores=function.cpu_millicores or 4000,
            memory_bytes=function.memory_bytes or 8 * 1024**3,
            outputs=(ArtifactOutput.directory("compiled-cutlass", "compiled-cutlass"),),
        )
        inputs = snapshot_inputs(args, kwargs)
        bound = dict(zip(self.argument_names, args, strict=False)) | kwargs
        for name in (*self.reset_to_zero, *self.restore_value):
            if not hasattr(bound[name], "untyped_storage"):
                raise TypeError("Reset/restore declarations must refer to tensors")
        identity = {
            "source": self.source,
            "arguments": self.argument_names,
            "variants": self.variants,
            "metadata": inputs["metadata"],
            "image": asdict(image),
            "env": function.env,
            "gpu_type": pool,
            "nvcc_flags": self.nvcc_flags,
            "include_dirs": self.include_dirs,
            "headers": self.headers.artifact_id if self.headers else None,
            "architecture": self.architecture,
            "reset": self.reset_to_zero,
            "restore": self.restore_value,
            "policy": self.tuning.request() if self.tuning else None,
        }
        import cloudpickle

        callback = portable_callable(self.tuning.evaluate if self.tuning else None)
        callbacks = cloudpickle.dumps((None, callback))
        identity["callbacks_sha256"] = hashlib.sha256(callbacks).hexdigest()
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        cache_key = (id(function.app.client), key)
        with self._lock:
            reused = cache_key in self._cache
            if not reused:
                probe = replace(
                    invocation.gpu_function, handler=triton_quick_runner.probe_target
                ).spawn()
                target = probe.wait()
                if (
                    self.architecture
                    and int(self.architecture.removeprefix("sm_").rstrip("af")) != target["arch"]
                ):
                    raise ValueError("Configured architecture does not match the selected GPU")
                artifacts, report = invocation._compile(target)
                report["call_ids"].insert(0, probe.call_id)
                request = invocation._request(inputs, callbacks, target, artifacts)
                if self.tuning:
                    report["benchmark"] = benchmark_shards(
                        invocation.gpu_function,
                        self.variants,
                        request,
                        self.tuning,
                        report["call_ids"],
                    )
                    winner = report["benchmark"]["best_configuration"]
                else:
                    winner = self.variants[0]
                artifact = next(a for a, ids in artifacts if winner["id"] in ids)
                cached = {"variant": winner, "artifacts": [artifact], "target": target}
                self._cache[cache_key] = cached
                report.update(specialization=key, input_metadata=inputs["metadata"])
                self._results[key] = _freeze(report)
            cached = self._cache[cache_key]
            if options is None:
                result = (
                    replace(invocation.gpu_function, handler=cutlass_runner.execute_winner)
                    .spawn(
                        source=self.source,
                        inputs=inputs,
                        argument_names=list(self.argument_names),
                        **cached,
                    )
                    .wait()
                )
                apply_writes(result, args, kwargs)
                return None
            policy = options.tuning_policy()
            request = invocation._request(
                inputs, callbacks, cached["target"], [(a, []) for a in cached["artifacts"]]
            )
            request["policy"] = policy.request()
            report = benchmark_shards(
                invocation.gpu_function, [cached["variant"]], request, policy, [], single_job=True
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
            return _freeze(report)

    def _request(
        self, inputs: Any, callbacks: bytes, target: Any, artifacts: Any
    ) -> dict[str, Any]:
        return {
            "backend": "cutlass",
            "source": self.source,
            "inputs": inputs,
            "metadata": inputs["metadata"],
            "callbacks": callbacks,
            "target": target,
            "artifacts": [a for a, _ in artifacts],
            "argument_names": list(self.argument_names),
            "reset_arguments": list(self.reset_to_zero),
            "restore_arguments": list(self.restore_value),
            "policy": self.tuning.request() if self.tuning else {},
        }

    def _compile(self, target: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        size = self.variants_per_job or max(
            1, math.ceil(len(self.variants) / self.max_concurrent_jobs)
        )
        chunks = [self.variants[i : i + size] for i in range(0, len(self.variants), size)]

        def submit(chunk: Any) -> Any:
            job = self.compiler.spawn(
                source=self.source,
                variants=json.dumps(chunk),
                target=target,
                architecture=self.architecture,
                nvcc_flags=list(self.nvcc_flags),
                include_dirs=list(self.include_dirs),
                headers=self.headers,
                workers=min(32, max(1, (self.compiler.cpu_millicores or 1000) // 1000)),
            )
            report = job.wait()
            if {r["id"] for r in report["results"]} != {v["id"] for v in chunk}:
                raise GfaasError("Incomplete CUTLASS compilation results")
            result = self.compiler.app.client.get_call_result(job.call_id)
            output = next(a for a in result["artifacts"] if a["name"] == "compiled-cutlass")
            return ArtifactRef(output["artifact_id"]), job.call_id, report

        with ThreadPoolExecutor(max_workers=self.max_concurrent_jobs) as pool:
            outcomes = list(pool.map(submit, chunks))
        report = {
            "schema": "vfunc.cutlass-kernel/v1",
            "variants": self.variants,
            "results": [r for _, _, shard in outcomes for r in shard["results"]],
            "call_ids": [call for _, call, _ in outcomes],
            "shards": [{"call_id": call, "report": shard} for _, call, shard in outcomes],
        }
        if not any(r["status"] == "compiled" for r in report["results"]):
            raise CutlassCompilationError(report)
        return [
            (artifact, [r["id"] for r in shard["results"]]) for artifact, _, shard in outcomes
        ], report
