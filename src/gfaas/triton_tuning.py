"""Submit one staged Call that prepares and benchmarks Triton candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import triton_tuning_runner
from .artifacts import ArtifactOutput
from .client import Client, RemoteResult
from .image import Image
from .stages import CallStage, StageArtifactBinding

TRITON_VERSION = "3.6.0"
_OUTPUT = ArtifactOutput.directory(
    "compiled-triton", "compiled-triton", kind="other", publish_on_failure=False
)
_ARTIFACT_ENV = "GFAAS_COMPILED_TRITON_ARTIFACT_ID"


@dataclass(frozen=True)
class TritonCandidate:
    name: str
    constexprs: dict[str, int | float | bool] = field(default_factory=dict)
    num_warps: int = 4
    num_stages: int = 2

    def request(self) -> dict[str, Any]:
        if not self.name or len(self.name) > 80:
            raise ValueError("candidate name must have 1 to 80 characters")
        if self.num_warps not in (1, 2, 4, 8, 16) or not 1 <= self.num_stages <= 8:
            raise ValueError("candidate num_warps or num_stages is out of range")
        return {
            "name": self.name,
            "constexprs": dict(self.constexprs),
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
        }


@dataclass(frozen=True)
class TritonCase:
    key: str
    params: dict[str, int | float | bool | str] = field(default_factory=dict)
    constexprs: dict[str, int | float | bool] = field(default_factory=dict)

    def request(self) -> dict[str, Any]:
        if not self.key or len(self.key) > 80:
            raise ValueError("tuning key must have 1 to 80 characters")
        return {
            "key": self.key,
            "params": dict(self.params),
            "constexprs": dict(self.constexprs),
        }


def staged_tuning_plan(*, cpu_millicores: int, memory_bytes: int) -> tuple[CallStage, CallStage]:
    """Keep compilation off the GPU and hand one transient tree to the tuning stage."""
    return (
        CallStage(
            name="compile",
            qualname="compile_stage",
            resources={
                "gpu": {"count": 0},
                "cpu_millicores": cpu_millicores,
                "memory_bytes": memory_bytes,
            },
            outputs=(_OUTPUT,),
        ),
        CallStage(
            name="tune",
            qualname="tune_stage",
            resources={"gpu": {"count": 1}},
            artifacts=(StageArtifactBinding("compile", _OUTPUT.name, _ARTIFACT_ENV),),
        ),
    )


def spawn_triton_tuning(
    *,
    source: str,
    kernel_name: str,
    signature: dict[str, str],
    candidates: list[TritonCandidate],
    cases: list[TritonCase],
    target_arch: int,
    image: Image | str,
    gpu: str,
    compile_workers: int = 4,
    compile_cpu_millicores: int = 4000,
    compile_memory_bytes: int = 8 * 1024**3,
    warmup: int = 2,
    trials: int = 5,
    max_adaptive_candidates: int = 0,
    timeout_s: int = 600,
    client: Client | None = None,
) -> RemoteResult:
    """Submit source with ``kernel``, ``make_inputs``, ``grid``, and ``validate`` hooks.

    The source may also define ``reset_inputs``, ``restore_inputs``, and
    ``suggest_candidates``.
    Each case is one tuning key. All initial candidates are prepared before the
    Call acquires its single GPU lease.
    """
    if not source or len(source.encode()) > 1024 * 1024:
        raise ValueError("Triton source must have 1 to 1048576 bytes")
    if not kernel_name.isidentifier():
        raise ValueError("kernel_name must be a Python identifier")
    if not signature or any(not name.isidentifier() for name in signature):
        raise ValueError("signature must name the kernel arguments")
    if not candidates or len(candidates) > 64 or not cases or len(cases) > 16:
        raise ValueError("provide 1-64 candidates and 1-16 cases")
    if len(candidates) * len(cases) > 128:
        raise ValueError("at most 128 initial variants may be prepared")
    candidate_requests = [candidate.request() for candidate in candidates]
    case_requests = [case.request() for case in cases]
    if len({candidate["name"] for candidate in candidate_requests}) != len(candidates):
        raise ValueError("candidate names must be distinct")
    if len({case["key"] for case in case_requests}) != len(cases):
        raise ValueError("tuning keys must be distinct")
    if not 1 <= compile_workers <= 8 or compile_cpu_millicores < compile_workers * 1000:
        raise ValueError("compile workers must fit the CPU stage resource envelope")
    if compile_memory_bytes < 1024**3 or target_arch < 70 or target_arch > 999:
        raise ValueError("invalid compile memory or CUDA target architecture")
    if not 0 <= warmup <= 20 or not 1 <= trials <= 50:
        raise ValueError("warmup or trial count is out of range")
    if not 0 <= max_adaptive_candidates <= 16:
        raise ValueError("max_adaptive_candidates must be from 0 to 16")

    publisher = client or Client()
    return publisher.submit(
        image=image,
        function=triton_tuning_runner.run,
        kwargs={
            "source": source,
            "kernel_name": kernel_name,
            "signature": dict(signature),
            "candidates": candidate_requests,
            "cases": case_requests,
            "target_arch": target_arch,
            "triton_version": TRITON_VERSION,
            "compile_workers": compile_workers,
            "warmup": warmup,
            "trials": trials,
            "max_adaptive_candidates": max_adaptive_candidates,
        },
        gpu=gpu,
        gpu_count=1,
        timeout_s=timeout_s,
        app_name="triton-autotune",
        stages=staged_tuning_plan(
            cpu_millicores=compile_cpu_millicores, memory_bytes=compile_memory_bytes
        ),
    )


def tune_triton(**kwargs: Any) -> dict[str, Any]:
    """Submit a tuning Call and wait for its candidate report."""
    return spawn_triton_tuning(**kwargs).wait()
