"""Raw-CUDA shortcut: submit a `.cu` string, get back stdout + (optional) ncu.

Sugar over ``Client.submit``. We bundle :mod:`gfaas.cuda_runner` on every
call so the GPU VM never needs a custom image — any CUDA-devel rootfs
with ``python3 + cloudpickle + nvcc + ncu`` works.

A profiled run asks the service for the ``gpu_counters`` capability, which needs a grant on the
API key, and returns a typed outcome beside the program's result. See :mod:`gfaas.profiling`.
"""

from __future__ import annotations

from typing import Any

from . import cuda_runner
from .artifacts import ArtifactOutput
from .client import Client, RemoteResult
from .compatibility import Compatibility
from .errors import CudaCompilationError, CudaProcessError, GfaasError
from .image import Image
from .profiling import GPU_COUNTERS, ProfilingOutcome, ProfilingRequest, outcome_from_result
from .stages import CallStage, StageArtifactBinding

CUDA_IMAGE = Image("cuda-nvcc")
_COMPILED_OUTPUT = ArtifactOutput.directory(
    "compiled-cuda",
    "compiled-cuda",
    kind="other",
    publish_on_failure=False,
)
_COMPILED_ARTIFACT_ENV = "GFAAS_COMPILED_ARTIFACT_ID"
#: Where a profiled run publishes its reports and their provenance record.
PROFILE_OUTPUT = ArtifactOutput.directory(
    "profiles",
    "profiles",
    kind="profile",
    required=False,
    publish_on_failure=True,
)


def profiling_request(
    profile: bool | ProfilingRequest,
    ncu_args: list[str] | None = None,
) -> ProfilingRequest | None:
    """The request ``profile`` and ``ncu_args`` stand for, or ``None`` for an unprofiled run."""
    if isinstance(profile, ProfilingRequest):
        if ncu_args:
            raise GfaasError("put profiler arguments in the ProfilingRequest, not in ncu_args")
        return profile
    if not profile:
        if ncu_args:
            raise GfaasError("ncu_args require profile=True")
        return None
    return ProfilingRequest.from_arguments(ncu_args or [])


def staged_execution_plan(*, profiled: bool = False) -> tuple[CallStage, CallStage]:
    """Return the compile and execute stages used by remote CUDA submissions.

    A profiled execute stage also publishes the profile reports.
    """
    return (
        CallStage(
            name="compile",
            qualname="compile_stage",
            resources={"gpu": {"count": 0}},
            outputs=(_COMPILED_OUTPUT,),
        ),
        CallStage(
            name="execute",
            qualname="execute_stage",
            outputs=(PROFILE_OUTPUT,) if profiled else (),
            artifacts=(
                StageArtifactBinding(
                    from_stage="compile",
                    output=_COMPILED_OUTPUT.name,
                    env=_COMPILED_ARTIFACT_ENV,
                ),
            ),
        ),
    )


def compile_and_run(
    source: str,
    *,
    gpu: str | None = None,
    gpu_count: int | None = None,
    gpu_type: str = "any",
    profile: bool | ProfilingRequest = False,
    ncu_args: list[str] | None = None,
    nvcc_flags: list[str] | None = None,
    program_args: list[str] | None = None,
    timeout_s: int = 600,
    image: Image | str = CUDA_IMAGE,
    compatibility: Compatibility | None = None,
    client: Client | None = None,
) -> dict[str, Any]:
    """Compile, run, and (optionally) ``ncu``-profile a CUDA source string.

    Returns ``{phase, stdout, stderr, returncode, compile_ms, run_ms, ncu_csv, profiling}``.
    A staged compiler failure terminates the Call, so ``wait()`` raises
    :class:`GfaasError`. An older single-stage service can still return a compile
    report, which this function converts to :class:`CudaCompilationError`.
    A failed compiled program raises :class:`CudaProcessError`.

    ``profile`` is ``True`` or a :class:`~gfaas.profiling.ProfilingRequest`. A profiled run needs
    the ``profiling`` grant on the API key and is refused before submission without it. Its result
    carries ``profiling``, the plain form of :class:`~gfaas.profiling.ProfilingOutcome`, with the
    Call identity and each report's Artifact added; read it with :func:`profiling_outcome`. The
    profiler capturing nothing is an outcome, not an error: this function raises only when the
    program itself failed.
    """
    remote = spawn(
        source,
        gpu=gpu,
        gpu_count=gpu_count,
        gpu_type=gpu_type,
        profile=profile,
        ncu_args=ncu_args,
        nvcc_flags=nvcc_flags,
        program_args=program_args,
        timeout_s=timeout_s,
        image=image,
        compatibility=compatibility,
        client=client,
    )
    report = remote.wait()
    if isinstance(report, dict) and report.get("profiling") is not None:
        report["profiling"] = locate_reports(remote, report["profiling"])
    _raise_for_failure(report)
    return report


def profiling_outcome(report: object) -> ProfilingOutcome | None:
    """The typed profiling outcome of a ``compile_and_run`` result, or ``None`` if unprofiled."""
    return outcome_from_result(report)


def locate_reports(remote: RemoteResult, outcome: Any) -> Any:
    """Add the Call identity and the published reports' Artifact identity to an outcome."""
    if not isinstance(outcome, dict):
        return outcome
    outcome["call_id"] = remote.call_id
    reports = [item for item in outcome.get("reports", []) if isinstance(item, dict)]
    if any(item.get("output_name") for item in reports):
        published = {
            item.get("name"): item.get("artifact", {}).get("id")
            for item in remote.artifacts().get("items", [])
            if isinstance(item, dict) and isinstance(item.get("artifact"), dict)
        }
        for item in reports:
            item["artifact_id"] = published.get(item.get("output_name"))
    return outcome


def spawn(
    source: str,
    *,
    gpu: str | None = None,
    gpu_count: int | None = None,
    gpu_type: str = "any",
    profile: bool | ProfilingRequest = False,
    ncu_args: list[str] | None = None,
    nvcc_flags: list[str] | None = None,
    program_args: list[str] | None = None,
    timeout_s: int = 600,
    image: Image | str = CUDA_IMAGE,
    compatibility: Compatibility | None = None,
    client: Client | None = None,
) -> RemoteResult:
    c = client or Client()
    request = profiling_request(profile, ncu_args)
    return c.submit(
        image=image,
        compatibility=compatibility,
        function=cuda_runner.run,  # SDK packages cuda_runner.py automatically
        kwargs={
            "source": source,
            "profiling": request.to_dict() if request else None,
            "nvcc_flags": nvcc_flags or [],
            "program_args": program_args or [],
        },
        profiling=GPU_COUNTERS if request else None,
        gpu=gpu,
        gpu_count=gpu_count,
        gpu_type=gpu_type,
        timeout_s=timeout_s,
        app_name="cuda-nvcc",
        stages=staged_execution_plan(profiled=request is not None),
    )


class CudaSource:
    """Optional fluent builder for ``compile_and_run``."""

    def __init__(self, source: str) -> None:
        self.source = source
        self._opts: dict[str, Any] = {}

    def gpu(self, model: str, *, gpu_type: str = "any") -> CudaSource:
        self._opts["gpu"] = model
        self._opts["gpu_type"] = gpu_type
        return self

    def nvcc(self, *flags: str) -> CudaSource:
        self._opts["nvcc_flags"] = list(flags)
        return self

    def profile(self, *ncu_args: str, request: ProfilingRequest | None = None) -> CudaSource:
        """Profile the run with ``ncu`` options, or with a full :class:`ProfilingRequest`."""
        self._opts["profile"] = request or True
        self._opts.pop("ncu_args", None)
        if ncu_args:
            self._opts["ncu_args"] = list(ncu_args)
        return self

    def run(self, client: Client | None = None) -> dict[str, Any]:
        return compile_and_run(self.source, client=client, **self._opts)


def _raise_for_failure(report: object) -> None:
    if not isinstance(report, dict):
        raise GfaasError("CUDA helper returned a non-object result")

    phase = report.get("phase")
    returncode = report.get("returncode")
    if phase not in {"compile", "run"}:
        raise GfaasError("CUDA helper result has no valid compile or run phase")
    if not isinstance(returncode, int) or isinstance(returncode, bool):
        raise GfaasError("CUDA helper result has no valid integer return code")
    if returncode == 0:
        return
    if phase == "compile":
        raise CudaCompilationError(report)
    outcome = report.get("profiling")
    if isinstance(outcome, dict) and outcome.get("program_returncode") in (None, 0):
        # The program did not fail; the profiler did. That is the outcome's to report.
        return
    raise CudaProcessError(report)
