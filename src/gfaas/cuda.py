"""Raw-CUDA shortcut: submit a `.cu` string, get back stdout + (optional) ncu.

Sugar over ``Client.submit``. We bundle :mod:`gfaas.cuda_runner` on every
call so the GPU VM never needs a custom image — any CUDA-devel rootfs
with ``python3 + cloudpickle + nvcc + ncu`` works.
"""

from __future__ import annotations

from typing import Any

from . import cuda_runner
from .client import Client, RemoteResult
from .errors import CudaCompilationError, CudaProcessError, GfaasError
from .image import Image

CUDA_IMAGE = Image("cuda-nvcc")


def compile_and_run(
    source: str,
    *,
    gpu: str | None = None,
    gpu_count: int | None = None,
    gpu_type: str = "any",
    profile: bool = False,
    ncu_args: list[str] | None = None,
    nvcc_flags: list[str] | None = None,
    program_args: list[str] | None = None,
    timeout_s: int = 600,
    image: Image | str = CUDA_IMAGE,
    client: Client | None = None,
) -> dict[str, Any]:
    """Compile, run, and (optionally) ``ncu``-profile a CUDA source string.

    Returns ``{phase, stdout, stderr, returncode, compile_ms, run_ms, ncu_csv?}``.
    Raises :class:`CudaCompilationError` when ``nvcc`` fails and
    :class:`CudaProcessError` when the compiled program exits unsuccessfully.
    Use :func:`spawn` and call ``wait()`` directly to inspect either raw result.
    """
    report = spawn(
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
        client=client,
    ).wait()
    _raise_for_failure(report)
    return report


def spawn(
    source: str,
    *,
    gpu: str | None = None,
    gpu_count: int | None = None,
    gpu_type: str = "any",
    profile: bool = False,
    ncu_args: list[str] | None = None,
    nvcc_flags: list[str] | None = None,
    program_args: list[str] | None = None,
    timeout_s: int = 600,
    image: Image | str = CUDA_IMAGE,
    client: Client | None = None,
) -> RemoteResult:
    c = client or Client()
    return c.submit(
        image=image,
        function=cuda_runner.run,  # SDK packages cuda_runner.py automatically
        kwargs={
            "source": source,
            "profile": profile,
            "ncu_args": ncu_args or [],
            "nvcc_flags": nvcc_flags or [],
            "program_args": program_args or [],
        },
        gpu=gpu,
        gpu_count=gpu_count,
        gpu_type=gpu_type,
        timeout_s=timeout_s,
        app_name="cuda-nvcc",
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

    def profile(self, *ncu_args: str) -> CudaSource:
        self._opts["profile"] = True
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
    raise CudaProcessError(report)
