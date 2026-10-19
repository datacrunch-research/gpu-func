"""Exceptions raised by the public gfaas client."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class GfaasError(RuntimeError):
    """Base class for errors raised by the gfaas Python implementation."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        self.status_code = status_code
        super().__init__(message)


class UnsupportedGpuPoolError(GfaasError):
    """The deployment does not configure the requested GPU pool."""

    def __init__(self, requested_gpu_pool: str, supported_gpu_pools: list[str]) -> None:
        self.requested_gpu_pool = requested_gpu_pool
        self.supported_gpu_pools = tuple(supported_gpu_pools)
        supported = ", ".join(self.supported_gpu_pools) or "none"
        super().__init__(
            f"GPU pool {requested_gpu_pool!r} is not configured; supported pools: {supported}"
        )


class ArtifactUploadError(GfaasError):
    """An interrupted transfer whose accepted parts can be reused with upload_id."""

    def __init__(self, upload_id: str, detail: str, *, phase: str = "transferring") -> None:
        self.upload_id = upload_id
        self.phase = phase
        super().__init__(
            f"Artifact upload {upload_id} failed: {detail.rstrip('.')}. "
            f"Resume with upload_id={upload_id!r}"
        )


class ArtifactTreeUploadError(GfaasError):
    """A child upload or tree registration failed; completed children remain available."""

    def __init__(
        self,
        child_artifact_ids: list[str],
        *,
        upload_ids: dict[str, str] | None = None,
        failures: dict[str, str] | None = None,
    ) -> None:
        self.upload_ids = dict(upload_ids or {})
        self.failures = dict(failures or {})
        self.child_artifact_ids = tuple(child_artifact_ids)
        super().__init__(
            "Artifact tree creation failed after "
            f"{len(self.child_artifact_ids)} child Artifact uploads"
        )


class CudaError(GfaasError):
    """Base class for a CUDA compiler or launched-process failure.

    ``report`` is the exact structured result returned by the remote CUDA
    harness. Its commonly inspected fields are also available as attributes.
    """

    operation = "CUDA command"

    def __init__(self, report: Mapping[str, Any]) -> None:
        self.report = dict(report)
        self.phase = str(report["phase"])
        self.returncode = int(report["returncode"])
        self.stdout = str(report.get("stdout", ""))
        self.stderr = str(report.get("stderr", ""))
        self.compile_ms = _optional_int(report.get("compile_ms"))
        self.run_ms = _optional_int(report.get("run_ms"))
        self.ncu_csv = report.get("ncu_csv")

        if self.returncode < 0:
            outcome = f"was terminated by signal {-self.returncode}"
        else:
            outcome = f"exited with status {self.returncode}"
        super().__init__(f"{self.operation} {outcome}")


class CudaCompilationError(CudaError):
    """Raised when ``nvcc`` rejects or cannot compile submitted CUDA source."""

    operation = "CUDA compilation"


class CudaProcessError(CudaError):
    """Raised when the compiled CUDA program or profiling command fails."""

    operation = "CUDA process"


class SerializationError(GfaasError):
    """Raised when job arguments or results cannot be serialized."""


def _optional_int(value: object) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None
