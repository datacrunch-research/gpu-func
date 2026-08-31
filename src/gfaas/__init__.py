"""gfaas: GPU Function-as-a-Service Python SDK.

Provides a small function API that routes ``.remote()`` calls through the
gfaas service.
"""

from __future__ import annotations

from .app import App, Function
from .artifacts import (
    ArtifactCheckpoint,
    ArtifactOutput,
    ArtifactRef,
    ArtifactUploadProgress,
    scratch_path,
)
from .errors import (
    ArtifactTreeUploadError,
    CudaCompilationError,
    CudaError,
    CudaProcessError,
    UnsupportedGpuPoolError,
)
from .image import Image

__all__ = [
    "App",
    "ArtifactRef",
    "scratch_path",
    "ArtifactOutput",
    "ArtifactCheckpoint",
    "ArtifactTreeUploadError",
    "ArtifactUploadProgress",
    "Client",
    "ClientConfig",
    "CudaCompilationError",
    "CudaError",
    "CudaProcessError",
    "CudaSource",
    "Function",
    "Image",
    "RemoteResult",
    "UnsupportedGpuPoolError",
    "compile_and_run",
]


def __getattr__(name: str):
    if name in {"Client", "RemoteResult"}:
        from .client import Client, RemoteResult

        return {"Client": Client, "RemoteResult": RemoteResult}[name]
    if name == "ClientConfig":
        from .config import ClientConfig

        return ClientConfig
    if name in {"CudaSource", "compile_and_run"}:
        from .cuda import CudaSource, compile_and_run

        return {"CudaSource": CudaSource, "compile_and_run": compile_and_run}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
