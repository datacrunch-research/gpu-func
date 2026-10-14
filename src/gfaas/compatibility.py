"""Hardware requirements carried by an Environment, independent of GPU pool names."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Compatibility:
    """Require a CPU architecture and/or exact GPU compute capabilities.

    Empty GPU capabilities permit any capability. These requirements do not
    establish driver, framework, or application compatibility.
    """

    cpu_architecture: str | None = None
    gpu_compute_capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.cpu_architecture not in (None, "amd64", "arm64"):
            raise ValueError("cpu_architecture must be amd64 or arm64")
        if isinstance(self.gpu_compute_capabilities, str):
            raise ValueError("gpu_compute_capabilities must be a sequence of capabilities")
        capabilities = tuple(self.gpu_compute_capabilities)
        if len(capabilities) > 32:
            raise ValueError("at most 32 GPU compute capabilities are supported")
        for value in capabilities:
            if not isinstance(value, str) or not re.fullmatch(
                r"[1-9][0-9]{0,2}\.(0|[1-9][0-9]?)", value
            ):
                raise ValueError("GPU compute capability must use major.minor, for example 10.3")
            if int(value.split(".")[0]) > 255:
                raise ValueError("GPU compute capability major version must not exceed 255")
        if len(set(capabilities)) != len(capabilities):
            raise ValueError("duplicate GPU compute capabilities")
        object.__setattr__(self, "gpu_compute_capabilities", capabilities)

    def request(self) -> dict[str, Any]:
        """Return the Environment compatibility fields declared by the caller."""
        result: dict[str, Any] = {}
        if self.cpu_architecture is not None:
            result["cpu_architecture"] = self.cpu_architecture
        if self.gpu_compute_capabilities:
            result["gpu_compute_capabilities"] = list(self.gpu_compute_capabilities)
        return result
