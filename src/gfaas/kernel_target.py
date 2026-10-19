"""Keep every phase of a managed kernel on the probed image and hardware target."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from .app import Function
from .compatibility import Compatibility
from .errors import GfaasError
from .image import Image
from .triton_quick_runner import probe_target


def resolve_target(function: Function) -> tuple[Function, dict[str, Any], str]:
    """Probe through the public API, then pin the selected native image and ISA.

    Resolve again on each invocation: a registry name can move between launches.
    The Call status supplies the actual image, including mixed-ISA selection.
    """
    probe = replace(function, handler=probe_target, outputs=()).spawn()
    target = probe.wait()
    return pin_target(function, target, probe.call_id), target, probe.call_id


def pin_target(function: Function, target: dict[str, Any], probe_id: str) -> Function:
    """Pin a completed probe, including probes that also check library versions."""
    if (
        not isinstance(target, dict)
        or target.get("backend") != "cuda"
        or type(target.get("arch")) is not int
        or target["arch"] < 10
        or target.get("warp_size") != 32
    ):
        raise GfaasError("GPU target discovery returned an invalid target")
    status = function.app.client.get_call(probe_id)
    image = (status.get("current_stage") or {}).get("image") or {}
    digest, architecture = image.get("digest"), image.get("architecture")
    if (
        not isinstance(digest, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
        or architecture not in ("amd64", "arm64")
    ):
        raise GfaasError(
            f"Target probe Call {probe_id} did not report its image digest and CPU architecture"
        )
    capability = f"{target['arch'] // 10}.{target['arch'] % 10}"
    requested = function.compatibility
    if requested is not None and (
        requested.cpu_architecture not in (None, architecture)
        or (
            requested.gpu_compute_capabilities
            and capability not in requested.gpu_compute_capabilities
        )
    ):
        raise GfaasError("Target probe did not satisfy the requested hardware compatibility")
    return replace(
        function,
        image=Image.from_registry(digest),
        compatibility=Compatibility(architecture, (capability,)),
    )


def compiler_compatibility(function: Function) -> Compatibility:
    """CPU compiler Calls need the selected host ISA, without reserving a GPU."""
    assert function.compatibility is not None
    return Compatibility(cpu_architecture=function.compatibility.cpu_architecture)
