"""CuTe entry points sharing vFunc's input rings, timing and replication runner."""

from __future__ import annotations

from typing import Any


def probe_environment() -> dict[str, Any]:
    import platform

    from gfaas.cute_backend import version
    from gfaas.triton_quick_runner import probe_target

    return {"target": probe_target(), "cute_version": version(), "cpu_arch": platform.machine()}


def benchmark_cycle(**kwargs: Any) -> dict[str, Any]:
    from gfaas.triton_quick_runner import benchmark_cycle as shared_cycle

    kwargs["backend"] = "cute"
    kwargs["triton_version"] = kwargs.pop("cute_version")
    return shared_cycle(**kwargs)


def benchmark_selected(**kwargs: Any) -> dict[str, Any]:
    kwargs["final_only"] = True
    return benchmark_cycle(**kwargs)


def execute_winner(
    *,
    source: str,
    kernel_name: str,
    variant: dict[str, Any],
    artifacts: list[Any],
    inputs: dict[str, Any],
    callbacks: bytes,
    target: dict[str, Any],
    cute_version: str,
    argument_names: list[str],
) -> dict[str, Any]:
    import torch  # type: ignore[import-not-found]

    from gfaas.cute_backend import candidate, load_artifacts
    from gfaas.triton_inputs import SnapshotInputs, snapshot_inputs
    from gfaas.triton_quick_runner import probe_target

    if probe_target() != target:
        raise RuntimeError("Execution GPU differs from compiled CuTe target")
    objects = load_artifacts(artifacts, source, target, cute_version, [variant])
    if variant["id"] not in objects:
        raise RuntimeError("Selected CuTe configuration has no compiled object")
    launch = candidate(objects[variant["id"]], variant, argument_names)
    factory = SnapshotInputs(inputs, [], [], argument_names)
    args, kwargs = factory(inputs["metadata"])
    launch(*args, **kwargs)
    torch.cuda.synchronize()
    return snapshot_inputs(args, kwargs)
