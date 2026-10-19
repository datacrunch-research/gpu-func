"""Bind Helion on the target GPU and capture launches without executing them."""

from __future__ import annotations

import hashlib
import inspect
import json
from typing import Any

from gfaas.helion_compat import installed_version


def prepare(
    *,
    source: str,
    kernel_name: str,
    settings: bytes,
    configurations: list[dict[str, Any]],
    inputs: dict[str, Any],
    helion_version: str,
) -> dict[str, Any]:
    import cloudpickle
    import helion  # type: ignore[import-not-found]
    import triton  # type: ignore[import-not-found]
    from torch._inductor.codecache import PyCodeCache  # type: ignore[import-not-found]

    from gfaas.triton_compat import argument_type
    from gfaas.triton_inputs import SnapshotInputs
    from gfaas.triton_quick_runner import probe_target

    if installed_version() != helion_version:
        raise RuntimeError("Preparation Helion version differs from the client")
    module = PyCodeCache.load(source)
    native = helion.kernel(
        getattr(module, kernel_name), settings=helion.Settings(**cloudpickle.loads(settings))
    )
    args, _ = SnapshotInputs(inputs, [], [], [])(inputs["metadata"])
    bound = native.bind(args)
    configs = configurations or [dict(bound.config_spec.default_config())]
    variants = []
    for config in configs:
        record: dict[str, Any] = {"configuration": config}
        record["id"] = hashlib.sha256(
            json.dumps(config, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        try:
            code = bound.to_triton_code(helion.Config(**config))
            run = bound.compile_config(helion.Config(**config))
            launches: list[dict[str, Any]] = []

            def capture(
                kernel: Any,
                grid: Any,
                *launch_args: Any,
                _launches: list[Any] = launches,
                **launch_kwargs: Any,
            ) -> None:
                if not hasattr(kernel, "params") or not inspect.isfunction(
                    getattr(kernel, "fn", None)
                ):
                    raise RuntimeError("Unsupported generated Helion launcher interface")
                parameters = {p.name: p for p in kernel.params}
                values = {k: v for k, v in launch_kwargs.items() if k in parameters}
                options = {k: v for k, v in launch_kwargs.items() if k not in parameters}
                binding = inspect.signature(kernel.fn).bind(*launch_args, **values)
                binding.apply_defaults()
                signature, constants = {}, {}
                for name, value in binding.arguments.items():
                    signature[name] = argument_type(value, parameters[name])
                    if signature[name] == "constexpr":
                        constants[name] = value
                unit = {
                    "kernel_name": kernel.fn.__name__,
                    "signature": signature,
                    "constants": constants,
                    "options": options,
                }
                unit["id"] = hashlib.sha256(
                    json.dumps(unit, sort_keys=True, allow_nan=False).encode()
                ).hexdigest()
                if unit not in _launches:
                    _launches.append(unit)

            # The adapter intercepts every generated launch, including multi-kernel
            # functions. It deliberately performs no GPU kernel execution.
            run(*args, _launcher=capture)
            if not launches:
                raise RuntimeError("Helion configuration emitted no GPU launches")
            record.update(
                status="prepared", source=code, launches=launches, kernel_name=kernel_name
            )
        except Exception as error:
            record.update(status="failed", diagnostics=f"{type(error).__name__}: {error}")
        variants.append(record)
    return {
        "schema": "vfunc.helion-preparation/v1",
        "variants": variants,
        "target": probe_target(),
        "helion_version": helion_version,
        "triton_version": triton.__version__,
    }
