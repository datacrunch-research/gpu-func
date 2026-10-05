"""Version-checked Helion source and configuration boundary."""

from __future__ import annotations

import ast
import importlib.metadata
import inspect
import json
import textwrap
import types
from typing import Any


class UnsupportedHelionKernelError(ValueError):
    pass


def installed_version() -> str:
    return importlib.metadata.version("helion")


def validate_kernel(kernel: Any) -> None:
    try:
        import helion  # type: ignore[import-not-found]
    except ImportError as error:
        raise UnsupportedHelionKernelError("Install Helion in the client environment") from error
    if type(kernel) is not helion.Kernel:
        raise UnsupportedHelionKernelError("Expected a kernel decorated with helion.kernel")
    if not callable(getattr(kernel, "bind", None)) or not inspect.isfunction(kernel.fn):
        raise UnsupportedHelionKernelError("Unrecognized Helion binding API")
    if kernel.settings.backend != "triton":
        raise UnsupportedHelionKernelError("HelionKernel requires Helion's Triton backend")
    if getattr(kernel, "_key_fn", None) is not None:
        raise UnsupportedHelionKernelError("Custom Helion specialization callbacks are unsupported")


def configuration_dicts(configs: Any) -> list[dict[str, Any]]:
    """Use Config's documented mapping API and deduplicate finite JSON values."""
    result = []
    identities = set()
    for config in configs:
        value = dict(config)
        identity = json.dumps(value, sort_keys=True, allow_nan=False)
        if identity not in identities:
            result.append(value)
            identities.add(identity)
    return result


def source_bundle(kernel: Any) -> str:
    """Bundle inspected function source without importing the author's module."""
    validate_kernel(kernel)
    function = kernel.fn
    if function.__closure__:
        raise UnsupportedHelionKernelError("Helion kernel closures are unsupported")
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    node.decorator_list = []
    references = {
        n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    local = {a.arg for a in ast.walk(node.args) if isinstance(a, ast.arg)}
    local.update(
        n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    )
    imports = []
    constants = []
    dtype_module = "__vfunc_torch_dtype"
    while dtype_module in references:
        dtype_module += "_"
    for name in sorted(references - local):
        if name not in function.__globals__ or name == function.__name__:
            continue
        value = function.__globals__[name]
        if isinstance(value, types.ModuleType) and value.__name__.split(".")[0] in (
            "torch",
            "helion",
            "math",
            "typing",
        ):
            imports.append(f"import {value.__name__} as {name}")
        elif type(value).__module__ == "torch" and type(value).__name__ == "dtype":
            dtype_name = str(value).removeprefix("torch.")
            if not dtype_name.isidentifier():
                raise UnsupportedHelionKernelError(f"Unsupported dtype constant: {name}")
            imports.append(f"import torch as {dtype_module}")
            constants.append(f"{name} = {dtype_module}.{dtype_name}")
        elif value is None or type(value) in (bool, int, float, str, tuple):
            try:
                ast.literal_eval(repr(value))
            except (ValueError, SyntaxError) as error:
                raise UnsupportedHelionKernelError(
                    f"Helion source constant cannot be transported: {name}"
                ) from error
            constants.append(f"{name} = {value!r}")
        elif (
            getattr(value, "__module__", "").split(".")[0] in ("torch", "helion", "typing")
            and getattr(value, "__name__", "").isidentifier()
        ):
            imports.append(f"from {value.__module__} import {value.__name__} as {name}")
        else:
            raise UnsupportedHelionKernelError(f"Unsupported Helion source dependency: {name}")
    return "\n".join(
        [
            "from __future__ import annotations",
            *imports,
            *constants,
            ast.unparse(ast.fix_missing_locations(node)),
        ]
    )
