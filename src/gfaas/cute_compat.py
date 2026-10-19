"""Source and metadata boundary for CuTe's public JIT/AOT APIs."""

from __future__ import annotations

import ast
import inspect
import textwrap
import types
from dataclasses import dataclass
from typing import Any


class UnsupportedCuteDSLKernelError(ValueError):
    pass


@dataclass(frozen=True)
class SourceEntry:
    source: str
    entrypoint: str
    signature: inspect.Signature
    cute_version: str


def source_entry(source: str, entrypoint: str, cute_version: str) -> SourceEntry:
    """Inspect a remote module's host signature without importing or evaluating it."""
    if not isinstance(source, str) or not source or len(source.encode()) > 1024 * 1024:
        raise ValueError("Source must be a nonempty Python module within 1 MiB")
    if not isinstance(entrypoint, str) or not entrypoint.isidentifier():
        raise ValueError("Invalid CuTe entrypoint")
    if not isinstance(cute_version, str) or not cute_version or len(cute_version) > 64:
        raise ValueError("Specify the CuTe version required in the selected image")
    nodes = [
        n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == entrypoint
    ]
    if len(nodes) != 1 or not any(
        isinstance(d, ast.Attribute) and d.attr == "jit" for d in nodes[0].decorator_list
    ):
        raise UnsupportedCuteDSLKernelError("Source must define one @cute.jit host entrypoint")
    args = nodes[0].args
    if args.vararg or args.kwarg:
        raise UnsupportedCuteDSLKernelError(
            "Source host entries must use explicit named parameters"
        )
    positional = [*args.posonlyargs, *args.args]
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    parameters = []
    for index, (arg, default) in enumerate(
        [
            *zip(positional, defaults, strict=True),
            *zip(args.kwonlyargs, args.kw_defaults, strict=True),
        ]
    ):
        try:
            value = inspect.Parameter.empty if default is None else ast.literal_eval(default)
        except (ValueError, TypeError):
            raise UnsupportedCuteDSLKernelError(
                "Source parameter defaults must be literals"
            ) from None
        kind = (
            inspect.Parameter.POSITIONAL_ONLY
            if index < len(args.posonlyargs)
            else inspect.Parameter.POSITIONAL_OR_KEYWORD
            if index < len(positional)
            else inspect.Parameter.KEYWORD_ONLY
        )
        annotation = (
            inspect.Parameter.empty if arg.annotation is None else ast.unparse(arg.annotation)
        )
        parameters.append(inspect.Parameter(arg.arg, kind, default=value, annotation=annotation))
    return SourceEntry(source, entrypoint, inspect.Signature(parameters), cute_version)


def kernel_function(kernel: Any) -> Any:
    if not callable(kernel):
        raise UnsupportedCuteDSLKernelError("Expected a callable CuTe host entry")
    function = kernel if inspect.isfunction(kernel) else kernel.__call__
    if not callable(function):
        raise UnsupportedCuteDSLKernelError("Expected a @cute.jit function or callable object")
    original = inspect.unwrap(function)
    if not inspect.isfunction(original):
        raise UnsupportedCuteDSLKernelError("Cannot inspect the CuTe host entry point")
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    definition = tree.body[0]
    if not isinstance(definition, ast.FunctionDef) or not any(
        isinstance(d, ast.Attribute) and d.attr == "jit" for d in definition.decorator_list
    ):
        raise UnsupportedCuteDSLKernelError("Wrap the @cute.jit host entry, not a device kernel")
    return function


def source_bundle(kernel: Any) -> tuple[str, str]:
    """Bundle definitions and dependencies without executing the user's whole module."""
    kernel_function(kernel)
    imports: dict[str, str] = {}
    definitions: dict[str, str] = {}
    bindings: dict[str, str] = {}
    visiting: set[str] = set()

    def expression(value: Any, name: str) -> str:
        if value is None or type(value) in (bool, int, float, str):
            return repr(value)
        if isinstance(value, (tuple, list)):
            items = ", ".join(expression(v, name) for v in value)
            return f"({items},)" if isinstance(value, tuple) else f"[{items}]"
        if isinstance(value, dict) and all(isinstance(k, str) for k in value):
            return "{" + ", ".join(f"{k!r}: {expression(v, name)}" for k, v in value.items()) + "}"
        if inspect.isclass(value) and value.__module__.startswith(("cutlass", "cuda")):
            alias = "_cute_type_" + value.__name__
            imports[alias] = f"from {value.__module__} import {value.__name__} as {alias}"
            return alias
        raise UnsupportedCuteDSLKernelError(
            f"Cannot bundle state/global {name}: {type(value).__name__}"
        )

    def visit(value: Any, name: str) -> None:
        if name in definitions or name in imports or name in bindings or name in visiting:
            return
        if isinstance(value, types.ModuleType):
            if value.__name__ == "__main__":
                raise UnsupportedCuteDSLKernelError(
                    "User-module imports must be explicit dependencies"
                )
            imports[name] = f"import {value.__name__} as {name}"
            return
        if (inspect.isclass(value) or inspect.isfunction(value)) and value.__module__.startswith(
            ("cutlass", "cuda", "math", "numpy")
        ):
            imports[name] = f"from {value.__module__} import {value.__name__} as {name}"
            return
        if not (inspect.isfunction(value) or inspect.isclass(value)):
            bindings[name] = f"{name} = {expression(value, name)}"
            return
        visiting.add(name)
        original = inspect.unwrap(value)
        source = textwrap.dedent(inspect.getsource(original))
        node = ast.parse(source).body[0]
        namespace = (
            original.__globals__
            if inspect.isfunction(original)
            else vars(__import__(original.__module__, fromlist=[original.__name__]))
        )
        if inspect.isfunction(original):
            namespace = {**namespace, **inspect.getclosurevars(original).nonlocals}
        names = {
            n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }
        for dependency in sorted(names):
            if dependency in namespace and dependency != original.__name__:
                visit(namespace[dependency], dependency)
        definitions[name] = source + (
            f"\n{name} = {original.__name__}\n" if name != original.__name__ else ""
        )
        visiting.remove(name)

    if inspect.isfunction(kernel):
        name = inspect.unwrap(kernel).__name__
        visit(kernel, name)
    else:
        name = "_vfunc_cute_entry"
        cls = type(kernel)
        visit(cls, cls.__name__)
        assignments = [f"{name} = {cls.__name__}.__new__({cls.__name__})"]
        for key, value in vars(kernel).items():
            if not key.isidentifier():
                raise UnsupportedCuteDSLKernelError("Invalid callable-object attribute")
            assignments.append(f"{name}.{key} = {expression(value, key)}")
        bindings[name] = "\n".join(assignments)
    # Constants precede functions; object construction follows the class definition.
    constants = [v for k, v in bindings.items() if k != "_vfunc_cute_entry"]
    return "\n".join(
        [
            *imports.values(),
            *constants,
            *definitions.values(),
            bindings.get("_vfunc_cute_entry", ""),
        ]
    ), name


def encode_constant(value: Any, depth: int = 0) -> Any:
    """Preserve constexpr tuple structure across the JSON compilation boundary."""
    import math

    if depth > 16:
        raise UnsupportedCuteDSLKernelError("Configuration nesting is too deep")
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) is tuple:
        return {"_vfunc_cute_tuple": [encode_constant(v, depth + 1) for v in value]}
    if type(value) is list:
        return [encode_constant(v, depth + 1) for v in value]
    raise UnsupportedCuteDSLKernelError(
        "CuTe configuration values must be literals or nested tuples/lists"
    )


def decode_constants(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"_vfunc_cute_tuple"}:
            return tuple(decode_constants(v) for v in value["_vfunc_cute_tuple"])
        return {k: decode_constants(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode_constants(v) for v in value]
    return value
