"""Small capability-checked boundary around Triton's evolving Python wrappers.

No run, warmup, benchmark, grid callable, or user hook is invoked here. Metadata
uses conservative types: runtime values/alignment are not promoted to constants.
"""

from __future__ import annotations

import ast
import inspect
import types
from typing import Any


class UnsupportedTritonKernelError(ValueError):
    """The kernel uses an unsupported wrapper, hook, or argument form."""


def _types() -> tuple[Any, Any]:
    try:
        from triton.runtime.autotuner import Autotuner  # type: ignore[import-not-found]
        from triton.runtime.jit import JITFunction  # type: ignore[import-not-found]
    except ImportError as error:
        raise UnsupportedTritonKernelError("Install Triton in the client environment") from error
    return JITFunction, Autotuner


def reset_arguments(kernel: Any) -> tuple[list[str], list[str]]:
    """Read documented names, with an index adapter for older Triton releases."""
    result = []
    for names_field, indices_field in (
        ("reset_to_zero", "reset_idx"),
        ("restore_value", "restore_idx"),
    ):
        names = getattr(kernel, names_field, None)
        if names is None:
            indices = getattr(kernel, indices_field, [])
            try:
                names = [kernel.arg_names[index] for index in indices]
            except (IndexError, TypeError) as error:
                raise UnsupportedTritonKernelError("Unrecognized reset/restore indices") from error
        if not isinstance(names, (list, tuple)) or any(
            type(name) is not str or name not in kernel.arg_names for name in names
        ):
            raise UnsupportedTritonKernelError(
                "Reset/restore declarations must name kernel arguments"
            )
        result.append(list(names))
    return result[0], result[1]


def validate_kernel(kernel: Any) -> tuple[Any, list[dict[str, Any]] | None]:
    """Accept exact JIT/vanilla autotuner classes; reject custom subclasses."""
    jit_type, auto_type = _types()
    configurations = None
    if type(kernel) is auto_type:
        if type(kernel.fn) is not jit_type:
            raise UnsupportedTritonKernelError("Only autotune directly wrapping jit is supported")
        # Instantiate the installed release's vanilla wrapper to compare defaults.
        # This runs Triton's constructor, never the user's callbacks or benchmark.
        reset, restore = reset_arguments(kernel)
        baseline = auto_type(kernel.fn, kernel.arg_names, kernel.configs, [], reset, restore)
        checks = [
            "reset_to_zero",
            "restore_value",
            "reset_idx",
            "restore_idx",
            "perf_model",
            "early_config_prune",
            "configs_top_k",
            "num_warmups",
            "num_reps",
            "use_cuda_graph",
            "_do_bench",
            "cache_results",
            "user_defined_pre_hook",
            "user_defined_post_hook",
        ]
        for field in checks:
            if hasattr(kernel, field) != hasattr(baseline, field):
                raise UnsupportedTritonKernelError(f"Unrecognized autotuner field: {field}")
            if hasattr(kernel, field) and getattr(kernel, field) != getattr(baseline, field):
                raise UnsupportedTritonKernelError(f"Autotuning modifier is unsupported: {field}")
        for field in ("pre_hook", "post_hook"):
            current, default = getattr(kernel, field, None), getattr(baseline, field, None)
            if not (
                inspect.isfunction(current)
                and inspect.isfunction(default)
                and current.__code__ is default.__code__
                and all(cell.cell_contents is kernel for cell in (current.__closure__ or ()))
            ):
                raise UnsupportedTritonKernelError(f"Autotuning callback is unsupported: {field}")
        # Fail closed when a newer release adds a modifier we do not know yet.
        # Ignore only identifiers and mutable state created by an ordinary run.
        ignored = {
            "fn",
            "base_fn",
            "arg_names",
            "configs",
            "keys",
            "key_idx",
            "cache",
            "pre_hook",
            "post_hook",
            "nargs",
            "best_config",
            "configs_timings",
            "bench_time",
            "restore_copies",
        }
        for field in set(vars(kernel)) | set(vars(baseline)):
            if field in ignored or field in checks:
                continue
            current = vars(kernel).get(field)
            default = vars(baseline).get(field)
            if current is default:
                continue
            if (
                type(current) in (bool, int, float, str, list, tuple, dict)
                and type(current) is type(default)
                and current == default
            ):
                continue
            raise UnsupportedTritonKernelError(
                f"Unrecognized autotuning modifier or state: {field}"
            )
        configurations = []
        import triton  # type: ignore[import-not-found]

        for config in kernel.configs:
            if type(config) is not triton.Config:
                raise UnsupportedTritonKernelError("Custom Config subclasses are unsupported")
            if getattr(config, "pre_hook", None) is not None:
                raise UnsupportedTritonKernelError("Configuration pre_hook is unsupported")
            if getattr(config, "ir_override", None) is not None:
                raise UnsupportedTritonKernelError("Configuration ir_override is unsupported")
            options = config.all_kwargs()
            if not isinstance(options, dict):
                raise UnsupportedTritonKernelError("Unrecognized Config interface")
            configurations.append(options)
        if not configurations:
            raise UnsupportedTritonKernelError("No configurations supplied")
        kernel = kernel.fn
    if type(kernel) is not jit_type:
        raise UnsupportedTritonKernelError("Only plain jit or vanilla autotune(jit) is supported")
    for field in ("pre_run_hooks", "launch_metadata"):
        if getattr(kernel, field, None):
            raise UnsupportedTritonKernelError(f"JIT callback is unsupported: {field}")
    # Custom representations are callbacks too; default repr handling differs by release.
    if hasattr(kernel, "_repr"):
        custom_repr = kernel._repr is not None
    else:
        representation = getattr(kernel, "repr", None)
        closure = (
            inspect.getclosurevars(representation).nonlocals
            if inspect.isfunction(representation)
            else {}
        )
        custom_repr = "repr" not in closure or closure["repr"] is not None
    if custom_repr:
        raise UnsupportedTritonKernelError("Custom JIT repr is unsupported")
    return kernel, configurations


def source_bundle(kernel: Any) -> str:
    """Emit just kernel/dependency definitions; do not import the user's module remotely."""
    jit_type, _ = _types()
    imports: dict[str, str] = {"triton": "import triton"}
    constants: dict[str, str] = {}
    functions: dict[str, str] = {}
    visiting: set[str] = set()

    def visit(jit: Any, name: str) -> None:
        validate_kernel(jit)
        if name in functions or name in visiting:
            return
        visiting.add(name)
        function = jit.fn
        if function.__closure__:
            raise UnsupportedTritonKernelError("Kernel closures are unsupported")
        import textwrap

        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        node = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
        if node is None:
            raise UnsupportedTritonKernelError("Kernel source must be inspectable Python")
        node.name = name
        node.decorator_list = []
        references = {
            n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }
        decorator_options = {}
        for field in (
            "version",
            "noinline",
            "debug",
            "do_not_specialize",
            "do_not_specialize_on_alignment",
        ):
            value = getattr(jit, field, None)
            if value not in (None, False, []):
                decorator_options[field] = value
        node.decorator_list = [
            ast.parse(
                "triton.jit(" + ",".join(f"{k}={v!r}" for k, v in decorator_options.items()) + ")",
                mode="eval",
            ).body
        ]
        local_names = {a.arg for a in ast.walk(node.args) if isinstance(a, ast.arg)}
        local_names.update(
            n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        )
        for referenced in references - local_names:
            if referenced not in function.__globals__ or referenced == name:
                continue
            value = function.__globals__[referenced]
            if type(value) is jit_type:
                visit(value, referenced)
            elif isinstance(value, types.ModuleType):
                if not (value.__name__ == "math" or value.__name__.startswith("triton")):
                    raise UnsupportedTritonKernelError(
                        f"Unsupported source dependency: {referenced}"
                    )
                imports[referenced] = f"import {value.__name__} as {referenced}"
            elif value is None or type(value) in (bool, int, float, str, tuple):
                constants[referenced] = f"{referenced} = {value!r}"
            else:
                raise UnsupportedTritonKernelError(f"Unsupported source dependency: {referenced}")
        functions[name] = ast.unparse(ast.fix_missing_locations(node))
        visiting.remove(name)

    visit(kernel, kernel.fn.__name__)
    return "\n\n".join([*imports.values(), *constants.values(), *functions.values()]) + "\n"


def argument_type(value: Any, parameter: Any) -> str:
    """Conservative scalar/pointer types; never read or transmit tensor contents."""
    if getattr(parameter, "is_constexpr", False):
        return "constexpr"
    annotation = getattr(parameter, "annotation_type", "")
    if annotation:
        return str(annotation)
    if value is None:
        return "constexpr"
    if type(value) is bool:
        return "i1"
    if type(value) is int:
        if not -(2**63) <= value < 2**64:
            raise UnsupportedTritonKernelError("Integer argument exceeds 64 bits")
        return "i32" if -(2**31) <= value < 2**31 else ("u64" if value >= 2**63 else "i64")
    if type(value) is float:
        return "fp32"
    if hasattr(value, "dtype") and callable(getattr(value, "data_ptr", None)):
        import triton.language as tl  # type: ignore[import-not-found]

        name = str(value.dtype).removeprefix("torch.").removeprefix("triton.language.")
        try:
            dtype = getattr(tl, name)
            scalar = str(dtype)
        except AttributeError as error:
            raise UnsupportedTritonKernelError(f"Unsupported tensor dtype: {name}") from error
        return ("*k" if getattr(parameter, "is_const", False) else "*") + scalar
    raise UnsupportedTritonKernelError(f"Unsupported argument type: {type(value).__name__}")
