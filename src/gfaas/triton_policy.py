"""Policy for vFunc's own tuning algorithm; Triton's autotuner is never run."""

from __future__ import annotations

import inspect
import math
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TritonTuning:
    quick_benchmark_delta: float | None = None
    evaluate: Callable[..., bool] | None = None
    pruning_min_runtime_us: float = 100.0
    quick_benchmark_group_size: int = 8
    quick_benchmark_variants_per_job: int = 256
    quick_benchmark_max_concurrent_jobs: int = 4

    def __post_init__(self) -> None:
        if self.quick_benchmark_delta is not None and (
            not math.isfinite(self.quick_benchmark_delta) or self.quick_benchmark_delta < 0
        ):
            raise ValueError("quick_benchmark_delta must be a finite nonnegative fraction or None")
        if not math.isfinite(self.pruning_min_runtime_us) or self.pruning_min_runtime_us < 0:
            raise ValueError("pruning_min_runtime_us must be finite and nonnegative")
        if (
            self.quick_benchmark_group_size < 1
            or self.quick_benchmark_variants_per_job < 1
            or not 1 <= self.quick_benchmark_max_concurrent_jobs <= 32
        ):
            raise ValueError("Invalid quick benchmark sharding limits")
        if self.evaluate is not None and not callable(self.evaluate):
            raise TypeError("evaluate must be callable or None")


def portable_callable(function: Any) -> Any:
    """Copy Python functions so cloudpickle sends their code, rather than user-module imports."""
    if not callable(function):
        return function
    copies: dict[int, Any] = {}

    def clone(fn: Any) -> Any:
        if not inspect.isfunction(fn):
            raise TypeError("Grid/evaluation callbacks must be ordinary Python functions")
        if id(fn) in copies:
            return copies[id(fn)]
        namespace = dict(fn.__globals__)
        closure = tuple(types.CellType(cell.cell_contents) for cell in fn.__closure__ or ()) or None
        result = types.FunctionType(fn.__code__, namespace, fn.__name__, fn.__defaults__, closure)
        result.__module__ = "__vfunc_callback__"
        result.__kwdefaults__ = fn.__kwdefaults__
        copies[id(fn)] = result
        for name, value in namespace.copy().items():
            if inspect.isfunction(value) and value.__module__ == fn.__module__:
                namespace[name] = clone(value)
        if closure:
            for cell in closure:
                if inspect.isfunction(cell.cell_contents):
                    cell.cell_contents = clone(cell.cell_contents)
        return result

    return clone(function)
