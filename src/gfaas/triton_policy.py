"""Policy for vFunc's own tuning algorithm; Triton's autotuner is never run."""

from __future__ import annotations

import inspect
import math
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TritonPruning:
    """Retain timings <= best + max(relative_delta * best, absolute_us)."""

    relative_delta: float = 0.05
    absolute_us: float = 0.1

    def __post_init__(self) -> None:
        for name in ("relative_delta", "absolute_us"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

    def request(self) -> dict[str, float]:
        return {"relative_delta": self.relative_delta, "absolute_us": self.absolute_us}


@dataclass(frozen=True)
class TritonTuning:
    pilot_pruning: TritonPruning | None = TritonPruning(relative_delta=0.25, absolute_us=1.0)
    refined_pruning: TritonPruning | None = TritonPruning()
    evaluate: Callable[..., bool] | None = None
    quick_benchmark_group_size: int = 8
    quick_benchmark_variants_per_job: int = 256
    quick_benchmark_max_concurrent_jobs: int = 4
    replication_factor: int = 3
    replication_max_attempts: int = 8
    max_input_sets: int = 65536
    max_ring_bytes: int = 8 * 1024**3

    def __post_init__(self) -> None:
        for name in ("pilot_pruning", "refined_pruning"):
            if getattr(self, name) is not None and not isinstance(
                getattr(self, name), TritonPruning
            ):
                raise TypeError(f"{name} must be TritonPruning or None")
        for name in (
            "quick_benchmark_group_size",
            "quick_benchmark_variants_per_job",
            "quick_benchmark_max_concurrent_jobs",
            "replication_factor",
            "replication_max_attempts",
            "max_input_sets",
            "max_ring_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.quick_benchmark_max_concurrent_jobs > 32:
            raise ValueError("At most 32 concurrent GPU jobs are supported")
        if self.evaluate is not None and not callable(self.evaluate):
            raise TypeError("evaluate must be callable or None")

    def request(self) -> dict[str, Any]:
        return {
            "pilot_pruning": self.pilot_pruning.request() if self.pilot_pruning else None,
            "refined_pruning": self.refined_pruning.request() if self.refined_pruning else None,
            "group_size": self.quick_benchmark_group_size,
            "max_input_sets": self.max_input_sets,
            "max_ring_bytes": self.max_ring_bytes,
        }


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
