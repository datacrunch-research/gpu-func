"""Kernel launch handles and replicated final-phase benchmarking."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .triton_policy import TritonBenchmark, TritonTuning


@dataclass(frozen=True)
class KernelBenchmark:
    """Fresh final measurements; durations in ms, memory limits in bytes."""

    estimate_trials: int = 3
    final_duration_ms: float = 25.0
    min_final_trials: int = 25
    max_final_trials: int = 1000
    graph_duration_ms: float = 1.0
    min_calls_per_graph: int = 10
    max_calls_per_graph: int = 100
    l2_flush_iterations: int = 100
    replication_factor: int = 3
    replication_max_attempts: int = 8
    max_concurrent_jobs: int = 4
    max_input_sets: int = 65536
    max_ring_bytes: int = 8 * 1024**3

    def __post_init__(self) -> None:
        self.tuning_policy()

    def tuning_policy(self) -> TritonTuning:
        return TritonTuning(
            pilot_pruning=None,
            refined_pruning=None,
            benchmark=TritonBenchmark(
                pilot_trials=self.estimate_trials,
                final_duration_ms=self.final_duration_ms,
                min_final_trials=self.min_final_trials,
                max_final_trials=self.max_final_trials,
                graph_duration_ms=self.graph_duration_ms,
                min_calls_per_graph=self.min_calls_per_graph,
                max_calls_per_graph=self.max_calls_per_graph,
                l2_flush_iterations=self.l2_flush_iterations,
            ),
            quick_benchmark_group_size=1,
            quick_benchmark_variants_per_job=1,
            quick_benchmark_max_concurrent_jobs=self.max_concurrent_jobs,
            replication_factor=self.replication_factor,
            replication_max_attempts=self.replication_max_attempts,
            max_input_sets=self.max_input_sets,
            max_ring_bytes=self.max_ring_bytes,
        )


class Kernel(ABC):
    """Base for vFunc-managed kernels and their launch handles."""

    def __getitem__(self, grid: Any) -> KernelCall:
        return KernelCall(self, grid)

    @abstractmethod
    def _invoke(
        self,
        grid: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        options: KernelBenchmark | None,
    ) -> Any:
        pass


@dataclass(frozen=True)
class KernelCall:
    kernel: Kernel
    grid: Any

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.kernel._invoke(self.grid, args, kwargs, None)


def benchmark(
    call: KernelCall | Kernel, *args: Any, options: KernelBenchmark | None = None, **kwargs: Any
) -> Any:
    """Autotune if necessary, then freshly measure the winner on distinct GPUs.

    Use benchmark(kernel[grid], *args, options=KernelBenchmark(...), **kwargs)
    for Triton, or benchmark(kernel, *args) for callable Helion kernels, inside
    app.function. Benchmarking leaves the caller's tensors unchanged.
    """
    if isinstance(call, Kernel) and callable(call):
        call = KernelCall(call, None)
    if not isinstance(call, KernelCall) or not isinstance(call.kernel, Kernel):
        raise TypeError("benchmark requires a vFunc Kernel launch, such as kernel[grid]")
    if options is not None and not isinstance(options, KernelBenchmark):
        raise TypeError("options must be KernelBenchmark")
    return call.kernel._invoke(call.grid, args, kwargs, options or KernelBenchmark())
