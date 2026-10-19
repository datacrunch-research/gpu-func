import pytest

from gfaas import KernelBenchmark, benchmark


@pytest.mark.parametrize(
    "options",
    [
        {"estimate_trials": 0},
        {"final_duration_ms": float("nan")},
        {"max_final_trials": 20},
        {"graph_duration_ms": -1},
        {"min_calls_per_graph": 101},
        {"l2_flush_iterations": 0},
        {"replication_factor": 0},
        {"replication_max_attempts": 0},
        {"max_concurrent_jobs": 33},
        {"max_input_sets": 0},
        {"max_ring_bytes": 0},
    ],
)
def test_invalid_benchmark_options_fail_before_submission(options):
    with pytest.raises(ValueError):
        KernelBenchmark(**options)


def test_benchmark_rejects_unmanaged_python_functions():
    with pytest.raises(TypeError, match="vFunc Kernel"):
        benchmark(lambda: None)
