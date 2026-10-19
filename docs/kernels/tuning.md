# Tuning controls

Tuning selects a configuration from the candidates you supply. It does not prove that your algorithm
is correct or suitable for another input. A benchmark measures the selected configuration again.

## Shared policy names

The public names below refer to the same policy classes:

| Preferred shared name | Equivalent names                                         |
| --------------------- | -------------------------------------------------------- |
| `KernelTuning`        | `TritonTuning`; `CutlassTuning` provides the same fields |
| `Pruning`             | `KernelPruning`, `TritonPruning`                         |
| `BenchmarkSettings`   | `KernelTiming`, `TritonBenchmark`                        |

`KernelBenchmark` is separate. It controls a fresh measurement through `benchmark()`, rather than
the search that selects a configuration. An uncached benchmark can run both operations.

## Start with one GPU

The default replication factor is three. Replication requires that many free GPUs on one worker. For
a one-GPU experiment, set the factor in both policies:

```python
import gfaas

search = gfaas.KernelTuning(replication_factor=1)
measurement = gfaas.KernelBenchmark(replication_factor=1)
# Pass search as the wrapper's tuning argument.
# Pass measurement as benchmark(..., options=measurement).
```

A one-GPU scope chooses the target. It does not override the policy's replication factor. A default
benchmark can wait for three GPUs even when the ordinary execution needs only one.

## Check candidate results

This evaluator is for an addition kernel with three runtime tensor arguments:

```python
def evaluate(candidate, x, y, output):
    import torch

    candidate(x, y, output)
    return bool(torch.allclose(output, x + y))
```

Pass it as `KernelTuning(evaluate=evaluate, ...)`. Match the evaluator's arguments to the wrapped
kernel. For Helion, the candidate can return tensors rather than only writing supplied outputs. Keep
the evaluator self-contained and supply its dependencies in the image.

Without an evaluator, the measurement process assumes numerical correctness. A successful CUDA
launch alone does not establish a correct answer. Benchmarks inherit the wrapper's evaluator.

## Restore mutable inputs

Repeated measurements use independent input copies. Within a copy, tensor views and storage aliases
remain intact. If a kernel accumulates into an input, later measurements need its original values.

For Triton, TLX, and Gluon, use native `reset_to_zero` and `restore_value` declarations. The CUDA,
ThunderKittens, CUTLASS, and CuTe constructors accept these argument names directly. Helion detects
mutated storage during warmup and restores it before reuse.

Reset and restore operations occur outside the timed region. They affect measurements, not the
values used for ordinary execution. The caller remains responsible for declaring the needed reset
behavior.

## Policy fields

`KernelTuning` accepts:

| Field                                 | Default               | Purpose                                        |
| ------------------------------------- | --------------------- | ---------------------------------------------- |
| `pilot_pruning`                       | `Pruning(0.25, 1.0)`  | Retain candidates near the pilot's best time   |
| `refined_pruning`                     | `Pruning(0.05, 0.1)`  | Retain candidates near the refined best time   |
| `evaluate`                            | `None`                | Optional numerical correctness callback        |
| `quick_benchmark_group_size`          | `8`                   | Variants measured together in a pilot group    |
| `quick_benchmark_variants_per_job`    | `256`                 | Maximum variants in a GPU batch                |
| `quick_benchmark_max_concurrent_jobs` | `4`                   | GPU batch concurrency, at most 32              |
| `replication_factor`                  | `3`                   | Distinct physical GPUs per completed candidate |
| `replication_max_attempts`            | `8`                   | Bound on replication attempts                  |
| `max_input_sets`                      | `65536`               | Maximum input-ring slots                       |
| `max_ring_bytes`                      | `8 * 1024**3`         | Maximum ring allocation in bytes               |
| `benchmark`                           | `BenchmarkSettings()` | Trial counts and timing budgets                |

`Pruning(relative_delta=0.05, absolute_us=0.1)` retains times no greater than
`best + max(relative_delta * best, absolute_us)`. Set a pruning policy to `None` to disable pruning
for that round. The thresholds are heuristics; use an unpruned comparison when selection quality
matters.

`BenchmarkSettings` accepts these controls:

| Field                    | Default | Unit or meaning                  |
| ------------------------ | ------- | -------------------------------- |
| `pilot_trials`           | `3`     | Event trials                     |
| `refinement_duration_ms` | `1.0`   | Estimated kernel compute time    |
| `min_refinement_trials`  | `10`    | Minimum count to run refinement  |
| `max_refinement_trials`  | `250`   | Refinement trial cap             |
| `final_duration_ms`      | `25.0`  | Estimated kernel compute time    |
| `min_final_trials`       | `25`    | Minimum direct-event trial count |
| `max_final_trials`       | `1000`  | Direct trials or graph replays   |
| `graph_duration_ms`      | `1.0`   | Target compute time per graph    |
| `min_calls_per_graph`    | `10`    | Minimum count to use a graph     |
| `max_calls_per_graph`    | `100`   | Captured launch cap              |
| `l2_flush_iterations`    | `100`   | Cache-flush operations           |

Durations are compute estimates, not wall-clock deadlines. Trial caps can shorten the requested
compute budget. Call timeouts remain separate. Durations must be finite and positive, counts must be
positive integers, and each minimum must not exceed its maximum.

`KernelBenchmark` uses the final timing, graph, flush, ring, and replication fields above. It adds
`estimate_trials=3` and `max_concurrent_jobs=4`; it has no pruning or evaluator argument. The
wrapper supplies the evaluator. Its concurrency setting does not parallelize devices inside a single
replica Call.

Read [measurement details](measurement.md) for input rings, graph replay, and ranking. Read
[benchmarking](benchmarking.md) for the returned report.
