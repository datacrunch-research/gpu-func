# Benchmark a selected kernel

Use `gfaas.benchmark()` to measure a managed kernel's selected configuration. This differs from
benchmarking a complete program through the CLI. The result describes GPU execution, not upload,
queue, compilation, or client-to-service latency.

The example below continues an application that already defines `app`, `kernel`, `grid`, and inputs.
See [the first kernel example](triton.md) for a complete program.

`benchmark` accepts vFunc-managed launch handles (`Kernel` and its adapters). It prepares an
uncached specialization and tunes it when the wrapper policy requires tuning. It then measures the
selected configuration. A cached specialization reuses its compiler artifacts and configuration,
then obtains fresh timings. Benchmarking leaves the caller's tensors unchanged and inherits the
kernel's optional correctness evaluator and native reset/restore declarations.

```python
import gfaas as vfunc

options = vfunc.KernelBenchmark(
    estimate_trials=3,
    final_duration_ms=25.0,
    min_final_trials=25,
    max_final_trials=1000,
    graph_duration_ms=1.0,
    min_calls_per_graph=10,
    max_calls_per_graph=100,
    l2_flush_iterations=100,
    replication_factor=3,
    replication_max_attempts=8,
    max_concurrent_jobs=4,
    max_input_sets=65536,
    max_ring_bytes=8 * 1024**3,
)
with app.function(gpu_count=1, gpu_type="gb300"):
    cold = vfunc.benchmark(kernel[grid], a, b, out, M=m, N=n, K=k, options=options)
    warm = vfunc.benchmark(kernel[grid], a, b, out, M=m, N=n, K=k, options=options)
print(warm["runtime_us"])           # Arithmetic mean of distinct GPU minima
print(warm["replicas"])             # GPU UUIDs, timings, trials, evaluation, Call IDs
print(warm["reused_specialization"]) # True: no repeat compilation or autotuning
```

Each new benchmark checks that the compiled kernel launches without error, optionally evaluates it
on scratch inputs, and estimates duration with `estimate_trials` rotating-input event measurements.
It uses the final autotuning phase: CUDA graphs for short kernels, direct events for longer kernels
or when graph padding would exceed memory limits. Estimates size the graph and trial counts; the
final duration is an estimated compute budget, rather than a wall-time deadline. `max_final_trials`
counts graph replays or individual direct calls. Standalone replication reserves all requested GPUs
in one Call. The job verifies and restores the selected configuration once, then estimates and
measures it on each GPU separately. Physical GPU UUIDs must be distinct. No configuration pruning is
needed. The selected compiler shard remains a compressed batch: its archive is scanned, but only the
winner's cache files are restored. Other compiler shards are not requested. The concurrency setting
does not parallelize devices within this job.

The returned report is immutable and includes `runtime_us`, `configuration`, `replicas`,
`specialization`, `autotuned`, `reused_specialization`, raw shard reports and benchmarking Call IDs.
Fresh benchmark results do not overwrite the original autotuning report in `kernel.tuning_results`.

See
[the complete matmul example](https://github.com/datacrunch-research/gpu-func/blob/main/examples/triton_matmul_benchmark.py)
for 1,024 configuration combinations, all tuning/benchmark options, and both cache paths.
