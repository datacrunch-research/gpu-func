# CuTe DSL kernels

Read [managed kernels](index.md) and [tuning controls](tuning.md) first. The fragments below show
adapter-specific configuration. Their input tensors, evaluator, source, and `app` come from your
application. Complete example files are linked at the end of this chapter.

`CuteDSLKernel` implements the same managed compilation, tuning, replicated benchmarking,
specialization cache and tensor-write contract as `TritonKernel`. It wraps the **`@cute.jit` host
entry point** that launches device kernels. CuTe host entries define their own launch grid, so call
the wrapper directly:

```python
import gfaas as vfunc

kernel = vfunc.CuteDSLKernel(
    add_one,
    configurations=[{"BLOCK": 64}, {"BLOCK": 128}, {"BLOCK": 256}],
    tuning=vfunc.KernelTuning(evaluate=evaluate_add, replication_factor=3),
    variants_per_job=2,
    max_concurrent_jobs=2,
)
with app.function(gpu_count=1, gpu_type="gb300", timeout=600, capacity_wait=300):
    cold = vfunc.benchmark(kernel, a, b)  # Compile/tune, then fresh R=3 measurement.
    warm = vfunc.benchmark(kernel, a, b)  # Fresh measurement of the saved winner.
    kernel(a, b)                        # Execute once and apply writes to b.
print(warm["runtime_us"], warm["configuration"]["constants"])
```

Services that receive kernel source can construct a wrapper without importing that source in the
client process:

```python
kernel = vfunc.CuteDSLKernel.from_source(
    source,
    entrypoint="kernel_entry",
    cute_version="4.8.0",
    configurations=[{"BLOCK": 64}, {"BLOCK": 128}],
    tuning=vfunc.KernelTuning(evaluate=evaluate_add, replication_factor=3),
)
```

`from_source` parses the host signature without executing the module. Host parameters must have
explicit names and literal defaults. The module executes only in remote compilation/execution jobs.
`cute_version` declares the expected version in the selected application image; the GPU probe
verifies it before compilation. This entry point does not require CuTe installed in the client.

For the callable entry point, install the **same `nvidia-cutlass-dsl` version** in the client and
the selected vFunc image. Images also need PyTorch, CUDA bindings and CuTe's runtime libraries. No
package installation occurs implicitly in the SDK. App/Function image, resource, target, timeout and
environment settings apply normally.

`KernelTuning`, `KernelPruning` and `KernelTiming` are shared aliases of `TritonTuning`,
`TritonPruning` and `TritonBenchmark`; all existing controls and defaults apply. Standalone
measurements use `KernelBenchmark`. `kernel.tuning_results` holds immutable reports by
specialization. Benchmarking preserves client tensors and the original tuning report. Ordinary calls
return `None` and preserve writes to the original tensors, including aliased views.

## CuTe-specific controls

- `configurations`: nonempty list of literal mappings naming `cutlass.Constexpr` parameters on the
  host entry. Scalar values and nested tuples/lists (such as tile shapes) are supported. Defaults to
  `[{}]`. Multiple configurations enable vFunc tuning automatically.
- `tuning`: shared `KernelTuning`, including an optional accuracy evaluator.
- `variants_per_job`: configurations per CPU job; `None` distributes the search across at most
  `max_concurrent_jobs` jobs.
- `max_concurrent_jobs`: maximum simultaneous CPU compiler jobs, default 8.
- `compile_options`: CuTe's public compilation option string, default empty. vFunc manages GPU
  target and the object ABI. Target and ABI overrides are rejected.
- `reset_to_zero` / `restore_value`: argument names requiring reset/restore before benchmark reuse.
  Defaults to empty tuples. Actual execution uses the supplied values. Restores use canonical host
  snapshots rather than extra GPU backups.

A typed `stream: cuda.CUstream` parameter is injected using the worker's current PyTorch stream;
callers omit it. This enables event timing and CUDA graph capture on the correct stream. Entries
without that parameter use direct-event final measurements. Source and helper definitions are
bundled without importing the user's entire script. Callable classes with literal/type state are
also supported; construct more complex DSL state inside the host entry or a bundled helper.

CPU workers compile from fake tensor descriptors, preserving shapes, strides and dtypes without
allocating GPU inputs. They export independent object files using CuTe's public AOT API. GPU workers
verify and load only requested objects, then reuse the shared input-ring and measurement pipeline.
Compilations use isolated processes to avoid sharing CuTe's compiler state between configurations.
Standalone benchmarks request only the winning compiler shard and measure all assigned replicas in
one GPU job, verifying the selected object files once.

See
[examples/cute_kernel.py](https://github.com/datacrunch-research/gpu-func/blob/main/examples/cute_kernel.py)
for the full client example and
[examples/cute_kernels.py](https://github.com/datacrunch-research/gpu-func/blob/main/examples/cute_kernels.py)
for vector addition, a callable class performing strided copy, and a small reference-style matmul.

The tensor/scalar transport restrictions of `TritonKernel` still apply. Arbitrary Python objects,
device-entry-only functions, non-literal configuration objects and host return values are outside
this launch contract. Compile failures are retained per variant; poisoned GPU processes fail with
retained Call evidence. Replication uses the existing verified-distinct-GPU scheme and placement
limits.
