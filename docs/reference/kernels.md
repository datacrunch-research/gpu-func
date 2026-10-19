# Kernel constructor reference

Read [managed kernels](../kernels/index.md) for the execution model and dependency table. All
wrappers inherit `Kernel`. They use the active `App.function()` scope for image, resources, and
target selection.

The signatures below show constructor parameters and defaults. `kernel` is a native callable;
`source` is source text. Configuration classes and accepted ABIs are described in the linked
chapters.

## TritonKernel

```text
TritonKernel(kernel, *, tuning=None, variants_per_job=None, max_concurrent_jobs=8, cache_compression_level=1)
```

See [TritonKernel](../kernels/triton.md) for accepted source, dependencies, examples, and return
behavior.

## TLXKernel

```text
TLXKernel(kernel, *, tuning=None, variants_per_job=None, max_concurrent_jobs=8, cache_compression_level=1)
```

See [TLXKernel](../kernels/tlx.md) for accepted source, dependencies, examples, and return behavior.

## GluonKernel

```text
GluonKernel(kernel, *, tuning=None, variants_per_job=None, max_concurrent_jobs=8, cache_compression_level=1)
```

See [GluonKernel](../kernels/gluon.md) for accepted source, dependencies, examples, and return
behavior.

## HelionKernel

```text
HelionKernel(kernel, *, configs=None, tuning=None, variants_per_job=None, max_concurrent_jobs=8, cache_compression_level=1)
```

See [HelionKernel](../kernels/helion.md) for accepted source, dependencies, examples, and return
behavior.

## CUDAKernel

```text
CUDAKernel(source, *, name, signature, configs=None, tuning=None, reset_to_zero=(), restore_value=(), nvcc_flags=(), variants_per_job=128, max_concurrent_jobs=8)
```

See [CUDAKernel](../kernels/cuda.md) for accepted source, dependencies, examples, and return
behavior.

## ThunderKittensKernel

```text
ThunderKittensKernel(source, entrypoint, *, signature, entrypoint_kind='launcher', headers, configs=(ThunderKittensConfig(),), tuning=None, reset_to_zero=(), restore_value=(), nvcc_flags=(), variants_per_job=16, max_concurrent_jobs=8)
```

See [ThunderKittensKernel](../kernels/thunderkittens.md) for accepted source, dependencies,
examples, and return behavior.

## CutlassKernel

```text
CutlassKernel(source, *, argument_names, configurations=None, tuning=None, headers=None, include_dirs=(), nvcc_flags=(), architecture=None, reset_to_zero=(), restore_value=(), variants_per_job=None, max_concurrent_jobs=8)
```

See [CutlassKernel](../kernels/cutlass.md) for accepted source, dependencies, examples, and return
behavior.

## CuteDSLKernel

```text
CuteDSLKernel(kernel, *, configurations=None, tuning=None, variants_per_job=None, max_concurrent_jobs=8, compile_options='', reset_to_zero=(), restore_value=())
```

See [CuteDSLKernel](../kernels/cute.md) for accepted source, dependencies, examples, and return
behavior.

## Configuration and launch objects

`CUDAConfig(block=(256,), defines={}, shared_memory=0)` sets block dimensions, numeric compiler
macros, and dynamic shared-memory bytes.
`ThunderKittensConfig(defines={}, block=(128,), shared_memory=0)` serves the same purpose with
ThunderKittens-specific validation.

`CutlassTuning` accepts the shared `KernelTuning` fields. `TritonInputMetadata` is a typed
dictionary with `args` and `kwargs` descriptions for captured scalar values and tensor layouts. It
does not contain a callable input generator.

`CuteDSLKernel.from_source(source, *, entrypoint, cute_version, **options)` accepts the
constructor's other options. It parses the entry signature locally without executing the source
module.

`CUDAKernel.compile()` compiles its declared configurations without tensor arguments or tuning. It
still probes the GPU target. The returned dictionary contains compiler results, Call IDs, target,
and artifact references for application-managed reuse.

`kernel[grid]` produces a managed launch handle for Triton, TLX, Gluon, CUDA, and ThunderKittens.
Helion, C++ CUTLASS, and CuTe control their own launch geometry and use `kernel(...)` directly.

`benchmark(call, *args, options=None, **kwargs)` accepts either form. It returns an immutable report
with timings and replica evidence. `options` defaults to `KernelBenchmark()`. Read
[benchmarking](../kernels/benchmarking.md) and [shared policies](../kernels/tuning.md) before using
its three-GPU default.

Every wrapper exposes immutable reports through `tuning_results`, indexed by specialization digest.
Ordinary launches apply tensor writes; benchmarking leaves client tensors unchanged. Only Helion
transports general supported Helion return values. Other wrappers return `None` for ordinary
execution.
