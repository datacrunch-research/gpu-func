# ThunderKittens kernels

Read [managed kernels](index.md) and [tuning controls](tuning.md) first. The fragments below show
adapter-specific configuration. Their input tensors, evaluator, source, and `app` come from your
application. Complete example files are linked at the end of this chapter.

`ThunderKittensKernel` compiles ThunderKittens CUDA C++ configurations on CPU workers. It shares the
tuning and measurement pipeline with `TritonKernel`. It inherits the image, GPU target and resources
selected in `app.function(...)`; it does not introduce another environment configuration.

```python
import gfaas as vfunc

kernel = vfunc.ThunderKittensKernel(
    source,                       # CUDA C++ source, including kittens.cuh
    "launch",                     # extern "C" host launcher
    signature={"A": "pointer", "B": "pointer", "C": "pointer", "N": "int32"},
    headers="/path/to/ThunderKittens/include",
    configs=[
        vfunc.ThunderKittensConfig({"TILE": 64}, block=(64,)),
        vfunc.ThunderKittensConfig({"TILE": 128}, block=(128,)),
    ],
    tuning=vfunc.TritonTuning(evaluate=evaluate, replication_factor=3),
    reset_to_zero=("C",),
)
with app.function(gpu_count=1, gpu_type="gb300"):
    result = vfunc.benchmark(kernel[grid], a, b, c, a.numel())
    # The same specialization reuses its compiled winner without repeating tuning.
    cached = vfunc.benchmark(kernel[grid], a, b, c, a.numel())
    kernel[grid](a, b, c, a.numel())   # Executes and writes back into c.

print(cached["runtime_us"], cached["configuration"], cached["replicas"])
print(kernel.tuning_results)
```

The shared `TritonTuning`, `TritonPruning`, `TritonBenchmark` and `KernelBenchmark` policy classes
are shared by both implementations. Every pruning, trial count/duration, grouping, concurrency,
replication and ring limit remains configurable. One configuration runs without tuning by default;
multiple configurations enable the default tuning policy. The evaluator is optional.

## CUDA entry points

The default `entrypoint_kind="launcher"` expects an exported host launcher returning `int` or
`cudaError_t`, where zero means success. Its parameters match the declared signature, followed by
`VFUNC_LAUNCH_ARGS`. The compiler supplies:

```cpp
extern "C" int launch(float* x, int n, VFUNC_LAUNCH_ARGS) {
    // Construct native ThunderKittens global layouts/TMA descriptors here.
    native_kernel<<<VFUNC_GRID, VFUNC_BLOCK, VFUNC_SHARED_BYTES, VFUNC_STREAM>>>(...);
    return static_cast<int>(cudaGetLastError());
}
```

`VFUNC_LAUNCH_ARGS` adds seven unsigned integers (grid x/y/z, block x/y/z, dynamic shared-memory
bytes) and `cudaStream_t`. Use the supplied stream so measurements and CUDA graph capture observe
the launch. The grid callable receives a dictionary containing the bound runtime arguments,
configuration macros, and `block`. Host launchers can preserve native descriptor construction and
custom launch setup without needing PyTorch C++ bindings.

Alternatively, `entrypoint_kind="kernel"` accepts an unmangled `extern "C" __global__ void` entry
point and launches its cubin directly through the CUDA Driver API. This mode has no extra
parameters. Struct arguments and arbitrary Python/C++ return objects are not transported. Pass
tensor pointers and supported scalar values, or construct structs in a host launcher.

Signatures preserve argument order and accept `pointer`, `int32`, `uint32`, `int64`, `uint64`,
`float32` and `float64`. CUDA compilation checks the exported entry point's ABI against this
declaration before any GPU runs. Pointer values must be tensors or `None`; CUDA source authors are
responsible for using the correct tensor dtype and layout. Ordinary launches return `None` and
preserve client tensor identity, strides, aliases and writes. Benchmarks leave client values
unchanged.

## Configuration and provenance

`ThunderKittensConfig(defines={}, block=(128,), shared_memory=0)` supplies numeric compile-time
macros, a one-to-three-dimensional CUDA block, and dynamic shared-memory bytes. The discovered GPU
selects the ThunderKittens architecture macro. Configuration macros cannot override `KITTENS_*`.

The constructor snapshots the supplied include directory into one immutable, SHA-256-verified header
artifact. Source, headers, flags, configurations, input metadata, callbacks and image identity
participate in specialization caching; changing local header files later does not change an existing
wrapper.

Compilation is sharded with `variants_per_job=16` and `max_concurrent_jobs=8` by default, with up to
four compiler processes per job. `nvcc_flags=()` accepts additional flags while vFunc manages output
mode and target architecture. The selected image must include nvcc, a C++20 host compiler, CUDA
headers and compatible PyTorch. NVIDIA's pip-distributed nvcc is supported even when it is absent
from PATH. No Triton installation is required on the client.

Cached execution and standalone benchmarking retain only the winning compiler shard and load only
the selected binary. Standalone R-GPU measurements use one job and load binary bytes once before
measuring each GPU. The scheduler requires those R GPU slots on one worker. Input reset/restore
remains per GPU.

`reset_to_zero` and `restore_value` specify pointer argument names. `restore_value` is needed for
inputs mutated by a kernel when every benchmarked invocation must see the original values; it is
applied outside timed regions. Normal execution uses the values supplied by the caller and copies
all tensor writes back. CUDA context poisoning still fails the worker Call; durable
checkpoint/restart is not implemented.

See
[examples/thunderkittens_kernels.py](https://github.com/datacrunch-research/gpu-func/blob/main/examples/thunderkittens_kernels.py)
for complete vector-add, in-place exp, BF16 matmul and raw CUDA definitions, correctness callbacks,
launchers and configuration sets.
