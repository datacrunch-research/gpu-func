# C++ CUTLASS kernels

Read [managed kernels](index.md) and [tuning controls](tuning.md) first. The fragments below show
adapter-specific configuration. Their input tensors, evaluator, source, and `app` come from your
application. Complete example files are linked at the end of this chapter.

`CutlassKernel` wraps C++ source defining `vfunc_launch`. Each configuration is a mapping of
preprocessor definitions (prefix names to avoid collisions with CUTLASS headers): CPU jobs build one
shared library per configuration. The active `app.function` supplies the image, GPU pool, resource
budgets and timeouts, as it does for `TritonKernel`.

```python
import gfaas as vfunc

kernel = vfunc.CutlassKernel(
    source,
    argument_names=("a", "b", "out"),
    configurations=[
        {"VF_TILE_M": 64, "VF_TILE_N": 64, "VF_STAGES": 2},
        {"VF_TILE_M": 128, "VF_TILE_N": 64, "VF_STAGES": 3},
    ],
    tuning=vfunc.CutlassTuning(evaluate=evaluate, replication_factor=3),
    include_dirs=("/opt/cutlass/include", "/opt/cutlass/tools/util/include"),
    nvcc_flags=(),
    architecture=None,          # Default: actual GPU's sm_ target.
    headers=None,               # Optional tar archive ArtifactRef.
    reset_to_zero=(),
    restore_value=(),
    variants_per_job=None,      # Automatic division across compiler jobs.
    max_concurrent_jobs=8,
)
with app.function(gpu_count=1, gpu_type="gb300", timeout=600, capacity_wait=600):
    cold = vfunc.benchmark(kernel, a, b, out)  # Compile + tune + benchmark.
    warm = vfunc.benchmark(kernel, a, b, out)  # Benchmark cached winner.
    kernel(a, b, out)                          # Execute and apply writes.
```

`CutlassTuning` has the same fields/defaults as `TritonTuning`: independent pilot/refined pruning
policies, optional evaluator, grouping/concurrency, replication, input-ring limits and a
`TritonBenchmark` timing policy. Standalone measurements accept the same `KernelBenchmark` controls.
Reports are immutable, stored by specialization in `kernel.tuning_results`, and contain
configurations, compiler diagnostics, per-GPU timings, numerical verdicts and Call IDs. Cache keys
include source, configurations, tensor metadata, image, environment, GPU pool, header identity,
compiler options, reset/restore declarations and evaluator.

The image must contain nvcc, a host C++ compiler, CUDA runtime libraries and PyTorch. CUTLASS
headers can be provided by immutable image include directories, or by `headers=ArtifactRef(...)`
referencing a tar archive containing `include/` and optionally `tools/util/include/`. Archive
revision selection belongs to the caller; it is not downloaded or changed during a kernel call.
`architecture` can select an architecture-specific target such as `sm_103a`, provided its numeric
architecture matches the assigned GPU. A single configuration compiles and executes without tuning
unless a tuning policy is explicitly supplied.

## C++ launch contract

The compiler prepends this ABI declaration to the source (do not redefine it):

```cpp
struct VFuncArgument {
    int32_t kind, dtype;
    void* data;
    int64_t ndim;
    const int64_t* shape;
    const int64_t* strides;
    int64_t i64;
    double f64;
};
extern "C" int vfunc_launch(
    const VFuncArgument* args, int64_t count, cudaStream_t stream);
```

Arguments follow `argument_names` order, including keyword arguments. `kind` is 0 for None, 1 for a
tensor, 2 for int64, 3 for float64 and 4 for bool. Tensor `data` addresses its first element,
including storage offset. Shapes and strides are int64; strides count elements. Scalar values use
`i64` or `f64`. Dtype codes are 1=float32, 2=float16, 3=bfloat16, 4=float64, 5=int8, 6=uint8,
7=int16, 8=int32, 9=int64 and 10=bool. Additional codes are 11=float8_e4m3fn, 12=float8_e5m2,
13=float8_e4m3fnuz, 14=float8_e5m2fnuz, 15=uint16, 16=uint32, 17=uint64, 18=complex64,
19=complex128, 20=complex32, 21=float8_e8m0fnu and 22=float4_e2m1fn_x2, when supported by the
installed PyTorch. These codes describe raw storage; the launcher chooses the corresponding C++
interpretation.

The launcher validates its expected dtype/layout, launches its CUTLASS operator on the supplied
current CUDA stream, and returns zero on success. It must consume host metadata before returning;
the arrays are temporary. Launch geometry and CUTLASS workspace handling belong in the C++ launcher
(workspace tensors can be supplied as ordinary arguments). Graph benchmarking requires a
capture-compatible launcher. During tuning, a nonzero launch status rejects that variant after a
healthy synchronization check; a CUDA synchronization fault terminates the job. An ordinary
selected-kernel call propagates a nonzero launch status.

The wrapper preserves tensor views, storage aliases and client object identity. Benchmark calls
measure scratch copies and leave client tensors unchanged; ordinary calls return None and apply all
tensor writes. If repeated calls need input resets, declare the argument names in `reset_to_zero` or
`restore_value`. Restores use the original host snapshot, without a device backup for every input
slot. GPU failures retain shard evidence but do not yet support automatic restart after a poisoned
CUDA context.

See
[examples/cutlass_kernel.py](https://github.com/datacrunch-research/gpu-func/blob/main/examples/cutlass_kernel.py)
and
[examples/cutlass/gemm.cu](https://github.com/datacrunch-research/gpu-func/blob/main/examples/cutlass/gemm.cu)
for a BF16 GEMM example, and
[examples/cutlass/vector_add.cu](https://github.com/datacrunch-research/gpu-func/blob/main/examples/cutlass/vector_add.cu)
for strided CuTe C++ tensor access. The Python CuTe DSL is a separate frontend and is not accepted
by this C++ class.
