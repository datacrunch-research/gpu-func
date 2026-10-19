# CUDA C++ kernels

Read [managed kernels](index.md) and [tuning controls](tuning.md) first. The fragments below show
adapter-specific configuration. Their input tensors, evaluator, source, and `app` come from your
application. Complete example files are linked at the end of this chapter.

`CUDAKernel` implements the same managed `Kernel` interface as `TritonKernel`. It compiles CUDA C++
to cubins on CPU jobs, tunes configurations with vFunc's benchmark pipeline, and executes the
selected kernel on the original arguments. The environment and GPU target come from
`app.function(...)`.

```python
import gfaas as vfunc
import torch

source = r'''
extern "C" __global__ void add(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
'''

def grid(meta):
    return ((meta["n"] + meta["block"][0] - 1) // meta["block"][0],)

def evaluate(candidate, a, b, out, n):
    expected = a + b
    candidate(a, b, out, n)
    return bool(torch.allclose(out, expected))

kernel = vfunc.CUDAKernel(
    source,
    name="add",
    signature={"a": "pointer", "b": "pointer", "out": "pointer", "n": "int32"},
    configs=[vfunc.CUDAConfig(block=(128,)), vfunc.CUDAConfig(block=(256,))],
    tuning=vfunc.KernelTuning(evaluate=evaluate, replication_factor=3),
)
with app.function(gpu_count=1, gpu_type="gb300"):
    cold = vfunc.benchmark(kernel[grid], a, b, out, n)
    warm = vfunc.benchmark(kernel[grid], a, b, out, n)
    kernel[grid](a, b, out, n)  # Returns None, preserves writes and client aliases.
```

The signature specifies ordered argument names and exact ABI types: `pointer`, `int32`, `uint32`,
`int64`, `uint64`, `float32`, or `float64`. It must match the C++ entry point. Pointer arguments
accept tensors or `None`; tensor element dtypes must match the C++ pointer types. CPU tensors work
through the existing storage transport. Strides and offsets are preserved; pass stride arguments
explicitly when your kernel needs them. Struct arguments and host launch wrappers are not supported;
provide an `extern "C" __global__` entry point.

`CUDAConfig(block=(256,), defines={"TILE": 128}, shared_memory=0)` specifies threads per block,
numeric `nvcc -D` definitions, and dynamic shared-memory bytes. The grid callback receives argument
values, defines, and the selected `block`. `CUDAKernel` also exposes `nvcc_flags=()`,
`variants_per_job=128`, and `max_concurrent_jobs=8`. The image must contain nvcc and PyTorch; the
client needs PyTorch but does not require a GPU, CUDA toolkit or Triton installation.

`reset_to_zero=("out",)` and `restore_value=("input",)` operate before timed reuse, using the
canonical host snapshot. They govern benchmarking, not ordinary execution. Input rings retain
aliasing and are subject to the same count/byte limits as Triton kernels. Omitting `tuning` uses the
normal policy for multiple configurations; a single configuration compiles and executes without
tuning.

`KernelTuning`, `BenchmarkSettings` and `Pruning` are shared-policy aliases for `TritonTuning`,
`TritonBenchmark` and `TritonPruning`. All pilot/refinement pruning, trial counts, durations,
grouping, replication and ring limits are available. Standalone `vfunc.benchmark` uses
`KernelBenchmark` and freshly measures the cached winner in one replicated job, loading only its
binary from the compiler shard. Reports are immutable and stored in `kernel.tuning_results` by
specialization.

For controllers that separate compilation from execution, `kernel.compile()` compiles every declared
configuration without tensor inputs or autotuning. Call it inside `app.function(...)`; it returns
the GPU target, per-configuration results, compiler Call IDs, and artifact references. These
artifacts can be retained and loaded by a separate GPU job.

The full example in
[examples/cuda_kernel.py](https://github.com/datacrunch-research/gpu-func/blob/main/examples/cuda_kernel.py)
exercises aliasing, strided reset/restore, matrix multiplication, rejected compilation and rejected
accuracy. Replication requires enough free GPUs on one worker. A poisoned CUDA process still fails
its Call; durable restart/checkpoint recovery remains follow-up work.
