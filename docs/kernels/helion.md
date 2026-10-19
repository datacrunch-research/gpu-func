# Helion kernels

Read [managed kernels](index.md) and [tuning controls](tuning.md) first. The fragments below show
adapter-specific configuration. Their input tensors, evaluator, source, and `app` come from your
application. Complete example files are linked at the end of this chapter.

`HelionKernel` accepts a native `@helion.kernel` using Helion's Triton backend. It retains ordinary
call syntax and returns the actual Helion return value, including nested tensor outputs and storage
aliases. Writes to supplied tensors are applied to the existing client objects. CPU inputs are
supported; new outputs return to the first input tensor's client device.

```python
import helion
import helion.language as hl
import torch
import gfaas as vfunc

@helion.kernel(configs=[helion.Config(block_sizes=[n]) for n in (64, 128, 256)])
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + y[tile]
    return out

def evaluate(candidate, x, y):
    return bool(torch.equal(candidate(x, y), x+y))

kernel = vfunc.HelionKernel(
    add,
    tuning=vfunc.KernelTuning(evaluate=evaluate, replication_factor=3),
    variants_per_job=16,
    max_concurrent_jobs=8,
    cache_compression_level=1,
)
with app.function(gpu_count=1, gpu_type="gb300"):
    out = kernel(x, y)
    measured = vfunc.benchmark(kernel, x, y, options=vfunc.KernelBenchmark())
print(measured["runtime_us"], measured["configuration"], measured["replicas"])
print(kernel.tuning_results)
```

The image must contain matching Helion/PyTorch/Triton versions. Image, hardware, resources, capacity
wait and timeout come from the existing `app.function` context. The client does not need a local
GPU.

The pipeline binds on the selected GPU and intercepts every generated device launch, then compiles
the emitted Triton kernels in bounded parallel CPU jobs. GPU measurement loads these binaries and
verifies that loading did not recompile them. vFunc uses its own interleaved pilot, refinement,
pruning, correctness checks, final CUDA graph/direct ring measurements and distinct-GPU replication.
It does not run Helion's autotuner. Mutated input storages are detected during warmup and restored
from the host snapshot before timed reuse. Compiled configurations containing several device
launches remain one candidate and are measured as a complete call.

`configs=` optionally overrides the native decorator's configurations. Duplicates are removed. When
neither supplies configurations, vFunc compiles Helion's bound default configuration; it does not
implicitly explore Helion's full search space. Native Helion autotuning effort/search settings do
not configure vFunc's tuning algorithm.

`KernelTuning`, `KernelPruning`, and `KernelTiming` are shared aliases for the existing
`TritonTuning`, `TritonPruning`, and `TritonBenchmark` controls, with identical defaults. The
standalone `KernelBenchmark` [shared controls](tuning.md) also apply to Helion. A cold benchmark
prepares/compiles/tunes, then measures the winner; a cached benchmark restores only the winner's
device-launch caches and measures all replicas in one job. Benchmark reports and tuning reports are
immutable; fresh measurements leave original inputs and the saved tuning report unchanged.

This implementation accepts flat tensor/scalar arguments, fixed configuration lists, and the Triton
backend. Custom specialization callbacks and closures are rejected. The adapter to Helion's
generated `_launcher` parameter is capability-checked; an unrecognized generated launcher fails
explicitly. See
[examples/helion_kernels.py](https://github.com/datacrunch-research/gpu-func/blob/main/examples/helion_kernels.py)
for vector addition, row reduction, BF16 matmul, in-place mutation, and a grid-barrier function
returning both intermediate and reduced tensors.
