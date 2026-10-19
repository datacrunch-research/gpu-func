# Gluon kernels

Read [managed kernels](index.md) and [tuning controls](tuning.md) first. The fragments below show
adapter-specific configuration. Their input tensors, evaluator, source, and `app` come from your
application. Complete example files are linked at the end of this chapter.

`GluonKernel` supports `@gluon.jit` kernels and vanilla `@triton.autotune` wrappers around them. It
shares `TritonKernel`'s CPU compilation shards, input snapshots, reset/restore declarations,
replicated tuning, specialization cache, tensor writeback and standalone benchmark API. Triton's
native autotuner is not executed.

```python
import gfaas as vfunc

import triton

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

@gluon.jit
def copy(X, Y, N: gl.constexpr, BLOCK: gl.constexpr):
    offsets = gl.program_id(0) * BLOCK + gl.arange(
        0, BLOCK, layout=gl.BlockedLayout([1], [32], [4], [0])
    )
    gl.store(Y + offsets, gl.load(X + offsets, offsets < N, other=0), offsets < N)

kernel = vfunc.GluonKernel(
    copy,
    tuning=None,
    variants_per_job=None,
    max_concurrent_jobs=8,
    cache_compression_level=1,
)
with app.function(gpu_count=1, gpu_type="gb300"):
    report = vfunc.benchmark(
        kernel[(triton.cdiv(n, 256),)], x, y, N=n, BLOCK=256,
        options=vfunc.KernelBenchmark(replication_factor=3),
    )
    kernel[(triton.cdiv(n, 256),)](x, y, N=n, BLOCK=256)
```

`TritonTuning`, `TritonPruning`, `TritonBenchmark` and `KernelBenchmark` retain the same options and
defaults for Gluon. Multi-configuration wrappers enable default tuning when `tuning` is omitted.
Ordinary calls return `None` and apply writes to the supplied tensors. Benchmarking preserves those
tensors and returns an immutable report. Reports remain keyed by specialization in
`kernel.tuning_results`.

The selected image and client must provide matching Triton versions with Gluon support. CPU
compilation and GPU preparation select the Gluon AST source; the adapter probes the public export
first and uses the runtime module on releases such as Triton 3.8. See Triton's
[Gluon runtime API](https://triton-lang.org/main/gluon/api/runtime.html). Source bundles preserve
Gluon JIT helpers and imports of layout constructors. Create layouts inside device code; non-scalar
constexpr arguments, host-side tensor descriptors, arbitrary callbacks, and non-vanilla autotuning
modifiers remain unsupported.

[Runnable Gluon examples](https://github.com/datacrunch-research/gpu-func/blob/main/examples/gluon_kernels.py)
cover strided copy, accumulating add with `reset_to_zero`, and row softmax. Each example compiles,
benchmarks on three distinct GPUs, benchmarks the cached specialization again, and executes the
selected kernel with tensor writeback.
