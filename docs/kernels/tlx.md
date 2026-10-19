# Meta TLX kernels

Read [managed kernels](index.md) and [tuning controls](tuning.md) first. The fragments below show
adapter-specific configuration. Their input tensors, evaluator, source, and `app` come from your
application. Complete example files are linked at the end of this chapter.

`TLXKernel` accepts Meta TLX kernels authored with `@triton.jit`, optionally wrapped in vanilla
`@triton.autotune`. It inherits the `TritonKernel` constructor, compiler sharding, tuning controls,
tensor transport, native reset/restore, replicated benchmarking and immutable specialization
reports.

Install the same pinned TLX-enabled Triton in the client and the selected image. Meta's distribution
is named `fbtriton`, while its import remains `triton`:

```sh
pip uninstall -y triton
pip install fbtriton==3.7.4
```

Use the existing App/Image/Function configuration:

```python
import gfaas as vfunc

kernel = vfunc.TLXKernel(
    native_tlx_kernel,
    variants_per_job=None,
    max_concurrent_jobs=8,
    cache_compression_level=1,
    tuning=vfunc.TritonTuning(evaluate=evaluate, replication_factor=3),
)
app = vfunc.App("tlx-example", image=vfunc.Image("your-tlx-enabled-image"))
with app.function(gpu_count=1, gpu_type="gb300"):
    measured = vfunc.benchmark(kernel[grid], *inputs, **kwargs)
    cached = vfunc.benchmark(kernel[grid], *inputs, **kwargs)
    kernel[grid](*inputs, **kwargs)  # Returns None; applies tensor writes.

print(cached["runtime_us"], cached["replicas"])
print(kernel.tuning_results[measured["specialization"]])
```

All `TritonTuning`, `TritonBenchmark`, `TritonPruning` and `KernelBenchmark` options apply
unchanged. A missing client TLX installation fails before work is submitted. Kernel source bundles
include the TLX import, so an incompatible remote runtime fails explicitly. TLX primitive imports
and JIT helper dependencies are reconstructed without importing the user's entire module remotely.

The shared argument rules apply: tensors, scalars and literal constexpr values. Host tensor
descriptors, arbitrary host objects and custom callbacks/pruning modifiers are unsupported. Create
descriptors and shared-memory layouts inside the JIT kernel where possible. Feature/hardware
compatibility still depends on the pinned Meta compiler release.

[examples/tlx_kernels.py](https://github.com/datacrunch-research/gpu-func/blob/main/examples/tlx_kernels.py)
provides warp-specialized dual addition, shared-memory row reduction, and shared-memory matmul,
including correctness evaluators.
