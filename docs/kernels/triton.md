# Triton kernels

`TritonKernel` wraps a native `@triton.jit` kernel. It also accepts vanilla `@triton.autotune`
directly around JIT. The SDK extracts the configurations and uses its own tuning process.

Install matching Triton versions on the client and in the registered image. The client also needs
PyTorch; the image needs CUDA-enabled PyTorch. This API does not inherit the Triton 3.6.0
restriction of the [separate staged helper](../triton-tuning.md).

## First kernel

Save this complete program as `remote_add.py`. Supply a registered image and pool through the two
variables read in `main()`. The image must provide the packages described above.

```python
import os

import gfaas
import torch
import triton
import triton.language as tl


@triton.jit
def add(X, Y, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    values = tl.load(X + offsets, mask, other=0) + tl.load(Y + offsets, mask, other=0)
    tl.store(OUT + offsets, values, mask)


def main():
    x = torch.arange(1024, dtype=torch.float32)
    y = torch.ones_like(x)
    output = torch.empty_like(x)
    kernel = gfaas.TritonKernel(add)
    image = gfaas.Image(os.environ["KERNEL_IMAGE"])
    with gfaas.Client() as client:
        app = gfaas.App("remote-add", image=image, client=client)
        with app.function(gpu_count=1, gpu_type=os.environ["GFAAS_GPU_TYPE"]):
            kernel[(triton.cdiv(x.numel(), 256),)](x, y, output, N=x.numel(), BLOCK=256)
    torch.testing.assert_close(output, x + y)
    print(output[:8])


if __name__ == "__main__":
    main()
```

This example uses CPU tensors on the client. It compiles one configuration, runs it remotely, and
copies the output back. It does not request the default three-GPU benchmark.

## Add configurations and correctness checks

For a native autotuned kernel, `TritonKernel` reads the decorator's configurations. Multiple
configurations enable default tuning when no policy is supplied. It does not run Triton's autotuner.

Use `TritonTuning(evaluate=...)` to compare candidates with an expected result. The evaluator
receives a candidate callable followed by the kernel's runtime arguments. It returns `True` for an
accepted result. See [shared tuning controls](tuning.md).

`reset_to_zero` and `restore_value` on the native autotune decorator describe mutable benchmark
inputs. Reset occurs outside timed intervals. Ordinary execution still uses the caller's supplied
values.

## Accepted source and arguments

The wrapper reconstructs kernel and helper source without importing your entire module remotely. It
accepts tensors, scalar arguments, literal constexpr values, and a fixed or callable launch grid.
`TritonInputMetadata` describes captured input metadata in reports; it is not a tensor transport
API.

Unsupported wrappers, heuristics, arbitrary autotuning callbacks, custom pruning, IR overrides, and
unsupported modifiers fail validation. Host tensor descriptors and arbitrary host objects are not
part of the argument contract. Source validation does not prove that arbitrary Python is safe.

## Compiler batching

The constructor accepts `variants_per_job=None`, `max_concurrent_jobs=8`, and
`cache_compression_level=1`. These control CPU compilation batches, not GPU replication.

With no CPU or memory override, each compiler Call requests 16 cores and 4 GiB. Threads follow the
CPU budget, up to 32. Small batches can reduce the reservation. Eight full batches can request 128
cores and 32 GiB in total. Set explicit budgets for a smaller service.

Automatic batches target 128 configurations and grow to 256 before another dispatch wave. Compiler
archives and manifests are checked before GPU use. See [cache and recovery](lifecycle.md) for
results and failures.

The
[complete matmul example](https://github.com/datacrunch-research/gpu-func/blob/main/examples/triton_matmul_benchmark.py)
shows tuning, fresh benchmarking, cache reuse, and configuration controls. It is a larger
experiment, not a minimum-capacity smoke test.
