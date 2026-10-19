# Managed kernels

A managed kernel wrapper accepts your kernel and its configurations. The SDK compiles them remotely,
can measure them on GPUs, and executes the selected configuration. You still supply the algorithm,
inputs, image, and correctness checks.

These wrappers are useful for kernel experiments and tuning. A remote launch includes data transfer
and service overhead. It is not a replacement for every launch inside a local GPU application. For a
complete Python workload with many launches, an ordinary [remote function](../python-functions.md)
can keep its data and control loop on the worker.

## Choose a wrapper

| Wrapper                                   | Input                                                   | Launch syntax       | Return value                            |
| ----------------------------------------- | ------------------------------------------------------- | ------------------- | --------------------------------------- |
| [TritonKernel](triton.md)                 | A native JIT kernel, optionally with vanilla autotuning | `kernel[grid](...)` | `None`; writes back tensors             |
| [TLXKernel](tlx.md)                       | A TLX-enabled Triton JIT kernel                         | `kernel[grid](...)` | `None`; writes back tensors             |
| [GluonKernel](gluon.md)                   | A Gluon JIT kernel                                      | `kernel[grid](...)` | `None`; writes back tensors             |
| [HelionKernel](helion.md)                 | A native Helion kernel using its Triton backend         | `kernel(...)`       | Helion result, including tensor outputs |
| [CUDAKernel](cuda.md)                     | CUDA C++ with an unmangled device entry point           | `kernel[grid](...)` | `None`; writes back tensors             |
| [ThunderKittensKernel](thunderkittens.md) | CUDA C++ and local ThunderKittens headers               | `kernel[grid](...)` | `None`; writes back tensors             |
| [CutlassKernel](cutlass.md)               | C++ source with a `vfunc_launch` host function          | `kernel(...)`       | `None`; writes back tensors             |
| [CuteDSLKernel](cute.md)                  | A CuTe DSL host entry point or source                   | `kernel(...)`       | `None`; writes back tensors             |

The shared `Kernel` base class identifies these managed wrappers. `benchmark()` accepts their launch
handles. It does not accept an arbitrary Python function or native kernel directly.

## Prepare the client and image

The basic SDK installation does not install GPU frameworks or compilers. For tensor arguments, the
client needs PyTorch. CPU tensors work; a local GPU is not required.

| Adapter        | Additional client packages                         | Image requirements                                                         |
| -------------- | -------------------------------------------------- | -------------------------------------------------------------------------- |
| Triton         | Matching Triton                                    | Triton, CUDA-enabled PyTorch                                               |
| TLX            | Matching TLX-enabled Triton (`fbtriton`)           | Same compiler distribution, CUDA-enabled PyTorch                           |
| Gluon          | Matching Triton with Gluon                         | Same Triton version, CUDA-enabled PyTorch                                  |
| Helion         | Helion and matching dependencies                   | Matching Helion, PyTorch, and Triton                                       |
| CUDA C++       | No local CUDA toolkit required                     | `nvcc`, host compiler, CUDA runtime, PyTorch                               |
| ThunderKittens | Local header directory                             | Compatible CUDA toolkit, C++20 compiler, PyTorch                           |
| C++ CUTLASS    | No local CUDA toolkit required                     | CUDA toolkit, host compiler, PyTorch, CUTLASS headers or a header Artifact |
| CuTe DSL       | Matching `nvidia-cutlass-dsl` for callable entries | CuTe DSL, PyTorch, CUDA bindings and runtime libraries                     |

CuTe's `from_source()` path does not require CuTe installed on the client. It declares the expected
remote version explicitly. Each adapter chapter describes its accepted source and package
requirements.

Choose a registered image with an immutable digest and reported CPU architecture. The target probe
must return that identity. Old servers or unregistered worker-local images that cannot provide it
cannot support these managed launches.

## Select the execution environment

All wrappers use an active `app.function()` scope:

```python
import gfaas

app = gfaas.App("kernels", image=gfaas.Image("YOUR_KERNEL_IMAGE"))

# Fragment: define kernel, grid, and tensors in your application first.
with app.function(gpu_count=1, gpu_type="YOUR_POOL", timeout=600):
    kernel[grid](x, y, output)
```

The block runs locally. Only the wrapper's submitted Calls run remotely. Select one GPU target in
the scope. Replication can request more GPUs later, according to the tuning or benchmark policy.

Image, environment variables, hardware requirements, and resource limits come from the scope.
Compilation requests no GPU. Read [function scopes](../python-functions.md#function-scopes) for
inheritance and client ownership.

## Understand a launch

1. The SDK copies input tensor storage into a host snapshot.
2. A GPU probe identifies the image, CPU architecture, and GPU compute capability.
3. For a new specialization, CPU Calls compile configurations. Tuning Calls measure candidates when
   tuning is enabled.
4. An execution Call runs the selected configuration on a fresh copy of the supplied inputs.
5. The client copies tensor writes into the original tensor objects.

A specialization is a compiled form for a particular source, configuration, target, and argument
layout. The cache also accounts for the image and environment. Tensor contents do not identify a
specialization, so new values can reuse a compiled winner.

Triton-derived wrappers, CUDA, ThunderKittens, CUTLASS, and CuTe skip default tuning for one
configuration. Helion uses its shared tuning policy even for its bound default configuration.
Explicit policies can request measurement where default behavior skips it.

## Timing, correctness, and state

An ordinary launch can modify the supplied tensors. `benchmark()` uses scratch copies and leaves the
caller's tensors unchanged. Helion also transports its return value; other wrappers use tensor
writes.

A successful compile or launch does not establish numerical correctness. Supply an evaluator when
you need to reject incorrect candidates. Declare input reset rules before repeated measurements.
Read [tuning controls](tuning.md) and [benchmarking](benchmarking.md).

Caches belong to the wrapper in the current client process. They are not a durable tuning database.
Every invocation still probes the image and target. Read [cache and recovery behavior](lifecycle.md)
before relying on cached execution in a long-running application.

The [older staged Triton helper](../triton-tuning.md) remains a separate API. It accepts source
text, explicit cases, and a target architecture. It creates one two-stage Call rather than several
managed Calls.
