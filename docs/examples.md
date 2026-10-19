# Examples

The repository contains standalone Python and CUDA examples in `examples/cli/`. Each example is a
complete source file that the `vfunc` command can submit.

## CUDA example

Run the CUDA example on a remote GPU:

```bash
vfunc run examples/cli/hello_cuda.cu --gpu-type "$GFAAS_GPU_TYPE"
```

Run the same source on a local NVIDIA GPU:

```bash
vfunc local run examples/cli/hello_cuda.cu
```

The local and remote commands use the same source file. You can compare their structured results
with the `--json` option.

## Python example

Run the Python example with a prepared PyTorch image:

```bash
vfunc run examples/cli/hello_python.py \
  --image "$GFAAS_PYTHON_IMAGE" \
  --gpu-type "$GFAAS_GPU_TYPE"
```

The selected image must contain each package that the script imports.

## More CLI examples

Read the [command-line interface](cli.md) chapter for local execution, output files, and Call
management. Read [CUDA development workflows](cuda-workflows.md) for profiling and course exercises.

## Nemotron LoRA examples

The repository contains two supervised fine-tuning examples:

| Example                               | Model family                           |
| ------------------------------------- | -------------------------------------- |
| `examples/nemotron_lora.py`           | Nemotron 3 Nano and the 9B pilot model |
| `examples/nemotron_lightning_lora.py` | Nemotron 3.5 Lightning 30B-A3B         |

Install the optional tokenizer dependencies before you run these examples:

```bash
uv sync --extra training-example
```

Read [Fine-tune Nemotron with LoRA](fine-tuning-nemotron.md) before you submit a training Call.

## Managed kernel examples

The kernel examples require local tensor/compiler packages and a compatible registered image. Read
[managed kernels](kernels/index.md) before running them. Some examples request three GPUs and large
compilation batches; they are not all suitable for a one-GPU service.

| Example                                                                                                        | Workload                                         |
| -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| [First Triton kernel](kernels/triton.md)                                                                       | A small one-configuration addition               |
| [Triton matmul](https://github.com/datacrunch-research/gpu-func/blob/main/examples/triton_matmul_benchmark.py) | Configuration search and fresh/cached benchmarks |
| [TLX](https://github.com/datacrunch-research/gpu-func/blob/main/examples/tlx_kernels.py)                       | Shared-memory and warp-specialized kernels       |
| [Gluon](https://github.com/datacrunch-research/gpu-func/blob/main/examples/gluon_kernels.py)                   | Copy, accumulating addition, and softmax         |
| [Helion](https://github.com/datacrunch-research/gpu-func/blob/main/examples/helion_kernels.py)                 | Tensor returns, mutation, reduction, and matmul  |
| [CUDA C++](https://github.com/datacrunch-research/gpu-func/blob/main/examples/cuda_kernel.py)                  | Raw device entry points and configuration tuning |
| [ThunderKittens](https://github.com/datacrunch-research/gpu-func/blob/main/examples/thunderkittens_kernels.py) | Host launchers and CUDA configurations           |
| [C++ CUTLASS](https://github.com/datacrunch-research/gpu-func/blob/main/examples/cutlass_kernel.py)            | Shared-library launchers and GEMM                |
| [CuTe DSL](https://github.com/datacrunch-research/gpu-func/blob/main/examples/cute_kernel.py)                  | Host entries compiled to reusable objects        |

See the [Python API reference](reference/index.md) for constructors, options, return values, and
errors.
