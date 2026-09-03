# Examples and API reference

The repository contains standalone Python and CUDA examples in `examples/cli/`.
Each example is a complete source file that the `vfunc` command can submit.

## CUDA example

Run the CUDA example on a remote GPU:

```bash
vfunc run examples/cli/hello_cuda.cu --gpu-type "$GFAAS_GPU_TYPE"
```

Run the same source on a local NVIDIA GPU:

```bash
vfunc local run examples/cli/hello_cuda.cu
```

The local and remote commands use the same source file. You can compare their
structured results with the `--json` option.

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

| Example | Model family |
| --- | --- |
| `examples/nemotron_lora.py` | Nemotron 3 Nano and the 9B pilot model |
| `examples/nemotron_lightning_lora.py` | Nemotron 3.5 Lightning 30B-A3B |

Install the optional tokenizer dependencies before you run these examples:

```bash
uv sync --extra training-example
```

Read [Fine-tune Nemotron with LoRA](fine-tuning-nemotron.md) before you submit a training Call.

## API reference

The SDK exports its public Python API from the `gfaas` package. Start with
these modules:

| Module | Purpose |
| --- | --- |
| `gfaas` | Applications, functions, Calls, Images, and Artifacts |
| `gfaas.client` | The lower-level HTTP client |
| `gfaas.cuda` | CUDA compilation, execution, and profiling |
| `gfaas.artifacts` | Artifact references and declared outputs |

The public REST contract is in the
[fast-container OpenAPI document](https://github.com/datacrunch-research/fast-container/blob/main/openapi/gfunc.yaml).
