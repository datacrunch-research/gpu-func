# Examples and API reference

The repository contains standalone Python and CUDA examples in `examples/cli/`.
Each example is a complete source file that the `gfaas` command can submit.

## CUDA example

Run the CUDA example on a remote GPU:

```bash
gfaas run examples/cli/hello_cuda.cu --gpu-type "$GFAAS_GPU_TYPE"
```

Run the same source on a local NVIDIA GPU:

```bash
gfaas local run examples/cli/hello_cuda.cu
```

The local and remote commands use the same source file. You can compare their
structured results with the `--json` option.

## Python example

Run the Python example with a prepared PyTorch image:

```bash
gfaas run examples/cli/hello_python.py \
  --image "$GFAAS_PYTHON_IMAGE" \
  --gpu-type "$GFAAS_GPU_TYPE"
```

The selected image must contain each package that the script imports.

## More CLI examples

Read the [CLI example guide](../examples/cli/README.md) for output files,
profiling, local execution, and Call management.

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
