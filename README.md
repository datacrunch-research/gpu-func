# gfaas SDK and command-line clients

This repository is the public home of the gfaas Python SDK and command-line
clients. It provides two command surfaces:

- `gfaas` runs Python and CUDA files, manages durable Calls and Artifacts, and
  can run CUDA directly on a local NVIDIA GPU.
- `gpu-func` operates CUDA exercises and custom kernels on a remote GPU through
  gfaas. The local computer does not need CUDA or an NVIDIA GPU.

The CUDA client provides these workflows:

- Compile, test, benchmark, profile, sanitize, or grade a CUDA course exercise.
- Compile, operate, or profile a custom CUDA program.
- Submit durable gfaas Calls that support cancellation and retained events.
- Publish Nsight Compute reports as gfaas Artifacts.
- Read an existing `.ncu-rep` file on a computer with Nsight Compute.

Read the [gfaas SDK guide](docs/introduction.md) for installation, concepts,
examples, and the `gfaas` command reference.
Read [GUIDE.md](GUIDE.md) for the `gpu-func` command reference.

## Install

Install the SDK and both commands from the public Git repository:

```bash
uv tool install "git+https://github.com/datacrunch-research/gpu-func.git"
gfaas --help
gpu-func --help
```

The legacy `gpu_func_cli` command remains available during the rename.

## Configure credentials

Set the gfaas API address and API key in the environment:

```bash
export GFAAS_API_BASE="https://gpu.example.com/api"
export GFAAS_API_KEY="..."
gpu-func pools
```

The CLI does not accept an API key argument. This rule keeps the key out of the
shell history and the process list.

## Use the Python SDK

The SDK provides a small function API and a lower-level durable Call client:

```python
import gfaas

app = gfaas.App("hello")


@app.function(image=gfaas.Image.from_registry("pytorch-cu130"), gpu_type="gb300")
def gpu_name() -> str:
    import torch

    return torch.cuda.get_device_name(0)


print(gpu_name.remote())
```

The general CLI can submit Python and CUDA source files:

```bash
gfaas run experiment.py --gpu-type gb300
gfaas run kernel.cu --gpu-type gb300 -- --problem-size 4096
```

Use `gfaas local run` to run trusted CUDA source on a local NVIDIA GPU:

```bash
gfaas local info
gfaas local run kernel.cu -- --problem-size 4096
```

Calls remain available after the submitting process disconnects. Use the CLI
to inspect or cancel them:

```bash
gfaas call show call_...
gfaas call logs call_... --follow
gfaas call cancel call_... --reason "no longer needed"
```

## Operate a custom CUDA program

Use `--gpu-type` if the coordinator has more than one GPU pool. The CLI selects
the pool automatically if the coordinator has exactly one pool.

```bash
gpu-func custom run kernel.cu
gpu-func custom run kernel.cu --harness harness.cu --gpu-type gb300
gpu-func custom profile kernel.cu --artifact-dir ./profiles
```

The worker detects its CUDA architecture by default. Use `--arch` only when the
source needs an explicit compilation target.

## Operate a course exercise

Run a command from a directory that contains `run.py` and `runner/cli.py`:

```bash
gpu-func compile
gpu-func test
gpu-func benchmark
gpu-func sanitizer
gpu-func profile --artifact-dir ./profiles
gpu-func grade
```

Use `--exercise-dir` to select an exercise from a different directory.

## Durable Calls

Use `--detach` to return after submission:

```bash
gpu-func custom run kernel.cu --detach
gfaas call watch call_...
gfaas call logs call_... --follow
gfaas call artifacts call_...
```

If you interrupt a foreground command, `gpu-func` requests Call cancellation.
The Call identity remains available in the coordinator.

## Remote data model

`gpu-func` sends the selected source files as an immutable tree Artifact. The
worker copies that tree to its scratch directory before compilation.

The CLI rejects symbolic links, hard links, unsafe paths, oversized workspaces,
and existing local output files. Binary exercise fixtures remain unchanged.

Nsight Compute reports do not travel in result JSON. The worker publishes them
through the declared `profiles` output Artifact.

## Develop

Create the locked development environment and run all checks from this
repository:

```bash
uv sync --extra dev --locked
uv run ruff format --check src tests
uv run ruff check src tests
uv run mypy src
uv run pytest -q
mdbook build docs
mdbook test docs
```
