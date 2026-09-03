# gfaas SDK and command-line clients

This repository is the public home of the gfaas Python SDK and its `gfaas`
command. The command runs Python and CUDA files, manages durable Calls and
Artifacts, and can run CUDA directly on a local NVIDIA GPU. It also operates
CUDA exercises and custom kernels on remote GPUs.

The CUDA client provides these workflows:

- Compile, test, benchmark, profile, sanitize, or grade a CUDA course exercise.
- Compile, operate, or profile a custom CUDA program.
- Submit durable gfaas Calls that support cancellation and retained events.
- Publish Nsight Compute reports as gfaas Artifacts.
- Read an existing `.ncu-rep` file on a computer with Nsight Compute.

Read the [gfaas SDK guide](docs/introduction.md) for installation, concepts,
examples, and the `gfaas` command reference.
Read [GUIDE.md](GUIDE.md) for the CUDA workflow guide.

## Install

Install the SDK and command from the public Git repository:

```bash
uv tool install "git+https://github.com/datacrunch-research/gpu-func.git"
gfaas --help
```

## Configure credentials

Set the gfaas API address and API key in the environment:

```bash
export GFAAS_API_BASE="https://gpu.example.com/api"
export GFAAS_API_KEY="..."
gfaas pool list
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

The [Nemotron LoRA guide](docs/fine-tuning-nemotron.md) covers bounded
fine-tuning Calls, checkpoints, resume, and adapter download.

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
gfaas call artifacts call_...
gfaas artifact download art_... ./result
gfaas call cancel call_... --reason "no longer needed"
```

Generate completion setup for Bash, Fish, Zsh, or PowerShell:

```bash
eval "$(gfaas completion bash)"
```

## Operate a custom CUDA program

Use `--gpu-type` if the coordinator has more than one GPU pool. The CLI selects
the pool automatically if the coordinator has exactly one pool.

```bash
gfaas custom run kernel.cu
gfaas custom run kernel.cu --harness harness.cu --gpu-type gb300
gfaas custom profile kernel.cu --artifact-dir ./profiles
```

The worker detects its CUDA architecture by default. Use `--arch` only when the
source needs an explicit compilation target.

## Operate a course exercise

Run a command from a directory that contains `run.py` and `runner/cli.py`:

```bash
gfaas compile
gfaas test
gfaas benchmark
gfaas sanitizer
gfaas profile --artifact-dir ./profiles
gfaas grade
```

Use `--exercise-dir` to select an exercise from a different directory.

## Durable Calls

Use `--detach` to return after submission:

```bash
gfaas custom run kernel.cu --detach
gfaas call watch call_...
gfaas call logs call_... --follow
gfaas call artifacts call_...
```

If you interrupt a foreground command, `gfaas` requests Call cancellation.
The Call identity remains available in the coordinator.

## Remote data model

`gfaas` sends the selected source files as an immutable tree Artifact. The
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
