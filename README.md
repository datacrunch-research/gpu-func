# gpu-func

`gpu-func` operates CUDA exercises and custom kernels on a remote GPU through
gfaas. The local computer does not need CUDA or an NVIDIA GPU.

The CLI provides these workflows:

- Compile, test, benchmark, profile, sanitize, or grade a CUDA course exercise.
- Compile, operate, or profile a custom CUDA program.
- Submit durable gfaas Calls that support cancellation and retained events.
- Publish Nsight Compute reports as gfaas Artifacts.
- Read an existing `.ncu-rep` file on a computer with Nsight Compute.

Read [GUIDE.md](GUIDE.md) for the complete command reference.

## Install

Install this repository with the current gfaas SDK checkout:

```bash
uv tool install --editable /path/to/gpu-func --with-editable /path/to/gfaas
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

Keep the gfaas checkout next to this repository. Then create the locked
development environment and run all checks:

```bash
uv sync --extra dev --locked
uv run ruff format --check src tests
uv run ruff check src tests
uv run mypy src
uv run pytest -q
```
