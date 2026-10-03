# gfaas SDK and vfunc CLI

This repository is the public home of the gfaas Python SDK and its `vfunc`
command. The command runs Python and CUDA files, manages durable Calls and
Artifacts, and can run CUDA directly on a local NVIDIA GPU. It also operates
CUDA exercises and custom kernels on remote GPUs.

`vfunc` replaces the previous `gfaas` command. The Python distribution and import namespace remain
`gfaas`.

The CUDA client provides these workflows:

- Compile, test, benchmark, profile, sanitize, or grade a CUDA course exercise.
- Compile, operate, or profile a custom CUDA program.
- Submit durable gfaas Calls that support cancellation and retained events.
- Publish Nsight Compute reports as gfaas Artifacts.
- Read an existing `.ncu-rep` file on a computer with Nsight Compute.

Read the [gfaas SDK guide](docs/introduction.md) for installation, concepts,
examples, and the `vfunc` command reference.
Read [GUIDE.md](GUIDE.md) for the CUDA workflow guide.

## Install

Install the SDK and command from the public Git repository:

```bash
uv tool install "git+https://github.com/datacrunch-research/gpu-func.git"
vfunc --help
```

## Configure credentials

Set the gfaas API address and API key in the environment:

```bash
export GFAAS_API_BASE="https://gpu.example.com/api"
export GFAAS_API_KEY="..."
vfunc pool list
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
vfunc run experiment.py --gpu-type gb300
vfunc run kernel.cu --gpu-type gb300 -- --problem-size 4096
```

Remote CUDA Calls use two durable stages. The compile stage uses CPU and memory, but it holds no GPU
lease. The execution stage requests the GPU after the compiled Artifact is durable.

The [Nemotron LoRA guide](docs/fine-tuning-nemotron.md) covers bounded
fine-tuning Calls, checkpoints, resume, and adapter download.

Use `vfunc local run` to run trusted CUDA source on a local NVIDIA GPU:

```bash
vfunc local info
vfunc local run kernel.cu -- --problem-size 4096
```

Calls remain available after the submitting process disconnects. Use the CLI
to inspect or cancel them:

```bash
vfunc call show call_...
vfunc call logs call_... --follow
vfunc call artifacts call_...
vfunc artifact download art_... ./result
vfunc call cancel call_... --reason "no longer needed"
```

Generate completion setup for Bash, Fish, Zsh, or PowerShell:

```bash
eval "$(vfunc completion bash)"
```

## Operate a custom CUDA program

Use `--gpu-type` if the coordinator has more than one GPU pool. The CLI selects
the pool automatically if the coordinator has exactly one pool.

```bash
vfunc custom run kernel.cu
vfunc custom run kernel.cu --harness harness.cu --gpu-type gb300
vfunc custom profile kernel.cu --artifact-dir ./profiles
```

The `run` and `profile` commands compile without a GPU lease. They publish the prepared workspace
as an internal Artifact. The GPU stage starts after the Artifact is durable and a GPU is reserved.
The `compile` command requests no GPU.

The worker detects its CUDA architecture by default. Use `--arch` only when the
source needs an explicit compilation target.

## Operate a course exercise

Run a command from a directory that contains `run.py` and `runner/cli.py`:

```bash
vfunc compile
vfunc test
vfunc benchmark
vfunc sanitizer
vfunc profile --artifact-dir ./profiles
vfunc grade
```

Use `--exercise-dir` to select an exercise from a different directory.

## Durable Calls

Use `--detach` to return after submission:

```bash
vfunc custom run kernel.cu --detach
vfunc call watch call_...
vfunc call logs call_... --follow
vfunc call artifacts call_...
```

If you interrupt a foreground command, `vfunc` requests Call cancellation.
The Call identity remains available in the coordinator.

## Remote data model

`vfunc` sends the selected source files as an immutable tree Artifact. The
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

## Compile-only Triton wrapper

```python
import gfaas as vfunc

app = vfunc.App("kernels", image=vfunc.Image("your-registered-triton-image"))
kernel = vfunc.TritonKernel(
    your_kernel, app=app, target_arch=103, cpu_millicores=16000,
)

try:
    kernel[grid](a, b, out, M=m, N=n, K=k)
except vfunc.TritonExecutionNotImplementedError as result:
    print(result.call_ids, result.report)
```

`your_kernel` may be plain `@triton.jit` with one supplied configuration or vanilla
`@triton.autotune` directly wrapping JIT. All declared unique configurations are
compiled in CPU-only Calls with bounded parallelism. No kernel,
benchmark, evaluator, or grid callable runs. Tensor contents are never uploaded.
A successful call deliberately raises `TritonExecutionNotImplementedError` with
its compilation report. Per-variant compiler failures raise
`TritonCompilationError`, preserving the report and successful artifacts.
Each Call retains the named `compiled-triton` output tree containing a packed cache
archive and SHA-256 manifest. Packing avoids publishing each compiler cache file
as a separate Artifact.

The defaults use 16 CPU compiler threads per job, 4 GiB of memory per job,
and at most eight concurrent jobs. Batches target 128 configurations per job;
within one dispatch window, chunks can grow to 256 before adding another wave.
For example, 1,024 configurations use eight jobs of 128, and 1,536 use eight jobs
of 192. Small batches use one job and reserve at most one CPU core per variant.
This can reserve up to 128 CPU cores and 32 GiB concurrently. Actual throughput
depends on available service capacity.

Environment and job settings use the existing `App`, `Image`, and `Function`
configuration path. Compiler jobs inherit `app.image` and `app.client`; an
`image=` override works like `@app.function(image=...)`. The wrapper forwards
`gpu_type`, `cpu_millicores`, `memory_bytes`, `timeout`, `capacity_wait`, `env`,
and storage/log/output limits through an ordinary CPU-only Function. The App
owns the client. Compilation threads follow the CPU budget (one thread per
whole core, up to 32), and small shards clamp their CPU reservation.

Set `cpu_millicores`, `memory_bytes`, and `max_concurrent_jobs` to control that budget.
An explicit `variants_per_job` sets a hard chunk limit instead of the automatic
policy; setting it at least as large as your configuration set uses a single
job. `cache_compression_level` defaults to 1 for faster lossless cache packing
and accepts gzip levels 0–9. Higher compression reduces transfer size at the
cost of packing time.

Multi-job reports preserve original variant order and include `call_ids` and
`shards`, with each shard's report and output name. Structural job failures stop
further submissions; active jobs are awaited and their artifacts remain
available. Individual variant failures are recorded without stopping other
configurations. The wrapper leaves the App-owned client open for reuse.

The report separates compiler imports, metadata decoding, source setup,
compilation, cache hashing, packing, and cleanup. It also records CPU time and
cache sizes. Configuration metadata travels as JSON, allowing large batches
without exhausting the SDK's bounded Artifact reference inspection.

Validation rejects additional wrappers/subclasses, heuristics, pruning,
callbacks, reset/restore behavior, custom benchmarking, IR overrides, and
unsupported launch modifiers before submission. Mutable wrappers are revalidated
on each invocation. Inspectable source is reconstructed from kernel/helper
functions and simple globals; importing the user's whole module remotely is
avoided. Closure dependencies and unsupported source/argument forms fail clearly.
Arbitrary Python source is not statically proven free of side effects.

Install Triton separately in the client environment. The remote image must have
the same Triton version and support the explicit GPU target. The compiler does not
initialize CUDA. Runtime scalars and pointer alignment use conservative,
unspecialized types; constexpr values retain their specified specialization.
Metadata-rich tensor descriptors, nested runtime arguments, and non-scalar
constexpr values are not implemented yet. This is a batch compiler on existing
vFunc jobs, not a deployed persistent compiler service.

The compatibility boundary probes `ASTSource` for the older `constants` interface
or newer `constexprs` interface. Tests cover both contracts; live CPU compilation
has been checked with Triton 3.8.0. Other releases require qualification before
claiming production support. See [examples/triton_compile.py](examples/triton_compile.py).
