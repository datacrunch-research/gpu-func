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

## Triton compilation and quick benchmarking

```python
import gfaas as vfunc

app = vfunc.App("kernels", image=vfunc.Image("registered-triton-image"))
kernel = vfunc.TritonKernel(
    native_kernel,
    tuning=vfunc.TritonTuning(
        quick_benchmark_delta=0.10,
        evaluate=evaluate,
        pruning_min_runtime_us=100,
    ),
)
with app.function(gpu="gb300", cpu_millicores=16000, memory_bytes=4 * 1024**3):
    try:
        kernel[grid](a, b, out, M=m, N=n, K=k)
    except vfunc.TritonExecutionNotImplementedError as result:
        print(result.call_ids, result.report)
```

`app.function(...)` works as its existing decorator and as a context manager.
The block executes on the client and its vFunc calls execute remotely. Nested
scopes on the same App inherit omitted settings; explicit values replace them,
including `None` to clear a setting. `env` dictionaries replace the parent
value. Contexts restore on exit, including exceptions, and are isolated across
threads and asynchronous tasks. Decorated functions capture their settings.
Kernel calls snapshot the current context; a kernel can be reused in different
contexts safely.

`TritonKernel` has no App, image, resource, or architecture arguments. A call
requires an active context with an image and one GPU target. A small GPU Call
discovers its CUDA architecture before CPU-only compilation. Benchmarking checks
that its assigned GPU matches. Image overrides, timeouts, capacity waits,
CPU/memory, environment variables and storage/log/output limits use the ordinary
Function submission path. The App owns the client. Phase outputs are managed
by the wrapper.

Plain JIT and vanilla autotuning directly wrapping JIT are accepted. vFunc
extracts the configurations and never runs Triton's autotuner. Additional
wrappers, heuristics, callbacks, pruning, custom benchmarking, IR overrides and
unsupported modifiers fail before submission. Mutable kernels are revalidated.
Kernel/helper source is reconstructed without importing the author's module;
unsupported dependencies fail clearly. Arbitrary Python source is not statically
proven free of side effects.

CPU compiler threads follow the CPU budget (one per whole core, up to 32).
Omitted compilation resources default to 16 cores and 4 GiB per job. Small
shards clamp CPU reservations. Defaults allow eight concurrent jobs with chunks
targeting 128 configurations and growing to 256 before another dispatch wave.
1,024 variants use eight chunks of 128; 1,536 use eight of 192. This can reserve
128 cores and 32 GiB concurrently. `variants_per_job`, `max_concurrent_jobs`,
and `cache_compression_level` remain compiler batching options (gzip level 1
by default). Per-shard archives and SHA-256 manifests are verified before GPU
use. Reports preserve variant order, Call IDs, phase timings, CPU time and cache
sizes. Structural failures stop queued jobs while active results are retained.

### Quick benchmark policy

`quick_benchmark_delta` is an optional finite nonnegative fraction. `0.10`
retains configurations within 10% of the final fastest accepted quick runtime.
`pruning_min_runtime_us` defaults to 100: faster configurations bypass pruning.
Set it to zero to disable that exemption. Invalid and failed kernels are excluded.

Each compiled kernel is synchronized, launched once and synchronized again to
catch execution errors. `quick_benchmark_group_size` controls interleaved groups
(default eight). Each group enqueues 100 L2 flushes, then five rounds of
start event → one kernel launch → end event → flush, visiting every kernel in
order each round. One synchronization completes the group's events.

All pilot groups finish before refinement. Take each kernel's pilot minimum;
exclude invalid kernels, then prune runtimes at least twice the fastest valid
pilot runtime, except those below `pruning_min_runtime_us`. This first cutoff
uses the best across batches in the same GPU shard. The client applies the final
configured delta against the global valid best when all shards return.

Regroup survivors for the second stage. Each kernel uses
`ceil(10_000 / pilot_us)` iterations to cover 10 ms of compute; fewer than ten
skips refinement. Groups enqueue another 100 flushes and interleave kernels
round by round until each has completed its own iteration count. Synchronize,
then return each kernel's minimum. Reports retain both stages and pruning status.
Flushes zero twice the reported L2 cache size outside timed intervals.

Quick GPU jobs reuse one input set across their configurations. Warmup and
repeated launches can mutate these inputs; evaluation uses fresh original inputs.
`quick_benchmark_variants_per_job` defaults to 256 and
`quick_benchmark_max_concurrent_jobs` to four, reserving up to four GPUs.

The optional `evaluate(candidate, *args, **kwargs) -> bool` runs remotely whenever
a result would establish a new best. `candidate` is an ordinary callable fixed
to one configuration and the current grid; compiled constexpr values cannot
change. Only a passing evaluation updates the best. Evaluator exceptions or
non-boolean returns fail the phase. Without an evaluator, correctness is assumed.
Without a valid best, the phase fails rather than returning a winner.

Quick input transport currently supports PyTorch tensors: `torch.save` snapshots
preserve strides and shared storage, and load onto the job's assigned GPU. Grid
and evaluator callbacks must be ordinary Python functions. Their code and Python
helpers are transported without requiring the author's module remotely. Use the
provided inputs instead of capturing CUDA tensors in callbacks.

Omitting `tuning` compiles only. `TritonTuning(quick_benchmark_delta=None)` skips
quick benchmarking and evaluation. Successful quick benchmarking also stops at
the full-benchmark boundary. These paths raise `TritonExecutionNotImplementedError`
with a report; full benchmarking, replication, final selection and execution
are deferred. `TritonQuickBenchmarkError` covers a failed GPU phase or no valid
best; `TritonCompilationError` retains compilation failures.

Install Triton on the client separately. The execution image must have the same
version and support the selected hardware, with PyTorch for quick benchmarking.
Runtime scalar/alignment specialization remains conservative. Tensor descriptors,
nested runtime arguments and non-scalar constexpr values are not supported yet.
The adapter probes old `ASTSource(constants=...)` and new
`ASTSource(constexprs=...)` interfaces; live qualification covers Triton 3.8.0 /
GB300. Other versions require qualification. See
[examples/triton_compile.py](examples/triton_compile.py).
