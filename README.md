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

## Triton compilation, tuning, and execution

```python
import gfaas as vfunc

app = vfunc.App("kernels", image=vfunc.Image("registered-triton-image"))
kernel = vfunc.TritonKernel(
    native_kernel,
    tuning=vfunc.TritonTuning(
        refined_pruning=vfunc.TritonPruning(relative_delta=0.05, absolute_us=0.1),
        evaluate=evaluate,
    ),
)
with app.function(gpu="gb300", cpu_millicores=16000, memory_bytes=4 * 1024**3):
    kernel[grid](a, b, out, M=m, N=n, K=k)
# out now contains the remote kernel's writes.
for specialization, result in kernel.tuning_results.items():
    print(specialization, result["benchmark"]["best_configuration"])
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

### Benchmark inputs, pruning and replication

`TritonKernel` snapshots the actual invocation tensors into host storage bytes.
Each GPU clones those storages into independent input-ring slots, preserving
values, shapes, strides, offsets, dtypes, and aliases within each slot. CUDA
storage is not serialized through cloudpickle. The client needs PyTorch for tensor
transport; scalar arguments remain ordinary JSON values.

Use [Triton's documented declarations](https://triton-lang.org/main/python-api/generated/triton.autotune.html),
`reset_to_zero=["output"]` and
`restore_value=["state"]` declarations on `@triton.autotune` to describe mutable
benchmark inputs. Reset runs before every timed reuse, outside the timed interval.
Restore uses the canonical host snapshot rather than a separate GPU backup for
every slot. Arbitrary hooks and custom autotuning modifiers remain unsupported.
Every mutated input that needs an initial value on reuse must be declared.
Graph warmup and capture are separated by another reset; graph slots are distinct,
and resets happen before replay rather than inside its timed region.

A launch compiles and tunes an uncached specialization, then executes its winner
once on a separate clone of the original invocation. It copies resulting tensor
storage back into the caller's existing tensors and returns `None`. Benchmark
reset/restore policies do not change the real invocation's supplied values.
A plain JIT kernel without tuning compiles and executes its single configuration.
A native autotuner with multiple configurations uses default `TritonTuning`
when no tuning policy is supplied.

`kernel.tuning_results` contains immutable reports indexed by specialization digest;
there is no `last_tuning`. Each report includes `input_metadata`, `argument_names`,
and `target` so the specialization can be identified without decoding its digest.
The key includes source, configurations, argument values
and tensor metadata, launch grid, image configuration, GPU pool, environment,
Triton version, reset declarations, evaluator, and tuning policy. Tensor contents
are excluded, so a subsequent call with new values and the same specialization
executes the cached winner without repeating compilation or tuning. Artifacts are
reused only with the same client/account. Reports and compiled artifacts are
cached in this wrapper's memory; registry image names should identify immutable
versions. Result reports describe tuning, while each invocation executes again.
Calls on one wrapper are serialized to prevent duplicate concurrent tuning.

```python
kernel = vfunc.TritonKernel(
    native_kernel,
    tuning=vfunc.TritonTuning(
        pilot_pruning=vfunc.TritonPruning(relative_delta=0.25, absolute_us=1.0),
        refined_pruning=vfunc.TritonPruning(relative_delta=0.05, absolute_us=0.1),
        evaluate=evaluate,
        replication_factor=3,
    ),
)
with app.function(gpu="gb300"):
    kernel[grid](*args, **kwargs)
for specialization, result in kernel.tuning_results.items():
    print(specialization, result["benchmark"]["best_configuration"])
```

Benchmark counts and estimated compute durations remain configurable separately
from pruning and resource limits:

```python
tuning = vfunc.TritonTuning(
    pilot_pruning=vfunc.TritonPruning(relative_delta=0.25, absolute_us=1.0),
    refined_pruning=vfunc.TritonPruning(relative_delta=0.05, absolute_us=0.1),
    benchmark=vfunc.TritonBenchmark(
        pilot_trials=3,
        refinement_duration_ms=1.0,
        min_refinement_trials=10,
        max_refinement_trials=250,
        final_duration_ms=25.0,
        min_final_trials=25,
        max_final_trials=1000,
        graph_duration_ms=1.0,
        min_calls_per_graph=10,
        max_calls_per_graph=100,
        l2_flush_iterations=100,
    ),
    quick_benchmark_group_size=8,
    quick_benchmark_variants_per_job=256,
    quick_benchmark_max_concurrent_jobs=4,
    replication_factor=3,
    replication_max_attempts=8,
    max_input_sets=65_536,
    max_ring_bytes=8 * 1024**3,
)
```

Durations describe estimated kernel compute, not wall-clock deadlines. Refinement
is skipped when its estimated trial count is below `min_refinement_trials`.
Final direct-event measurements use at least `min_final_trials`; graph replay
counts use at least one replay. Both are bounded by `max_final_trials` (counting
replays on the graph path). Iteration caps can shorten the requested compute
budget. Durations must be finite and positive, counts must be positive integers,
and minimum counts must not exceed their maximums.

Each round retains times <= `best + max(relative_delta * best, absolute_us)`.
Defaults allow 25% or 1 us in the three-trial pilot, then 5% or 0.1 us in
refinement and global replica reduction, whichever allowance is larger. The
pilot leaves more room for noisy early measurements. The smaller refined
absolute allowance avoids replicating almost every configuration of a short
kernel. These are heuristic starting points tested on GB300; increase either
allowance for noisier workloads or use an unpruned control to check selection
quality. Replication still defaults to three distinct GPUs.
Set either pruning policy to `None` to disable that round's pruning while still
measuring it. The old percentage-only delta and runtime-exemption options are
replaced by these independent policies.

The GPU builds independent sets until their logical tensor footprint exceeds its
reported L2 size. Shared storages are counted conservatively once. This is an
allocation and scheduling contract, not a guarantee of cache eviction: kernels
may access only a subset, and resets can warm cache lines. The supplied
inputs must exercise the intended working set. Ring allocation and graph padding
are bounded by `max_input_sets` (65,536) and `max_ring_bytes` (8 GiB); exceeding
the initial ring limits fails clearly. If graph padding would exceed a ring limit,
final benchmarking uses direct events with per-call reset instead. Reports include
the ring footprint and allocation size.

Pilot groups (eight variants by default) enqueue 100 L2 zeroing operations, then
interleave three single-launch event trials per variant, rotating inputs without
intermediate cache flushes. Synchronize once, take minima, and validate proposed
new bests before using them for pruning. All pilot groups finish before survivors
are regrouped. Refinement uses up to `min(250, ceil(1_000 / pilot_us))` launches per variant when
at least ten are required, returning each minimum. Reset runs outside timed
intervals. Refined pruning uses a validated best; every survivor passes the
optional evaluator before final timing. Without an evaluator correctness is assumed.

Final timing handles one configuration at a time. Set
`Z = max(1, min(100, floor(1_000 / refined_us)))`. For `Z >= 10`, capture Z calls
per graph and measure `ceil(25_000 / (Z * refined_us))` replays. Cycle distinct
graphs spanning the ring because CUDA graphs fix input pointers. Pad to whole
graphs so every captured call uses a distinct slot within its graph; this also
allows resets outside replay timing. Return the minimum event duration divided
by Z. For `Z < 10`, use direct events for
`max(25, ceil(25_000 / refined_us))` launches. Both paths begin with 100 L2 flushes;
final results include their trials and timing method.

Initial shards report all candidates and their final timings. Client reduction
applies refined pruning against the global best, then schedules only surviving
configurations for additional final-only measurements. Every completed replica
must report a different physical GPU UUID. The initial measurement counts toward
`replication_factor` (three by default); final ranking uses the arithmetic mean.
After each replica shard completes, partial means are reduced and pruned again;
this is a heuristic, and partial results are never selected as completed winners.
Replication is also re-evaluated for correctness on each device.

The public vFunc scheduler selects a GPU model but has no device-exclusion field.
Replica Calls therefore reserve `replication_factor` GPUs on one worker and
measure their distinct device UUIDs separately. Devices already represented in
that candidate's measurements are skipped. This avoids repeatedly scheduling
single-GPU retries onto the same available device. Each final-only shard requires
that many free GPUs on one worker; current GB300 trays have four GPUs, so higher
replication factors require a worker with more GPUs. Calls are bounded by the
normal capacity deadline and `replication_max_attempts`. Failed capacity or lack
of enough distinct devices raises `TritonBenchmarkError` with retained reports
and Call IDs. Generic single-device dispatch also verifies identities and never
counts duplicate UUID placements as independent replicas.

`quick_benchmark_variants_per_job` defaults to 256 and GPU concurrency to four;
these are configurable batching limits. All phases reuse existing App/Image/
Function settings and CPU-prepared artifacts; no service deployment is required.

### Benchmark a selected kernel configuration

`benchmark` accepts vFunc-managed launch handles (`Kernel`, currently
`TritonKernel`). It autotunes an uncached specialization before measuring its
selected configuration. A cached specialization reuses its compiler artifacts
and configuration, then obtains fresh timings. Benchmarking leaves the caller's
tensors unchanged and inherits the kernel's optional correctness evaluator and
native reset/restore declarations.

```python
options = vfunc.KernelBenchmark(
    estimate_trials=3,
    final_duration_ms=25.0,
    min_final_trials=25,
    max_final_trials=1000,
    graph_duration_ms=1.0,
    min_calls_per_graph=10,
    max_calls_per_graph=100,
    l2_flush_iterations=100,
    replication_factor=3,
    replication_max_attempts=8,
    max_concurrent_jobs=4,
    max_input_sets=65536,
    max_ring_bytes=8 * 1024**3,
)
with app.function(gpu="gb300"):
    cold = vfunc.benchmark(kernel[grid], a, b, out, M=m, N=n, K=k, options=options)
    warm = vfunc.benchmark(kernel[grid], a, b, out, M=m, N=n, K=k, options=options)
print(warm["runtime_us"])           # Arithmetic mean of distinct GPU minima
print(warm["replicas"])             # GPU UUIDs, timings, trials, evaluation, Call IDs
print(warm["reused_specialization"]) # True: no repeat compilation or autotuning
```

Each new benchmark checks that the compiled kernel launches without error,
optionally evaluates it on scratch inputs, and estimates duration with
`estimate_trials` rotating-input event measurements. It uses the final autotuning
phase: CUDA graphs for short kernels, direct events for longer kernels or when
graph padding would exceed memory limits. Estimates size the graph and trial
counts; the final duration is an estimated compute budget, rather than a wall-time
deadline. `max_final_trials` counts graph replays or individual direct calls.
Standalone replication reserves all R GPUs in one job. The job verifies and
restores the selected configuration once, then estimates and measures it on each
GPU separately. Physical GPU UUIDs must be distinct. No configuration pruning
is needed. The selected compiler shard remains a compressed batch: its archive
is scanned, but only the winner's cache files are restored. Other compiler
shards are not requested. The concurrency setting does not parallelize devices
within this job.

The returned report is immutable and includes `runtime_us`, `configuration`,
`replicas`, `specialization`, `autotuned`, `reused_specialization`, raw shard
reports and benchmarking Call IDs. Fresh benchmark results do not overwrite the
original autotuning report in `kernel.tuning_results`.

See [the complete matmul example](examples/triton_matmul_benchmark.py) for 1,024
configuration combinations, all tuning/benchmark options, and both cache paths.

## CuTe DSL kernels

`CuteDSLKernel` implements the same managed compilation, tuning, replicated
benchmarking, specialization cache and tensor-write contract as `TritonKernel`.
It wraps the **`@cute.jit` host entry point** that launches device kernels. CuTe
host entries define their own launch grid, so call the wrapper directly:

```python
kernel = vfunc.CuteDSLKernel(
    add_one,
    configurations=[{"BLOCK": 64}, {"BLOCK": 128}, {"BLOCK": 256}],
    tuning=vfunc.KernelTuning(evaluate=evaluate_add, replication_factor=3),
    variants_per_job=2,
    max_concurrent_jobs=2,
)
with app.function(gpu="gb300", timeout=600, capacity_wait=300):
    cold = vfunc.benchmark(kernel, a, b)  # Compile/tune, then fresh R=3 measurement.
    warm = vfunc.benchmark(kernel, a, b)  # Fresh measurement of the saved winner.
    kernel(a, b)                        # Execute once and apply writes to b.
print(warm["runtime_us"], warm["configuration"]["constants"])
```

Install the **same `nvidia-cutlass-dsl` version** in the client and the selected
vFunc image. Images also need PyTorch, CUDA bindings and CuTe's runtime libraries.
No package installation occurs implicitly in the SDK. App/Function image,
resource, target, timeout and environment settings apply normally.

`KernelTuning`, `KernelPruning` and `KernelTiming` are shared aliases of
`TritonTuning`, `TritonPruning` and `TritonBenchmark`; all existing controls and
defaults apply. Standalone measurements use `KernelBenchmark`.
`kernel.tuning_results` holds immutable reports by specialization. Benchmarking
preserves client tensors and the original tuning report. Ordinary calls return
`None` and preserve writes to the original tensors, including aliased views.

### CuTe-specific controls

- `configurations`: nonempty list of literal mappings naming
  `cutlass.Constexpr` parameters on the host entry. Scalar values and nested
  tuples/lists (such as tile shapes) are supported. Defaults to `[{}]`.
  Multiple configurations enable vFunc tuning automatically.
- `tuning`: shared `KernelTuning`, including an optional accuracy evaluator.
- `variants_per_job`: configurations per CPU job; `None` distributes the search
  across at most `max_concurrent_jobs` jobs.
- `max_concurrent_jobs`: maximum simultaneous CPU compiler jobs, default 8.
- `compile_options`: CuTe's public compilation option string, default empty.
  vFunc manages GPU target and the object ABI. Target and ABI overrides are rejected.
- `reset_to_zero` / `restore_value`: argument names requiring reset/restore before
  benchmark reuse. Defaults to empty tuples. Actual execution uses the supplied
  values. Restores use canonical host snapshots rather than extra GPU backups.

A typed `stream: cuda.CUstream` parameter is injected using the worker's current
PyTorch stream; callers omit it. This enables event timing and CUDA graph capture
on the correct stream. Entries without that parameter use direct-event final
measurements. Source and helper definitions are bundled without importing the
user's entire script. Callable classes with literal/type state are also supported;
construct more complex DSL state inside the host entry or a bundled helper.

CPU workers compile from fake tensor descriptors, preserving shapes, strides and
dtypes without allocating GPU inputs. They export independent object files using
CuTe's public AOT API. GPU workers verify and load only requested objects, then
reuse the shared input-ring and measurement pipeline. Compilations use isolated
processes to avoid sharing CuTe's compiler state between configurations.
Standalone benchmarks request only the winning compiler shard and measure all
assigned replicas in one GPU job, verifying the selected object files once.

See `examples/cute_kernel.py` for the full client example and
`examples/cute_kernels.py` for vector addition, a callable class performing
strided copy, and a small reference-style matmul.

The tensor/scalar transport restrictions of `TritonKernel` still apply. Arbitrary
Python objects, device-entry-only functions, non-literal configuration objects and
host return values are outside this launch contract. Compile failures are retained
per variant; poisoned GPU processes fail with retained Call evidence. Replication
uses the existing verified-distinct-GPU scheme and placement limits.
