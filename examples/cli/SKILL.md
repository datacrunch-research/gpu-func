---
name: gfaas-cli
description: >-
  Use the gfaas CLI to run self-contained CUDA on a local NVIDIA GPU or to run
  CUDA and Python workloads on a remote GPU fleet. Use it to compare local and
  remote CUDA results, manage durable Calls, and manage Artifacts. Do not use it
  for long-lived services or multi-host distributed jobs.
---

# Use the gfaas CLI

Use `gfaas` to run a bounded GPU workload on a remote GPU fleet. The service accepts the workload
through HTTPS and selects a worker from the requested GPU pool.

The CLI can also compile and run CUDA directly on the local host. Local execution does not contact
the service.

The primary use is CUDA kernel development. You can compile, run, profile, and revise a kernel
without SSH access to a GPU host.

The CLI also runs self-contained Python programs in operator-provided images. These images can
contain PyTorch, CUDA tools, or other qualified packages.

## Understand the execution model

Each submission creates a durable **Call**. A Call remains available after the client disconnects.

The service records these parts of a Call:

- Lifecycle state and placement Attempts.
- Standard output and standard error, within retention limits.
- The serialized result.
- Declared output Artifacts.
- Preparation, worker, and timing information.

The caller selects a GPU pool and resource limits. The service selects the worker and the physical
GPU devices.

`--gpu-count 4` requests four GPUs on one worker. It does not create a job across four workers.

## Use gfaas for these tasks

- Compile and run a self-contained `.cu` file.
- Compare CUDA kernel variants on real hardware.
- Collect an Nsight Compute profile for a CUDA program.
- Run a self-contained Python GPU experiment.
- Run one module-level Python callable with string arguments.
- Submit a long Call, disconnect, and reconnect with its Call ID.
- Publish files or directory trees as output Artifacts.
- Read retained logs and download result Artifacts.

Use the Python SDK instead when a workload needs typed arguments, keyword arguments, Artifact
inputs, or a more complex application wrapper.

Use a workflow system for dependent jobs. Use a container orchestrator for a long-lived service. Use
a distributed training system for jobs that span multiple workers.

## Select local or remote execution

Use local execution for rapid CUDA development on an available NVIDIA GPU:

```bash
gfaas local info
gfaas local run kernel.cu --json
```

Use remote execution to run in an operator-managed image on a selected GPU pool:

```bash
gfaas run kernel.cu --gpu-type gb300 --json
```

CAUTION: Local execution does not use a container. Run only reviewed source because the program can
read host files and inherited environment values.

Local execution does not create a Call or an Artifact. Declared outputs remain ordinary local files.
Remote execution creates a durable Call and publishes declared outputs as Artifacts.

Keep generated local files under `.local/gfaas/`. Do not leave benchmark results in the repository
root:

```bash
mkdir -p .local/gfaas
gfaas local run kernel.cu \
  --output benchmark=.local/gfaas/results.csv \
  -- --output .local/gfaas/results.csv
```

Configure the program to write the same path that you declare with `--output`.

Before a local run, use `gfaas local info`. Make sure that the selected device and detected
architecture match the intended test.

Local mode detects `nvcc`, `nvidia-smi`, the host compiler, and `ncu`. Use `--nvcc`, `--ccbin`, or
`--ncu` only when automatic discovery selects the wrong tool.

The local program inherits the current environment. Use `--env NAME=VALUE` to add or replace one
value. Use `--device INDEX_OR_UUID` to select a GPU.

## Compare a kernel on local and remote GPUs

Keep the source, compiler options, and program arguments identical. Change only the execution target
and required architecture option.

```bash
mkdir -p .local/gfaas

gfaas local run kernel.cu \
  --nvcc-flag=-O3 \
  --json > .local/gfaas/local.json

gfaas run kernel.cu \
  --gpu-type gb300 \
  --nvcc-flag=-O3 \
  --json > .local/gfaas/remote.jsonl
```

Read the final result records. Compare the device, architecture, compiler time, execution time,
standard output, and profiler metrics.

Do not compare elapsed times until both paths use equivalent warm-up and iteration counts. Record
hardware differences when you interpret the result.

## Make sure that access is configured

Make sure that the `gfaas` command is available:

```bash
gfaas --help
```

The CLI reads its endpoint and credential from the environment:

```bash
export GFAAS_API_BASE=https://gpu.example.com/api
export GFAAS_API_KEY='provided-separately'
export GFAAS_GPU_TYPE=operator-provided-pool
```

CAUTION: Do not print, log, or commit `GFAAS_API_KEY`. Do not pass the key as a command argument.

Do not forward `GFAAS_API_KEY` with `--env`. Use `--env` only for non-secret workload values.

List the configured GPU pools before you select one:

```bash
gfaas pool list
```

Use an explicit pool name. The value `any` is a literal pool name, not a wildcard.

Ask the operator for available image names, GPU architecture values, and deployment limits.

## Run a CUDA program

The CUDA file must contain a complete program with `main()`. The CLI compiles it with `nvcc` in the
`cuda-nvcc` image.

Start with the included example:

```bash
gfaas run examples/cli/hello_cuda.cu \
  --gpu-type gb300 \
  --nvcc-flag=-O3
```

The worker selects the architecture for its GPU. Use `--arch` only when the source needs an
explicit target.

Arguments after `--` go to the compiled program:

```bash
gfaas run kernel.cu \
  --gpu-type gb300 \
  --nvcc-flag=-O3 \
  -- --problem-size 4096
```

Add `--profile` to collect an Nsight Compute CSV report. Add each profiler option with a separate
`--ncu-arg` argument.

The compiled program starts in the workload output directory. Use a relative path when the program
writes a file for publication:

```bash
gfaas run kernel.cu \
  --gpu-type gb300 \
  --output benchmark=results.csv
```

The program must write `results.csv` in its current directory.

## Run a Python program

The Python file must be self-contained. The CLI does not package a source tree or install local
dependencies.

Select an image that contains every imported third-party package:

```bash
gfaas run examples/cli/hello_python.py \
  --image pytorch-cu130 \
  --gpu-type gb300 \
  -- --size 2048
```

The CLI does not import the program on the client. It invokes the program with unbuffered output in
the workload output directory.

Use `file.py:callable` to run one module-level callable:

```bash
gfaas run experiment.py:train \
  --image pytorch-cu130 \
  --gpu-type gb300 \
  -- input.json
```

Each argument after `--` becomes one string positional argument.

## Select resource limits

Request only the resources that the workload needs. The selected worker policy sets the maximum
values.

Common options are:

- `--gpu-type NAME`: Select the GPU pool.
- `--gpu-count COUNT`: Request GPUs on one worker.
- `--timeout SECONDS`: Set the execution deadline after worker acceptance.
- `--capacity-wait SECONDS`: Set the deadline for placement and preparation.
- `--cpu-millicores COUNT`: Set the CPU request.
- `--memory SIZE`: Set the aggregate memory limit.
- `--storage SIZE`: Set the writable scratch reservation.
- `--shared-memory SIZE`: Set the `/dev/shm` size.
- `--max-log SIZE`: Set the retained log limit.
- `--max-output SIZE`: Set the output Artifact limit.

Size options accept bytes or the `KiB`, `MiB`, `GiB`, and `TiB` suffixes.

Capacity wait and execution timeout are different limits. Artifact staging occurs during the
capacity-wait period.

## Publish output Artifacts

Declare a required output file with `--output NAME=PATH`:

```bash
gfaas run experiment.py \
  --image pytorch-cu130 \
  --gpu-type gb300 \
  --output report=reports/result.json
```

Declare a required directory with `--output-directory NAME=PATH`.

Python and CUDA programs start in the workload output directory. The Call fails if a required path
does not exist when the program stops.

List the Artifacts from a Call:

```bash
gfaas call artifacts CALL_ID
```

Download an Artifact to a new path:

```bash
gfaas artifact download ARTIFACT_ID result.bin
```

The download command does not replace an existing path.

## Detach and reconnect

Use foreground mode for a short experiment. The CLI shows lifecycle progress and workload output.

If you interrupt a foreground run, the CLI requests cancellation of the remote Call.

Use `--detach` when the Call must continue after the client exits:

```bash
call_id="$(gfaas run kernel.cu --gpu-type gb300 --detach)"
```

Use the Call ID to reconnect:

```bash
gfaas call show "$call_id"
gfaas call watch "$call_id"
gfaas call logs "$call_id"
gfaas call logs "$call_id" --follow
gfaas call artifacts "$call_id"
```

Request cancellation only for a Call that the user authorized you to control:

```bash
gfaas call cancel "$call_id" --reason superseded
```

## Use machine-readable output

Add `--json` for scripts and automation:

```bash
gfaas run kernel.cu --gpu-type gb300 --json
gfaas pool list --json
```

A foreground run emits JSON Lines. It includes the submission, complete event records, and the final
result.

Without `--json`, progress goes to standard error. Program output remains on its original output
stream.

## Diagnose errors

Use the exact error before you retry.

- `authentication rejected`: Set the correct `GFAAS_API_BASE` and `GFAAS_API_KEY`.
- `GPU pool 'any' is not configured`: Set `GFAAS_GPU_TYPE` or use `--gpu-type`.
- `gpu_occupancy`: The selected workers are busy. Wait for capacity or change the request.
- `ResourcePolicyViolation`: Reduce the request or ask the operator about the worker policy.
- Missing Python package: Select an image that contains the package.
- Missing output path: Make sure that the program writes every declared output.
- Local interruption: Use the printed Call ID to inspect cancellation convergence.

Do not automatically retry authentication, invalid-input, or policy errors. Change the credential,
source, image, or resource request first.

A CUDA compiler or program error can occur inside a successful service Call. Read the final CLI
status, compiler output, and process return code.

## Preserve authorization boundaries

Pool discovery and Call inspection are read-only operations. A submission consumes fleet resources.

Submit a live workload only when the user authorizes remote execution. Cancel only Calls that are in
the user's scope.

Do not assume that a free GPU is dedicated to this work. The service makes the final placement
decision.

For more details, read [the CLI guide](../../docs/cli.md).
