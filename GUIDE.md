# gfaas guide

`gfaas` is for people who have CUDA source code but do not have the right
GPU available locally. It sends a small, selected workspace to a remote GPU
worker. The worker compiles the code, runs it, and returns the result.

You can use this guide in either of these situations:

- You received a CUDA course exercise as a ZIP file and want to test your
  solution on the course hardware.
- You have a standalone CUDA experiment and want to compile, run, benchmark,
  or profile it on a remote GPU.

You do not need a local CUDA installation, `nvcc`, or an NVIDIA GPU. You also
do not need to understand the gfaas API before you start. You need the
`gfaas` command, service credentials, and either an exercise or CUDA source
file.

The CLI supports two types of work:

1. **CUDA course exercises.** Start with an exercise directory from a course.
   Use its tests, benchmarks, sanitizer rules, profiler, and grading logic.
2. **Custom CUDA programs.** Start with any `.cu` file. Supply a `main()`
   function in that file or in a separate harness.

Both types can create an Nsight Compute report. The worker stores this report
as a gfaas Artifact. You can download the report and inspect it locally.

gfaas is the remote execution service behind this CLI. It schedules each Call
on a compatible worker and keeps the Call state after the local command ends.
`gfaas` turns the service into a CUDA development workflow. It does not
generate kernels or decide whether a custom program produced the right answer.

This guide starts with the purpose of each workflow. The command reference and
the gfaas data flow come later.

## Contents

1. [Choose a workflow](#1-choose-a-workflow)
2. [Install and configure the CLI](#2-install-and-configure-the-cli)
3. [Work on a course exercise](#3-work-on-a-course-exercise)
4. [Work on a custom CUDA program](#4-work-on-a-custom-cuda-program)
5. [Complete custom-program walkthrough](#5-complete-custom-program-walkthrough)
6. [Measure performance](#6-measure-performance)
7. [Use durable Calls and Artifacts](#7-use-durable-calls-and-artifacts)
8. [Select GPU and system resources](#8-select-gpu-and-system-resources)
9. [Read Nsight Compute reports](#9-read-nsight-compute-reports)
10. [Understand the remote data flow](#10-understand-the-remote-data-flow)
11. [Command reference](#11-command-reference)
12. [Workspace and trust rules](#12-workspace-and-trust-rules)
13. [Exit status](#13-exit-status)
14. [Troubleshooting](#14-troubleshooting)

## 1. Choose a workflow

### Course exercise

When an exercise supplies `run.py` and `runner/cli.py`, use the course workflow.
The exercise defines what correctness and performance mean for that problem.

The available actions have different purposes:

| Action | Purpose |
| --- | --- |
| `compile` | Find compiler and linker errors without running the program. |
| `test` | Compare the solution with the correctness cases from the exercise. |
| `benchmark` | Measure the solution with the benchmark cases from the exercise. |
| `sanitizer` | Find CUDA memory errors with the sanitizer workflow from the exercise. |
| `profile` | Collect Nsight Compute metrics and exercise-specific feedback. |
| `grade` | Run the complete assessment that the exercise defines. |

Use these actions as a development loop:

1. Run `compile` after a structural code change.
2. Run `test` until all correctness cases pass.
3. Run `sanitizer` before you trust the result.
4. Run `benchmark` to compare implementations.
5. When timing alone does not explain a result, run `profile`.

The exercise owns its tests and measurements. `gfaas` supplies the remote
GPU and transports the files and results.

### Custom CUDA program

Use the custom workflow for a standalone experiment, a kernel prototype, or a
small reproduction. The CLI does not know the correct answer for custom code.
Your program or harness must detect errors and return a nonzero exit code.

The custom actions are:

| Action | Purpose |
| --- | --- |
| `custom compile` | Compile and link the program. Do not run it. |
| `custom run` | Compile, link, and run the program. |
| `custom profile` | Compile the program and run it through Nsight Compute. |

A custom program is useful for a focused question. Examples include a new
memory layout, a launch-configuration comparison, or a reduced compiler error.

## 2. Install and configure the CLI

### Install

Install the SDK and CLI from this repository:

```bash
uv tool install /path/to/gpu-func
```

Make sure that the command is available:

```bash
gfaas --help
```

Your local computer does not need CUDA for remote work. The local report
commands need the Python module from Nsight Compute, but they do not need a GPU.

### Configure access

Set the coordinator address and API key in the environment:

```bash
export GFAAS_API_BASE="https://gpu.example.com/api"
export GFAAS_API_KEY="..."
```

The CLI does not accept an API key argument. This rule keeps the key out of the
shell history and process list.

Show the GPU pools that the coordinator provides:

```bash
gfaas pool list
```

A pool is a class of GPU capacity, such as `gb300`. It is not a specific
worker. The coordinator selects a worker when it places the Call.

If the coordinator has one pool, `gfaas` selects it automatically. If it
has multiple pools, use `--gpu-type` to select one.

## 3. Work on a course exercise

### 3.1 Understand the exercise directory

A distributed course exercise usually arrives as a ZIP file. After extraction,
the directory has this general structure:

```text
01-haxpy/
├── haxpy.cu          # your solution
├── tester.cu         # the exercise driver
├── run.py            # the exercise entry point
├── runner/           # compile, test, benchmark, and profile support
├── tests/            # correctness cases
└── benchmarks/       # performance cases
```

The exact names depend on the exercise. The important markers are `run.py` and
`runner/cli.py`. The CLI uses these markers to find the exercise root.

The worker receives the exercise runner with your solution. This design keeps
the remote result consistent with the exercise that you received.

### 3.2 Start the development loop

Change to the extracted exercise directory:

```bash
cd 01-haxpy
```

Compile your current solution:

```bash
gfaas compile
```

Run the correctness suite:

```bash
gfaas test
```

The CLI finds the exercise from the current directory. You can start in a
subdirectory because the CLI also examines parent directories.

### 3.3 Select a test or benchmark

With no spec argument, the exercise runner uses all applicable specs. Supply
one or more paths to shorten an iteration:

```bash
gfaas test tests/01_corner_n1.txt
gfaas benchmark benchmarks/01_aligned_small.txt
```

Use a narrow spec while you diagnose a specific error. Before completion,
run the complete suite without spec arguments.

### 3.4 Use an exercise from another directory

When you do not want to change directories, use `--exercise-dir`:

```bash
gfaas benchmark --exercise-dir ~/Downloads/01-haxpy
```

Use `--file` to replace the solution file for one remote Call:

```bash
gfaas test \
  --exercise-dir ~/Downloads/01-haxpy \
  --file ~/src/my-haxpy.cu
```

This command does not modify the extracted exercise. The CLI inserts the
selected file into the uploaded workspace.

### 3.5 Use a course checkout

The explicit form works with a complete course checkout:

```bash
gfaas exercise 01-haxpy benchmark \
  --course-root ~/src/cuda-course
```

The exercise identifier selects `exercises/01-haxpy`. The CLI also uploads the
shared course runner. It excludes every `solutions/` directory.

### 3.6 Interpret the exercise result

The exercise runner controls the detailed output. A typical result contains:

- Compiler output for `compile`.
- A pass or error for each correctness spec in `test`.
- Runtime and throughput measurements for `benchmark`.
- Memory-access errors for `sanitizer`.
- Hardware counters and recommendations for `profile`.
- A combined assessment for `grade`.

Read the result as evidence from that exercise, not as a general CUDA score.
Different exercises use different inputs, warmup rules, and success criteria.

## 4. Work on a custom CUDA program

### 4.1 Decide whether you need a harness

The worker builds a real executable. One submitted source file must define
`main()`.

When the source file defines `main()`, submit it without a harness:

```bash
gfaas custom run vecadd.cu
```

When the kernel source does not define `main()`, use a harness:

```bash
gfaas custom run scale_kernel.cu \
  --harness scale_harness.cu
```

The kernel file normally contains device code and a launch function. The
harness normally allocates memory, initializes inputs, launches the kernel, and
checks the result.

One harness can compare multiple kernel implementations. Different harnesses
can also run one kernel with different shapes or data types.

### 4.2 Make correctness observable

`gfaas` treats exit code zero as a successful custom run. It does not inspect
the numerical output.

Make the program compare its output with an expected result. Return a nonzero
exit code when the comparison fails. Print enough context to diagnose the
first error.

This rule separates two questions:

1. Did the CUDA program run without a process error?
2. Did the CUDA program calculate the correct result?

Your harness answers the second question.

### 4.3 Pass program arguments

Repeat `--arg` for each argument to `main()`:

```bash
gfaas custom run scale_kernel.cu \
  --harness scale_harness.cu \
  --arg 1048576 \
  --arg 2.5
```

The worker preserves the argument order.

### 4.4 Change compiler flags

The default compiler flags are:

```text
-std=c++20 -O3 -lineinfo
```

Replace them with `--nvcc-flags`:

```bash
gfaas custom run kernel.cu \
  --nvcc-flags "-std=c++20 -O2 -lineinfo --use_fast_math"
```

The worker adds a CUDA architecture flag when you do not supply one. It reads
the compute capability from the selected GPU.

### 4.5 Profile a custom program

By default, `custom profile` selects an NVTX range named `profile_kernel`.
Place this range around the code that you want to measure:

```cpp
nvtxRangePush("profile_kernel");
launch_my_kernel(...);
cudaDeviceSynchronize();
nvtxRangePop();
```

The synchronization keeps the asynchronous kernel launch inside the measured
range.

When the program has an NVTX structure, use its range name:

```bash
gfaas custom profile kernel.cu \
  --nvtx-range attention_forward \
  --artifact-dir ./profiles
```

If the program has no NVTX range, profile the complete executable:

```bash
gfaas custom profile kernel.cu \
  --no-nvtx-filter \
  --artifact-dir ./profiles
```

This mode can collect startup kernels and library kernels. When you need a
focused report, use an NVTX range.

## 5. Complete custom-program walkthrough

This walkthrough creates a self-contained vector-add program. The program
checks one output value and prints a short result.

### 5.1 Create the source file

Save this text as `vecadd.cu`:

```cpp
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                           \
    do {                                                           \
        cudaError_t error = (call);                                \
        if (error != cudaSuccess) {                                \
            std::fprintf(stderr, "%s failed: %s\n",               \
                         #call, cudaGetErrorString(error));         \
            return 1;                                              \
        }                                                          \
    } while (0)

__global__ void vecadd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : (1 << 20);
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    std::vector<float> a(n, 1.0f);
    std::vector<float> b(n, 2.0f);
    std::vector<float> c(n, 0.0f);

    float* da = nullptr;
    float* db = nullptr;
    float* dc = nullptr;
    CUDA_CHECK(cudaMalloc(&da, bytes));
    CUDA_CHECK(cudaMalloc(&db, bytes));
    CUDA_CHECK(cudaMalloc(&dc, bytes));
    CUDA_CHECK(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));

    const int block = 256;
    const int grid = (n + block - 1) / block;
    nvtxRangePush("profile_kernel");
    vecadd<<<grid, block>>>(da, db, dc, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    nvtxRangePop();

    CUDA_CHECK(cudaMemcpy(c.data(), dc, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));

    if (std::fabs(c.front() - 3.0f) > 1e-6f ||
        std::fabs(c.back() - 3.0f) > 1e-6f) {
        std::fprintf(stderr, "incorrect result: first=%f last=%f\n", c.front(), c.back());
        return 2;
    }

    std::printf("vecadd passed: n=%d first=%f last=%f\n", n, c.front(), c.back());
    return 0;
}
```

### 5.2 Compile the program

```bash
gfaas custom compile vecadd.cu
```

This action checks that `nvcc` can compile and link the source. It does not
allocate GPU memory or launch the kernel.

### 5.3 Run the program

```bash
gfaas custom run vecadd.cu --arg 1048576
```

A successful result contains text similar to:

```text
vecadd passed: n=1048576 first=3.000000 last=3.000000
Custom run passed
```

### 5.4 Create a profile

```bash
gfaas custom profile vecadd.cu \
  --arg 1048576 \
  --artifact-dir ./profiles
```

The worker stores `vecadd.ncu-rep` in the declared output Artifact. The CLI
downloads that file into `./profiles` after the Call finishes.

### 5.5 Read the profile

If Nsight Compute is installed locally, summarize the report:

```bash
gfaas report summary ./profiles/vecadd.ncu-rep --per-kernel
```

The summary can show duration, DRAM traffic, throughput, occupancy,
instructions, and launch-specific metrics.

## 6. Measure performance

### Benchmark before you profile

A benchmark answers, "How long did this implementation take for this input?"
Use the course benchmark workflow or a timing loop in a custom harness.

A profiler answers, "What did the hardware do during this kernel?" Nsight
Compute can replay a kernel many times to collect counters. As a result, the
profile duration is not a benchmark result.

Use this sequence:

1. Establish a correct implementation.
2. Measure a stable benchmark result.
3. Change one implementation detail.
4. Measure the benchmark again.
5. Use the profiler to explain an important difference.

### Interpret common metrics

No single metric proves that a kernel is good. Interpret counters together with
the algorithm and benchmark result.

- **Duration** shows time for captured launches. Profiler replay can affect it.
- **DRAM read and write bytes** show traffic to device memory.
- **DRAM throughput** shows use of available memory bandwidth.
- **SM throughput** shows use of streaming-multiprocessor execution capacity.
- **Occupancy** shows active warps relative to the hardware limit.
- **Instructions** help compare the amount of executed work.
- **Loads and stores** help explain memory behavior.

Low occupancy is not automatically an error. A kernel can be fast with low
occupancy when it has enough independent work or uses specialized hardware.

### Select an Nsight metric set

Custom profiling uses `--set basic` by default. When the basic report cannot
answer the question, use a larger metric set:

```bash
gfaas custom profile vecadd.cu \
  --ncu-args "--set full" \
  --artifact-dir ./profiles-full
```

The full set takes longer because Nsight Compute uses more replay passes. Do
not use it for every edit.

## 7. Use durable Calls and Artifacts

### Call identity

Every remote command creates a durable gfaas Call. The CLI prints its identity
immediately:

```text
call: call_...
```

The Call exists independently of the local terminal. A network interruption
does not erase the Call or its retained events.

### Detach from a Call

Use `--detach` to return after submission:

```bash
gfaas custom run vecadd.cu --detach
```

Inspect the Call with the same `gfaas` command:

```bash
gfaas call show call_...
gfaas call watch call_...
gfaas call logs call_... --follow
gfaas call artifacts call_...
```

When the local terminal must not remain open, detach from a long benchmark or
profile.

### Cancel a Call

Cancel a detached Call explicitly:

```bash
gfaas call cancel call_... --reason "experiment no longer needed"
```

If you interrupt a foreground `gfaas` command, the CLI requests remote Call
cancellation. The CLI prints the Call identity so that you can inspect its final
state with `gfaas call show`.

### Understand the two time limits

`--timeout` sets the worker execution limit. The coordinator also applies the
worker policy ceiling.

`--wait-timeout` limits the local wait. If this limit expires, the remote Call
can continue. Use its Call identity to inspect or cancel it.

### Avoid duplicate submissions

When a script can repeat the same submission, use an idempotency key:

```bash
gfaas custom run vecadd.cu \
  --idempotency-key vecadd-baseline-2026-08-27
```

The coordinator uses this key to identify a repeated Call request.

### Profile Artifacts

Nsight reports do not travel inside result JSON. They can be large, and they
must remain useful after the local process ends.

The worker publishes reports through a declared `profiles` output Artifact.
Use `--artifact-dir` to download them immediately:

```bash
gfaas custom profile vecadd.cu --artifact-dir ./profiles
```

Without `--artifact-dir`, the CLI prints the Artifact identity. Download it
later with the same command:

```bash
gfaas artifact download art_... ./profiles
```

The CLI does not replace an existing local report. Select a new directory for
each experiment, or remove the old file first.

## 8. Select GPU and system resources

### Select a GPU pool

If the coordinator has multiple pools, select the required pool:

```bash
gfaas custom run vecadd.cu --gpu-type gb300
```

The pool name describes scheduler capacity. It does not select a specific tray
or worker.

The compatibility option `--gpu GB300` derives both `gb300` and `sm_103`.
For new scripts, use `--gpu-type`. Omit `--arch` unless the source needs it.

### Select the CUDA architecture

For custom programs, the worker detects the compute capability when `--arch`
is absent. Use an explicit architecture for reproducibility or special code:

```bash
gfaas custom run vecadd.cu \
  --gpu-type gb300 \
  --arch sm_103
```

The architecture controls code generation. The GPU pool controls placement.
These values describe related but different requirements.

### Request multiple GPUs

```bash
gfaas custom run multi_gpu.cu \
  --gpu-type gb300 \
  --gpu-count 4
```

This option makes four GPUs visible to one process. It does not make a
single-GPU program parallel. The program must initialize and use each device.

### Request host resources

All remote workflows accept these options:

| Option | Purpose |
| --- | --- |
| `--cpu-millicores N` | Request CPU capacity. |
| `--memory SIZE` | Request host memory. |
| `--storage SIZE` | Request temporary storage. |
| `--shared-memory SIZE` | Request shared memory. |
| `--max-log SIZE` | Set the retained log limit. |
| `--max-output SIZE` | Set the output Artifact limit. |
| `--env NAME=VALUE` | Add an environment variable to the worker process. |

Sizes accept values such as `512MiB`, `4GiB`, and `1TiB`.

Example:

```bash
gfaas custom profile vecadd.cu \
  --gpu-type gb300 \
  --memory 64GiB \
  --storage 32GiB \
  --capacity-wait 1800 \
  --timeout 600 \
  --artifact-dir ./profiles
```

The worker policy can reject a request that exceeds its configured ceiling.

## 9. Read Nsight Compute reports

### Generic summary

The summary command reads a local `.ncu-rep` file:

```bash
gfaas report summary ./profiles/vecadd.ncu-rep
gfaas report summary ./profiles/vecadd.ncu-rep --per-kernel
```

This command needs `ncu_report.py`. Nsight Compute supplies this module. The
command does not need a local GPU.

If Python cannot find the module, add its directory to `PYTHONPATH`:

```bash
export PYTHONPATH="/opt/nvidia/nsight-compute/<version>/extras/python:$PYTHONPATH"
```

### Course-specific feedback

Some courses define rules that convert counters into exercise-specific advice.
The feedback command uses those rules:

```bash
gfaas report feedback ./profiles/haxpy.ncu-rep \
  --course-dir ~/src/cuda-course \
  --exercise 01-haxpy \
  --benchmark benchmarks/01_aligned_small.txt \
  --trust-course-code
```

CAUTION: Inspect the course code before you use `--trust-course-code`. This
command imports and executes the selected exercise `run.py` in the local Python
process.

The generic summary does not execute course code.

## 10. Understand the remote data flow

This section explains the remote system after the user workflows are clear.

### Submission

1. The CLI locates the selected source files.
2. The CLI rejects unsafe paths, links, and oversized workspaces.
3. The CLI uploads the workspace as an immutable tree Artifact.
4. The gfaas SDK packages the worker function.
5. The coordinator creates a durable Call with the requested resources.

Source files do not travel as a large JSON object. Binary exercise fixtures
remain byte-for-byte unchanged in the tree Artifact.

### Placement and preparation

The coordinator offers the Call to a compatible GPU pool. A worker can reject
the offer when GPUs or other resources are busy.

While a Call waits, `gfaas` shows capacity events. After placement, the
worker resolves the image and stages the source, input, and workspace Artifacts.

### Execution

The worker copies the staged workspace into a private scratch directory. It
checks each file hash before it starts a compiler or course runner.

For a course exercise, the worker runs the submitted `run.py`. For a custom
program, the worker invokes `nvcc` and then runs the executable.

The worker starts each compiler or program in a process group. A timeout stops
the complete group, not only its parent process.

### Results and cleanup

Standard output and standard error become retained Call events. The small
structured result becomes the Call result.

Nsight reports become output Artifacts. The worker removes its temporary
scratch copy when the worker function ends.

## 11. Command reference

### Pools

```bash
gfaas pool list
gfaas pools         # compatibility shortcut
gfaas workers       # compatibility shortcut
```

### Course exercises

```bash
# Find the exercise from the current directory.
gfaas <compile|test|benchmark|sanitizer|profile|grade> [specs...] [options]

# Select an exercise from a complete course checkout.
gfaas exercise EXERCISE_ID \
  <compile|test|benchmark|sanitizer|profile|grade> \
  [specs...] [options]
```

Important exercise options:

| Option | Purpose |
| --- | --- |
| `--exercise-dir DIR` | Select an extracted exercise directory. |
| `--course-root DIR` | Select a complete course checkout. |
| `--file PATH` | Replace the submitted solution for this Call. |
| `--exercise-id ID` | Set the report label for an auto-detected exercise. |
| `--json PATH` | Save the structured result. |
| `--artifact-dir DIR` | Download profile reports. |
| `--verbose` | Show additional exercise-runner output. |

### Custom programs

```bash
gfaas custom compile SOURCE.cu [--harness HARNESS.cu] [options]
gfaas custom run SOURCE.cu [--harness HARNESS.cu] [options]
gfaas custom profile SOURCE.cu [--harness HARNESS.cu] [options]
```

Important custom options:

| Option | Purpose |
| --- | --- |
| `--harness PATH` | Add the source file that defines `main()`. |
| `--arg VALUE` | Add one program argument. Repeat this option for more arguments. |
| `--nvcc-flags TEXT` | Replace the default compiler flags. |
| `--output NAME` | Set the remote executable name. |
| `--ncu-args TEXT` | Set Nsight Compute arguments. |
| `--nvtx-range NAME` | Select an NVTX range. |
| `--no-nvtx-filter` | Profile the complete executable. |
| `--report-name NAME` | Set the base name for the `.ncu-rep` file. |
| `--json PATH` | Save the structured result. |
| `--artifact-dir DIR` | Download profile reports. |

### Common remote options

| Option | Purpose |
| --- | --- |
| `--gpu-type NAME` | Select a GPU pool. |
| `--gpu-count N` | Request the specified number of GPUs. |
| `--image NAME` | Select a registered worker image. |
| `--arch ARCH` | Set the CUDA compilation architecture. |
| `--timeout SEC` | Set the worker execution limit. |
| `--capacity-wait SEC` | Set the maximum wait for capacity. |
| `--wait-timeout SEC` | Set the local wait limit. |
| `--idempotency-key KEY` | Identify repeat submissions. |
| `--detach` | Return after Call creation. |
| `--json-events` | Write retained Call events as JSON Lines. |

Global connection options must appear before the subcommand:

```bash
gfaas --api-base https://gpu.example.com/api \
  --request-timeout 60 \
  --poll-interval 1 \
  custom run vecadd.cu
```

Use environment variables for normal operation. Do not put API keys in command
arguments.

## 12. Workspace and trust rules

The CLI uploads only the selected exercise or custom source files. It rejects:

- Symbolic links.
- Hard links in an exercise workspace.
- Absolute paths and paths that contain `..`.
- More than 10,000 files.
- More than 1GiB of file data.
- Special file types.

The CLI skips common version-control, cache, build, and virtual-environment
directories. It also excludes course solution directories.

The remote worker runs submitted code inside the selected worker image.
Treat every submitted source tree as executable code.

The local `report feedback` command has a different trust boundary. It imports
course Python code into the local process. It requires `--trust-course-code`.

The CLI does not replace existing JSON files or downloaded profile reports.
This rule prevents one experiment from silently overwriting another.

## 13. Exit status

The course and custom workflow commands use these exit codes:

| Code | Meaning |
| --- | --- |
| `0` | The workflow passed, or the CLI detached successfully. |
| `1` | Compilation or linking failed. |
| `2` | The program stopped with an error. |
| `3` | A correctness test reported a wrong answer. |
| `4` | The workflow exceeded a time limit. |
| `5` | Setup, API access, remote execution, or report parsing failed. |
| `130` | The user interrupted the command. |

A course runner controls its detailed exit result. A custom harness must return
a nonzero exit code for an incorrect numerical result.

The `run` and `local run` commands return the program status when the program stops with an error.
They limit a positive program status to `125`. They return `1` for a negative status or a CLI error.

The Call, Artifact, pool, and completion commands return `0` after success. They return `1` after
a client or service error.

## 14. Troubleshooting

### The CLI cannot reach the coordinator

Make sure that `GFAAS_API_BASE` contains the public API path. Make sure that
`GFAAS_API_KEY` contains a valid key.

Show the available pools:

```bash
gfaas pool list
```

### The requested GPU pool is not configured

Run `gfaas pool list` and select one of the reported names with
`--gpu-type`.

### The Call waits for capacity

The selected workers have insufficient free GPUs or other resources. Keep the
Call queued, select another pool, or cancel it.

Use `--capacity-wait` to set the maximum scheduler wait.

### The CLI cannot find the exercise

Make sure that the exercise directory contains `run.py` and `runner/cli.py`.
Run the command from that directory, or use `--exercise-dir`.

### The linker reports `undefined reference to main`

The custom source contains a kernel but no host entry point. Add a harness that
defines `main()`, then pass it with `--harness`.

### A custom profile contains no selected kernel

The program did not create the default `profile_kernel` NVTX range. Add that
range, select the correct name with `--nvtx-range`, or use
`--no-nvtx-filter`.

### Nsight profiling is slow

Nsight Compute replays kernels to collect metric groups. `--set full` needs
more passes than `--set basic`.

Use a benchmark for timing. Use a profile to explain the benchmark.

### The local report parser cannot import `ncu_report.py`

Install Nsight Compute locally, or add its `extras/python` directory to
`PYTHONPATH`. A local GPU is not necessary for report parsing.

### A result file already exists

Select a new output path. The CLI does not replace existing JSON or profile
files.

### A local wait ended but the Call still runs

`--wait-timeout` limits only the local wait. Inspect the Call with
`gfaas call show`. Then watch or cancel it with the same command.
