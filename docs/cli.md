# Command-line interface

The `gfaas` command runs Python files, CUDA files, CUDA course exercises, and custom CUDA
workflows. It also manages durable Calls and Artifacts. See [CUDA development
workflows](cuda-workflows.md) for exercise, benchmark, sanitizer, profiler, and grading commands.

The `run` command supports one self-contained source file. It does not package a source tree or
install local dependencies. The course and custom commands package the selected workspace.

The `.py` and `.cu` suffixes select a runtime. Use `--runtime python` or `--runtime cuda` to
override the suffix or to use a different suffix.

The command groups have these purposes:

| Command | Purpose |
| --- | --- |
| `gfaas run` | Submit one Python or CUDA source file. |
| `gfaas local` | Inspect or use a CUDA GPU on the local host. |
| `gfaas call` | Inspect, watch, or cancel a durable Call. |
| `gfaas artifact` | Download an Artifact. |
| `gfaas pool` | List the configured GPU pools. |
| `gfaas completion` | Generate shell-completion setup. |
| `gfaas custom` | Compile, run, or profile a custom CUDA program. |
| `gfaas exercise` | Run an action for a CUDA course exercise. |
| `gfaas report` | Inspect a local Nsight Compute report. |

The top-level `compile`, `test`, `benchmark`, `sanitizer`, `profile`, and `grade` commands
auto-detect a course exercise from the current directory.

## Enable shell completion

The CLI can generate completion setup for Bash, Fish, Zsh, and PowerShell. Completion includes
commands, nested subcommands, options, option values, and Python or CUDA source paths.

Enable completion in the current Bash session:

```bash
eval "$(gfaas completion bash)"
```

Add the same line to `~/.bashrc` to enable it in new Bash sessions.

Install completion for Fish:

```fish
mkdir -p ~/.config/fish/completions
gfaas completion fish > ~/.config/fish/completions/gfaas.fish
```

Fish loads that file in new sessions. Run `gfaas completion fish | source` to enable it in the
current session.

Enable completion in the current Zsh session:

```zsh
eval "$(gfaas completion zsh)"
```

Add the same line to `~/.zshrc` to enable it in new Zsh sessions.

Enable completion in the current PowerShell session:

```powershell
gfaas completion powershell | Out-String | Invoke-Expression
```

Completion generation is local. It does not read credentials or connect to the service.

## Run CUDA directly on the local host

Local mode uses the NVIDIA GPU and CUDA toolkit on the current host. It does not create a Call or
use a container.

CAUTION: Run only trusted CUDA source in local mode. The compiled program can read host files and
the inherited process environment.

Inspect the local toolchain before execution:

```bash
gfaas local info
gfaas local info --json
```

The command reports the GPUs, compute capabilities, CUDA compiler, host compiler, and Nsight Compute
path. It also reports the effective `CUDA_VISIBLE_DEVICES` value.

Compile and run a local CUDA program:

```bash
gfaas local run examples/cli/hello_cuda.cu \
  --nvcc-flag=-O3 \
  -- --problem-size 4096
```

Local mode detects the selected GPU architecture. It adds the applicable `-arch=sm_NNN` compiler
option unless another architecture option is present.

Use `--arch sm_NNN` to select an architecture. Use `--device INDEX_OR_UUID` to select a physical GPU
and set `CUDA_VISIBLE_DEVICES` for the program.

Tool discovery uses this order:

1. The `--nvcc`, `--ncu`, and `--ccbin` options.
2. `CUDACXX`, `CUDA_HOME`, `GFAAS_NCU`, `GFAAS_NVCC_CCBIN`, and `CXX`.
3. The current `PATH`.
4. The conventional `/usr/local/cuda/bin` directory.

`GFAAS_CUDA_ARCH` sets the default architecture. `GFAAS_NVIDIA_SMI` selects a nonstandard
`nvidia-smi` executable.

Use `--env NAME=VALUE` to add or replace an environment value. Local mode inherits other values from
the current process.

Declared local outputs remain ordinary files. Keep generated benchmark files under `.local/gfaas/`
so they do not add files to the repository root:

```bash
mkdir -p .local/gfaas
gfaas local run kernel.cu \
  --output benchmark=.local/gfaas/results.csv \
  -- --output .local/gfaas/results.csv
```

The program must write the declared path. The CLI refuses to replace a declared path that already
exists.

Use JSON results to compare the same source on local and remote GPUs:

```bash
mkdir -p .local/gfaas
gfaas local run kernel.cu --json > .local/gfaas/local.json
gfaas run kernel.cu --gpu-type gb300 --json > .local/gfaas/remote.jsonl
```

The local result records the device, architecture, toolchain, compiler time, execution time, and
captured output. The remote command produces a JSON Lines event stream with a final result record.

## Run the included examples

Run the standalone CUDA example:

```bash
gfaas run examples/cli/hello_cuda.cu \
  --gpu-type gb300 \
  --nvcc-flag=-O3
```

The remote `run` command passes each `--nvcc-flag` value to `nvcc`. If you omit an architecture
flag, the `nvcc` configuration in the selected image controls the compilation target.

Run the standalone PyTorch example:

```bash
gfaas run examples/cli/hello_python.py \
  --image pytorch-cu130 \
  --gpu-type gb300 \
  -- --size 2048
```

The arguments after `--` go to the Python program.

## Credentials

The CLI does not store an API key. It reads the same environment variables as the Python SDK:

```bash
export GFAAS_API_BASE=https://gpu.example.com/api
export GFAAS_API_KEY='provided-separately'
```

Do not pass an API key as a command argument. Command arguments can appear in shell history and
process lists.

The CLI does not copy `GFAAS_API_KEY` into the workload environment. Use `--env` only for non-secret
workload values. The coordinator stores workload environment values in the Environment resource.

The global connection options must occur before the command name:

```bash
gfaas \
  --api-base https://gpu.example.com/api \
  --request-timeout 60 \
  --poll-interval 0.5 \
  pool list
```

Use the environment variables for normal operation. The CLI has no API-key option.

## Run a CUDA file

Create a self-contained `kernel.cu` file. Then submit it:

```bash
uv run gfaas run kernel.cu \
  --gpu-type gb300 \
  --nvcc-flag=-O3 \
  -- --problem-size 4096
```

The `.cu` suffix selects the CUDA runner and the `cuda-nvcc` image. The arguments after `--` go to
the compiled program.

The remote `run` command has no `--arch` option. Use a compiler flag to select an explicit target:

```bash
uv run gfaas run kernel.cu \
  --gpu-type gb300 \
  --nvcc-flag=-arch=sm_103
```

Add `--profile` to collect an Nsight Compute CSV report. With no profiler arguments, the command
uses `--set full`.

To replace the default, repeat `--ncu-arg` for each profiler argument:

```bash
uv run gfaas run kernel.cu \
  --profile \
  --ncu-arg=--set \
  --ncu-arg=basic
```

The compiled program starts in the workload output directory. Declare a file that the Call must
publish:

```bash
uv run gfaas run kernel.cu \
  --gpu-type gb300 \
  --output benchmark=results.csv
```

The CUDA program must write `results.csv` in its current directory. Use a relative path for each
declared output.

## Run a Python script

Submit a self-contained Python script:

```bash
uv run gfaas run experiment.py \
  --image pytorch-cu130 \
  --gpu-type gb300 \
  --memory 128GiB \
  -- --steps 10
```

The `.py` suffix selects the Python script runner. The runner invokes the script with unbuffered
output and passes the arguments after `--`.

The selected image must contain all imported third-party packages. The CLI does not import or run
the script on the client.

The script starts with the output directory as its current directory. Declare files and directories
that the Call must publish:

```bash
uv run gfaas run experiment.py \
  --image pytorch-cu130 \
  --output report=reports/result.json \
  --output-directory profiles=profiles
```

The paths are relative to the workload output directory. A required output that does not exist
causes the Call to fail.

## Run a Python callable

Add the callable name after the source path:

```bash
uv run gfaas run experiment.py:train \
  --image pytorch-cu130 \
  -- input.json
```

The callable must be at module scope. Each value after `--` becomes one string positional argument.
The CLI prints the serialized return value after the Call completes.

Use a Python SDK program when the callable needs typed arguments or keyword arguments.

## Select resources

The `run` command accepts these resource options:

| Option             | Meaning                                      |
| ------------------ | -------------------------------------------- |
| `--gpu-type`       | GPU pool name                                |
| `--gpu-count`      | GPU count on one worker                      |
| `--timeout`        | Execution deadline in seconds                |
| `--capacity-wait`  | Preparation and capacity deadline in seconds |
| `--cpu-millicores` | CPU request                                  |
| `--memory`         | Aggregate memory limit                       |
| `--storage`        | Writable scratch capacity                    |
| `--shared-memory`  | `/dev/shm` size                              |
| `--max-log`        | Retained output limit                        |
| `--max-output`     | Published Artifact limit                     |

Size options accept bytes or the `KiB`, `MiB`, `GiB`, and `TiB` suffixes.

If `--gpu-type` is absent, the CLI reads `GFAAS_GPU_TYPE`. If that variable is absent, the CLI uses
the literal `any` pool.

## Detach and reconnect

Use `--detach` to return after Call creation:

```bash
call_id="$(uv run gfaas run kernel.cu --gpu-type gb300 --detach)"
```

The command writes only the Call ID to standard output. The remote Call continues after the client
exits.

Use the Call commands to reconnect:

```bash
uv run gfaas call show "$call_id"
uv run gfaas call watch "$call_id"
uv run gfaas call logs "$call_id"
uv run gfaas call logs "$call_id" --follow
uv run gfaas call artifacts "$call_id"
uv run gfaas call cancel "$call_id" --reason superseded
```

The Call subcommands have these purposes:

| Subcommand | Purpose |
| --- | --- |
| `show` | Show the current Call record. |
| `watch` | Follow retained and new Call events. |
| `logs` | Read retained standard output and standard error. |
| `cancel` | Request Call cancellation. |
| `artifacts` | List the Artifacts that the Call published. |

The `watch` command follows retained and new events. Use `--after CURSOR` to resume after a known
event. The `logs` command returns retained output and then stops. Add `--follow` to wait for new
output. All Call subcommands accept `--json`.

If you interrupt a foreground `gfaas run`, the CLI requests remote Call cancellation. Use `--detach`
before a long Call when the Call must survive a client interruption.

## Machine-readable output

By default, the CLI shows short status lines for state changes, preparation, placement, and
Artifacts. Workload output remains on its original output stream.

Add `--json` to produce complete event records. A foreground `run` command produces JSON Lines
because it reports the submission, Call events, and final result.

```bash
uv run gfaas run kernel.cu --gpu-type gb300 --json
uv run gfaas pool list --json
```

Human-readable progress goes to standard error. This separation lets scripts capture detached Call
IDs and JSON output from standard output.

## Download an Artifact

Download one Artifact to a new path:

```bash
uv run gfaas artifact download art_abc result.bin
```

The command refuses to replace an existing path. Directory Artifacts retain their tree layout and
file modes.

If you omit the destination, the CLI uses the Artifact filename. If the Artifact has no filename,
the CLI uses its Artifact ID.

Add `--json` to return the Artifact ID, destination, and media type as a JSON object.
