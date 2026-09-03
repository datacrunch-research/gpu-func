# vfunc CLI examples and agent skill

This directory contains two standalone GPU programs and the `vfunc-cli` agent skill.

- `hello_cuda.cu` compiles and runs a CUDA kernel.
- `hello_python.py` runs a PyTorch matrix multiplication.
- `SKILL.md` teaches an agent when and how to use the `vfunc` command.

## Requirements

Install these tools before you continue:

- Python 3.11 or newer.
- [uv](https://docs.astral.sh/uv/).
- Git, for installation from the repository.

You also need a gfaas API endpoint, an API key, and a GPU pool name.

## Install the vfunc command from a checkout

From the repository root, install the command as an editable uv tool:

```bash
uv tool install --editable .
uv tool update-shell
```

Restart the shell after `uv tool update-shell` changes the shell configuration.

An editable installation uses the code in the checkout. Local source changes apply without another
installation.

Make sure that the command is available:

```bash
vfunc --help
```

If the editable tool was installed before its dependencies changed, reinstall it once:

```bash
uv tool install --force --editable .
```

## Enable shell completion

Install completion for Fish:

```fish
mkdir -p ~/.config/fish/completions
vfunc completion fish > ~/.config/fish/completions/vfunc.fish
```

For Bash or Zsh, add the applicable line to the shell startup file:

```bash
eval "$(vfunc completion bash)"
```

```zsh
eval "$(vfunc completion zsh)"
```

Use `vfunc completion --help` to see all supported shells. Completion generation does not require an
API key.

## Install the vfunc command from GitHub

Install the current default branch directly from GitHub:

```bash
uv tool install git+https://github.com/datacrunch-research/gpu-func.git
uv tool update-shell
```

Upgrade a non-editable installation after a new version becomes available:

```bash
uv tool upgrade gfaas
```

## Install the skill in Codex

Keep a repository checkout when you use a symbolic link. Skill changes then apply without another
copy operation.

From the repository root, create the Codex skill directory:

```bash
skill_root="${CODEX_HOME:-$HOME/.codex}/skills"
mkdir -p "$skill_root"
ln -s "$(pwd)/examples/cli" "$skill_root/vfunc-cli"
```

The `ln` command stops if `vfunc-cli` already exists. Inspect the existing path before you replace
it.

Start a new Codex session after you install the skill. Codex can then select the skill
automatically.

You can also request the skill explicitly:

```text
Use $vfunc-cli to run this CUDA program on the gb300 pool.
```

For another skill-compatible agent, copy or link this directory into that agent's skill directory.
Keep the installed directory name as `vfunc-cli`.

## Configure access

Set the endpoint, API key, and default GPU pool in the shell environment:

```bash
export GFAAS_API_BASE=https://gpu.example.com/api
export GFAAS_API_KEY='provided-separately'
export GFAAS_GPU_TYPE=operator-provided-pool
```

CAUTION: Do not put the API key in this directory. Do not commit the API key to a repository.

Make sure that the client can read the service capabilities:

```bash
vfunc pool list
```

## Run the examples

Inspect a local CUDA host:

```bash
vfunc local info
```

Run the CUDA example directly on the local GPU:

```bash
vfunc local run examples/cli/hello_cuda.cu --nvcc-flag=-O3
```

Keep generated benchmark results under `.local/vfunc/`:

```bash
mkdir -p .local/vfunc
vfunc local run kernel.cu \
  --output benchmark=.local/vfunc/results.csv \
  -- --output .local/vfunc/results.csv
```

Local mode requires an NVIDIA GPU, `nvidia-smi`, `nvcc`, and a host C++ compiler. It does not use
gfaas credentials or a remote worker.

CAUTION: Local mode does not use a container. Run only source that you trust on the local host.

Run the same example through the GPU service. Use the architecture for the selected pool:

```bash
vfunc run examples/cli/hello_cuda.cu \
  --gpu-type gb300 \
  --nvcc-flag=-O3 \
  --nvcc-flag=-arch=sm_103
```

Ask the operator for the correct CUDA architecture. Do not infer it from the pool name.

Run the PyTorch example:

```bash
vfunc run examples/cli/hello_python.py \
  --image pytorch-cu130 \
  --gpu-type gb300 \
  -- --size 2048
```

Read [SKILL.md](SKILL.md) for Calls, Artifacts, resource limits, profiling, and error handling.
