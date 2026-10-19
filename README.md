# vFunc Python SDK and CLI

vFunc runs functions on remote GPU machines. Submit Python code or CUDA programs, choose the
hardware and software environment, and collect results without managing the worker yourself. Calls
have durable IDs, retained logs, and input and output Artifacts.

This repository contains the public Python SDK and the `vfunc` command. The Python package and
imports remain `gfaas`. The service and its credentials come from your operator.

Read the [user guide](docs/introduction.md) for installation and examples.

## Install

For the command-line client:

```bash
uv tool install "git+https://github.com/datacrunch-research/gpu-func.git"
vfunc --help
```

For the Python SDK in an existing uv project:

```bash
uv add "git+https://github.com/datacrunch-research/gpu-func.git"
```

Read [installation](docs/install.md) for standalone environments, repository examples, and upgrades.

## Submit a Python function

Set `GFAAS_API_BASE` to your service's API URL, ending in `/api`. Supply `GFAAS_API_KEY` through
your normal secret mechanism. Replace the image and pool below with values supplied by your
operator. The image must contain compatible Python and PyTorch versions.

```python
import gfaas

app = gfaas.App("hello", image=gfaas.Image("pytorch-cu130"))


@app.function(gpu_count=1, gpu_type="gb300")
def gpu_name() -> str:
    import torch

    return torch.cuda.get_device_name(0)


if __name__ == "__main__":
    print(gpu_name.remote())
```

Save this as a Python file and run it with `uv run python your_program.py`. The SDK packages that
file. It does not package your entire project or install dependencies remotely. Read
[Python functions](docs/python-functions.md) for outputs, resource limits, and serialization safety.

## Run source files

```bash
vfunc run experiment.py --image YOUR_PYTHON_IMAGE --gpu-type YOUR_POOL
vfunc run kernel.cu --gpu-type YOUR_POOL --nvcc-flag=-arch=sm_103
```

Use the CUDA compilation target for your GPU. CUDA programs need `main()`. Remote CUDA compilation
uses a CPU stage before the Call requests GPU capacity.

To submit a Call and disconnect:

```bash
vfunc run kernel.cu --gpu-type YOUR_POOL --nvcc-flag=-arch=sm_103 --detach
vfunc call show call_...
vfunc call logs call_... --follow
vfunc call artifacts call_...
vfunc artifact download art_... ./result
```

## Choose a workflow

| Task                                   | Guide                                                |
| -------------------------------------- | ---------------------------------------------------- |
| Run ordinary Python functions          | [Python functions](docs/python-functions.md)         |
| Compile, tune, and benchmark kernels   | [Managed kernels](docs/kernels/index.md)             |
| Run CUDA programs and course exercises | [CUDA workflows](GUIDE.md)                           |
| Select images for ARM or x86 workers   | [Images and hardware](docs/images/hardware.md)       |
| Transfer files or recover an upload    | [Artifacts](docs/artifacts/upload-download.md)       |
| Reconnect to a Call                    | [Call lifecycle](docs/calls/lifecycle.md)            |
| Fine-tune a model with LoRA            | [Nemotron walkthrough](docs/fine-tuning-nemotron.md) |
| Find API parameters and return values  | [Python API reference](docs/reference/index.md)      |
| Install the CLI skill for an agent     | [Agent setup](examples/cli/README.md)                |

Managed kernels support Triton, TLX, Gluon, Helion, CUDA C++, ThunderKittens, C++ CUTLASS, and CuTe
DSL. Their wrappers compile configurations and can tune or benchmark them on remote GPUs. The guide
explains image requirements, correctness checks, tensor writes, caching, and resource use.

## Develop

From a checkout:

```bash
uv sync --extra dev --locked
uv run ruff format --check src tests
uv run ruff check src tests
uv run mypy src
uv run pytest -q
dprint check
python scripts/check-docs.py
```

Some kernel tests require PyTorch and skip without it. CI installs CPU PyTorch to run those tests.
Remote GPU tests require a suitable image, service access, and explicit authorization to consume
capacity.

For API changes, update the relevant reference and user chapter in the same PR. Keep complete
examples import-safe and distinguish executable examples from fragments. Use simple English and
format Markdown with `dprint fmt` before committing. The documentation check builds a clean book,
checks local links and anchors, and parses Python snippets without submitting remote work.
