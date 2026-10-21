# Install and authenticate

The Python package is `gfaas`. The command is `vfunc`. Both come from this repository. You need
Python 3.11 or newer, Git, and [uv](https://docs.astral.sh/uv/getting-started/installation/). Remote
work also needs a service URL, an API key, and a registered image.

## Install the CLI

For command-line use, install an isolated tool:

```bash
uv tool install "git+https://github.com/datacrunch-research/gpu-func.git"
vfunc --help
```

This makes `vfunc` available on your command path. It does not install `gfaas` into another Python
project. If the shell cannot find `vfunc`, run `uv tool update-shell`, then start a new shell.

To update this installation:

```bash
uv tool upgrade gfaas
```

## Install the Python SDK

In an existing uv project, add the SDK as a dependency:

```bash
uv add "git+https://github.com/datacrunch-research/gpu-func.git"
uv run python your_program.py
```

For a standalone virtual environment:

```bash
uv venv
uv pip install "git+https://github.com/datacrunch-research/gpu-func.git"
.venv/bin/python your_program.py
```

Git installations follow the default branch unless you select a revision. For reproducible
applications, retain your lockfile and the SDK commit used for a run.

## Run repository examples

Examples use paths relative to the repository root. Obtain the files and install their environment:

```bash
git clone https://github.com/datacrunch-research/gpu-func.git
cd gpu-func
uv sync --locked
uv run vfunc --help
```

From this checkout, use `uv run python examples/NAME.py` or `uv run vfunc ...`. Some examples need
additional local packages. The training examples use `uv sync --extra training-example`. Kernel
examples specify their compiler and tensor dependencies in their respective chapters.

For SDK development, use `uv sync --extra dev --locked`.

## Get an API key

The service has no self-service key issuance. Request a key and API base URL from a service
operator. The operator delivers the key through a private channel.

Do not share, commit, or paste the key into source code. Supply it through your normal secret
mechanism. If you enter it in a shell, prevent that command from entering shell history.

## Configure the client

```bash
export GFAAS_API_BASE=https://gpu.example.com/api
# Supply GFAAS_API_KEY through your normal secret mechanism.
```

The SDK reads these variables:

| Variable                          | Meaning                                            | Default                                     |
| --------------------------------- | -------------------------------------------------- | ------------------------------------------- |
| `GFAAS_API_BASE`                  | Public API base ending in `/api`                   | `http://127.0.0.1:8000/api`                 |
| `GFAAS_API_KEY`                   | Your API key                                       | none                                        |
| `GFAAS_POLL_INTERVAL`             | Result polling interval, in seconds                | `0.5`                                       |
| `GFAAS_REQUEST_TIMEOUT`           | HTTP request timeout, in seconds                   | `60`                                        |
| `GFAAS_UPLOAD_COMPLETION_TIMEOUT` | Multipart publication response timeout, in seconds | `600`                                       |
| `GFAAS_UPLOAD_CONCURRENCY`        | Concurrent upload limit per client, from 1 to 16   | `4`                                         |
| `GFAAS_DOWNLOAD_CONCURRENCY`      | Concurrent directory child downloads, from 1 to 16 | `4`                                         |
| `GFAAS_TRANSFER_CACHE_DIR`        | Base directory for private transfer recovery data  | `$XDG_CACHE_HOME/vfunc` or `~/.cache/vfunc` |

The CLI also reads `GFAAS_GPU_TYPE` as its default pool selection. Python SDK calls use their
explicit `gpu_type` argument. Neither interface stores the API key in a file.

## Check access and capacity

List the configured pools through the authenticated API:

```bash
vfunc pool list
```

The coordinator readiness endpoint needs no key:

```bash
curl --fail --silent --show-error "${GFAAS_API_BASE%/api}/ready"
```

Readiness checks the coordinator's dependencies. A ready coordinator does not guarantee an idle GPU
or enough capacity for your request. Ask the operator for image names and resource limits.

Before a Python function Call, check the image's Python and package versions. Read the
[serialization requirements](python-functions.md#serialization-and-trust).
