# Install and authenticate

You need Python 3.11 or newer, `uv`, and an API key.

## Get the SDK

Install the SDK and command from the public Git repository:

```bash
uv tool install "git+https://github.com/datacrunch-research/gpu-func.git"
```

For development, clone the repository and install its locked environment:

```bash
uv sync --extra dev --locked
```

Make sure that the command is available:

```bash
gfaas --help
```

## Get an API key

The service has no self-service key issuance. Request a key from a service operator. The operator
delivers the key through a private channel.

CAUTION: Do not share, commit, or paste the key into source code.

## Configure the client

```bash
export GFAAS_API_BASE=https://gpu.example.com/api
export GFAAS_API_KEY='provided-separately'
```

The SDK reads these variables:

| Variable                | Meaning                                | Default                 |
| ----------------------- | -------------------------------------- | ----------------------- |
| `GFAAS_API_BASE`        | Public API base that ends in `/api`    | `http://127.0.0.1:8000/api` |
| `GFAAS_API_KEY`         | Your API key                           | none                    |
| `GFAAS_POLL_INTERVAL`   | Poll interval for `wait()`, in seconds | `0.5`                   |
| `GFAAS_REQUEST_TIMEOUT` | HTTP request timeout, in seconds       | `60`                    |

The command-line interface reads the same variables. It does not store the API key in a file.

## Make sure the service is ready

The readiness endpoint needs no key:

```bash
curl --fail --silent --show-error "${GFAAS_API_BASE%/api}/ready"
```

The endpoint returns a non-success status when the service is not ready. Its response can also
report the observed worker inventory.
