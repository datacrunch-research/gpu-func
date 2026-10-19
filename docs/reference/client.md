# Client and Call handles

## Connection configuration

`Client(config=None)` uses `ClientConfig.from_env()` when no configuration is supplied. Use it as a
context manager, or call `close()` when you finish. Closing the client closes its HTTP connections;
it does not cancel remote Calls.

`ClientConfig` is a frozen dataclass with these constructor fields:

| Field                         | Meaning                                                   |
| ----------------------------- | --------------------------------------------------------- |
| `api_base`                    | Required API base URL ending in `/api`                    |
| `api_key`                     | Required constructor argument; use a key string or `None` |
| `poll_interval_s`             | Required polling interval in seconds                      |
| `request_timeout_s`           | Required HTTP timeout in seconds                          |
| `upload_concurrency`          | Default 4; integer from 1 to 16                           |
| `upload_completion_timeout_s` | Default 600; positive finite seconds                      |

`ClientConfig.from_env(env=None)` fills all fields from environment variables and defaults. To
change one field, use `dataclasses.replace()` on that configuration. See
[installation](../install.md#configure-the-client) for environment variable names.

`get_capabilities()` returns the service's advertised capability dictionary. The SDK checks relevant
capabilities before submitting explicit hardware requirements or native image variants.

## Submit a function

`submit()` accepts keyword arguments and returns a `RemoteResult`:

| Argument                                         | Default        | Meaning                                                                              |
| ------------------------------------------------ | -------------- | ------------------------------------------------------------------------------------ |
| `image`                                          | required       | Registered string reference or `Image`                                               |
| `function`                                       | required       | A file-defined callable, or a `(module, qualname)` tuple                             |
| `args`, `kwargs`                                 | `()`, `None`   | Arguments serialized for remote execution                                            |
| `source_file`                                    | inferred       | Optional source path; a tuple without a file refers to a module already in the image |
| `app_name`                                       | `"gfaas"`      | Function resource name                                                               |
| `gpu_count`, `gpu`                               | `None`, `None` | Explicit count or older selector; mutually exclusive                                 |
| `gpu_type`                                       | `"any"`        | Pool restriction; does not request a GPU by itself                                   |
| `compatibility`                                  | `None`         | CPU architecture and GPU capability requirements                                     |
| `timeout_s`                                      | `300`          | Active execution timeout                                                             |
| `capacity_wait_s`                                | `None`         | Optional placement/preparation deadline                                              |
| `cpu_millicores`, `memory_bytes`                 | `None`         | CPU and memory requests                                                              |
| `ephemeral_storage_bytes`, `shared_memory_bytes` | `None`         | Scratch and shared-memory requests                                                   |
| `max_log_bytes`, `max_output_bytes`              | `None`         | Retained-log and declared-output limits                                              |
| `env`                                            | `None`         | Stored Environment variables; do not put secrets here                                |
| `outputs`                                        | `()`           | Terminal or checkpoint output declarations                                           |
| `stages`                                         | `()`           | Optional sequential `CallStage` declarations                                         |
| `idempotency_key`                                | generated      | Key for retrying the same Call request                                               |
| `qualification_image_digest`                     | `None`         | Shares that digest after a successful qualification Call                             |

The SDK packages one source file and its own runner support. It does not package a whole application
or install local dependencies. With `function=(module, qualname)` and a supplied source file, the
module must match that file's module name.

Use [ordinary functions](../python-functions.md) for the common case. Read
[staged Calls](../calls/stages.md) before supplying `stages`. The
[sharing rules](../images/qualification.md) apply to `qualification_image_digest`.

## Inspect a Call

All dictionaries below are service responses. Fields can vary with the Call state and server
version.

| Method                                                             | Return value                                                     |
| ------------------------------------------------------------------ | ---------------------------------------------------------------- |
| `get_call(call_id)`                                                | Current Call status dictionary                                   |
| `cancel_call(call_id, *, reason=None)`                             | Call dictionary after the cancellation request                   |
| `get_call_result(call_id)`                                         | Result envelope, without deserializing the Python output         |
| `wait_for_result(call_id, *, timeout_s=None)`                      | Decoded output after success; raises on failure or local timeout |
| `list_attempts(call_id)`                                           | Page of infrastructure Attempts                                  |
| `list_events(call_id, *, after=None, limit=100)`                   | Bounded event page with `items` and `next_cursor`                |
| `iter_events(call_id, *, after=None, follow=True, timeout_s=None)` | Iterator of event dictionaries                                   |
| `get_call_logs(call_id)`                                           | Retained `stdout`, `stderr`, and `truncated` flag                |
| `list_call_artifacts(call_id)`                                     | Publication page with `items`; no SDK pagination arguments       |

A result envelope reports `outcome` and, on success, an `output` that can be inline or an Artifact
reference. `wait_for_result()` follows that reference and decodes it. The
[trust rules](../python-functions.md#serialization-and-trust) apply to decoding.

The wait timeout ends local polling. It does not cancel the Call and is not a hard deadline for
result download or decoding. HTTP operations have their own timeout. See
[reconnection](../calls/lifecycle.md#reconnect-from-another-process).

`iter_events()` automatically resumes after temporary connection errors. Its total deadline includes
recovery. `iter_logs()` uses the same recovery rules. See
[event recovery](../calls/events.md#automatic-recovery) for retry limits, HTTP errors, and cursor
handling. Single-request methods such as `list_events()` and `get_call()` do not gain automatic
retries.

## RemoteResult

`submit()` and `Function.spawn()` create this handle. It retains `client`, `job_id`, `gpu_type`, and
optional `environment_id` and `function_id`. `call_id` is the preferred property for the durable ID;
`job_id` is its compatibility name.

| Method                                                    | Equivalent client operation                       |
| --------------------------------------------------------- | ------------------------------------------------- |
| `wait(*, timeout_s=None)`                                 | `wait_for_result(call_id, ...)`                   |
| `status()`                                                | `get_call(call_id)`                               |
| `cancel(*, reason=None)`                                  | `cancel_call(call_id, ...)`                       |
| `iter_events(*, after=None, follow=True, timeout_s=None)` | `iter_events(call_id, ...)`                       |
| `iter_logs(*, after=None, follow=True, timeout_s=None)`   | Event iterator filtered to `(stream, text)` pairs |
| `logs()`                                                  | `get_call_logs(call_id)`                          |
| `artifacts()`                                             | `list_call_artifacts(call_id)`                    |

For a new process with a saved ID, use the Client methods directly; reconstructing a handle is not
required.

## Upload Artifacts

The methods return Artifact metadata, except session creation, which returns the upload session.

| Method                                   | Options                                                                                                          |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `upload_artifact(data, ...)`             | Bytes; `kind="other"`, `media_type="application/octet-stream"`, `filename=None`, `upload_id=None`, `cancel=None` |
| `upload_artifact_file(path, ...)`        | Same options, plus `progress=None`                                                                               |
| `create_artifact_file_upload(path, ...)` | `kind="other"`, `media_type="application/octet-stream"`, `filename=None`                                         |
| `upload_artifact_directory(path, ...)`   | `kind="input"`, `filename=None`, `progress=None`, `cancel=None`                                                  |

All options after the first argument are keyword-only. `upload_id` resumes the same file or byte
session. Directory uploads do not accept an aggregate resume ID. `cancel` is a `threading.Event`.
File and directory progress callbacks receive `ArtifactUploadProgress`.

See [transfer and recovery](../artifacts/upload-download.md) for accepted file types, concurrency,
publication phases, timeouts, and partial-failure receipts.

## List, download, and delete Artifacts

| Method                                                  | Return value                                            |
| ------------------------------------------------------- | ------------------------------------------------------- |
| `get_artifact(artifact_id)`                             | Metadata dictionary                                     |
| `list_artifacts(*, after=None, limit=100)`              | Owner-scoped page with `items` and `next_cursor`        |
| `iter_artifacts(*, limit=100)`                          | Iterator across owner-scoped pages                      |
| `download_artifact(artifact_id)`                        | `(bytes, media_type)`; buffers the whole response       |
| `download_artifact_file(artifact_id, destination)`      | `(Path, media_type)` after size and digest verification |
| `download_artifact_directory(artifact_id, destination)` | `Path` after manifest and file verification             |
| `delete_artifact(artifact_id)`                          | `None`; raises if the Artifact remains in use           |

Use file downloads for large results. Destination paths must not exist. A directory download retains
the tree layout and read-only modes. Deleting a tree does not delete its child blobs.

## Lower-level resource creation

`create_environment(definition, *, idempotency_key=None)`,
`create_function(definition, *, idempotency_key=None)`, and
`create_call(request, *, idempotency_key=None)` send dictionaries directly to the service. They
return the created or reused resource dictionary. `get_environment(id)` and `get_function(id)`
return existing definitions.

The following fragment shows the minimum shape for a callable that already exists in an image.
Replace the image, module, and callable names with your own values:

```python
from gfaas import Client

with Client() as client:
    environment = client.create_environment({
        "name": "example",
        "source": {"kind": "registered_image", "name": "YOUR_IMAGE"},
        "variables": {},
    })
    function = client.create_function({
        "name": "example",
        "environment_id": environment["id"],
        "executable": {"kind": "python_callable", "module": "your_module", "qualname": "run"},
        "default_resources": {"gpu": {"count": 1, "models": ["YOUR_POOL"]}, "timeout_seconds": 300},
    })
    call = client.create_call({"function_id": function["id"]})
    print(call["id"])
```

Use `submit()` when the SDK must package local code or serialize arguments. Raw resource methods do
not perform that work or the SDK's hardware capability preflight. The service validates requests.
The resource methods use `/v1/environments`, `/v1/functions`, and `/v1/calls` beneath `api_base`.
