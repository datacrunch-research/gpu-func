# Errors and limits

## Error types

| Exception                 | When it occurs                                              |
| ------------------------- | ----------------------------------------------------------- |
| `GfaasError`              | An API request fails, or a Call ends in a non-success state |
| `ArtifactTreeUploadError` | Tree creation fails after one or more child uploads         |
| `CudaCompilationError`    | A single-stage CUDA helper returns a compile error report   |
| `CudaProcessError`        | The compiled program or profiler exits unsuccessfully       |
| `SerializationError`      | The SDK cannot serialize the arguments or result            |
| `TimeoutError`            | The local `wait()` reaches its deadline                     |

CUDA exceptions keep the full report as `.report`. They also expose `.stdout`, `.stderr`,
`.returncode`, `.compile_ms`, and `.run_ms`.

Remote staged CUDA compilation fails the Call before GPU assignment. `wait()` raises `GfaasError`
for this terminal state. Read the retained Call logs to get the complete compiler output.

The compiled program returns a process report from the GPU stage. `compile_and_run()` converts an
unsuccessful process report to `CudaProcessError`.

`TimeoutError` only stops the local wait. It does not cancel the remote Call. Keep the
`RemoteResult`, then read its status or request cancellation.

```python
try:
    result = call.wait(timeout_s=30)
except TimeoutError:
    print("still running:", call.status()["state"])
    call.cancel(reason="client deadline")
```

If one handler processes CUDA report errors, catch `CudaError`:

```python
import gfaas

try:
    report = gfaas.compile_and_run(source, gpu="any", gpu_type="gb300")
except gfaas.CudaError as error:
    print("phase:", error.phase)
    print("return code:", error.returncode)
    print(error.stderr)
```

Do not retry every `GfaasError` automatically. Authentication, invalid input, and policy errors need
a configuration or code change.

## GPU requests

| Request            | Meaning                                  |
| ------------------ | ---------------------------------------- |
| `gpu_count=0`      | No GPU                                   |
| `gpu_count=1`      | One GPU from the selected pool           |
| `gpu_count=4`      | Four GPUs on one worker                  |
| `gpu_type="gb300"` | Restrict the request to the `gb300` pool |

Use `gpu_count` for new code. The worker selects and leases that number of idle GPUs as one atomic
assignment. The selected worker and its policy must provide the complete request.

The older `gpu` option remains available. Values such as `gpu="any"` request one GPU. Numeric tokens
such as `gpu="0,1"` determine only the count and do not select physical device IDs. Do not use `gpu`
and `gpu_count` together. The SDK rejects `gpu="all"` as ambiguous.

The default `gpu_type="any"` names the literal `any` broker route. It is not a wildcard. Use a pool
name that the operator provides.

## Limits

Limits are deployment-specific. Ask your operator for the current values.

| Limit                 | Scope                                                       |
| --------------------- | ----------------------------------------------------------- |
| Inline Artifact bytes | Small single-request uploads and serialized values          |
| Upload session bytes  | Unfinished bytes reserved by one principal                  |
| Log retention         | Combined stdout and stderr per Call                         |
| Capacity wait         | Time before worker acceptance, including input preparation  |
| Execution timeout     | One active wall-clock deadline per run                      |
| CPU and memory        | Per-Call request, bounded by the selected worker policy     |
| Disk scratch          | Per-Call reservation and periodic logical-size check        |
| Shared memory         | Per-Call `/dev/shm` size. Pages also count as memory.       |
| Declared outputs      | File bytes across terminal outputs and checkpoint versions  |
| Artifact references   | 32 unique Artifacts per Call                                |
| Tree entries          | Deployment-specific maximum number of entries                |
| Tree path length      | Deployment-specific maximum UTF-8 path length                |

The `wait()` deadline is a client-side limit. The function `capacity_wait` limits all time before
worker acceptance. This time includes image preparation and Artifact staging. The function `timeout`
is a worker-side execution limit. These limits are independent.

The service can apply more limits than this table shows. Ask the service operator for the limits of
your account and GPU pool.
