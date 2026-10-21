# Troubleshooting and limits

For the complete exception list and retained report fields, read the
[error reference](reference/errors.md).

## Error types

| Exception                 | When it occurs                                              |
| ------------------------- | ----------------------------------------------------------- |
| `GfaasError`              | An API request fails, or a Call ends in a non-success state |
| `ArtifactTreeUploadError` | A child upload or final tree creation fails                 |
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

## Execution isolation errors

A `sandbox` request currently fails as unsupported. The SDK also rejects explicit isolation
requirements when the service cannot advertise enforcement support. Removing the requirement changes
the requested boundary. It is not an automatic retry strategy.

`execution_policy_denied` means that your principal cannot use the Environment's resolved policy. An
administrator must restore permission, or you must use an Environment with a permitted policy. A new
default does not change existing Environments.

Waiting Calls can report `execution_unknown` or `execution_mismatch` in their `compatibility`
details. These states mean that matching hardware lacks reported enforcement support or has an
incompatible policy. More time does not change that policy. Ask the operator to inspect worker
versions and configuration. The capacity deadline still bounds the wait.

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

With coordinator placement, `gpu_type="any"` permits matching workers across pools. The deployment
must enable `any`. A named pool restricts placement. Setting `gpu_type` alone does not request a
GPU; set `gpu_count=1` for one GPU or `gpu_count=0` for a CPU-only Call.

## Limits

Limits are deployment-specific. Ask your operator for the current values.

| Limit                 | Scope                                                                   |
| --------------------- | ----------------------------------------------------------------------- |
| Inline Artifact bytes | Small single-request uploads; larger SDK uploads use multipart requests |
| Upload session bytes  | Unfinished bytes reserved by one principal                              |
| Log retention         | Combined stdout and stderr per Call                                     |
| Capacity wait         | Time before worker acceptance, including input preparation              |
| Execution timeout     | One active wall-clock deadline per run                                  |
| CPU and memory        | Per-Call request, bounded by the selected worker policy                 |
| Disk scratch          | Per-Call reservation and periodic logical-size check                    |
| Shared memory         | Per-Call `/dev/shm` size. Pages also count as memory.                   |
| Declared outputs      | File bytes across terminal outputs and checkpoint versions              |
| Artifact references   | 32 unique Artifacts per Call                                            |
| Tree entries          | Deployment-specific maximum number of entries                           |
| Tree path length      | Deployment-specific maximum UTF-8 path length                           |

The `wait()` deadline is a client-side limit. The function `capacity_wait` limits all time before
worker acceptance. This time includes image preparation and Artifact staging. The function `timeout`
is a worker-side execution limit. These limits are independent.

The service can apply more limits than this table shows. Ask the service operator for the limits of
your account and GPU pool.

## Common problems

| Symptom                                     | What to inspect                                                                                     |
| ------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `import gfaas` fails after CLI installation | Install the SDK in the Python project's environment; a uv tool is isolated                          |
| CUDA is unavailable in an ordinary function | Set `gpu_count=1`; selecting `gpu_type` alone does not request a GPU                                |
| Image reference is unknown                  | Ask for an accessible registered name or digest; worker-local fallback is not supported             |
| Call waits without starting                 | Read preparation and compatibility status, then compare requested resources with available capacity |
| Managed benchmark waits for several GPUs    | Check both tuning and benchmark replication factors; the default is three on one worker             |
| Kernel target probe lacks image identity    | Use a registered image and a service that reports image digest and architecture                     |
| Result decoding fails                       | Compare Python/package versions and confirm that the returned format is supported                   |
| Upload reaches 100% but does not return     | Read its publication phase; accepted bytes are not yet a published Artifact                         |
| Client wait expires                         | Save the Call ID and reconnect; a local timeout does not cancel remote work                         |
| An Artifact cannot be deleted               | Check durable references and tree membership before attempting cleanup                              |

Use `vfunc call show CALL_ID` and `vfunc call logs CALL_ID` to retain the exact status and error. A
queued Call does not prove that every GPU is busy. Preparation, hardware mismatch, missing reports,
and other resource requirements can also delay it.
