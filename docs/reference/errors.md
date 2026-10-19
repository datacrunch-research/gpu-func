# Exceptions and recovery evidence

API and execution errors derive from `GfaasError`, except the local validation errors noted below.
Import `GfaasError` and `SerializationError` from `gfaas.errors`. Other listed exception classes are
also exported by `gfaas`.

| Exception                        | Meaning and useful fields                                                                                                        |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `GfaasError`                     | API failure or unsuccessful Call; optional `status_code`                                                                         |
| `SerializationError`             | Argument encoding or result decoding failed                                                                                      |
| `UnsupportedGpuPoolError`        | Unsupported pool; `requested_gpu_pool`, `supported_gpu_pools`                                                                    |
| `ArtifactUploadError`            | Transfer or publication could not complete; `upload_id`, `phase`                                                                 |
| `ArtifactTreeUploadError`        | Directory operation failed; `child_artifact_ids`, `upload_ids`, `failures`                                                       |
| `CudaError`                      | Base class for CUDA helper report errors; `report`, `phase`, `stdout`, `stderr`, `returncode`, `compile_ms`, `run_ms`, `ncu_csv` |
| `CudaCompilationError`           | Compile failure represented by a CUDA report; staged compile failures normally fail the Call instead                             |
| `CudaProcessError`               | Program or profiler failed in a CUDA helper report                                                                               |
| `TritonCompilationError`         | Compiler variants failed; `report`, `call_id`, `call_ids`                                                                        |
| `TritonQuickBenchmarkError`      | Early benchmark failed; `report`, `call_id`, `call_ids`                                                                          |
| `TritonBenchmarkError`           | Replicated benchmark failed; `report`, `call_ids`                                                                                |
| `HelionCompilationError`         | Helion compilation failed; `report`, `call_ids`                                                                                  |
| `ThunderKittensCompilationError` | ThunderKittens compilation failed; `report`, `call_ids`                                                                          |
| `CutlassCompilationError`        | C++ CUTLASS compilation failed; `report`, `call_ids`                                                                             |
| `CuteDSLCompilationError`        | CuTe DSL compilation failed; `report`, `call_ids`                                                                                |

`UnsupportedTritonKernelError`, `UnsupportedHelionKernelError`, and `UnsupportedCuteDSLKernelError`
derive from `ValueError`. They indicate unsupported source or configuration. Ordinary `TypeError`
and `ValueError` can also report invalid arguments before submission.

`TritonExecutionNotImplementedError` remains exported for compatibility with earlier compiler-only
work. Current managed kernels implement execution; do not use that exception as a capability query.

A local `TimeoutError` from waiting does not cancel the remote Call. Keep its ID and inspect status.
HTTP failures and decoding failures are different from a workload's reported execution error.

## Recover deliberately

For an upload failure, retain its upload ID before retrying. A lost completion response can leave
publication uncertain. Read
[upload recovery](../artifacts/upload-download.md#publication-timeouts-and-recovery).

For a kernel failure, retain report and Call IDs. Read logs before repeating compilation or tuning.
Partial successes do not establish that a winner exists. See
[kernel recovery](../kernels/lifecycle.md).

Do not automatically retry authentication, invalid input, or policy failures. Change the request or
credentials first. An image or package mismatch needs a compatible environment, not more retries.
