# vFunc user guide

vFunc runs functions on remote GPU machines. You send code and inputs through an HTTPS API, then
read the results and logs. The service selects a worker and applies the requested resource limits.

This guide is for application developers and researchers who submit workloads. It does not require
SSH access or knowledge of the service's deployment tools.

The Python SDK is imported as `gfaas`. Its command-line tool is `vfunc`. The service operator
supplies your API URL, API key, available image names, and GPU-pool names.

## Choose an interface

| Your workload                                             | Interface                                    |
| --------------------------------------------------------- | -------------------------------------------- |
| A Python function with typed arguments and return values  | `App.function`, then `remote()` or `spawn()` |
| A self-contained Python script or CUDA program            | `vfunc run`                                  |
| A kernel with configurations to compile, tune, or measure | A managed kernel wrapper and `benchmark()`   |
| A CUDA course exercise or a multi-file CUDA program       | The course and `custom` CLI commands         |
| An application that manages durable resources directly    | `Client`                                     |

Python functions can perform GPU computation, training, or CPU-only preparation. The selected image
must contain the required software. The SDK does not supply machine-learning models or install your
local dependencies on the worker.

## Understand the resources

| Resource    | Meaning                                                              |
| ----------- | -------------------------------------------------------------------- |
| Image       | The software filesystem used by a workload                           |
| Environment | An image reference, environment variables, and hardware requirements |
| Function    | An executable workload in an Environment                             |
| Call        | One durable invocation, optionally with sequential stages            |
| Artifact    | An immutable file or directory tree                                  |

A Call can have placement Attempts, events, logs, a Python result, and declared output Artifacts.
The SDK creates most of these resources for normal function submissions.

A function's `remote()` method waits for its result. Its `spawn()` method returns a Call handle that
you can inspect, wait for, or cancel. Saving the Call ID lets another process reconnect later.

Managed kernels can create several Calls for discovery, compilation, tuning, and execution. They
provide ordinary kernel syntax on the client, but each launch can involve network transfers and
remote work. Read [managed kernels](kernels/index.md) before using them in a tight loop.

## Hardware and isolation

A GPU count requests devices on one worker. A pool restricts the available worker class. Hardware
requirements can further restrict CPU architecture and GPU compute capability.

Calls run in containers with the deployment's isolation policy. Do not assume that every deployment
accepts untrusted code. Python result decoding also has a separate
[client-side trust requirement](python-functions.md#serialization-and-trust).

The service does not manage long-lived web services or distributed jobs across multiple workers. A
multi-GPU Call uses GPUs on a single worker. Sequential stages and client-side orchestration can
coordinate bounded tasks; they do not provide a general distributed workflow engine.

## Start here

1. [Install and authenticate](install.md).
2. Choose [Python functions](python-functions.md) or [a first CUDA program](first-cuda-job.md).
3. Read [Calls](calls/index.md) and [Artifacts](artifacts/index.md) for progress, recovery, and
   files.
4. Use the [API reference](reference/index.md) for parameters and return values.
