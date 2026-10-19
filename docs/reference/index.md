# Python API reference

Import the public API as `gfaas`. The package name remains unchanged even though the command is
`vfunc`. These pages describe the client API; they do not require access to the service
implementation.

| Area                                                   | Reference                               |
| ------------------------------------------------------ | --------------------------------------- |
| Applications, functions, images, hardware, and outputs | [Core types](core.md)                   |
| HTTP client, submission, transfer, and Call handles    | [Client](client.md)                     |
| CUDA programs and explicit Triton candidate tuning     | [CUDA and staged tuning](cuda.md)       |
| Managed kernel constructors                            | [Kernel adapters](kernels.md)           |
| Tuning and measurement options                         | [Shared policies](../kernels/tuning.md) |
| Exceptions and recovery evidence                       | [Errors](errors.md)                     |

`GfaasError` and `SerializationError` are imported from `gfaas.errors`. Other names listed here are
available from the package root unless stated otherwise. Module-level helpers used internally by
runners are not a promise of a stable application API.

Most client methods perform synchronous HTTP requests. `spawn()` returns without waiting for remote
execution, but packaging, upload, and Call creation happen before it returns. It is not an asyncio
API.
