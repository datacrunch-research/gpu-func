# Kernel caches and recovery

Managed kernel operations are synchronous Python calls. One launch can create several durable
service Calls. There is no single aggregate Call ID that controls the entire operation.

## Image and hardware identity

Each invocation probes its selected GPU before it uses compiled artifacts. The probe returns the
image digest, CPU architecture, and CUDA target. Later phases use the same image and CPU
architecture. GPU phases require the probed compute capability; CPU compiler Calls do not reserve
GPUs.

The probe runs even when the wrapper has a cached specialization. A changed image alias, target, or
relevant environment creates a separate cache entry. Reusing the cache avoids compilation and
tuning, not the probe or the execution Call.

Hardware matching does not establish compatibility with every CUDA driver or library behavior. Use a
controlled image and retain the target details for experiments.

## Cache lifetime

Compiled artifacts and tuning results are cached by the wrapper in client memory. Reuse requires the
same client/account. Recreating the wrapper or restarting the Python process does not restore that
in-memory tuning state automatically.

The specialization identity includes source, configurations, argument metadata, image, environment,
and relevant compiler and policy inputs. Tensor contents do not identify the specialization. A new
launch with the same layout and new values can execute the cached winner.

`tuning_results` maps specialization digests to immutable reports. It has no `last_tuning` shortcut.
A fresh `benchmark()` returns a new measurement without replacing the saved tuning report. Keep the
reports you need for comparisons; do not treat the wrapper cache as an experiment database.

## Resources and deadlines

The active function scope supplies resource limits. Compilation concurrency and GPU batching can
create several simultaneous Calls. Their combined reservations can exceed the limits of one Call.
The adapter chapters describe compiler defaults; set budgets appropriate for your service.

`timeout` and `capacity_wait` apply to individual submitted Calls. They are not a deadline for the
whole compile/tune/execute sequence. Default replication requires three GPUs on one worker.

## Diagnose failures

Read the exception's report and Call IDs before retrying. Compiler and measurement exceptions can
retain successful partial work and failed-call evidence. The
[error reference](../reference/errors.md) lists the available fields. Inspect retained logs with
`vfunc call logs CALL_ID`.

An interrupted client does not establish that every submitted Call stopped. Use retained Call IDs to
inspect and, where appropriate, cancel outstanding work. These wrappers do not expose an aggregate
cancellation handle or a durable resume operation for an interrupted tuning session.

A poisoned CUDA context fails its Call. The wrapper does not recover that context and continue the
same measurement. Repeating a whole operation can create new Calls and consume additional capacity.

`CUDAKernel.compile()` is an explicit compilation-only interface. It returns reports and artifact
references for callers that manage later execution themselves. It still probes a GPU target before
CPU compilation. It is not a generic compile-only method on every wrapper.
