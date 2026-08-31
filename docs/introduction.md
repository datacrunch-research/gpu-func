# Introduction

gfaas runs Python and CUDA workloads on a fleet of GPU machines. You submit each workload through
one HTTPS API.

The service gives application developers one API for several GPU workers. It handles job delivery,
pool routing, execution, logs, results, and file transfer.

You do not need SSH access to a GPU machine. You do not need to select a specific worker or operate
the job queue.

## What gfaas is for

The current primary use is CUDA kernel development. A developer sends CUDA source to a real GPU,
reads the compiler and runtime results, and then submits the next revision.

The SDK includes these workload paths:

- CUDA compilation and kernel experiments.
- GPU profiling with Nsight Compute.
- Repeated tests against a selected GPU pool.
- Python wrapper functions around a kernel experiment.
- Read-only Artifact inputs for test data.

A workload can use a prepared container image with CUDA, Python packages, and other tools. A
workload can also use a content-addressed image from object storage.

gfaas does not provide models or machine-learning frameworks by itself. Those components need an
operator-qualified image.

The service operator can qualify CUDA compilation, execution, retained logs, Artifact input, and
the configured image path.

## A typical kernel-development workflow

Suppose that you need to develop and measure a new CUDA kernel. The operator provides a prepared
image with `nvcc` and the required CUDA tools.

You then complete these steps:

1. Write a self-contained CUDA program.
2. Submit the source with `compile_and_run()` or `spawn()`.
3. Request one GPU from the target pool.
4. Read compiler diagnostics, program output, and execution timings.
5. If you need hardware counters, add an Nsight Compute profile.
6. Change the source and submit another Call.

Each Call runs in a new container on a worker in the requested pool. You can repeat this cycle
without SSH access or a persistent shell on the GPU host.

## Why use it

Direct GPU access creates repeated operational work. Each user needs machine access, an environment,
input transfer, process supervision, and result collection.

gfaas puts those tasks behind one API. The API also gives each Call a durable identity. A client can
disconnect and return later without losing that identity.

The service provides these behaviors:

- The caller requests GPU capacity instead of a named machine.
- The service records the Call state and its Attempts.
- The SDK transfers function code, arguments, results, and other Artifacts.
- The event stream retains lifecycle events, standard output, and standard error within configured
  limits.
- The same idempotency key and request return the same Call.
- Prepared images select an operator-installed container filesystem by name.
- Remote images fetch filesystem objects through a worker cache when a Call reads them.

## Workloads outside the current scope

gfaas is not a general container platform. It does not manage long-lived web services, interactive
shells, or distributed training jobs across many hosts.

The isolation policy depends on the service deployment. Ask the service operator before you submit
code from an untrusted source.

If you need an interactive debugger or full host control, use direct machine access. If a workload
coordinates dependent tasks, use a workflow system.

If you only need durable file storage, use an object store directly. If you need a continuous
network service, use a container orchestrator.

## Two ways to submit work

The raw CUDA interface submits kernels, compiler experiments, and profiling runs. The SDK sends the
CUDA source to a prepared image that contains `nvcc`.

```python
import gfaas

report = gfaas.compile_and_run(
    source,
    gpu="any",
    gpu_type="gb300",
)
print(report["stdout"])
```

The Python function interface supports helper code around a kernel experiment. The raw CUDA helper
itself uses this path. Custom library workloads need an operator-qualified image.

Ask the service operator for the available image names and GPU pool names.

## The execution model

Each submission creates a **Call**. A Call is an asynchronous request to run one **Function** inside
one **Environment**.

The service uses four public resource types:

| Resource    | Meaning                                           |
| ----------- | ------------------------------------------------- |
| Environment | One image selection and its environment variables |
| Function    | One executable workload inside an Environment     |
| Call        | One invocation of a Function                      |
| Artifact    | An immutable blob or directory tree               |

A Call can produce Attempts, Events, a result, and output Artifacts. The SDK hides most resource
creation during normal use.

For a Python function, the SDK does this work:

1. It packages the source file.
2. It uploads the source and arguments as Artifacts.
3. It creates or reuses the Environment and Function resources.
4. It creates a Call.
5. It waits for the result or returns a `RemoteResult` handle.

The service records the Call and routes it to the requested GPU pool.

The worker starts a namespaced container with configured resource limits. It stages the required
Artifacts and returns events, logs, and the result.

## Synchronous and asynchronous use

If the caller can wait, use `.remote()` on a decorated Function named `fn`:

```python
result = fn.remote(...)
```

If the caller needs the Call identity immediately, use `.spawn()`:

```python
call = fn.spawn(...)
print(call.call_id)
result = call.wait()
```

The asynchronous handle also provides status, cancellation, events, and logs.

## Scope of this guide

This guide is for people who write and submit GPU workloads. It covers the Python SDK, Images,
Artifacts, Calls, errors, and deployment-specific limits.

## Next steps

1. [Install and authenticate](install.md).
2. [Run the first CUDA job](first-cuda-job.md).
3. [Use a Python wrapper function](python-functions.md).
