# Python wrapper functions

A Python function runs inside a prepared image. If a kernel experiment needs custom preparation,
Artifact inputs, or result processing, use this path.

For ordinary CUDA source, use `gfaas.compile_and_run()` first. It already uses the Python function
path and the `cuda-nvcc` image.

The SDK packages the Python file that defines the function. It does not package an entire project or
install local dependencies.

## Define a function

```python
from pathlib import Path

import gfaas

image = gfaas.Image.from_registry("cuda-nvcc")
app = gfaas.App("kernel-wrapper", image=image)

@app.function(gpu="any", gpu_type="gb300", timeout=600)
def compile_variant(source: str, flags: list[str]) -> dict:
    from gfaas.cuda_runner import run
    return run(source=source, nvcc_flags=flags)

if __name__ == "__main__":
    source = Path("kernel.cu").read_text()
    report = compile_variant.remote(source, ["-lineinfo"])
    print(report["stdout"])
```

`compile_variant.remote(...)` submits the Call and blocks until completion. It returns the
function's serialized return value.

This direct wrapper returns CUDA failures in the report. Inspect `phase` and `returncode`. The
`compile_and_run()` helper converts those failures to CUDA exceptions.

Define remote functions at module scope. Put imported third-party packages in the image. Put other
required data in Artifacts.

The worker imports the submitted module. Put submission code and other import-time side effects
under `if __name__ == "__main__":`.

Network access depends on the service policy. Put required software in the image. Pass required
files as Artifacts.

The SDK serializes arguments and results with Cloudpickle. Each serialized Artifact must fit the
configured Artifact size limit.

## Decorator options

| Option                    | Meaning                                                                         |
| ------------------------- | ------------------------------------------------------------------------------- |
| `image`                   | Image for this function. The `App` image is the default.                        |
| `gpu_count`               | Number of GPUs that one worker must lease. Use 0 for a CPU-only Call.           |
| `gpu`                     | Older GPU request syntax. Do not combine it with `gpu_count`.                   |
| `gpu_type`                | GPU pool, such as `gb300`. The default is the literal `any` pool.               |
| `timeout`                 | Active execution deadline in seconds. Default 300.                              |
| `capacity_wait`           | Maximum pre-assignment time. This time includes image and Artifact preparation. |
| `cpu_millicores`          | Requested CPU time. A value of 1000 equals one CPU.                             |
| `memory_bytes`            | Aggregate memory limit for the function and its helper processes.               |
| `ephemeral_storage_bytes` | Requested capacity for the disk-backed scratch directory.                       |
| `shared_memory_bytes`     | Size of the `/dev/shm` tmpfs. Its pages also count against `memory_bytes`.      |
| `max_log_bytes`           | Combined retained standard output and standard error.                           |
| `max_output_bytes`        | Combined bytes for declared output Artifacts.                                   |
| `env`                     | Environment variables for the function.                                         |

The `env` values become part of the durable Environment definition. The coordinator stores these
values. Do not put secrets in this mapping.

Use `gfaas.scratch_path()` for writable temporary files. The path exists only inside a remote
function. The worker reserves its requested capacity and checks logical file sizes during the run.
This check is not a filesystem quota. A short write can exceed the limit before the next check. The
worker then fails the Call and removes the scratch directory.

`gpu_type="any"` is not a wildcard across all pools. Ask the operator for a pool name. A CPU-only
request also uses the `any` pool, so that pool needs an available worker.

The capacity wait and execution timeout measure different stages. A Call uses `capacity_wait` before
worker acceptance. This period includes image preparation and Artifact staging. The function
`timeout` starts after a worker accepts the Call.

## Submit without blocking

`fn.spawn(...)` returns a `RemoteResult` without waiting. Put submission code in the guarded main
block:

```python
if __name__ == "__main__":
    source = Path("kernel.cu").read_text()
    job = compile_variant.spawn(source, ["-lineinfo"])
    result = job.wait(timeout_s=900)
```

Then read the result or the logs. See [Events and live logs](calls/events.md).

## Use the same client for several Calls

If an application submits several Calls, create one `Client`.

```python
from pathlib import Path

import gfaas

def compile_variant(source: str, flags: list[str]) -> dict:
    from gfaas.cuda_runner import run
    return run(source=source, nvcc_flags=flags)

if __name__ == "__main__":
    source = Path("kernel.cu").read_text()

    with gfaas.Client() as client:
        calls = [
            client.submit(
                image="cuda-nvcc",
                function=compile_variant,
                args=(source, [optimization]),
                gpu="any",
                gpu_type="gb300",
                timeout_s=600,
                app_name="kernel-batch",
            )
            for optimization in ("-O0", "-O3")
        ]
        results = [call.wait() for call in calls]

    print(results)
```

This example submits the calls one after another. The service can run them concurrently when the
selected pool has enough capacity.

## Call the local function

The decorated object remains callable. A normal call runs the function in the current Python
process.

Inside the guarded main block, use both forms as follows:

```python
if __name__ == "__main__":
    source = Path("kernel.cu").read_text()
    local_report = compile_variant(source, ["-O3"])
    remote_report = compile_variant.remote(source, ["-O3"])
```

The local call does not use the gfaas service. This example needs local CUDA tools and a local GPU.
