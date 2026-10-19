# Sequential stages in one Call

A staged Call runs several callables in sequence. Each stage can use different resources, and a
later stage can read a declared output from an earlier stage. This supports CPU preparation followed
by GPU execution without holding a GPU during preparation.

Use `Client.submit(stages=...)` with `CallStage` declarations. `StageArtifactBinding` connects an
earlier output to a later stage. This is a sequential pipeline, not a distributed task graph.

## Complete example

Save this as one Python module. Select an image containing compatible Python and PyTorch. The CPU
stage writes a value; the GPU stage reads it and performs a small tensor operation.

```python
import os

import gfaas

prepared = gfaas.ArtifactOutput("prepared", "prepared.txt", media_type="text/plain")


def prepare(value: int) -> None:
    with prepared.open("w", encoding="utf-8") as output:
        output.write(str(value * 2))


def execute(value: int) -> int:
    import torch

    data = gfaas.ArtifactRef(os.environ["PREPARED_ARTIFACT_ID"])
    tensor = torch.tensor(int(data.read_text()), device="cuda")
    return int((tensor + 1).item())


if __name__ == "__main__":
    with gfaas.Client() as client:
        call = client.submit(
            image=os.environ["STAGED_IMAGE"],
            function=prepare,
            args=(20,),
            gpu_count=1,
            gpu_type=os.environ["GFAAS_GPU_TYPE"],
            stages=(
                gfaas.CallStage(
                    name="prepare",
                    qualname="prepare",
                    resources={"gpu": {"count": 0}},
                    outputs=(prepared,),
                ),
                gfaas.CallStage(
                    name="execute",
                    qualname="execute",
                    artifacts=(gfaas.StageArtifactBinding(
                        from_stage="prepare",
                        output="prepared",
                        env="PREPARED_ARTIFACT_ID",
                    ),),
                ),
            ),
        )
        print(call.call_id)
        print(call.wait())  # 41
```

Both functions accept the original Call arguments. A stage's Python return value is not passed as
the next stage's arguments. Use declared Artifacts for that handoff. The final stage supplies the
Call's Python result.

The binding exposes the Artifact ID through the named environment variable. `ArtifactRef` resolves
it to the staged read-only file or directory. Finish writing the output before its stage ends.

## Stage fields

| Field       | Meaning                                                                    |
| ----------- | -------------------------------------------------------------------------- |
| `name`      | Unique stage name used by later bindings                                   |
| `qualname`  | Callable name within the module                                            |
| `module`    | Optional module override; defaults to the submitted source module          |
| `resources` | Resource overrides, using API field names such as `gpu` and `memory_bytes` |
| `outputs`   | Stage output declarations                                                  |
| `artifacts` | Bindings to earlier stage outputs                                          |

A binding contains `from_stage`, `output`, and `env`. The service validates the stage graph and
resource requests. All stages use the selected Environment and CPU architecture. A stage can run on
a different worker, so it cannot rely on an earlier stage's local files or process state.

## Inspect and recover

`get_call()` reports the active stage while the Call runs. Events and logs retain stage progress.
The service publishes the handoff Artifact before the next stage can consume it.

A failed stage fails the Call. The service does not rerun successful preparation merely because a
later stage waits for placement. Internal handoff Artifacts follow the service's transient retention
rules. Declare user-visible outputs separately when you need lasting application data.

The CUDA helpers and the [explicit Triton tuning helper](../triton-tuning.md) use this mechanism.
Managed kernel wrappers use several Calls and have a different
[recovery model](../kernels/lifecycle.md).
