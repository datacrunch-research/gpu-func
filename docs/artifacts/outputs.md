# Output Artifacts, logs, and retention

## Declare output files

A function can publish files in addition to its Python return value. Declare each file when you
define the function.

```python
import json

import gfaas

app = gfaas.App("kernel-profiler", image=gfaas.Image("pytorch-cu130"))
profile = gfaas.ArtifactOutput(
    "kernel-profile",
    "profiles/kernel.json",
    kind="profile",
    media_type="application/json",
)

@app.function(gpu_count=1, gpu_type="gb300", outputs=(profile,))
def profile_kernel() -> dict[str, float]:
    measurements = {"duration_ms": 12.4}
    with profile.open("w", encoding="utf-8") as file:
        json.dump(measurements, file)
    return measurements

if __name__ == "__main__":
    job = profile_kernel.spawn()
    print(job.wait())
    for publication in job.artifacts()["items"]:
        print(publication["name"], publication["artifact"]["id"])
```

The service sets `GFAAS_OUTPUT_ROOT` inside the function. `ArtifactOutput.path` resolves below this
directory. `ArtifactOutput.open()` also creates missing parent directories.

The worker accepts regular files only. It rejects symbolic links and paths outside the output root.
The worker also applies the resolved `max_output_bytes` limit to all declared files together. Set
this request on the function when its output needs a larger limit:

```python
@app.function(
    gpu_count=1,
    gpu_type="gb300",
    outputs=(profile,),
    max_output_bytes=2 * 1024**3,
)
def profile_large_kernel() -> None:
    ...
```

The worker rejects a request above its deployment-policy ceiling.

An output is required by default. A successful function fails if it does not create a required file.
Set `required=False` for an optional file.

The worker publishes declared files after a failure by default. This behavior can preserve a
diagnostic file from an unsuccessful run. Set `publish_on_failure=False` if a partial file is not
useful.

The subscriber streams each file from the worker to the coordinator. The function does not receive
coordinator or object-storage credentials.

## Choose how long to keep an output

`ArtifactOutput`, `ArtifactOutput.directory`, and `ArtifactCheckpoint` accept the same retention
options:

```python
temporary = gfaas.ArtifactOutput("scratch", "scratch.bin", retention="call")
kept = gfaas.ArtifactOutput.directory("adapter", "adapter", retention="keep")
expiring = gfaas.ArtifactCheckpoint("snapshots", "snapshots", ttl_seconds=86400)
```

| Option             | Lifetime                                                              |
| ------------------ | --------------------------------------------------------------------- |
| `retention="keep"` | Retain until the owner explicitly deletes the Artifact.               |
| `retention="call"` | Collect after the Call finishes and other consumers release the data. |
| `ttl_seconds=N`    | Allow collection after N seconds, once references release the data.   |

You can also specify `retention="expire"` with `ttl_seconds`. Keep and Call scope reject a duration.
The server limits the expiry duration. Expiry starts at each Artifact's creation, including each
checkpoint version. Result replay does not extend it.

Without either option, final outputs and checkpoints stay kept. Intermediate stage outputs retain
their existing temporary default. An explicit Keep policy preserves an intermediate output too.
These options do not change the Python return value, stdout, stderr, or unpublished scratch files.

Call-scoped outputs stay protected during all stages of the Call. Success, failure, cancellation,
and timeout end that protection. Other consumers and kept directories can still prevent collection.
Temporary outputs can disappear before you download them after completion. Keep checkpoints that you
need for recovery in a later Call.

A directory declaration covers its root and files. Existing explicit Artifact policies remain in
effect. Shared data with an older Keep policy can remain kept. You can preserve a published
temporary Artifact before collection:

```python
client.set_artifact_retention(artifact_id)
```

Keeping a directory root protects its children. Collection removes its live Call listing, but the
Call result and events retain the historical ID. That ID can refer to deleted bytes. See
[Artifact retention](upload-download.md#keep-an-artifact-or-set-its-expiry) for expiry, references,
and manual cleanup.

## Publish a directory

Use a directory output for model adapters, profiler reports, or another file set. The service keeps
the relative file layout.

```python
import gfaas

adapter = gfaas.ArtifactOutput.directory("adapter", "adapter")

@app.function(gpu_count=1, gpu_type="gb300", outputs=(adapter,))
def train_adapter() -> str:
    adapter.path.mkdir(parents=True)
    (adapter.path / "adapter_config.json").write_text("{}\n")
    (adapter.path / "adapter_model.bin").write_bytes(b"model data")
    return "complete"
```

The worker publishes the directory as a tree Artifact after the function stops. It rejects links,
special files, invalid paths, and trees above the deployment limits.

## Publish checkpoints during a Call

Use `ArtifactCheckpoint` when a long Call must publish recovery state before it ends. Each version
is an immutable directory.

```python
import gfaas

checkpoint = gfaas.ArtifactCheckpoint(
    "training-checkpoint",
    "checkpoints",
    maximum_versions=8,
)

@app.function(gpu_count=1, gpu_type="gb300", outputs=(checkpoint,))
def train() -> str:
    version = checkpoint.path / "step-001000"
    version.mkdir(parents=True)
    (version / "adapter_model.bin").write_bytes(b"checkpoint data")
    checkpoint.publish("step-001000")
    return "complete"
```

Finish all writes before you call `publish()`. Do not change that version directory after the call.
The method creates a local atomic marker. It does not wait for remote storage.

The worker checks markers during execution and once after the function stops. A published version
appears as `training-checkpoint.00000001` in `job.artifacts()`.

Treat a checkpoint as durable after it appears in the Call Artifact list. You can inspect this list
while the Call runs. The event stream also reports each Artifact publication.

```python
versions = [
    item
    for item in job.artifacts()["items"]
    if item.get("role") == "output"
    and str(item.get("name", "")).startswith("training-checkpoint.")
]
latest = max(versions, key=lambda item: item["name"])
resume_from = gfaas.ArtifactRef(latest["artifact"]["id"])
```

Pass `resume_from` to a later function Call. The function receives the checkpoint as a read-only
directory at `resume_from.path`.

Checkpoint generations must be contiguous. A declaration accepts at most 64 versions. The worker
counts all checkpoint file bytes and terminal output file bytes against one `max_output_bytes`
limit.

## List published Artifacts

The Call Artifact endpoint lists the Python result and all named output files and trees.

```python
page = job.artifacts()
for publication in page["items"]:
    artifact = publication["artifact"]
    print(publication["role"], publication.get("name"), artifact["id"])
```

`Client.submit()` uses Artifacts for the source, input, result, and named outputs. The public result
format supports an inline value or an Artifact reference.

Each publication includes the role, optional name, Attempt identity, and publication time. Its
nested `artifact` object includes the media type, digest, size, creation time, and optional
expiration time. Current SDK uploads do not set an expiration time.

The public endpoint can paginate this list. The current `Client.list_call_artifacts()` method does
not expose the pagination controls.

## Logs

The worker writes the function's stdout and stderr to the Call's durable event stream. The worker
applies one finite combined byte budget to both streams. The worker discards output beyond the
budget. The stream then records a `retention.truncated` event.

`job.logs()` returns the retained text and the truncation flag:

```python
logs = job.logs()
print("stdout:", logs["stdout"])
print("stderr:", logs["stderr"])
print("truncated:", logs["truncated"])
```

The `truncated` flag is true when the workload exceeded the combined byte budget.

## Built-in profiles

A CUDA job can request an Nsight Compute profile. The Python result carries the report in `ncu_csv`.
See [Examples](../examples.md).
