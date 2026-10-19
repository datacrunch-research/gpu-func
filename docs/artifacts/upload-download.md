# Upload and download

Use the `Client` for direct Artifact operations.

```python
from gfaas.client import Client

payload = b"hello from an Artifact\n"

with Client() as client:
    artifact = client.upload_artifact(
        payload,
        kind="other",
        media_type="text/plain",
        filename="hello.txt",
    )
    downloaded, media_type = client.download_artifact(artifact["id"])
    metadata = client.get_artifact(artifact["id"])
    print(metadata["digest"], metadata["size_bytes"], metadata["media_type"])
    client.delete_artifact(artifact["id"])

assert downloaded == payload
```

Byte uploads larger than 1 MiB use multipart requests automatically. This includes serialized
function arguments. The byte methods still hold the supplied bytes in your process. Use the file
methods for large data:

```python
with Client() as client:
    artifact = client.upload_artifact_file(
        "dataset.bin",
        kind="input",
        media_type="application/octet-stream",
    )
    path, media_type = client.download_artifact_file(
        artifact["id"],
        "dataset-copy.bin",
    )
```

The file upload uses a durable session. The coordinator returns the required part size and records
each verified part. Create the session separately if your application must recover its identity:

```python
session = client.create_artifact_file_upload("dataset.bin", kind="input")
artifact = client.upload_artifact_file(
    "dataset.bin",
    kind="input",
    upload_id=session["id"],
)
```

Store `session["id"]` in durable application state. Supply it again after an interrupted transfer.

An interrupted byte or file upload raises `ArtifactUploadError` with its `upload_id`. Supply that ID
and the same content and metadata to resume the upload. The client reuses verified parts. A
completed session returns its original Artifact, including after a lost completion response. An
expired session requires a new upload.

```python
from gfaas import ArtifactUploadError

try:
    artifact = client.upload_artifact(payload, kind="input")
except ArtifactUploadError as error:
    # Save the ID if the application must resume after restart.
    artifact = client.upload_artifact(payload, kind="input", upload_id=error.upload_id)
```

`wait_for_result()` streams the result to a private file, verifies its size and digest, and decodes
from that file. The decoded Python object must still fit in client memory. The temporary file is
removed after decoding or failure.

## Upload concurrency

The client uploads file parts and directory children concurrently. The default limit is four.

Set `GFAAS_UPLOAD_CONCURRENCY` to an integer from 1 to 16. Use 1 for serial transfers. The CLI uses
the same environment variable. To set the limit in Python:

```python
from dataclasses import replace
from gfaas import Client, ClientConfig

config = replace(ClientConfig.from_env(), upload_concurrency=4)
with Client(config) as client:
    artifact = client.upload_artifact_file("dataset.bin")
```

One `Client` shares this limit across its automatic uploads. With a limit of N:

- At most N upload requests can run at once, including session creation and completion.
- At most N file or byte multipart uploads can hold active transfer slots.
- At most N source files remain open for upload or hashing.
- Active part buffers use at most N times the negotiated part size. The client accepts part sizes up
  to 64 MiB.
- File hashing can use another 2N MiB. Caller-owned byte inputs, HTTP overhead, and tree metadata
  are separate.

Workers acquire a request slot before they read a part. Waiting parts do not hold payload buffers.
Separate `Client` instances have separate limits.

The service still enforces its session and unfinished-byte quotas. The client reports quota
rejection and stops the failed operation. It does not automatically retry rejected requests or
create replacement sessions. Existing requests finish before the method returns an error. A
directory failure also waits for active child uploads.

## Progress and cancellation

Use a progress callback for a large file or directory:

```python
import gfaas


def show_progress(item: gfaas.ArtifactUploadProgress) -> None:
    percent = 100 if item.total_bytes == 0 else item.completed_bytes * 100 // item.total_bytes
    print(
        f"{item.phase}: {percent}% of bytes accepted "
        f"files={item.completed_files}/{item.total_files} "
        f"transferred={item.transferred_bytes} "
        f"reused={item.reused_bytes}"
    )


with gfaas.Client() as client:
    artifact = client.upload_artifact_file(
        "model.safetensors",
        kind="input",
        progress=show_progress,
    )
```

The client reports these phases:

| Phase          | Meaning                                                                              |
| -------------- | ------------------------------------------------------------------------------------ |
| `transferring` | The client sends parts or checks parts accepted by an earlier attempt.               |
| `publishing`   | All parts arrived. The client waits for the service to publish the Artifact.         |
| `checking`     | The client checks a saved publication or looks up the outcome after a lost response. |
| `published`    | The service confirmed publication. `artifact_id` identifies the Artifact.            |

**100% of bytes accepted does not mean publication succeeded.** The service can still be verifying
content or writing to storage. Wait for `published` or the upload method's successful return.
`upload_id` identifies the resumable file session. `artifact_id` is set only on a `published`
update.

The client calls the callback after each completed part and when the phase changes. Parts can finish
in any order. `completed_files` increases only after publication succeeds. A resumed upload reports
accepted parts as `reused_bytes`. New parts contribute to `transferred_bytes`. A completed session
reports all its bytes as reused.

For a directory upload, the byte and file counts cover the complete directory. The `path` field
identifies the file that produced the update, and `phase` describes that file. A child's `published`
update does not confirm publication of the directory. After all children publish, the client reports
`publishing` and then `published` for the directory's own path. The final update contains the tree
Artifact ID. This also applies to empty directories.

Directory callbacks are serialized but can use different threads. Keep callbacks short and avoid
callback exceptions. Do not start another upload on the same client from a callback.

### Publication timeouts and recovery

File and multipart byte uploads use a separate completion-response timeout. The default is 600
seconds. Set `GFAAS_UPLOAD_COMPLETION_TIMEOUT` or `ClientConfig.upload_completion_timeout_s` to
change it. Ordinary requests and completion connection/write timeouts still use
`GFAAS_REQUEST_TIMEOUT`, which defaults to 60 seconds. These are HTTP timeouts, not GPU execution
timeouts. A proxy can end a request earlier than the client timeout.

After a completion transport error or HTTP 5xx response, the client looks up the session once. If
the session contains an Artifact ID, the client fetches its metadata and returns success. This
recovers a publication whose response was lost. It does not repeat the completion POST
automatically.

If publication cannot be confirmed, `ArtifactUploadError` includes `upload_id` and
`phase="publishing"`. The message distinguishes a completion timeout from a rejected request. An
open session does not prove that publication stopped. A failed status lookup leaves the outcome
unknown. Retain the upload ID and resume the same session as shown above. The service's session
expiry still applies. Cancellation and transfer errors also retain the upload ID.

Directory child failures appear in `ArtifactTreeUploadError.failures` and `upload_ids`. The final
tree registration has no upload session; it uses the ordinary request timeout and does not use this
completion recovery check.

### JSON progress

To emit machine-readable progress:

```python
import json
from dataclasses import asdict


def json_progress(item):
    print(json.dumps({**asdict(item), "path": str(item.path)}), flush=True)
```

Supply a `threading.Event` as `cancel` to a byte, file, or directory upload. Set the event from
another thread to stop new parts and prevent completion. Requests already in progress finish or
reach their HTTP timeout. Cancellation cannot undo a completion request that already reached the
service.

A cancelled multipart transfer raises `ArtifactUploadError` with its session ID. A directory failure
raises `ArtifactTreeUploadError`. Clear the event before resuming an interrupted file.

The file download writes to a temporary path first. It verifies the size and digest before it
renames the file to the requested path.

## Directory trees

Use a tree Artifact when your input has several files or nested directories:

```python
with Client() as client:
    tree = client.upload_artifact_directory("training-data", kind="input")
    client.download_artifact_directory(tree["id"], "training-data-copy")
```

The upload accepts regular files and directories. It rejects symbolic links, hard links, and special
files. It preserves only a read-only mode and an optional executable bit.

Each file becomes a blob Artifact. The tree Artifact contains a canonical manifest that refers to
those blobs. This structure permits file-level deduplication across trees.

The upload response includes `child_artifact_ids` for cleanup bookkeeping. Deleting the tree does
not delete these blob Artifacts. A blob can belong to another tree or another application resource.

If a child upload or final tree creation fails, `ArtifactTreeUploadError.child_artifact_ids`
contains the completed child uploads. This list includes children that finish while another child
fails. `upload_ids` maps failed relative file paths to their resumable session IDs. `failures` maps
failed relative file paths to error descriptions. Your application can resume these files
individually or remove completed child Artifacts. The directory method does not automatically resume
an entire tree.

The client registers the tree only after all children succeed. It sorts manifest entries by path, so
concurrent completion does not change tree identity.

The directory download verifies the manifest and every file. It builds a temporary read-only tree
and then renames that tree to the requested destination.

The destination must not exist. A failed download removes its temporary tree.

The metadata can also contain `filename` and `expires_at`.

The coordinator can reuse an Artifact when its owner, content, metadata, and retention policy match.
Different policies can produce different Artifact IDs that share stored bytes.

## Upload options

| Option        | Meaning                                                                                         |
| ------------- | ----------------------------------------------------------------------------------------------- |
| `kind`        | One of `source`, `build_context`, `input`, `output`, `log`, `profile`, `diagnostic`, or `other` |
| `media_type`  | MIME type of the bytes                                                                          |
| `ttl_seconds` | Optional positive lifetime in seconds; omit to keep the Artifact                                |
| `filename`    | Optional file name                                                                              |
| `progress`    | File or directory callback that receives `ArtifactUploadProgress` after parts and publication   |
| `cancel`      | Optional `threading.Event` that stops new transfer work                                         |

The deployment keeps finite limits for inline bytes, part size, open sessions, unfinished bytes, and
session lifetime. It does not apply the inline byte limit to the completed file Artifact.

## Keep an Artifact or set its expiry

Uploads keep their data by default. To request expiry, pass `ttl_seconds` to a byte, file,
directory, or session-creation method:

```python
artifact = client.upload_artifact_file("scratch.bin", ttl_seconds=86400)

# Keep a useful result after the experiment.
client.set_artifact_retention(artifact["id"])

# Or allow collection one day from now.
client.set_artifact_retention(artifact["id"], ttl_seconds=86400)
```

The server limits the requested duration. Metadata reports `retention.mode` and, for an expiring
Artifact, `expires_at`. Updates start a new duration when the server accepts them. For a resumable
upload, the duration starts at session creation. Supply the same `ttl_seconds` when resuming; a
retry does not extend it. Allow enough time for upload and use.

Directory uploads apply the requested policy to the tree and each uploaded file. Each duration
starts when that file's session or the tree is created. Updating a tree later changes only its own
policy. A kept tree protects its children while it exists, even if a child's expiry passes.

Expiry prevents new references. Existing Calls, Functions, Environments, and trees protect stored
bytes and downloads after expiry. The collector removes an expired Artifact only after those
references are released. A finished Call still holds references until it is retired. Expiry does not
promise immediate deletion.

`set_artifact_retention(id)` cancels automatic cleanup while the Artifact's metadata still exists.
It cannot restore data after deletion has started. Explicit keep does not prevent an owner from
using `delete_artifact()` once the Artifact is no longer in use.

## Delete rules

`delete_artifact()` removes an Artifact that no durable resource still requires. The API rejects
deletion while an Environment, Function, Call input, published result, or nonterminal Call still
references the Artifact. The API also rejects deletion of a blob while a tree refers to it. Delete
the tree first when you no longer need either resource.

A finished Call can still hold input and output references. Use
[resource retirement](../reference/client.md#resource-retirement) to release them while retaining
history. Retiring a Call does not delete its independently retained outputs. Functions and
Environments can hold additional source references.

## Find Artifacts for your account

`list_artifacts(after=None, limit=100)` returns one page of metadata. Pass its `next_cursor` as
`after` for the next page. `iter_artifacts()` follows those pages for you:

```python
from gfaas import Client

with Client() as client:
    for artifact in client.iter_artifacts():
        print(artifact["id"], artifact.get("filename"), artifact["size_bytes"])
```

The list is scoped to the principal identified by your API key. Keys with the same principal see the
same owner-scoped Artifacts. This is not a fleet-wide operator listing.

Record the IDs of data your application creates. Before cleanup, check which resources still use
those IDs. Do not delete every listed Artifact merely because one experiment is complete. Deletion
can fail with `artifact_in_use` when a durable resource still references the data.

For a tree you own and no longer need, delete the tree before its unneeded child blobs. Retain the
upload response's `child_artifact_ids` for that bookkeeping. Some children can still belong to other
trees.
