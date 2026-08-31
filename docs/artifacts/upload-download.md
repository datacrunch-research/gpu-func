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

The byte methods are for small values. Use the file methods for large data:

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

Use a progress callback for a large file or directory:

```python
import gfaas


def show_progress(item: gfaas.ArtifactUploadProgress) -> None:
    percent = 100 if item.total_bytes == 0 else item.completed_bytes * 100 // item.total_bytes
    print(
        f"{percent}% "
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

The client calls the callback after each completed part. A resumed upload reports completed parts as
`reused_bytes`. A new transfer reports its completed bytes as `transferred_bytes`.

For a directory upload, the byte and file counts cover the complete directory. The `path` field
identifies the current file.

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

If final tree creation fails, `ArtifactTreeUploadError.child_artifact_ids` contains the completed
child uploads. Your application can retain them for a retry or remove them.

The directory download verifies the manifest and every file. It builds a temporary read-only tree
and then renames that tree to the requested destination.

The destination must not exist. A failed download removes its temporary tree.

The metadata can also contain `filename` and `expires_at`.

The coordinator deduplicates uploads by principal and digest. If that principal uploads the same
bytes again, the original metadata remains.

## Upload options

| Option       | Meaning                                                                                         |
| ------------ | ----------------------------------------------------------------------------------------------- |
| `kind`       | One of `source`, `build_context`, `input`, `output`, `log`, `profile`, `diagnostic`, or `other` |
| `media_type` | MIME type of the bytes                                                                          |
| `filename`   | Optional file name                                                                              |
| `progress`   | Callback that receives `ArtifactUploadProgress` after each completed part                       |

The deployment keeps finite limits for inline bytes, part size, open sessions, unfinished bytes, and
session lifetime. It does not apply the inline byte limit to the completed file Artifact.

## Delete rules

`delete_artifact()` removes an Artifact that no durable resource still requires. The API rejects
deletion while an Environment, Function, Call input, published result, or nonterminal Call still
references the Artifact. The API also rejects deletion of a blob while a tree refers to it. Delete
the tree first when you no longer need either resource.
