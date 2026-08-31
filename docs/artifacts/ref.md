# Pass an ArtifactRef into a function

If a function must read an existing Artifact, use an `ArtifactRef`. The SDK adds the Artifact to the
Call and preserves the reference in the arguments.

## Complete example

This example uploads a text file, reads it in a function, and removes it after the Call reaches a
terminal state.

The example uses the verified `cuda-nvcc` prepared image.

```python
import os

import gfaas

def count_lines(document: gfaas.ArtifactRef) -> int:
    text = document.read_text()
    return len(text.splitlines())

if __name__ == "__main__":
    gpu_type = os.environ["GFAAS_GPU_TYPE"]

    with gfaas.Client() as client:
        uploaded = client.upload_artifact(
            b"first line\nsecond line\n",
            kind="input",
            media_type="text/plain",
            filename="input.txt",
        )
        reference = gfaas.ArtifactRef(uploaded["id"])

        try:
            call = client.submit(
                image="cuda-nvcc",
                function=count_lines,
                args=(reference,),
                gpu="any",
                gpu_type=gpu_type,
                app_name="artifact-example",
            )
            print("call:", call.call_id)
            print(call.wait())
        finally:
            client.delete_artifact(uploaded["id"])
```

The function receives a local read-only file.

CAUTION: Wait for the Call to reach a terminal state before you remove the Artifact. The API rejects
removal while a nonterminal Call uses it.

## Nested references

You can put `ArtifactRef` values in these containers:

- Lists and tuples.
- Sets and frozen sets.
- Dictionary keys and values.
- Dataclass fields.

The SDK searches these containers before submission. It attaches each unique Artifact once.

```python
from dataclasses import dataclass

import gfaas

@dataclass
class Inputs:
    weights: gfaas.ArtifactRef
    labels: list[gfaas.ArtifactRef]

inputs = Inputs(
    weights=gfaas.ArtifactRef(weights_id),
    labels=[gfaas.ArtifactRef(labels_id)],
)

call = consume.spawn(inputs)
```

A Call can reference at most 32 unique Artifacts. The SDK also stops its search after 10,000
inspected values.

## Read methods

Inside a running function, `ArtifactRef` provides these members:

| Member                 | Meaning                                          |
| ---------------------- | ------------------------------------------------ |
| `path`                 | The staged file or directory path                |
| `os.fspath(reference)` | The staged path as a string                      |
| `open()`               | A standard Python file object                    |
| `read_bytes()`         | The complete file as bytes                       |
| `read_text()`          | The complete file as text, with UTF-8 by default |

For a tree Artifact, `path` is a read-only directory. Use normal `pathlib` operations to read its
files. The file-only methods do not apply to a directory.

The coordinator authorizes each reference for the Call. The subscriber transfers the Artifact
through the coordinator API.

The worker makes sure that the digest matches and stages the file read-only. The function never
receives coordinator or object-storage credentials.
