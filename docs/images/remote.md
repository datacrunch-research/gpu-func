# Use an immutable image reference

A published image has a SHA-256 digest. Use that digest to select the same image in later Calls:

```python
import gfaas

# Replace this placeholder with the digest from your image publisher.
image = gfaas.Image.from_registry("sha256:<64 lowercase hex characters>")
```

The image must be registered and accessible to your API key's principal. Knowing a digest does not
grant access to a private image.

The coordinator resolves names when it accepts each Call. An Environment that contains a name does
not freeze that name forever. The accepted Call retains its resolved image through placement
retries.

An image digest identifies software content. It does not establish compatibility with every GPU,
CUDA driver, or input. Declare [hardware requirements](hardware.md) where applicable.

## Explicit descriptors

`Image.from_remote(name, descriptor)` also accepts a remote-image descriptor. This is an advanced
interface for applications that already retain publication metadata.

```python
import json
from pathlib import Path

import gfaas

descriptor = json.loads(Path("remote-image.json").read_text())
image = gfaas.Image.from_remote("research-image", descriptor)
```

The descriptor must match an accessible registered image. It cannot authorize an arbitrary storage
location or bypass publication. Obtain it from the publisher rather than constructing it yourself.

Its structure is:

```json
{
  "schema": "fast-container-remote-image/v1",
  "image_digest": "sha256:...",
  "index": {
    "storage_key": "sha256/ab/...",
    "sha256": "ab...",
    "size_bytes": 123
  },
  "cas_object_count": 123,
  "cas_total_bytes": 123456789
}
```

For most applications, a registered digest is sufficient. The service handles object transfer and
checks content hashes. Your function does not receive object-storage credentials.
