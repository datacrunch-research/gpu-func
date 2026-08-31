# Remote images

A remote image stores its filesystem as content-addressed objects at an object origin. Its
descriptor identifies an image index. The index names the objects in the image.

## Select a catalog image

An operator catalog maps approved names to immutable remote-image descriptors. Select a catalog
image by name:

```python
import gfaas

image = gfaas.Image("pytorch-cu130")
```

The coordinator stores the resolved descriptor in the Environment. Thus, later catalog changes do
not change that Environment.

If the name is absent from the catalog, the service treats it as a prepared worker image.

## Select an explicit descriptor

You can also supply an immutable descriptor:

```python
import json
from pathlib import Path

import gfaas

descriptor = json.loads(Path("remote-image.json").read_text())
image = gfaas.Image.from_remote("cuda-nvcc-s3", descriptor)
```

Get this descriptor from the service operator. Do not edit its digest, storage key, object count, or
byte count. An explicit descriptor remains useful for a fixed historical image.

The descriptor has this shape:

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

## What the worker does

The worker reads image files through a lazy FUSE filesystem. A local file server downloads missing
objects into a cache.

Before it stores an object, the file server makes sure that its digest matches. Your function never
receives object-storage credentials.

Operators publish remote images and add approved descriptors to the catalog.
