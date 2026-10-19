# Images

An image supplies the filesystem, Python interpreter, libraries, and tools for a workload. The SDK
selects an image registered with the service. It does not install your local packages remotely.

Ask the operator or image publisher for an accessible image name or immutable digest. Use a name for
a maintained image. Use a digest to keep the same image across submissions.

```python
import gfaas

image = gfaas.Image.from_registry("research-python")
app = gfaas.App("research", image=image)
```

Registered images can be private to a service principal or available to other principals. An unknown
or inaccessible reference fails. It does not fall back to a worker-local filesystem.

Workers fetch missing filesystem objects into a local cache. The SDK does not receive storage
credentials. Cache placement and storage configuration belong to the operator.

- [Select a registered image](prepared.md).
- [Use an immutable image reference](remote.md).
- [Select native images and hardware](hardware.md).
- [Check and share a published image](qualification.md).
