# Select a registered image

Use a name supplied by your operator or image publisher:

```python
import gfaas

image = gfaas.Image.from_registry("research-python")
# This form selects the same registered image.
image = gfaas.Image("research-python")
```

Names such as `cuda-nvcc` and `pytorch-cu130` in examples are deployment choices, not bundled
images. The selected image must exist in your service and contain the packages your function
imports.

The coordinator checks your access to the image. The image's CPU architecture also restricts
placement. A GPU-pool name does not select an image automatically.

A registered name can move to a newer image. Each accepted Call retains its resolved image, but a
later Call can resolve the same name differently. Use an immutable digest for repeatable
submissions.

## Additional software

The SDK does not build images during submission. `Image.from_container(...)` constructs a build
specification, but `Client.submit()` rejects it because dynamic builds are not enabled on this path.

Ask an operator or publisher to provide an image with the required software. Then use its registered
name or digest. For mixed ARM and x86 workers, supply [native image variants](hardware.md).
