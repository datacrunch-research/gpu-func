# Prepared images

A prepared image is already present on the GPU machines. The machine lists it in
`/etc/gfaas/images.json`. Select it by name:

```python
image = gfaas.Image.from_registry("cuda-nvcc")
# or
image = gfaas.Image("cuda-nvcc")
```

The verified fleet path in this guide uses `cuda-nvcc`. Ask your operator which prepared images the
deployment offers.

## Build-spec images

`gfaas.Image.from_container(...)` creates a build specification. The current SDK rejects that
specification before submission because dynamic builds are not enabled.

Use a prepared image or a remote image. Ask an operator to publish a new image when your workload
needs more packages.
