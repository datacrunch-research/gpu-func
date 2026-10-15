# Hardware requirements and native images

A Function can run on ARM or x86 workers in the same deployment. Each worker needs an image built
for its CPU architecture.

CPU architecture and GPU compute capability are separate requirements. An ARM image does not imply a
particular GPU model.

## Select native images

Publish one image for each CPU architecture that the Function supports. Use distinct names or
immutable digests for the images.

```python
import gfaas

image = gfaas.Image.from_registry_variants(
    "research",
    {
        "amd64": "research-x86",
        "arm64": "research-arm",
    },
)
app = gfaas.App("research", image=image)

@app.function(
    gpu_count=1,
    gpu_type="any",
    compatibility=gfaas.Compatibility(
        gpu_compute_capabilities=("10.0", "10.3"),
    ),
)
def run(value: int) -> int:
    return value + 1

if __name__ == "__main__":
    call = run.spawn(41)
    print(gfaas.call_status_summary(call.status()))
    print(call.wait())
```

The image label `research` names the Environment. The map values identify registered images. Each
image's registered architecture must match its map key.

The service resolves image names to immutable digests when it accepts the Call. Later changes to
those names do not change that Call.

The coordinator selects a matching worker and its native image. It does not emulate another CPU
architecture or translate binaries.

For one native image, keep using `Image.from_registry("research-arm")` or a string with
`Client.submit()`. The image's architecture still restricts placement.

## Declare hardware requirements

Use `gfaas.Compatibility` with `@app.function()`, `Client.submit()`, `compile_and_run()`, or
`spawn_triton_tuning()`.

```python
requirements = gfaas.Compatibility(
    cpu_architecture="arm64",
    gpu_compute_capabilities=("10.3",),
)
```

| Field                      | Meaning                                                                                  |
| -------------------------- | ---------------------------------------------------------------------------------------- |
| `cpu_architecture`         | Require `amd64` or `arm64`. Omit it to allow any architecture represented by the image.  |
| `gpu_compute_capabilities` | Accept the listed exact capabilities. An empty tuple adds no GPU capability requirement. |

Capabilities use `major.minor`, such as `10.0` or `10.3`. They are exact alternatives, not minimum
versions. The list can contain up to 32 distinct values.

Every GPU assigned to the Call must match. CPU-only stages retain the CPU requirement but do not
require a GPU capability.

The GPU pool remains a separate constraint. A named pool restricts placement to that pool. With
coordinator placement, `gpu_type="any"` permits matching workers across pools. The deployment must
enable `any` in its configured pool list.

The SDK checks advertised service support before uploading a request with explicit requirements or
image variants. It reports an error instead of silently removing requirements.

Hardware matching does not establish CUDA driver, framework, or library compatibility. Those are
properties of the workload and its images.

The CLI accepts the same requirements through `--cpu-architecture` and repeatable
`--gpu-compute-capability` options. See
[CLI hardware requirements](../cli.md#require-compatible-hardware) for examples and their
relationship to CUDA compiler flags.

## Keep stages and inputs compatible

All stages of one Call use the same selected CPU architecture. Placement retries retain it too.
Thus, an x86 compile stage cannot send a native binary to an ARM execution stage.

Before selecting between images, the coordinator checks reported hardware for later stages. Busy
hardware counts, but this check does not reserve it. Missing or stale reports can delay the first
stage.

Package source and inputs that work on every offered architecture. Put native Python extensions in
each native image. Do not bundle a local x86 shared library for an ARM worker.

GPU compilation targets remain your responsibility. Set CUDA flags or Triton's `target_arch` to a
target supported by the requested GPU capabilities.

## Inspect placement

Read `call.status()` or `Client.get_call(call_id)`. A waiting Call can report:

| `compatibility.state`  | Meaning                                                                                     |
| ---------------------- | ------------------------------------------------------------------------------------------- |
| `hardware_mismatch`    | Current reports contain no matching hardware in the requested pool.                         |
| `hardware_unknown`     | Required observations are missing, stale, or from workers that cannot enforce requirements. |
| `waiting_for_capacity` | Matching hardware exists, but placement is waiting for resources or worker availability.    |
| `assigned`             | A matching worker was selected; acceptance is pending.                                      |

The `required`, `observed`, and `image_architectures` fields explain unassigned placement decisions.
Calls retain their capacity-wait deadline. A timeout result retains the last observation under
`error.details.compatibility`.

After placement, `current_stage.image` contains the selected name, digest, and CPU architecture.
Inside the workload, these values are available as:

- `GFAAS_BUILD_IMAGE_NAME`
- `GFAAS_BUILD_IMAGE_DIGEST`
- `GFAAS_BUILD_IMAGE_ARCHITECTURE`

An image qualification Call runs only the variant matching its requested digest. It does not qualify
every variant in the Environment.
