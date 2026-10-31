# Build an image from a container specification

Describe the software your function needs with `Image.from_container`. The deployment's image build
service builds that specification and publishes it for your API key's principal. You need no Docker
build host, container runtime, or publication key.

```python
import gfaas

image = gfaas.Image.from_container(
    "nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04",
    python_version="3.12",
    pip_packages=["cloudpickle==3.1.2", "flashinfer-python[cu13]==0.7.0.post1"],
    name="flashinfer-cu13",
)
app = gfaas.App("exp", image=image)


@app.function(gpu_count=1, gpu_type="gb300")
def version() -> str:
    import flashinfer

    return flashinfer.__version__
```

The published name is `name` followed by six characters derived from the build fields:
`flashinfer-cu13-eazw5y`. Without `name`, it starts with `img`, as in `img-eazw5y`.

The suffix comes from the field values, not the source text. Reformatting the call keeps the name.
Changing any value, such as a package version, gives a new name, so a name never stands for two
different images. Two specifications that differ only in `name` share one build.

## Publish the image

Save the code as `exp.py` and ask the service to build its `image` variable:

```bash
vfunc image publish exp.py:image
```

The command copies the build log to stderr. When the image is ready, it prints the image name on
stdout. Omit `:image` when the file declares exactly one container image. Use `--no-wait` to return
after the service accepts the build, and `vfunc image status exp.py:image` to check it later.

The CLI does not import `exp.py`, so `flashinfer` does not need to be installed locally. It reads
module-level assignments of `Image.from_container(...)`. Their arguments must be literals or names
assigned to literals earlier in the module.

Publishing the same specification again returns the existing build. A failed build is retried.
`vfunc image list` shows your principal's builds, newest first, with their state and digest.

## Watch builds on the dashboard

The build service also has a dashboard of every build, with its specification and the end of its
log. It has no authentication, so it listens only on the build host. Forward it to your machine
over SSH, then open it:

```bash
ssh -N -L 8091:127.0.0.1:8091 BUILD_HOST &
vfunc image-dashboard
```

Ask your operator for `BUILD_HOST`. The command checks that the dashboard answers and opens
`http://127.0.0.1:8091/` in your browser. Over SSH without a display, or with `--no-browser`, it
prints the address instead. Set `GFAAS_IMAGE_DASHBOARD_URL` or pass `--url` for another address.

## Run with the image

After publication, the SDK selects the image by its name:

```bash
python -c 'import exp; print(exp.version.remote())'
```

`vfunc run exp.py` and `vfunc run exp.py:version` also use the image that the file declares, unless
you pass `--image`.

A Call fails at submission if the image is not yet published for your principal. The error names
the `vfunc image publish` command.

## What the build does

| Field            | Build step                                                                            |
| ---------------- | ------------------------------------------------------------------------------------- |
| `base_image`     | Starts from this image for the service's CPU architecture.                            |
| `apt_packages`   | Installs Debian or Ubuntu packages.                                                   |
| `pip_packages`   | Installs packages into the Python that workers run, adding Python 3 if it is missing. |
| `python_version` | With `pip_packages`, fails the build unless that Python has this version.             |
| `env`            | Sets image environment variables.                                                     |
| `commands`       | Runs each shell command in order.                                                     |

Functions need `cloudpickle` in the image, so list it in `pip_packages` when the base image lacks
it. Do not put secrets in `env` or `commands`: they become part of the image.

The service builds images for its own CPU architecture. Its operator decides which principals it
serves; a request from another principal fails with HTTP 403.

## Service location

The CLI and SDK send build requests and your API key to `$GFAAS_API_BASE/v1/image-builds`. An
operator can run the service elsewhere; set `GFAAS_IMAGE_BUILDS_URL` only to a service that your
operator names, because it receives your API key.

The SDK exposes the same operations as `Client.request_image_build(image)`,
`Client.get_image_build(name)`, and `Client.read_image_build_log(name, offset=0)`.
