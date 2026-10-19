# Core types

## App and Function

`App(name, *, image=None, client=None)` groups functions and owns their default image and client.
Pass an `Image` as `image`. Supply a `Client` when your application needs explicit connection
lifetime.

`App.function(...)` returns a decorator or function scope. Its accepted options are listed in
[Python functions](../python-functions.md#decorator-options). Defaults are `gpu_count=None`,
`gpu=None`, `gpu_type="any"`, and `timeout=300`. With neither GPU selector, an ordinary function
requests no GPU.

A decorated `Function` provides:

| Member                                      | Result                                                        |
| ------------------------------------------- | ------------------------------------------------------------- |
| `fn(*args, **kwargs)`                       | Local execution in the current Python process                 |
| `fn.remote(*args, **kwargs)`                | Remote execution and decoded return value                     |
| `fn.spawn(*args, **kwargs)`                 | A `RemoteResult` after submission                             |
| `fn.qualify(image_digest, *args, **kwargs)` | A `RemoteResult` for a Call that shares the digest on success |
| `name`                                      | Handler name                                                  |
| `source_file`                               | Local file that the SDK packages                              |

`qualify()` changes image visibility after success. Read
[check and share an image](../images/qualification.md) before using it as a smoke test. Use
`spawn()` for an owner-only test.

## Image

| Constructor                                  | Meaning                                                                            |
| -------------------------------------------- | ---------------------------------------------------------------------------------- |
| `Image(name)` or `Image.from_registry(name)` | Select a registered name or immutable digest                                       |
| `Image.from_registry_variants(name, images)` | Map `amd64` and/or `arm64` to registered references                                |
| `Image.from_remote(name, descriptor)`        | Select an existing registration through its exact descriptor                       |
| `Image.from_container(base_image, ...)`      | Construct a build specification; submission rejects it on the current service path |

`from_container` accepts `python_version="3.11"`, `apt_packages=()`, `pip_packages=()`, `env=None`,
`commands=()`, and `name=None`. These describe a specification; they do not perform an image build.
See [images](../images/index.md) for the supported publication and selection boundary.

## Compatibility

`Compatibility(cpu_architecture=None, gpu_compute_capabilities=())` restricts worker selection. CPU
architecture accepts `amd64` or `arm64`. GPU capabilities are exact alternatives such as
`("10.0", "10.3")`, not minimum versions. `request()` returns the corresponding API fields.

Image architecture and pool selection remain additional constraints. See
[hardware requirements](../images/hardware.md).

## ArtifactRef and scratch_path

`ArtifactRef(artifact_id)` refers to an existing accessible file or directory Artifact. Inside a
remote function, `path`, `open()`, `read_bytes()`, and `read_text()` access the staged data. It also
supports `os.fspath()`. File methods do not apply to a directory.

`scratch_path()` returns the remote function's writable scratch directory as a `Path`. It fails
outside a running remote function. Scratch files disappear with the run unless published as outputs.

See [Artifact inputs](../artifacts/ref.md) for nested references and attachment limits.

## Declared outputs

`ArtifactOutput(name, relative_path, kind="output", media_type="application/octet-stream", required=True, publish_on_failure=True, layout="blob")`
describes a terminal file output.

Use
`ArtifactOutput.directory(name, relative_path, *, kind="output", required=True, publish_on_failure=True)`
for a tree. The class sets its tree media type. `path` gives the writable remote destination. For
file outputs, `open()` creates parent directories. `request()` returns the API output declaration.

`ArtifactCheckpoint(name, relative_path, kind="output", maximum_versions=8)` describes directory
checkpoints. `publish(relative_directory)` marks a completed child directory and returns its
generation. The marker does not confirm remote durability. Check the Call's Artifact list before
relying on it.

See [outputs and checkpoints](../artifacts/outputs.md) for required files, failure publication, and
limits.

## ArtifactUploadProgress

Callbacks receive a frozen `ArtifactUploadProgress` record. Important fields are `path`,
`completed_bytes`, `total_bytes`, `completed_files`, `total_files`, `transferred_bytes`,
`reused_bytes`, `upload_id`, `phase`, and `artifact_id`.

The phase distinguishes accepted bytes from completed publication. A directory child's `published`
update does not establish that the directory itself is published. See
[upload progress](../artifacts/upload-download.md#progress-and-cancellation).

## CallStage and StageArtifactBinding

`CallStage(name, qualname, module=None, resources=None, outputs=(), artifacts=())` describes one
sequential callable. `StageArtifactBinding(from_stage, output, env)` connects a prior output to a
later stage. Their `request()` methods produce API declarations; `CallStage.request()` also needs
`default_module` as a keyword argument.

Read the [complete staged example](../calls/stages.md) for argument handling, resource inheritance,
and output transfer.

## call_status_summary

`call_status_summary(status)` returns a short string from a Call status dictionary. It describes
lifecycle, preparation, and placement where the server supplies those fields. It does not contact
the service. Obtain the dictionary with `RemoteResult.status()` or `Client.get_call()`.
