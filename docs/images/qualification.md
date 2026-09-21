# Qualify a published image

A publisher can run a normal GPU Function to check an immutable image. The image stays private
until the qualification Call succeeds. The publisher can use it before qualification.

Configure the Function with the publisher-scoped image name. Pass the digest returned by image
publication to `Function.qualify`:

```python
import gfaas

digest = "sha256:<64 lowercase hex characters>"
app = gfaas.App("image-check", image=gfaas.Image("candidate"))


@app.function(gpu_type="gb300", timeout=300)
def check_image() -> dict[str, str]:
    import torch

    assert torch.cuda.is_available()
    return {"device": torch.cuda.get_device_name(0)}


if __name__ == "__main__":
    qualification = check_image.qualify(digest)
    print(f"qualification call: {qualification.call_id}")
    for event in qualification.iter_events():
        print(event)
    print(qualification.wait())
```

The client pins the Function's Environment to `digest` and submits the qualification marker with
the Call. The returned `RemoteResult` supports the normal event, log, cancellation, and result
methods.

A successful Call makes the immutable digest available to other service principals. It does not
make the publisher's movable image name public. A failed, cancelled, or timed-out Call leaves the
digest private. It also leaves earlier qualified digests available.

The coordinator does not add object-store, deployment, or repository credentials to the workload.
The Function result is the publisher's functional check. It is not an operator security approval.
