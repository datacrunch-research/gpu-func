from types import SimpleNamespace

import pytest

from gfaas import App, Compatibility, Image
from gfaas.errors import GfaasError
from gfaas.kernel_target import compiler_compatibility, resolve_target


class Client:
    def __init__(self, image):
        self.image = image
        self.request = None

    def submit(self, **request):
        self.request = request
        return SimpleNamespace(
            call_id="call_probe",
            wait=lambda: {
                "target": {"backend": "cuda", "arch": 103, "warp_size": 32},
                "image": self.image,
            },
        )

    def get_call(self, identity):
        raise AssertionError("Completed Calls have no current stage; use the probe result")


def test_mixed_isa_probe_pins_image_and_hardware_without_gpu_requirement_on_compiler():
    image = Image.from_registry_variants("kernel", {"amd64": "x86", "arm64": "arm"})
    digest = "sha256:" + "a" * 64
    client = Client({"digest": digest, "architecture": "arm64"})
    app = App("test", image=image, client=client)
    requested = Compatibility(gpu_compute_capabilities=("10.0", "10.3"))
    # Nested scopes must retain the outer hardware requirements.
    with (
        app.function(gpu="any", compatibility=requested),
        app.function(env={"COMPILER_FLAG": "1"}) as scope,
    ):
        function, target, identity = resolve_target(scope.bind(lambda: None))
    assert client.request["kwargs"] == {"include_environment": True}
    assert client.request["image"] == image
    assert client.request["compatibility"] == requested
    assert function.image.name == digest
    assert function.compatibility == Compatibility("arm64", ("10.3",))
    assert compiler_compatibility(function) == Compatibility("arm64")
    assert function.env == {"COMPILER_FLAG": "1"}
    assert target["arch"] == 103 and identity == "call_probe"


@pytest.mark.parametrize("image", [{}, {"digest": "alias", "architecture": "arm64"}])
def test_probe_requires_immutable_image_evidence(image):
    app = App("test", image=Image("mutable"), client=Client(image))
    with pytest.raises(GfaasError, match="image digest and CPU architecture"):
        resolve_target(app.function(gpu="any").bind(lambda: None))


def test_probe_rejects_hardware_outside_requested_set():
    client = Client({"digest": "sha256:" + "a" * 64, "architecture": "arm64"})
    app = App("test", image=Image("kernel"), client=client)
    function = app.function(gpu="any", compatibility=Compatibility("amd64")).bind(lambda: None)
    with pytest.raises(GfaasError, match="requested hardware"):
        resolve_target(function)


@pytest.mark.parametrize("architecture", ["amd64", "arm64"])
def test_probe_retains_coordinator_image_identity_in_result(monkeypatch, architecture):
    from gfaas import triton_quick_runner as runner

    class Driver:
        def cuInit(self, flags):
            return 0

        def cuDeviceGet(self, device, index):
            device._obj.value = index
            return 0

        def cuDeviceComputeCapability(self, major, minor, device):
            major._obj.value, minor._obj.value = 10, 3
            return 0

    monkeypatch.setattr(runner.ctypes, "CDLL", lambda _: Driver())
    digest = "sha256:" + "b" * 64
    monkeypatch.setenv("GFAAS_BUILD_IMAGE_DIGEST", digest)
    monkeypatch.setenv("GFAAS_BUILD_IMAGE_ARCHITECTURE", architecture)
    target = {"backend": "cuda", "arch": 103, "warp_size": 32}
    assert runner.probe_target() == target
    assert runner.probe_target(include_environment=True) == {
        "target": target,
        "image": {"digest": digest, "architecture": architecture},
    }
