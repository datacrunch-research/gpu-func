"""Check hardware declarations at the public SDK and HTTP boundaries."""

import json

import httpx
import pytest

import gfaas
from gfaas.errors import GfaasError

CAPABILITIES = {
    "environment_compatibility": {
        "cpu_architectures": ["amd64", "arm64"],
        "image_variants": True,
        "gpu_compute_capabilities": "exact",
    }
}


@pytest.mark.parametrize("qualify", [False, True])
@pytest.mark.parametrize("architecture", [None, "arm64"])
def test_decorated_call_sends_native_images_and_requirements(qualify, architecture):
    requests = {}
    digest = "sha256:" + "a" * 64
    variants = {"amd64": "research-x86", "arm64": digest}
    image = gfaas.Image.from_registry_variants("research", variants)
    variants["arm64"] = "changed-after-construction"
    requirements = gfaas.Compatibility(
        cpu_architecture=architecture, gpu_compute_capabilities=("10.0", "10.3")
    )

    def handler(request):
        path = request.url.path
        if path.endswith("/capabilities"):
            return httpx.Response(200, json=CAPABILITIES)
        if path.endswith("/artifacts"):
            return httpx.Response(201, json={"id": "art_source"})
        requests[path] = json.loads(request.content)
        if path.endswith("/environments"):
            return httpx.Response(201, json={"id": "env_test"})
        if path.endswith("/functions"):
            return httpx.Response(201, json={"id": "fn_test"})
        assert path.endswith("/calls")
        return httpx.Response(202, json={"id": "call_test"})

    with gfaas.Client(
        gfaas.ClientConfig(
            api_base="https://example.test/api",
            api_key="test-key",
            poll_interval_s=0.01,
            request_timeout_s=1,
        )
    ) as client:
        client._http.close()
        client._http = httpx.Client(
            base_url=client.cfg.api_base, transport=httpx.MockTransport(handler)
        )
        app = gfaas.App("research", image=image, client=client)

        @app.function(compatibility=requirements, gpu_count=1)
        def run():
            return 42

        result = run.qualify(digest) if qualify else run.spawn()
        assert result.call_id == "call_test"

    environment = requests["/api/v1/environments"]
    assert environment["source"] == {
        "kind": "registered_images",
        "images": {"amd64": "research-x86", "arm64": digest},
    }
    expected = {"gpu_compute_capabilities": ["10.0", "10.3"]}
    if architecture is not None:
        expected["cpu_architecture"] = architecture
    assert environment["compatibility"] == expected
    assert requests["/api/v1/functions"]["environment_id"] == "env_test"
    call = requests["/api/v1/calls"]
    if qualify:
        assert call["qualification"] == {"image_digest": digest}
    else:
        assert "qualification" not in call


@pytest.mark.parametrize(
    ("capabilities", "image", "requirements", "message"),
    [
        ({}, "native", gfaas.Compatibility(cpu_architecture="arm64"), "does not advertise"),
        (
            {"environment_compatibility": {}},
            "native",
            gfaas.Compatibility(cpu_architecture="arm64"),
            "CPU architecture",
        ),
        (
            {"environment_compatibility": {}},
            "native",
            gfaas.Compatibility(gpu_compute_capabilities=("10.3",)),
            "exact GPU",
        ),
        (
            {"environment_compatibility": {}},
            gfaas.Image.from_registry_variants("native", {"amd64": "x86"}),
            None,
            "native image variants",
        ),
    ],
)
def test_unsupported_service_rejected_before_upload(capabilities, image, requirements, message):
    def handler(request):
        assert request.method == "GET"
        assert request.url.path.endswith("/capabilities")
        return httpx.Response(200, json=capabilities)

    with gfaas.Client(
        gfaas.ClientConfig(
            api_base="https://example.test/api",
            api_key="test-key",
            poll_interval_s=0.01,
            request_timeout_s=1,
        )
    ) as client:
        client._http.close()
        client._http = httpx.Client(
            base_url=client.cfg.api_base, transport=httpx.MockTransport(handler)
        )
        with pytest.raises(GfaasError, match=message):
            client.submit(image=image, compatibility=requirements, function=("work", "run"))


@pytest.mark.parametrize("capability", ["sm_103", "10.03", "0.0", "256.0", "10.100", "10.3\n"])
def test_noncanonical_compute_capabilities_are_rejected(capability):
    with pytest.raises(ValueError, match="compute capability"):
        gfaas.Compatibility(gpu_compute_capabilities=(capability,))


def test_conflicting_image_declarations_are_rejected_before_submission():
    with pytest.raises(ValueError, match="duplicate"):
        gfaas.Compatibility(gpu_compute_capabilities=("10.3", "10.3"))
    with pytest.raises(ValueError, match="registry variants"):
        gfaas.Image.from_registry_variants("bad", {"x86_64": "native"})
    with pytest.raises(ValueError, match="cannot be combined"):
        gfaas.Image("bad", build_spec={}, registry_variants={"arm64": "native"})
    with (
        gfaas.Client(
            gfaas.ClientConfig(
                api_base="https://example.invalid/api",
                api_key=None,
                poll_interval_s=0.01,
                request_timeout_s=1,
            )
        ) as client,
        pytest.raises(GfaasError, match="no variant"),
    ):
        client.submit(
            image=gfaas.Image.from_registry_variants("native", {"amd64": "x86"}),
            compatibility=gfaas.Compatibility(cpu_architecture="arm64"),
            function=("work", "run"),
        )


def test_waiting_summary_explains_hardware_mismatch_without_preparation():
    call = {
        "state": "queued",
        "compatibility": {
            "state": "hardware_mismatch",
            "message": "No matching hardware in the requested pool.",
        },
    }
    summary = gfaas.call_status_summary(call)
    assert "compatibility=hardware_mismatch" in summary
    assert "No matching hardware" in summary
    call["state"] = "failed"
    assert gfaas.call_status_summary(call) == "state=failed"
