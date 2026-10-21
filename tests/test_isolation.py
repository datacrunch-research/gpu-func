"""Explicit isolation must survive SDK submission or fail before any upload."""

import json

import httpx
import pytest

import gfaas
from gfaas.errors import GfaasError


@pytest.mark.parametrize(
    ("isolation", "capability", "error"),
    [
        ("container", None, "cannot enforce"),
        ("container", {"protocol_version": 2}, "cannot enforce"),
        ("sandbox", {"protocol_version": 1, "supported": ["container"]}, "does not support"),
        (
            "container",
            {"protocol_version": 1, "supported": ["container"], "allowed": []},
            "permission denied",
        ),
        ("runc", None, "container or sandbox"),
    ],
)
@pytest.mark.parametrize("low_level", [False, True])
def test_isolation_rejected_without_sending_work(isolation, capability, error, low_level):
    def handle(request):
        if request.method == "POST":
            assert request.url.path.endswith("/environments")
            assert capability["allowed"] == []
            return httpx.Response(
                403,
                json={"code": "execution_policy_denied", "message": "principal permission denied"},
            )
        assert request.method == "GET"
        assert request.url.path.endswith("/capabilities")
        return httpx.Response(200, json={"environment_isolation": capability})

    with gfaas.Client(
        gfaas.ClientConfig(
            api_base="https://example.test/api",
            api_key="test",
            poll_interval_s=0.01,
            request_timeout_s=1,
        )
    ) as client:
        client._http.close()
        client._http = httpx.Client(
            base_url=client.cfg.api_base, transport=httpx.MockTransport(handle)
        )
        with pytest.raises(GfaasError, match=error):
            if low_level:
                client.create_environment({"name": "test", "isolation": isolation})
            else:
                client.submit(image="cuda", function=("work", "run"), isolation=isolation)


@pytest.mark.parametrize("isolation", [None, "container"])
def test_decorator_preserves_isolation_and_omitted_default_works_with_old_servers(isolation):
    bodies = {}

    def handle(request):
        path = request.url.path.rsplit("/", 1)[-1]
        if path == "capabilities":
            assert isolation is not None
            return httpx.Response(
                200,
                json={
                    "environment_isolation": {
                        "protocol_version": 1,
                        "supported": ["container"],
                        "allowed": ["container"],
                        "default": "container",
                    }
                },
            )
        if path == "artifacts":
            return httpx.Response(201, json={"id": "art_source"})
        bodies[path] = json.loads(request.content)
        return httpx.Response(
            201,
            json={
                "id": {"environments": "env_test", "functions": "fn_test", "calls": "call_test"}[
                    path
                ]
            },
        )

    with gfaas.Client(
        gfaas.ClientConfig(
            api_base="https://example.test/api",
            api_key="test",
            poll_interval_s=0.01,
            request_timeout_s=1,
        )
    ) as client:
        client._http.close()
        client._http = httpx.Client(
            base_url=client.cfg.api_base, transport=httpx.MockTransport(handle)
        )
        app = gfaas.App("test", image=gfaas.Image.from_registry("cuda"), client=client)

        @app.function(isolation=isolation)
        def run():
            return 42

        run.spawn()
    assert bodies["environments"].get("isolation") == isolation
    assert "isolation" not in bodies["calls"]
    if isolation is None:
        assert "isolation" not in bodies["environments"]


def test_environment_replay_reaches_server_after_permission_revocation():
    def handle(request):
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "environment_isolation": {
                        "protocol_version": 1,
                        "supported": ["container"],
                        "allowed": [],
                        "default": None,
                    }
                },
            )
        assert request.url.path.endswith("/environments")
        assert request.headers["idempotency-key"] == "original-request"
        return httpx.Response(200, json={"id": "env_original"})

    config = gfaas.ClientConfig(
        api_base="https://example.test/api",
        api_key="test",
        poll_interval_s=0.01,
        request_timeout_s=1,
    )
    with gfaas.Client(config) as client:
        client._http.close()
        client._http = httpx.Client(base_url=config.api_base, transport=httpx.MockTransport(handle))
        result = client.create_environment(
            {"name": "test", "isolation": "container"}, idempotency_key="original-request"
        )
    assert result["id"] == "env_original"
