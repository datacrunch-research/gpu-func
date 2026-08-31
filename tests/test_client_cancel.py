from __future__ import annotations

import json
import pickle
from collections.abc import Callable

import cloudpickle
import httpx
import pytest

from gfaas.artifacts import ArtifactOutput, ArtifactRef
from gfaas.client import (
    CLOUDPICKLE_MEDIA_TYPE,
    PICKLE_MEDIA_TYPE,
    Client,
    GfaasError,
    RemoteResult,
)
from gfaas.config import ClientConfig
from gfaas.errors import UnsupportedGpuPoolError
from gfaas.image import Image


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> Client:
    client = Client(
        ClientConfig(
            api_base="https://gpu.example.com/api",
            api_key="key-1",
            poll_interval_s=0.001,
            request_timeout_s=5.0,
        )
    )
    client._http.close()
    client._http = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url=client.cfg.api_base,
        headers={"X-API-Key": client.cfg.api_key},
    )
    return client


def test_remote_result_cancel_uses_call_cancellation_endpoint():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/v1/calls/call_1/cancellation"
        assert request.headers["X-API-Key"] == "key-1"
        assert json.loads(request.content) == {"reason": "no longer needed"}
        return httpx.Response(202, json={"id": "call_1", "state": "running"})

    with _client(handler) as client:
        result = RemoteResult(client=client, job_id="call_1", gpu_type="b200")
        response = result.cancel(reason="no longer needed")

    assert result.call_id == "call_1"
    assert response["state"] == "running"


@pytest.mark.parametrize("status_code", [401, 403])
def test_high_level_submit_reports_authentication_rejection_without_leaking_key(status_code: int):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/artifacts"
        return httpx.Response(status_code, json={"detail": "invalid API key key-1"})

    with _client(handler) as client, pytest.raises(GfaasError) as captured:
        client.submit(image="image", function=("module", "function"))

    assert str(captured.value) == (
        "artifact upload failed: authentication rejected by "
        f"https://gpu.example.com/api ({status_code})"
    )
    assert "key-1" not in str(captured.value)


@pytest.mark.parametrize(
    ("gpu", "gpu_count", "message"),
    [
        ("any", 4, "gpu and gpu_count cannot be used together"),
        (None, -1, "gpu_count must be a non-negative integer"),
        (None, True, "gpu_count must be a non-negative integer"),
        (None, 1.5, "gpu_count must be a non-negative integer"),
    ],
)
def test_submit_rejects_invalid_explicit_gpu_counts_before_network_access(
    gpu,
    gpu_count,
    message: str,
):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    with _client(handler) as client, pytest.raises(GfaasError, match=message):
        client.submit(
            image="image",
            function=("module", "function"),
            gpu=gpu,
            gpu_count=gpu_count,
        )


@pytest.mark.parametrize(
    ("operation", "invoke"),
    [
        ("artifact lookup", lambda client: client.get_artifact("art_1")),
        ("artifact download", lambda client: client.download_artifact("art_1")),
        ("artifact deletion", lambda client: client.delete_artifact("art_1")),
        ("environment lookup", lambda client: client.get_environment("env_1")),
        ("function lookup", lambda client: client.get_function("fn_1")),
        ("call status", lambda client: client.get_call("call_1")),
        ("call cancellation", lambda client: client.cancel_call("call_1")),
        ("call result", lambda client: client.get_call_result("call_1")),
        ("attempt listing", lambda client: client.list_attempts("call_1")),
        ("event listing", lambda client: client.list_events("call_1")),
        ("call artifact listing", lambda client: client.list_call_artifacts("call_1")),
        ("capability lookup", lambda client: client.get_capabilities()),
    ],
)
def test_client_operations_translate_authentication_rejection(operation: str, invoke):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "not authenticated"})

    with (
        _client(handler) as client,
        pytest.raises(GfaasError, match=rf"{operation} failed: authentication rejected"),
    ):
        invoke(client)


def test_client_transport_errors_remain_distinct():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with _client(handler) as client, pytest.raises(httpx.ConnectError, match="connection refused"):
        client.get_call("call_1")


def test_client_reports_configured_gpu_pool_capabilities():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/v1/capabilities"
        return httpx.Response(
            200,
            json={
                "gpu_pools": [
                    {
                        "name": "gb300",
                        "status": "busy",
                        "connected_workers": 2,
                        "available_workers": 0,
                    }
                ]
            },
        )

    with _client(handler) as client:
        capabilities = client.get_capabilities()

    assert capabilities["gpu_pools"][0]["status"] == "busy"


def test_client_raises_stable_error_for_an_unconfigured_gpu_pool():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={
                "code": "unsupported_gpu_pool",
                "detail": 'requested GPU pool "gb30" is not configured',
                "compatibility": {
                    "requested_gpu_pool": "gb30",
                    "supported_gpu_pools": ["gb300"],
                },
            },
        )

    with _client(handler) as client, pytest.raises(UnsupportedGpuPoolError) as captured:
        client.create_call(
            {
                "function_id": "fn_1",
                "resources": {"gpu": {"count": 1, "models": ["gb30"]}},
            }
        )

    assert captured.value.requested_gpu_pool == "gb30"
    assert captured.value.supported_gpu_pools == ("gb300",)
    assert str(captured.value) == "GPU pool 'gb30' is not configured; supported pools: gb300"


def test_client_deletes_an_owner_scoped_artifact():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path == "/api/v1/artifacts/art_nonce"
        assert request.headers["X-API-Key"] == "key-1"
        return httpx.Response(204)

    with _client(handler) as client:
        assert client.delete_artifact("art_nonce") is None


def test_remote_result_retrieves_and_follows_stdout_and_stderr():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.params.get("follow") == "true":
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=(
                    b'id: 3\nevent: stdout\ndata: {"cursor":"3","type":"stdout",'
                    b'"stream_data":"live\\n"}\n\n'
                    b'id: 4\nevent: state\ndata: {"cursor":"4","type":"state",'
                    b'"state":"succeeded"}\n\n'
                ),
            )
        return httpx.Response(
            200,
            json={
                "items": [
                    {"cursor": "1", "type": "stdout", "stream_data": "out\n"},
                    {"cursor": "2", "type": "stderr", "stream_data": "err\n"},
                    {"cursor": "3", "type": "retention.truncated"},
                ],
                "truncated": False,
            },
        )

    with _client(handler) as client:
        result = RemoteResult(client=client, job_id="call_1", gpu_type="gb300")
        assert result.logs() == {
            "stdout": "out\n",
            "stderr": "err\n",
            "truncated": True,
        }
        assert list(result.iter_logs(after="2", follow=True)) == [("stdout", "live\n")]

    assert requests[0].url.params["limit"] == "1000"
    assert requests[1].url.params["after"] == "2"


def test_log_iteration_with_timeout_polls_until_terminal_state():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.path == "/api/v1/calls/call_1/events"
        if request.url.params.get("after") is None:
            return httpx.Response(
                200,
                json={
                    "items": [{"cursor": "0", "type": "state", "state": "queued"}],
                    "next_cursor": None,
                    "truncated": False,
                },
            )
        return httpx.Response(
            200,
            json={
                "items": [
                    {"cursor": "1", "type": "stdout", "stream_data": "ready\n"},
                    {"cursor": "2", "type": "state", "state": "succeeded"},
                ],
                "next_cursor": None,
                "truncated": False,
            },
        )

    with _client(handler) as client:
        result = RemoteResult(client=client, job_id="call_1", gpu_type="gb300")
        assert list(result.iter_logs(timeout_s=1)) == [("stdout", "ready\n")]

    assert len(requests) == 2
    assert all(request.url.params["follow"] == "false" for request in requests)
    assert requests[1].url.params["after"] == "0"


def test_log_iteration_total_timeout_is_bounded():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/calls/call_1/events"
        return httpx.Response(
            200,
            json={"items": [], "next_cursor": None, "truncated": False},
        )

    with _client(handler) as client:
        result = RemoteResult(client=client, job_id="call_1", gpu_type="gb300")
        with pytest.raises(TimeoutError, match="event stream exceeded 0s"):
            list(result.iter_logs(timeout_s=0))


def test_submit_and_wait_map_python_convenience_onto_public_resources():
    requests: list[httpx.Request] = []
    artifact_ids = iter(["art_source", "art_input"])
    call_polls = iter(["queued", "succeeded"])
    encoded_result = cloudpickle.dumps({"answer": 42})

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = request.url.path
        if request.method == "POST" and path == "/api/v1/artifacts":
            assert request.headers["Content-Digest"].startswith("sha-256=:")
            return httpx.Response(201, json={"id": next(artifact_ids)})
        if request.method == "POST" and path == "/api/v1/environments":
            body = json.loads(request.content)
            assert body == {
                "name": "cuda-nvcc",
                "source": {"kind": "registered_image", "name": "cuda-nvcc"},
                "variables": {"MODE": "test"},
            }
            return httpx.Response(201, json={"id": "env_1"})
        if request.method == "POST" and path == "/api/v1/functions":
            body = json.loads(request.content)
            assert body["executable"] == {
                "kind": "python_callable",
                "module": "module",
                "qualname": "function",
                "source_artifact_id": "art_source",
            }
            assert body["default_resources"] == {
                "gpu": {"count": 4, "models": ["gb300"]},
                "timeout_seconds": 120,
                "cpu_millicores": 8000,
                "memory_bytes": 64 * 1024**3,
                "ephemeral_storage_bytes": 128 * 1024**3,
                "shared_memory_bytes": 8 * 1024**3,
                "max_log_bytes": 32 * 1024**2,
                "max_output_bytes": 2 * 1024**3,
            }
            return httpx.Response(201, json={"id": "fn_1"})
        if request.method == "POST" and path == "/api/v1/calls":
            assert request.headers["Idempotency-Key"] == "request-1"
            assert json.loads(request.content) == {
                "function_id": "fn_1",
                "capacity_wait_seconds": 1800,
                "artifacts": [{"artifact_id": "art_attached"}],
                "outputs": [
                    {
                        "name": "profile",
                        "path": "profiles/kernel.json",
                        "kind": "profile",
                        "media_type": "application/json",
                        "layout": "blob",
                        "publication": "terminal",
                        "maximum_versions": 1,
                        "required": False,
                        "publish_on_failure": True,
                    }
                ],
                "input": {
                    "storage": "artifact",
                    "media_type": CLOUDPICKLE_MEDIA_TYPE,
                    "artifact_id": "art_input",
                },
            }
            return httpx.Response(202, json={"id": "call_1", "state": "queued"})
        if request.method == "GET" and path == "/api/v1/calls/call_1":
            return httpx.Response(200, json={"id": "call_1", "state": next(call_polls)})
        if request.method == "GET" and path == "/api/v1/calls/call_1/result":
            return httpx.Response(
                200,
                json={
                    "call_id": "call_1",
                    "outcome": "succeeded",
                    "output": {
                        "storage": "artifact",
                        "media_type": CLOUDPICKLE_MEDIA_TYPE,
                        "artifact_id": "art_result",
                    },
                },
            )
        if request.method == "GET" and path == "/api/v1/artifacts/art_result/content":
            return httpx.Response(
                200,
                content=encoded_result,
                headers={"Content-Type": CLOUDPICKLE_MEDIA_TYPE},
            )
        raise AssertionError(f"unexpected request: {request.method} {path}")

    with _client(handler) as client:
        remote = client.submit(
            image="cuda-nvcc",
            function=("module", "function"),
            args=(41,),
            kwargs={"increment": 1, "attachment": ArtifactRef("art_attached")},
            gpu_count=4,
            gpu_type="gb300",
            app_name="example",
            timeout_s=120,
            capacity_wait_s=1800,
            cpu_millicores=8000,
            memory_bytes=64 * 1024**3,
            ephemeral_storage_bytes=128 * 1024**3,
            shared_memory_bytes=8 * 1024**3,
            max_log_bytes=32 * 1024**2,
            max_output_bytes=2 * 1024**3,
            env={"MODE": "test"},
            idempotency_key="request-1",
            outputs=(
                ArtifactOutput(
                    "profile",
                    "profiles/kernel.json",
                    kind="profile",
                    media_type="application/json",
                    required=False,
                ),
            ),
        )
        assert remote.wait(timeout_s=1) == {"answer": 42}

    assert remote.job_id == "call_1"
    artifact_uploads = [
        request
        for request in requests
        if request.method == "POST" and request.url.path.endswith("/artifacts")
    ]
    assert artifact_uploads[0].headers["X-Gfunc-Artifact-Kind"] == "source"
    assert artifact_uploads[1].headers["X-Gfunc-Artifact-Kind"] == "input"


def test_submit_includes_remote_image_descriptor_in_environment_source():
    sha256 = "a" * 64
    descriptor = {
        "schema": "fast-container-remote-image/v1",
        "image_digest": f"sha256:{sha256}",
        "index": {
            "storage_key": f"sha256/aa/{'a' * 62}",
            "sha256": sha256,
            "size_bytes": 123,
        },
        "cas_object_count": 2,
        "cas_total_bytes": 456,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/artifacts":
            return httpx.Response(201, json={"id": "art_source"})
        if request.url.path == "/api/v1/environments":
            assert json.loads(request.content) == {
                "name": "cuda-nvcc-s3",
                "source": {
                    "kind": "registered_image",
                    "name": "cuda-nvcc-s3",
                    "remote": descriptor,
                },
                "variables": {},
            }
            return httpx.Response(201, json={"id": "env_1"})
        if request.url.path == "/api/v1/functions":
            return httpx.Response(201, json={"id": "fn_1"})
        if request.url.path == "/api/v1/calls":
            return httpx.Response(202, json={"id": "call_1", "state": "queued"})
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    with _client(handler) as client:
        remote = client.submit(
            image=Image.from_remote("cuda-nvcc-s3", descriptor),
            function=("module", "function"),
        )

    assert remote.job_id == "call_1"


def test_wait_decodes_explicit_python_pickle_result_media_type():
    encoded_result = pickle.dumps({"answer": 42})

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/calls/call_1":
            return httpx.Response(200, json={"id": "call_1", "state": "succeeded"})
        if request.url.path == "/api/v1/calls/call_1/result":
            return httpx.Response(
                200,
                json={
                    "call_id": "call_1",
                    "outcome": "succeeded",
                    "output": {
                        "storage": "artifact",
                        "media_type": PICKLE_MEDIA_TYPE,
                        "artifact_id": "art_result",
                    },
                },
            )
        if request.url.path == "/api/v1/artifacts/art_result/content":
            return httpx.Response(
                200,
                content=encoded_result,
                headers={"Content-Type": PICKLE_MEDIA_TYPE},
            )
        raise AssertionError(f"unexpected request: {request.method} {request.url.path}")

    with _client(handler) as client:
        assert client.wait_for_result("call_1", timeout_s=1) == {"answer": 42}


def test_wait_bounds_a_large_failure_journey():
    journey = [
        {"component": "subscriber", "stage": "tree_entry_uploaded", "item": index}
        for index in range(45)
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/calls/call_1":
            return httpx.Response(200, json={"id": "call_1", "state": "failed"})
        if request.url.path == "/api/v1/calls/call_1/result":
            return httpx.Response(
                200,
                json={
                    "call_id": "call_1",
                    "outcome": "failed",
                    "error": {
                        "message": "transfer failed",
                        "details": {"journey": journey},
                    },
                },
            )
        raise AssertionError(f"unexpected request: {request.method} {request.url.path}")

    with _client(handler) as client, pytest.raises(GfaasError) as captured:
        client.wait_for_result("call_1", timeout_s=1)

    message = str(captured.value)
    assert "5 journey events omitted" in message
    assert "item=0" in message
    assert "item=44" in message
    assert "item=20" not in message


def test_dynamic_image_builds_fail_before_network_access():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    image = Image.from_container("python:3.11")
    with _client(handler) as client, pytest.raises(GfaasError, match="registered image"):
        client.submit(image=image, function=("module", "function"))
