"""Transfers above the old agent limit, with accepted parts and lost completion responses."""

from __future__ import annotations

import base64
import hashlib
import json
import pickle
from pathlib import Path

import httpx
import pytest

from gfaas.client import Client
from gfaas.config import ClientConfig
from gfaas.errors import ArtifactUploadError


def digest(data: bytes) -> str:
    return "sha-256=:" + base64.b64encode(hashlib.sha256(data).digest()).decode() + ":"


@pytest.mark.parametrize("file_source", [False, True])
def test_large_bytes_and_files_resume_after_part_and_completion_failures(
    tmp_path: Path,
    file_source: bool,
) -> None:
    size = 128 * 1024 * 1024 + 17
    data = b"z" * size
    source = tmp_path / "payload"
    if file_source:
        source.write_bytes(data)
    part_size = 16 * 1024 * 1024
    session: dict = {}
    received: dict[int, dict] = {}
    requests: list[int] = []
    failed_part = failed_completion = False
    metadata: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal session, failed_part, failed_completion, metadata
        path = request.url.path
        if path == "/v1/artifacts":
            pytest.fail("large content reached the single-request endpoint")
        if path == "/v1/artifact-uploads":
            assert not session, "retry created a second upload session"
            metadata = json.loads(request.content)
            assert metadata["size_bytes"] == size
            assert metadata["digest"] == digest(data)
            session = {**metadata, "id": "upl_large", "part_size_bytes": part_size}
            return httpx.Response(201, json=session)
        if path == "/v1/artifact-uploads/upl_large":
            return httpx.Response(200, json={**session, "parts": list(received.values())})
        if "/parts/" in path:
            number = int(path.rsplit("/", 1)[1])
            requests.append(number)
            assert 0 < len(request.content) <= part_size
            assert request.headers["Content-Digest"] == digest(request.content)
            assert request.content == data[number * part_size : (number + 1) * part_size]
            received[number] = {
                "part_number": number,
                "size_bytes": len(request.content),
                "digest": digest(request.content),
            }
            if number == 1 and not failed_part:
                failed_part = True
                raise httpx.ReadError("lost part receipt")
            return httpx.Response(200, json={})
        if path.endswith("/completion"):
            assert sum(part["size_bytes"] for part in received.values()) == size
            session["artifact_id"] = "art_large"
            assert not failed_completion
            failed_completion = True
            raise httpx.ReadError("lost completion receipt")
        if path == "/v1/artifacts/art_large":
            return httpx.Response(200, json={**metadata, "id": "art_large"})
        raise AssertionError(path)

    with Client(
        ClientConfig(
            api_base="https://example.invalid",
            api_key="test",
            poll_interval_s=0.01,
            request_timeout_s=30,
        )
    ) as client:
        client._http.close()
        client._http = httpx.Client(
            transport=httpx.MockTransport(handler), base_url="https://example.invalid"
        )

        def upload(upload_id: str | None = None) -> dict:
            if file_source:
                return client.upload_artifact_file(source, kind="input", upload_id=upload_id)
            return client.upload_artifact(data, kind="input", upload_id=upload_id)

        with pytest.raises(ArtifactUploadError) as interrupted:
            upload()
        with pytest.raises(ArtifactUploadError):
            upload(interrupted.value.upload_id)
        result = upload(interrupted.value.upload_id)
        assert result["id"] == "art_large"
        assert sorted(requests) == list(range(9)), "accepted parts were transferred again"


def test_wait_streams_large_result_and_checks_integrity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = pickle.dumps(b"q" * (128 * 1024 * 1024 + 1), protocol=5)

    class Chunks(httpx.SyncByteStream):
        def __iter__(self):
            for offset in range(0, len(payload), 64 * 1024):
                yield payload[offset : offset + 64 * 1024]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/calls/call_1":
            return httpx.Response(200, json={"state": "succeeded"})
        if request.url.path.endswith("/result"):
            return httpx.Response(
                200,
                json={
                    "outcome": "succeeded",
                    "output": {
                        "artifact_id": "art_result",
                        "media_type": "application/vnd.python.pickle",
                    },
                },
            )
        return httpx.Response(
            200,
            stream=Chunks(),
            headers={
                "Content-Type": "application/vnd.python.pickle",
                "Content-Length": str(len(payload)),
                "Content-Digest": digest(payload),
            },
        )

    with Client(
        ClientConfig(
            api_base="https://example.invalid",
            api_key="test",
            poll_interval_s=0.01,
            request_timeout_s=30,
        )
    ) as client:
        client._http.close()
        client._http = httpx.Client(
            transport=httpx.MockTransport(handler), base_url="https://example.invalid"
        )
        monkeypatch.setattr(
            client, "download_artifact", lambda _: pytest.fail("buffered result download")
        )
        result = client.wait_for_result("call_1")
        assert len(result) == 128 * 1024 * 1024 + 1
        assert result[:1] == result[-1:] == b"q"


@pytest.mark.parametrize("body", [b"evil", b"da", b"data-too-long"])
def test_result_integrity_failure_precedes_decoding_and_removes_temporary_files(
    body: bytes,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gfaas import client as client_module
    from gfaas.errors import GfaasError

    monkeypatch.setattr(client_module.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(
        client_module,
        "decode_result_file",
        lambda *args, **kwargs: pytest.fail("decoded corrupt data"),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/calls/call_1":
            return httpx.Response(200, json={"state": "succeeded"})
        if request.url.path.endswith("/result"):
            return httpx.Response(
                200,
                json={
                    "outcome": "succeeded",
                    "output": {
                        "artifact_id": "art_result",
                        "media_type": "application/vnd.python.pickle",
                    },
                },
            )
        return httpx.Response(
            200,
            content=body,
            headers={
                "Content-Length": "4",
                "Content-Digest": digest(b"data"),
            },
        )

    with Client(
        ClientConfig(
            api_base="https://example.invalid",
            api_key="test",
            poll_interval_s=0.01,
            request_timeout_s=30,
        )
    ) as client:
        client._http.close()
        client._http = httpx.Client(
            transport=httpx.MockTransport(handler), base_url="https://example.invalid"
        )
        with pytest.raises(GfaasError, match="integrity|declared size"):
            client.wait_for_result("call_1")
    assert list(tmp_path.iterdir()) == []
