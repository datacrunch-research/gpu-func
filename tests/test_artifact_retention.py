"""Upload and update paths must carry the owner's lifetime choice consistently."""

import json
from pathlib import Path

import httpx
import pytest
from test_upload_concurrency import UploadService

from gfaas import Client, ClientConfig
from gfaas.errors import GfaasError
from gfaas_cli import main as cli


def make_client(handler):
    client = Client(
        ClientConfig(
            api_base="https://example.test", api_key="key", request_timeout_s=5, poll_interval_s=0.1
        )
    )
    client._http.close()
    client._http = httpx.Client(
        transport=httpx.MockTransport(handler), base_url=client.cfg.api_base
    )
    return client


@pytest.mark.parametrize("ttl_seconds", [None, 600])
def test_directory_and_resumed_file_keep_the_requested_policy(
    tmp_path: Path, ttl_seconds, monkeypatch
):
    service = UploadService()
    source = tmp_path / "data.bin"
    source.write_bytes(b"test data")
    expected = (
        {"mode": "keep"} if ttl_seconds is None else {"mode": "expire", "ttl_seconds": ttl_seconds}
    )
    monkeypatch.setattr("gfaas.client.MAXIMUM_INLINE_ARTIFACT_BYTES", 1)
    with make_client(service) as client:
        client.upload_artifact(b"chunked data", ttl_seconds=ttl_seconds)
        session = client.create_artifact_file_upload(source, ttl_seconds=ttl_seconds)
        assert session["retention"] == expected
        with pytest.raises(GfaasError, match="does not match retention"):
            client.upload_artifact_file(source, upload_id=session["id"], ttl_seconds=60)
        assert all(uid != session["id"] for uid, _ in service.puts)
        client.upload_artifact_file(source, upload_id=session["id"], ttl_seconds=ttl_seconds)
        client.upload_artifact_directory(tmp_path, ttl_seconds=ttl_seconds)
    assert all(session["retention"] == expected for session in service.sessions.values())
    assert service.trees[0]["retention"] == expected


def test_inline_upload_and_cli_retention_update(capsys):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"id": "art_test"})

    with make_client(handler) as client:
        client.upload_artifact(b"keep")
        client.upload_artifact(b"expire", ttl_seconds=60)
    assert seen[0].headers["x-gfunc-retention"] == "keep"
    assert "x-gfunc-ttl-seconds" not in seen[0].headers
    assert seen[1].headers["x-gfunc-ttl-seconds"] == "60"
    assert "x-gfunc-retention" not in seen[1].headers
    for extra, expected in [
        ([], {"mode": "keep"}),
        (["--ttl-seconds", "600"], {"mode": "expire", "ttl_seconds": 600}),
    ]:
        assert (
            cli.main(
                ["artifact", "retain", "art_test", *extra],
                client_factory=lambda: make_client(handler),
            )
            == 0
        )
        assert seen[-1].method == "PUT"
        assert seen[-1].url.path == "/v1/artifacts/art_test/retention"
        assert json.loads(seen[-1].content) == expected
    assert "art_test" in capsys.readouterr().out


@pytest.mark.parametrize("ttl_seconds", [0, -1, True, 1.5])
def test_invalid_retention_does_not_start_an_upload(tmp_path, ttl_seconds):
    def handler(request):
        raise AssertionError("invalid retention must fail before a request")

    with make_client(handler) as client:
        for operation in [
            lambda: client.upload_artifact(b"data", ttl_seconds=ttl_seconds),
            lambda: client.upload_artifact_file(tmp_path / "missing", ttl_seconds=ttl_seconds),
            lambda: client.upload_artifact_directory(tmp_path, ttl_seconds=ttl_seconds),
            lambda: client.create_artifact_file_upload(
                tmp_path / "missing", ttl_seconds=ttl_seconds
            ),
            lambda: client.set_artifact_retention("art_test", ttl_seconds=ttl_seconds),
        ]:
            with pytest.raises(ValueError, match="positive integer"):
                operation()
