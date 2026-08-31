from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import httpx
import pytest

from gfaas.client import Client
from gfaas.config import ClientConfig
from gfaas.errors import ArtifactTreeUploadError


def digest(data: bytes) -> str:
    encoded = base64.b64encode(hashlib.sha256(data).digest()).decode("ascii")
    return f"sha-256=:{encoded}:"


def test_file_upload_resumes_verified_parts(tmp_path: Path) -> None:
    source = tmp_path / "input.bin"
    source.write_bytes(b"abcdefghij")
    uploaded_parts: list[int] = []
    progress = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/artifact-uploads/upl_test":
            return httpx.Response(
                200,
                json={
                    "id": "upl_test",
                    "state": "open",
                    "kind": "input",
                    "media_type": "application/octet-stream",
                    "filename": "input.bin",
                    "digest": digest(b"abcdefghij"),
                    "size_bytes": 10,
                    "part_size_bytes": 4,
                    "parts": [{"part_number": 0, "size_bytes": 4, "digest": digest(b"abcd")}],
                },
            )
        if "/parts/" in request.url.path:
            uploaded_parts.append(int(request.url.path.rsplit("/", 1)[1]))
            return httpx.Response(200, json={})
        if request.url.path.endswith("/completion"):
            return httpx.Response(201, json={"id": "art_test", "size_bytes": 10})
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    client = make_test_client(handler)
    artifact = client.upload_artifact_file(
        source,
        kind="input",
        upload_id="upl_test",
        progress=progress.append,
    )

    assert artifact["id"] == "art_test"
    assert uploaded_parts == [1, 2]
    assert progress[-1].path == source
    assert progress[-1].completed_bytes == 10
    assert progress[-1].total_bytes == 10
    assert progress[-1].transferred_bytes == 6
    assert progress[-1].reused_bytes == 4
    assert progress[-1].completed_files == 1
    assert progress[-1].total_files == 1


def test_file_download_streams_and_verifies_bytes(tmp_path: Path) -> None:
    payload = b"large Artifact bytes"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/artifacts/art_test/content"
        return httpx.Response(
            200,
            headers={
                "Content-Length": str(len(payload)),
                "Content-Digest": digest(payload),
                "Content-Type": "application/example",
            },
            content=payload,
        )

    client = make_test_client(handler)
    destination = tmp_path / "download.bin"

    path, media_type = client.download_artifact_file("art_test", destination)

    assert path == destination
    assert path.read_bytes() == payload
    assert media_type == "application/example"


def test_directory_upload_is_canonical_and_preserves_executable_mode(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "dataset"
    (source / "empty").mkdir(parents=True)
    (source / "bin").mkdir()
    (source / "z.txt").write_bytes(b"z")
    program = source / "bin" / "prepare"
    program.write_bytes(b"#!/bin/sh\n")
    program.chmod(0o755)
    uploaded_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/artifact-trees"
        body = json.loads(request.content)
        assert [entry["path"] for entry in body["entries"]] == [
            "bin",
            "bin/prepare",
            "empty",
            "z.txt",
        ]
        assert body["entries"][1]["executable"] is True
        assert body["entries"][3]["executable"] is False
        return httpx.Response(201, json={"id": "art_tree", "layout": "tree"})

    client = make_test_client(handler)

    def upload(path: Path, **_kwargs) -> dict[str, str]:
        relative = path.relative_to(source).as_posix()
        uploaded_paths.append(relative)
        return {"id": f"art_{relative.replace('/', '_')}"}

    monkeypatch.setattr(client, "upload_artifact_file", upload)

    artifact = client.upload_artifact_directory(source)

    assert artifact["id"] == "art_tree"
    assert artifact["child_artifact_ids"] == ["art_bin_prepare", "art_z.txt"]
    assert sorted(uploaded_paths) == ["bin/prepare", "z.txt"]


def test_directory_upload_reports_aggregate_progress(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    source.mkdir()
    (source / "a.bin").write_bytes(b"abc")
    (source / "b.bin").write_bytes(b"defgh")
    progress = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/artifact-uploads":
            body = json.loads(request.content)
            upload_id = f"upl_{body['filename']}"
            return httpx.Response(
                201,
                json={
                    "id": upload_id,
                    **body,
                    "part_size_bytes": 4,
                    "parts": [],
                },
            )
        if "/parts/" in request.url.path:
            return httpx.Response(200, json={})
        if request.url.path.endswith("/completion"):
            upload_id = request.url.path.split("/")[-2]
            return httpx.Response(201, json={"id": f"art_{upload_id}"})
        if request.url.path == "/v1/artifact-trees":
            return httpx.Response(201, json={"id": "art_tree", "layout": "tree"})
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    client = make_test_client(handler)

    artifact = client.upload_artifact_directory(source, progress=progress.append)

    assert artifact["id"] == "art_tree"
    assert progress[-1].path == source / "b.bin"
    assert progress[-1].completed_bytes == 8
    assert progress[-1].total_bytes == 8
    assert progress[-1].transferred_bytes == 8
    assert progress[-1].reused_bytes == 0
    assert progress[-1].completed_files == 2
    assert progress[-1].total_files == 2


def test_directory_upload_rejects_links(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    source.mkdir()
    (source / "target").write_bytes(b"target")
    (source / "link").symlink_to("target")
    client = make_test_client(lambda _request: httpx.Response(500))

    with pytest.raises(ValueError, match="symbolic link"):
        client.upload_artifact_directory(source)


def test_directory_upload_reports_children_when_tree_creation_fails(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "dataset"
    source.mkdir()
    (source / "input.bin").write_bytes(b"input")
    client = make_test_client(lambda _request: httpx.Response(500, json={"detail": "failed"}))
    monkeypatch.setattr(
        client,
        "upload_artifact_file",
        lambda _path, **_kwargs: {"id": "art_child"},
    )

    with pytest.raises(ArtifactTreeUploadError) as failure:
        client.upload_artifact_directory(source)

    assert failure.value.child_artifact_ids == ("art_child",)


def test_directory_download_materializes_verified_read_only_tree(tmp_path: Path) -> None:
    payload = b"training data"
    manifest = json.dumps(
        {
            "schema": "gfaas.artifact-tree/v1",
            "entries": [
                {"path": "data", "type": "directory", "mode": 0o555},
                {
                    "path": "data/train.bin",
                    "type": "file",
                    "artifact_id": "art_child",
                    "size_bytes": len(payload),
                    "digest": digest(payload),
                    "mode": 0o444,
                },
            ],
        },
        separators=(",", ":"),
    ).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/artifacts/art_tree":
            return httpx.Response(
                200,
                json={
                    "id": "art_tree",
                    "layout": "tree",
                    "media_type": "application/vnd.gfaas.artifact-tree.v1+json",
                },
            )
        if request.url.path == "/v1/artifacts/art_tree/content":
            return artifact_response(manifest, "application/vnd.gfaas.artifact-tree.v1+json")
        if request.url.path == "/v1/artifacts/art_child/content":
            return artifact_response(payload)
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    client = make_test_client(handler)
    destination = tmp_path / "download"

    result = client.download_artifact_directory("art_tree", destination)

    assert result == destination
    assert (destination / "data/train.bin").read_bytes() == payload
    assert (destination / "data/train.bin").stat().st_mode & 0o777 == 0o444
    assert (destination / "data").stat().st_mode & 0o777 == 0o555


def artifact_response(
    payload: bytes, media_type: str = "application/octet-stream"
) -> httpx.Response:
    return httpx.Response(
        200,
        headers={
            "Content-Length": str(len(payload)),
            "Content-Digest": digest(payload),
            "Content-Type": media_type,
        },
        content=payload,
    )


def make_test_client(handler) -> Client:
    client = Client(
        ClientConfig(
            api_base="https://api.example/v1-placeholder",
            api_key=None,
            poll_interval_s=0.01,
            request_timeout_s=1,
        )
    )
    client._http.close()
    client._http = httpx.Client(
        base_url="https://api.example",
        transport=httpx.MockTransport(handler),
    )
    return client
