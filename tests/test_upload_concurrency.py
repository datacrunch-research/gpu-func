"""Concurrent uploads retain integrity and recovery evidence under interleaved failures."""

from __future__ import annotations

import base64
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event, Lock

import httpx
import pytest

from gfaas import ArtifactTreeUploadError, ArtifactUploadError, Client, ClientConfig


def digest(data: bytes) -> str:
    return "sha-256=:" + base64.b64encode(hashlib.sha256(data).digest()).decode() + ":"


class UploadService:
    def __init__(self) -> None:
        self.lock = Lock()
        self.sessions: dict[str, dict] = {}
        self.parts: dict[str, dict[int, bytes]] = {}
        self.artifacts: dict[str, dict] = {}
        self.active = self.peak = self.open_sessions = self.peak_sessions = 0
        self.puts: list[tuple[str, int]] = []
        self.trees: list[dict] = []
        self.before = lambda request: None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            response = self.before(request)
            if response is not None:
                return response
            time.sleep(0.002)
            with self.lock:
                return self.respond(request)
        finally:
            with self.lock:
                self.active -= 1

    def respond(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/artifact-uploads":
            uid = str(len(self.sessions))
            session = {**json.loads(request.content), "id": uid, "part_size_bytes": 4}
            self.sessions[uid] = session
            self.parts[uid] = {}
            self.open_sessions += 1
            self.peak_sessions = max(self.peak_sessions, self.open_sessions)
            return httpx.Response(201, json=session)
        if "/parts/" in path:
            uid, _, number = path.split("/")[-3:]
            index = int(number)
            assert request.headers["Content-Digest"] == digest(request.content)
            self.puts.append((uid, index))
            self.parts[uid][index] = request.content
            return httpx.Response(200, json={})
        if path.endswith("/completion"):
            uid = path.split("/")[-2]
            data = b"".join(self.parts[uid][i] for i in sorted(self.parts[uid]))
            session = self.sessions[uid]
            assert len(data) == session["size_bytes"]
            assert digest(data) == session["digest"]
            aid = hashlib.sha256(data).hexdigest()
            self.artifacts[aid] = {**session, "id": aid}
            session["artifact_id"] = aid
            self.open_sessions -= 1
            return httpx.Response(201, json=self.artifacts[aid])
        if path.startswith("/v1/artifact-uploads/"):
            uid = path.split("/")[-1]
            return httpx.Response(
                200,
                json={
                    **self.sessions[uid],
                    "parts": [
                        {"part_number": number, "size_bytes": len(data), "digest": digest(data)}
                        for number, data in self.parts[uid].items()
                    ],
                },
            )
        if path.startswith("/v1/artifacts/"):
            return httpx.Response(200, json=self.artifacts[path.split("/")[-1]])
        if path == "/v1/artifact-trees":
            tree = json.loads(request.content)
            assert all(
                e["artifact_id"] in self.artifacts for e in tree["entries"] if e["type"] == "file"
            )
            self.trees.append(tree)
            return httpx.Response(201, json={"id": digest(request.content)})
        raise AssertionError(path)


def client_for(service: UploadService, concurrency: int = 2) -> Client:
    client = Client(ClientConfig("https://example.invalid", None, 0.01, 5, concurrency))
    client._http.close()
    client._http = httpx.Client(
        base_url=client.cfg.api_base, transport=httpx.MockTransport(service)
    )
    return client


def test_parts_overlap_and_resume_lost_receipt_without_retransmission(tmp_path: Path) -> None:
    service = UploadService()
    first_pair = Barrier(2)
    failed = Event()
    source = tmp_path / "input"
    source.write_bytes(b"abcdefghijklmnopqrst")

    def intercept(request):
        if "/parts/" in request.url.path and not failed.is_set():
            first_pair.wait(timeout=5)
            if request.url.path.endswith("/1"):
                with service.lock:
                    service.respond(request)  # Accepted, but the response was lost.
                failed.set()
                raise httpx.ReadError("lost receipt")
        return None

    service.before = intercept
    with client_for(service) as client:
        with pytest.raises(ArtifactUploadError, match="lost receipt") as error:
            client.upload_artifact_file(source)
        service.before = lambda request: None
        progress = []
        result = client.upload_artifact_file(
            source, upload_id=error.value.upload_id, progress=progress.append
        )
        again = client.upload_artifact_file(source, upload_id=error.value.upload_id)
    assert result == again
    assert sorted(number for _, number in service.puts) == list(range(5))
    assert service.peak == 2
    assert progress[-1].completed_files == 1
    assert progress[-1].reused_bytes >= 8


def test_shared_budget_bounds_multiple_directories_and_keeps_tree_identity(tmp_path: Path) -> None:
    service = UploadService()
    for index in range(8):
        (tmp_path / f"{index}.bin").write_bytes(bytes([index]) * 16)
    progress = []
    with client_for(service) as client, ThreadPoolExecutor(2) as callers:
        first = callers.submit(client.upload_artifact_directory, tmp_path, progress=progress.append)
        second = callers.submit(client.upload_artifact_directory, tmp_path)
        assert first.result()["id"] == second.result()["id"]
    assert 1 < service.peak <= 2
    assert service.peak_sessions <= 2
    assert progress[-1].completed_bytes == 128
    assert progress[-1].completed_files == 8
    assert [p.completed_bytes for p in progress] == sorted(p.completed_bytes for p in progress)
    assert service.trees[0] == service.trees[1]
    assert [e["path"] for e in service.trees[0]["entries"]] == [f"{i}.bin" for i in range(8)]


def test_partial_directory_failure_reports_completed_children_and_failed_sessions(
    tmp_path: Path,
) -> None:
    service = UploadService()
    (tmp_path / "a").write_bytes(b"aaaa")
    (tmp_path / "b").write_bytes(b"bbbb")
    pair = Barrier(2)

    def intercept(request):
        if "/parts/" in request.url.path:
            pair.wait(timeout=5)
            uid = request.url.path.split("/")[-3]
            if service.sessions[uid]["filename"] == "b":
                return httpx.Response(429, json={"detail": "upload quota"})
        return None

    service.before = intercept
    with client_for(service) as client, pytest.raises(ArtifactTreeUploadError) as error:
        client.upload_artifact_directory(tmp_path)
    assert error.value.child_artifact_ids == (hashlib.sha256(b"aaaa").hexdigest(),)
    assert set(error.value.upload_ids) == {"b"}
    assert "429" in error.value.failures["b"]
    assert service.trees == []


@pytest.mark.parametrize("fault", ["cancel", "digest", "completion"])
def test_failure_never_publishes_a_tree_and_retains_resume_id(tmp_path: Path, fault: str) -> None:
    service = UploadService()
    cancel = Event()
    (tmp_path / "data").write_bytes(b"abcdefghijklmnop")

    def intercept(request):
        if "/parts/" in request.url.path:
            if fault == "cancel":
                cancel.set()
            elif fault == "digest":
                return httpx.Response(400, json={"detail": "digest mismatch"})
        if fault == "completion" and request.url.path.endswith("/completion"):
            return httpx.Response(503, json={"detail": "storage unavailable"})
        return None

    service.before = intercept
    with client_for(service) as client, pytest.raises(ArtifactTreeUploadError) as error:
        client.upload_artifact_directory(tmp_path, cancel=cancel)
    assert set(error.value.upload_ids) == {"data"}
    assert {"cancel": "cancelled", "digest": "digest mismatch", "completion": "503"}[
        fault
    ] in error.value.failures["data"]
    assert service.trees == []
    assert service.artifacts == {}


def test_part_buffers_are_not_read_ahead_of_request_budget(tmp_path: Path, monkeypatch) -> None:
    service = UploadService()
    blocked = Barrier(3)
    release = Event()
    source = tmp_path / "input"
    source.write_bytes(b"x" * 80)
    reads = []
    original_open = Path.open

    class ObservedFile:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def read(self, size):
            if size == 4:
                reads.append(size)
            return self.stream.read(size)

    def intercept(request):
        if "/parts/" in request.url.path and not release.is_set():
            blocked.wait(timeout=5)
            assert release.wait(5)
        return None

    service.before = intercept
    monkeypatch.setattr(
        Path, "open", lambda self, *a, **kw: ObservedFile(original_open(self, *a, **kw))
    )
    with client_for(service) as client, ThreadPoolExecutor(1) as callers:
        upload = callers.submit(client.upload_artifact_file, source)
        try:
            blocked.wait(timeout=5)
            assert reads == [4, 4]
        finally:
            release.set()
        upload.result()
    assert len(reads) == 20


def test_conflicting_resume_part_stops_before_completion(tmp_path: Path) -> None:
    service = UploadService()
    source = tmp_path / "input"
    source.write_bytes(b"abcdefgh")
    with client_for(service) as client:
        session = client.create_artifact_file_upload(source)
        service.parts[session["id"]][0] = b"evil"
        with pytest.raises(ArtifactUploadError, match="conflicting part 0") as error:
            client.upload_artifact_file(source, upload_id=session["id"])
    assert error.value.upload_id == session["id"]
    assert service.artifacts == {}


def test_session_quota_rejection_stops_tree_without_creating_more_sessions(tmp_path: Path) -> None:
    service = UploadService()
    for index in range(8):
        (tmp_path / str(index)).write_bytes(b"x")
    attempted = []

    def reject(request):
        assert request.url.path == "/v1/artifact-uploads"
        attempted.append(request.url.path)
        return httpx.Response(429, json={"detail": "session limit"})

    service.before = reject
    with client_for(service) as client, pytest.raises(ArtifactTreeUploadError) as error:
        client.upload_artifact_directory(tmp_path)
    assert 1 <= len(attempted) <= 2
    assert all("429" in failure for failure in error.value.failures.values())
    assert error.value.upload_ids == {}
    assert service.trees == []


@pytest.mark.parametrize("operation", ["inline", "file", "tree"])
def test_cancellation_while_waiting_to_publish_does_not_send_request(
    tmp_path: Path, monkeypatch, operation: str
) -> None:
    from gfaas.errors import GfaasError

    service = UploadService()
    cancel = Event()
    source = tmp_path / "payload"
    source.write_bytes(b"abcd")
    with client_for(service) as client:
        original_request = client._upload_request

        def delayed_request(method, path, **kwargs):
            # Cancellation arrives after the caller's check, before its request gets a slot.
            if (
                (operation == "inline" and path == "/v1/artifacts")
                or (operation == "file" and path.endswith("/completion"))
                or (operation == "tree" and path == "/v1/artifact-trees")
            ):
                cancel.set()
            return original_request(method, path, **kwargs)

        monkeypatch.setattr(client, "_upload_request", delayed_request)
        with pytest.raises(GfaasError):
            if operation == "inline":
                client.upload_artifact(b"abcd", cancel=cancel)
            elif operation == "file":
                client.upload_artifact_file(source, cancel=cancel)
            else:
                client.upload_artifact_directory(tmp_path, cancel=cancel)
    assert cancel.is_set()
    assert service.trees == []
    assert len(service.artifacts) == (1 if operation == "tree" else 0)


@pytest.mark.parametrize("failure", ["timeout", "disconnect", "gateway"])
def test_completion_recovers_lost_response_without_repeating_publication(
    tmp_path: Path, failure: str
) -> None:
    service = UploadService()
    source = tmp_path / "weights"
    source.write_bytes(b"abcdefgh")
    progress = []
    completions = []

    def intercept(request):
        if request.url.path.endswith("/completion"):
            completions.append(request)
            assert progress[-1].phase == "publishing"
            assert progress[-1].completed_bytes == 8
            assert progress[-1].completed_files == 0
            assert progress[-1].artifact_id is None
            assert request.extensions["timeout"]["read"] == 600
            with service.lock:
                service.respond(request)
            if failure == "gateway":
                return httpx.Response(504)
            error = httpx.ReadTimeout if failure == "timeout" else httpx.ReadError
            raise error("response lost", request=request)
        return None

    service.before = intercept
    with client_for(service) as client:
        result = client.upload_artifact_file(source, progress=progress.append)
    assert len(completions) == 1
    assert len(service.puts) == 2
    assert [item.phase for item in progress][-3:] == ["publishing", "checking", "published"]
    assert progress[-1].artifact_id == result["id"]
    assert progress[-1].completed_files == 1
    assert all(item.upload_id == "0" for item in progress)


@pytest.mark.parametrize("lookup_fails", [False, True])
def test_completion_timeout_keeps_resume_id_without_claiming_publication(
    tmp_path: Path, lookup_fails: bool
) -> None:
    service = UploadService()
    source = tmp_path / "weights"
    source.write_bytes(b"abcdefgh")
    requests = []
    progress = []

    def intercept(request):
        requests.append((request.method, request.url.path))
        if request.url.path.endswith("/completion"):
            raise httpx.ReadTimeout("", request=request)
        if lookup_fails and request.method == "GET":
            return httpx.Response(503)
        return None

    service.before = intercept
    with client_for(service) as client:
        with pytest.raises(ArtifactUploadError, match="publication did not respond") as error:
            client.upload_artifact_file(source, progress=progress.append)
        assert error.value.phase == "publishing"
        assert error.value.upload_id == "0"
        assert "600s" in str(error.value)
        assert "GFAAS_UPLOAD_COMPLETION_TIMEOUT" in str(error.value)
        assert "Resume with upload_id='0'" in str(error.value)
        assert ("could not be confirmed" if lookup_fails else "may still be running") in str(
            error.value
        )
        assert requests[-1] == ("GET", "/v1/artifact-uploads/0")
        assert len([p for _, p in requests if p.endswith("/completion")]) == 1
        assert all(item.phase != "published" for item in progress)
        assert len(service.parts["0"]) == 2
        service.before = lambda request: None
        result = client.upload_artifact_file(source, upload_id="0", progress=progress.append)
    assert len(service.puts) == 2
    assert progress[-1].phase == "published"
    assert progress[-1].artifact_id == result["id"]
    assert progress[-1].reused_bytes == 8


def test_completion_rejection_preserves_reason_without_status_lookup(tmp_path: Path) -> None:
    service = UploadService()
    source = tmp_path / "weights"
    source.write_bytes(b"abcdefgh")

    def intercept(request):
        assert request.method != "GET"
        if request.url.path.endswith("/completion"):
            return httpx.Response(400, json={"detail": "digest mismatch"})
        return None

    service.before = intercept
    with client_for(service) as client, pytest.raises(ArtifactUploadError) as error:
        client.upload_artifact_file(source)
    assert error.value.phase == "publishing"
    assert "400: digest mismatch" in str(error.value)


@pytest.mark.parametrize("empty", [False, True])
def test_tree_publication_has_its_own_progress_after_child_files(
    tmp_path: Path, empty: bool
) -> None:
    service = UploadService()
    if not empty:
        (tmp_path / "data").write_bytes(b"abcdefgh")
    progress = []

    def intercept(request):
        if request.url.path == "/v1/artifact-trees":
            assert progress[-1].path == tmp_path
            assert progress[-1].phase == "publishing"
            assert progress[-1].artifact_id is None
        return None

    service.before = intercept
    with client_for(service) as client:
        tree = client.upload_artifact_directory(tmp_path, progress=progress.append)
    assert progress[-1].phase == "published"
    assert progress[-1].path == tmp_path
    assert progress[-1].artifact_id == tree["id"]
    assert progress[-1].completed_files == (0 if empty else 1)
    assert progress[-1].completed_bytes == (0 if empty else 8)
