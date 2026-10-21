"""Python client for the gfaas public resource API.

The decorator and ``Client.submit`` experience remains deliberately small;
packaging and cloudpickle are Python adapter details layered over typed
Environment, Function, Call, and Artifact resources.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from threading import BoundedSemaphore, Event, Lock
from typing import Any, Literal, Self

import httpx

from .artifacts import (
    TREE_ARTIFACT_MEDIA_TYPE,
    ArtifactCheckpoint,
    ArtifactOutput,
    ArtifactUploadPhase,
    ArtifactUploadProgress,
    collect_artifact_ids,
)
from .bundle import empty_bundle, package_single_file
from .compatibility import Compatibility
from .config import ClientConfig
from .errors import (
    ArtifactTreeUploadError,
    ArtifactUploadError,
    GfaasError,
    UnsupportedGpuPoolError,
)
from .events import TERMINAL_CALL_STATES, EventWatcher
from .image import Image
from .serialization import CLOUDPICKLE_CODEC, PICKLE_CODEC, decode_result_file, encode_args
from .stages import CallStage
from .transfers import RETRYABLE_STATUS, TRANSIENT_ERRORS, TransferManager, retry_request
from .uploads import check_cancelled, run_uploads

MAXIMUM_INLINE_ARTIFACT_BYTES = 1024 * 1024
MAXIMUM_UPLOAD_PART_BYTES = 64 * 1024 * 1024

MAX_ERROR_JOURNEY_EVENTS = 40
CLOUDPICKLE_MEDIA_TYPE = "application/vnd.gfunc.cloudpickle"
PICKLE_MEDIA_TYPE = "application/vnd.python.pickle"


@dataclass
class RemoteResult:
    """Handle returned by ``Client.submit``. Block with ``.wait()``."""

    client: Client
    job_id: str
    gpu_type: str
    environment_id: str | None = None
    function_id: str | None = None

    @property
    def call_id(self) -> str:
        """Public Call identity (``job_id`` is retained for SDK compatibility)."""
        return self.job_id

    def wait(self, *, timeout_s: float | None = None) -> Any:
        return self.client.wait_for_result(self.call_id, timeout_s=timeout_s)

    def status(self) -> dict[str, Any]:
        return self.client.get_call(self.call_id)

    def cancel(self, *, reason: str | None = None) -> dict[str, Any]:
        """Request cancellation and return the current public Call state."""
        return self.client.cancel_call(self.call_id, reason=reason)

    def iter_events(
        self,
        *,
        after: str | None = None,
        follow: bool = True,
        timeout_s: float | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield resumable lifecycle and output events for this call."""
        return self.client.iter_events(
            self.call_id,
            after=after,
            follow=follow,
            timeout_s=timeout_s,
        )

    def iter_logs(
        self,
        *,
        after: str | None = None,
        follow: bool = True,
        timeout_s: float | None = None,
    ) -> Iterator[tuple[str, str]]:
        """Yield ``(stream, text)`` chunks for stdout and stderr."""
        for event in self.iter_events(after=after, follow=follow, timeout_s=timeout_s):
            if event.get("type") in {"stdout", "stderr"}:
                yield str(event["type"]), str(event.get("stream_data", ""))

    def logs(self) -> dict[str, Any]:
        """Return retained stdout/stderr text and whether retention was truncated."""
        return self.client.get_call_logs(self.call_id)

    def artifacts(self) -> dict[str, Any]:
        """Return the Artifacts published by this Call."""
        return self.client.list_call_artifacts(self.call_id)


class Client:
    """Thin, thread-shareable REST client for the gfunc public API."""

    def __init__(self, config: ClientConfig | None = None) -> None:
        self.cfg = config or ClientConfig.from_env()
        self._upload_slots = BoundedSemaphore(self.cfg.upload_concurrency)
        self._upload_files = BoundedSemaphore(self.cfg.upload_concurrency)
        headers = {"X-API-Key": self.cfg.api_key} if self.cfg.api_key else {}
        self._http = httpx.Client(
            base_url=self.cfg.api_base,
            headers=headers,
            timeout=self.cfg.request_timeout_s,
        )

        self._transfers = TransferManager(
            lambda: self._http,
            f"{self.cfg.api_base}\0{self.cfg.api_key or ''}",
            self.cfg.transfer_cache_dir,
            self.cfg.download_concurrency,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        operation: str,
        **kwargs: Any,
    ) -> httpx.Response:
        response = self._http.request(method, path, **kwargs)
        if response.is_success:
            return response
        if response.status_code in {401, 403}:
            raise GfaasError(
                f"{operation} failed: authentication rejected by "
                f"{self.cfg.api_base} ({response.status_code})"
            )

        problem = _problem_body(response)
        if problem is not None and problem.get("code") == "unsupported_gpu_pool":
            compatibility = problem.get("compatibility")
            if isinstance(compatibility, dict):
                requested = compatibility.get("requested_gpu_pool")
                supported = compatibility.get("supported_gpu_pools")
                if (
                    isinstance(requested, str)
                    and isinstance(supported, list)
                    and all(isinstance(pool, str) for pool in supported)
                ):
                    raise UnsupportedGpuPoolError(requested, [str(pool) for pool in supported])

        detail = _response_detail(response, problem)
        suffix = f": {detail}" if detail else ""
        raise GfaasError(
            f"{operation} failed: {response.status_code}{suffix}", status_code=response.status_code
        )

    def _upload_request(
        self, method: str, path: str, *, cancel: Event | None = None, **kwargs: Any
    ) -> httpx.Response:
        # Metadata, completion, and inline uploads share the same request budget as parts.
        with self._upload_slots:
            check_cancelled(cancel)

            def action() -> httpx.Response:
                return self._request(method, path, **kwargs)

            return retry_request(action, cancel) if method in {"GET", "PUT"} else action()

    # --- public resources -------------------------------------------------

    def get_capabilities(self) -> dict[str, Any]:
        """Return configured GPU pools and their current availability."""
        return self._request(
            "GET",
            "/v1/capabilities",
            operation="capability lookup",
        ).json()

    def upload_artifact(
        self,
        data: bytes,
        *,
        kind: str = "other",
        media_type: str = "application/octet-stream",
        filename: str | None = None,
        ttl_seconds: int | None = None,
        upload_id: str | None = None,
        cancel: Event | None = None,
    ) -> dict[str, Any]:
        retention = _artifact_retention(ttl_seconds)
        check_cancelled(cancel)
        if len(data) > MAXIMUM_INLINE_ARTIFACT_BYTES or upload_id is not None:
            expected = {
                "kind": kind,
                "media_type": media_type,
                "filename": filename,
                "digest": _content_digest(data),
                "size_bytes": len(data),
                "retention": retention,
            }
            with self._upload_files:
                check_cancelled(cancel)
                session = self._artifact_upload_session(expected, upload_id)
                view = memoryview(data)
                return self._upload_artifact_parts(
                    session,
                    lambda offset, length: bytes(view[offset : offset + length]),
                    len(data),
                    cancel=cancel,
                )
        digest = base64.b64encode(hashlib.sha256(data).digest()).decode("ascii")
        headers = {
            "Content-Type": media_type,
            "Content-Digest": f"sha-256=:{digest}:",
            "X-Gfunc-Artifact-Kind": kind,
        }
        if ttl_seconds is None:
            headers["X-Gfunc-Retention"] = "keep"
        else:
            headers["X-Gfunc-Ttl-Seconds"] = str(ttl_seconds)
        if filename is not None:
            headers["X-Gfunc-Filename"] = filename
        response = self._upload_request(
            "POST",
            "/v1/artifacts",
            operation="artifact upload",
            cancel=cancel,
            content=data,
            headers=headers,
        )
        return response.json()

    def upload_artifact_file(
        self,
        path: Path | str,
        *,
        kind: str = "other",
        media_type: str = "application/octet-stream",
        filename: str | None = None,
        ttl_seconds: int | None = None,
        upload_id: str | None = None,
        progress: Callable[[ArtifactUploadProgress], None] | None = None,
        cancel: Event | None = None,
    ) -> dict[str, Any]:
        """Upload a file with resumable, bounded-memory transfer requests."""
        _artifact_retention(ttl_seconds)
        with self._upload_files:
            check_cancelled(cancel)
            return self._upload_artifact_file(
                path,
                kind=kind,
                media_type=media_type,
                filename=filename,
                ttl_seconds=ttl_seconds,
                upload_id=upload_id,
                progress=progress,
                cancel=cancel,
            )

    def _upload_artifact_file(
        self,
        path: Path | str,
        *,
        kind: str = "other",
        media_type: str = "application/octet-stream",
        filename: str | None = None,
        ttl_seconds: int | None = None,
        upload_id: str | None = None,
        progress: Callable[[ArtifactUploadProgress], None] | None = None,
        cancel: Event | None = None,
    ) -> dict[str, Any]:
        """Upload a file with resumable, bounded-memory transfer requests."""
        retention = _artifact_retention(ttl_seconds)
        source = Path(path)
        size_bytes, digest = _file_identity(source)
        expected: dict[str, Any] = {
            "kind": kind,
            "media_type": media_type,
            "filename": filename if filename is not None else source.name,
            "digest": digest,
            "size_bytes": size_bytes,
            "retention": retention,
        }
        session = self._artifact_upload_session(expected, upload_id)
        _upload_part_size(session)

        last_progress = (0, 0, 0)

        def report(
            completed: int,
            transferred: int,
            reused: int,
            phase: ArtifactUploadPhase = "transferring",
            *,
            artifact_id: str | None = None,
        ) -> None:
            nonlocal last_progress
            last_progress = completed, transferred, reused
            if progress is not None:
                progress(
                    ArtifactUploadProgress(
                        path=source,
                        completed_bytes=completed,
                        total_bytes=size_bytes,
                        transferred_bytes=transferred,
                        reused_bytes=reused,
                        completed_files=int(phase == "published"),
                        phase=phase,
                        upload_id=str(session["id"]),
                        artifact_id=artifact_id,
                    )
                )

        with source.open("rb") as stream:
            read_lock = Lock()

            def read_part(offset: int, length: int) -> bytes:
                with read_lock:
                    stream.seek(offset)
                    chunk = stream.read(length)
                    if os.fstat(stream.fileno()).st_size != size_bytes:
                        raise GfaasError("Artifact source changed size during upload")
                    return chunk

            artifact = self._upload_artifact_parts(session, read_part, size_bytes, report, cancel)
        report(*last_progress, "published", artifact_id=artifact["id"])
        return artifact

    def _artifact_upload_session(
        self,
        expected: dict[str, Any],
        upload_id: str | None,
    ) -> dict[str, Any]:
        key = "upload:" + json.dumps(expected, sort_keys=True, separators=(",", ":"))
        with self._transfers.workspace(key) as directory:
            receipt = directory / "session.json"
            automatic = upload_id is None
            if automatic and receipt.exists():
                upload_id = json.loads(receipt.read_bytes())["id"]
            session = None
            if upload_id is not None:
                try:
                    session = self._upload_request(
                        "GET",
                        f"/v1/artifact-uploads/{upload_id}",
                        operation="Artifact upload session lookup",
                    ).json()
                except GfaasError as error:
                    if not automatic or error.status_code not in {404, 410}:
                        raise
                if session is not None:
                    for field, value in expected.items():
                        if session.get(field) != value:
                            raise GfaasError(
                                f"Artifact upload session {upload_id} does not match {field}"
                            )
            if automatic and session is not None and session.get("artifact_id"):
                try:
                    published = self.get_artifact(session["artifact_id"])
                except GfaasError as error:
                    if error.status_code not in {404, 410}:
                        raise
                    session = None
                else:
                    if _artifact_expired(published):
                        session = None
            if session is None:
                session = self._upload_request(
                    "POST",
                    "/v1/artifact-uploads",
                    operation="Artifact upload session creation",
                    json=expected,
                ).json()
                temporary = receipt.with_suffix(".new")
                with temporary.open("w") as output:
                    json.dump({"id": session["id"]}, output)
                    output.flush()
                    os.fsync(output.fileno())
                temporary.replace(receipt)
            session["_recovery_key"] = key
            return session

    def _upload_artifact_parts(
        self,
        session: dict[str, Any],
        read_part: Callable[[int, int], bytes],
        size_bytes: int,
        progress: Callable[[int, int, int, ArtifactUploadPhase], None] | None = None,
        cancel: Event | None = None,
    ) -> dict[str, Any]:
        result = self._transfer_artifact_parts(session, read_part, size_bytes, progress, cancel)
        if key := session.get("_recovery_key"):
            with self._transfers.workspace(key) as directory:
                self._transfers.finish(directory)
        return result

    def _transfer_artifact_parts(
        self,
        session: dict[str, Any],
        read_part: Callable[[int, int], bytes],
        size_bytes: int,
        progress: Callable[[int, int, int, ArtifactUploadPhase], None] | None = None,
        cancel: Event | None = None,
    ) -> dict[str, Any]:
        session_id = str(session["id"])
        part_size = _upload_part_size(session)
        uploaded: dict[int, dict[str, Any]] = {}
        for part in session.get("parts", []):
            number = int(part["part_number"])
            if (
                number < 0
                or number >= (size_bytes + part_size - 1) // part_size
                or number in uploaded
            ):
                raise GfaasError(
                    "Artifact upload session contains invalid or duplicate part numbers"
                )
            uploaded[number] = part
        completed_bytes = transferred_bytes = reused_bytes = 0
        stopped = Event()
        transfer_errors: list[BaseException] = []

        def transfer(part_number: int) -> tuple[int, bool]:
            # Acquire before reading: waiting workers hold no part buffers.
            with self._upload_slots:
                check_cancelled(cancel)
                check_cancelled(stopped)
                try:
                    offset = part_number * part_size
                    expected_size = min(part_size, size_bytes - offset)
                    chunk = read_part(offset, expected_size)
                    if len(chunk) != expected_size:
                        raise GfaasError("Artifact source changed size during upload")
                    chunk_digest = _content_digest(chunk)
                    existing = uploaded.get(part_number)
                    if existing is None:

                        def send_part() -> httpx.Response:
                            check_cancelled(stopped)
                            try:
                                return self._request(
                                    "PUT",
                                    f"/v1/artifact-uploads/{session_id}/parts/{part_number}",
                                    operation="Artifact upload part transfer",
                                    content=chunk,
                                    headers={
                                        "Content-Type": "application/octet-stream",
                                        "Content-Digest": chunk_digest,
                                    },
                                )
                            except (GfaasError, *TRANSIENT_ERRORS) as error:
                                if (
                                    isinstance(error, GfaasError)
                                    and error.status_code not in RETRYABLE_STATUS
                                ):
                                    raise
                                # A lost response does not mean the PUT was rejected. Check its receipt
                                # before sending this part again. This lookup uses the held request slot.
                                status = self._request(
                                    "GET",
                                    f"/v1/artifact-uploads/{session_id}",
                                    operation="Artifact part reconciliation",
                                )
                                for part in status.json().get("parts", []):
                                    if part.get("part_number") == part_number:
                                        if (
                                            part.get("size_bytes") != len(chunk)
                                            or part.get("digest") != chunk_digest
                                        ):
                                            raise GfaasError(
                                                "Artifact upload receipt conflicts with local bytes"
                                            ) from error
                                        return status
                                raise

                        retry_request(send_part, cancel)
                    elif (
                        int(existing.get("size_bytes", -1)) != len(chunk)
                        or existing.get("digest") != chunk_digest
                    ):
                        raise GfaasError(
                            f"Artifact upload session {session_id} has a conflicting part {part_number}"
                        )
                    return len(chunk), existing is not None
                except BaseException as error:
                    transfer_errors.append(error)
                    stopped.set()
                    raise
                finally:
                    # Exceptions retain frames. Do not retain part buffers in an error traceback.
                    chunk = b""

        def completed(result: tuple[int, bool]) -> None:
            nonlocal completed_bytes, transferred_bytes, reused_bytes
            size, reused = result
            completed_bytes += size
            reused_bytes += size if reused else 0
            transferred_bytes += 0 if reused else size
            if progress is not None:
                progress(completed_bytes, transferred_bytes, reused_bytes, "transferring")

        phase = "transferring"
        try:
            check_cancelled(cancel)
            if session.get("artifact_id"):
                phase = "checking"
                if progress is not None:
                    progress(size_bytes, 0, size_bytes, "checking")
                artifact = self._upload_request(
                    "GET", f"/v1/artifacts/{session['artifact_id']}", operation="artifact lookup"
                ).json()
                return artifact
            run_uploads(
                range((size_bytes + part_size - 1) // part_size),
                transfer,
                completed,
                self.cfg.upload_concurrency,
            )
            check_cancelled(cancel)
        except (GfaasError, httpx.HTTPError, OSError) as error:
            cause = transfer_errors[0] if transfer_errors else error
            raise ArtifactUploadError(session_id, str(cause), phase=phase) from cause

        def report_phase(phase: ArtifactUploadPhase) -> None:
            if progress is not None:
                progress(completed_bytes, transferred_bytes, reused_bytes, phase)

        report_phase("publishing")
        return self._complete_artifact_upload(session_id, report_phase, cancel)

    def _complete_artifact_upload(
        self,
        session_id: str,
        report_phase: Callable[[ArtifactUploadPhase], None],
        cancel: Event | None,
    ) -> dict[str, Any]:
        timeout = self.cfg.upload_completion_timeout_s
        try:
            return self._upload_request(
                "POST",
                f"/v1/artifact-uploads/{session_id}/completion",
                operation="Artifact upload completion",
                cancel=cancel,
                timeout=httpx.Timeout(self.cfg.request_timeout_s, read=timeout),
            ).json()
        except (GfaasError, httpx.HTTPError) as error:
            ambiguous = isinstance(error, httpx.TransportError) or (
                isinstance(error, GfaasError)
                and error.status_code is not None
                and error.status_code >= 500
            )
            if isinstance(error, httpx.ReadTimeout):
                detail = (
                    f"All parts reached the service, but publication did not respond within {timeout:g}s. "
                    "Set GFAAS_UPLOAD_COMPLETION_TIMEOUT to allow a longer publication wait."
                )
            elif isinstance(error, httpx.TransportError):
                detail = (
                    "All parts reached the service, but the publication request failed without a response "
                    f"({type(error).__name__})."
                )
            else:
                detail = f"All parts reached the service. Publication request failed: {error}."
            if ambiguous:
                report_phase("checking")
                try:
                    session = self._upload_request(
                        "GET",
                        f"/v1/artifact-uploads/{session_id}",
                        operation="Artifact upload status check",
                    ).json()
                    if session.get("artifact_id"):
                        return self._upload_request(
                            "GET",
                            f"/v1/artifacts/{session['artifact_id']}",
                            operation="artifact lookup",
                        ).json()
                    detail += " The session has no published Artifact yet; publication may still be running."
                except (GfaasError, httpx.HTTPError) as lookup_error:
                    detail += (
                        f" Publication could not be confirmed ({type(lookup_error).__name__})."
                    )
            raise ArtifactUploadError(session_id, detail, phase="publishing") from error

    def create_artifact_file_upload(
        self,
        path: Path | str,
        *,
        kind: str = "other",
        media_type: str = "application/octet-stream",
        filename: str | None = None,
        ttl_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Create a durable upload session before a resumable file transfer."""
        retention = _artifact_retention(ttl_seconds)
        with self._upload_files:
            source = Path(path)
            size_bytes, digest = _file_identity(source)
            return self._upload_request(
                "POST",
                "/v1/artifact-uploads",
                operation="Artifact upload session creation",
                json={
                    "kind": kind,
                    "media_type": media_type,
                    "filename": filename if filename is not None else source.name,
                    "digest": digest,
                    "size_bytes": size_bytes,
                    "retention": retention,
                },
            ).json()

    def upload_artifact_directory(
        self,
        path: Path | str,
        *,
        kind: str = "input",
        filename: str | None = None,
        ttl_seconds: int | None = None,
        progress: Callable[[ArtifactUploadProgress], None] | None = None,
        cancel: Event | None = None,
    ) -> dict[str, Any]:
        """Upload a directory, retaining completed children if publication fails."""
        key = "tree-upload:" + json.dumps([str(Path(path).resolve()), kind, filename, ttl_seconds])
        with self._transfers.workspace(key) as receipts:
            result = self._upload_artifact_directory(
                path,
                kind=kind,
                filename=filename,
                ttl_seconds=ttl_seconds,
                progress=progress,
                cancel=cancel,
                receipts=receipts,
            )
            self._transfers.finish(receipts)
            return result

    def _upload_artifact_directory(
        self,
        path: Path | str,
        *,
        kind: str = "input",
        filename: str | None = None,
        ttl_seconds: int | None = None,
        progress: Callable[[ArtifactUploadProgress], None] | None = None,
        cancel: Event | None = None,
        receipts: Path,
    ) -> dict[str, Any]:
        """Upload a directory as one immutable tree Artifact."""
        retention = _artifact_retention(ttl_seconds)
        source = Path(path)
        metadata = source.lstat()
        if source.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"Artifact tree source is not a directory: {source}")
        local_entries: list[tuple[str, str, Path, os.stat_result]] = []
        for current, directory_names, file_names in os.walk(source, followlinks=False):
            directory_names.sort()
            file_names.sort()
            current_path = Path(current)
            for name in directory_names:
                child = current_path / name
                child_metadata = child.lstat()
                if child.is_symlink():
                    raise ValueError(f"Artifact trees cannot contain a symbolic link: {child}")
                if not stat.S_ISDIR(child_metadata.st_mode):
                    raise ValueError(f"Artifact tree entry is not a directory: {child}")
                relative = child.relative_to(source).as_posix()
                _validate_local_tree_path(relative)
                local_entries.append((relative, "directory", child, child_metadata))
            for name in file_names:
                child = current_path / name
                child_metadata = child.lstat()
                if child.is_symlink():
                    raise ValueError(f"Artifact trees cannot contain a symbolic link: {child}")
                if not stat.S_ISREG(child_metadata.st_mode):
                    raise ValueError(f"Artifact trees can contain only regular files: {child}")
                if child_metadata.st_nlink != 1:
                    raise ValueError(f"Artifact trees cannot contain hard-linked files: {child}")
                relative = child.relative_to(source).as_posix()
                _validate_local_tree_path(relative)
                local_entries.append((relative, "file", child, child_metadata))
        if len(local_entries) > 1_000_000:
            raise ValueError("Artifact tree contains more than 1000000 entries")
        local_entries.sort(key=lambda entry: entry[0])
        file_entries = [entry for entry in local_entries if entry[1] == "file"]
        total_bytes = sum(entry[3].st_size for entry in file_entries)
        total_files = len(file_entries)
        progress_lock = Lock()
        file_progress: dict[Path, ArtifactUploadProgress] = {}
        totals = [0, 0, 0, 0]
        uploaded_files: dict[str, dict[str, Any]] = {}
        failed_uploads: dict[str, str] = {}
        failures: dict[str, str] = {}
        stopped = Event()

        def upload_file(entry: tuple[str, str, Path, os.stat_result]) -> tuple[str, dict[str, Any]]:
            relative, _, child, _ = entry
            check_cancelled(cancel)
            check_cancelled(stopped)

            def report_file(item: ArtifactUploadProgress) -> None:
                # Callbacks for a directory are serialized, including aggregate counter updates.
                with progress_lock:
                    previous = file_progress.get(child)
                    for index, field in enumerate(
                        ("completed_bytes", "transferred_bytes", "reused_bytes", "completed_files")
                    ):
                        totals[index] += getattr(item, field) - (
                            getattr(previous, field) if previous else 0
                        )
                    file_progress[child] = item
                    if progress is not None:
                        progress(
                            ArtifactUploadProgress(
                                path=child,
                                completed_bytes=totals[0],
                                total_bytes=total_bytes,
                                transferred_bytes=totals[1],
                                reused_bytes=totals[2],
                                completed_files=totals[3],
                                total_files=total_files,
                                phase=item.phase,
                                upload_id=item.upload_id,
                                artifact_id=item.artifact_id,
                            )
                        )

            receipt = receipts / (hashlib.sha256(relative.encode()).hexdigest() + ".json")
            size, digest = _file_identity(child)
            if receipt.exists():
                saved = json.loads(receipt.read_bytes())
                if saved["size"] == size and saved["digest"] == digest:
                    try:
                        artifact = self.get_artifact(saved["artifact"]["id"])
                    except GfaasError as error:
                        if error.status_code not in {404, 410}:
                            raise
                    else:
                        if not _artifact_expired(artifact):
                            report_file(
                                ArtifactUploadProgress(
                                    path=child,
                                    completed_bytes=size,
                                    total_bytes=size,
                                    transferred_bytes=0,
                                    reused_bytes=size,
                                    completed_files=1,
                                    phase="published",
                                    artifact_id=artifact["id"],
                                )
                            )
                            return relative, artifact

            try:
                artifact = self.upload_artifact_file(
                    child,
                    kind=kind,
                    filename=child.name,
                    ttl_seconds=ttl_seconds,
                    progress=report_file if progress is not None else None,
                    cancel=cancel,
                )
                temporary = receipt.with_suffix(".new")
                with temporary.open("w") as output:
                    json.dump({"size": size, "digest": digest, "artifact": artifact}, output)
                    output.flush()
                    os.fsync(output.fileno())
                temporary.replace(receipt)
                return relative, artifact
            except Exception as error:
                stopped.set()
                with progress_lock:
                    failures[relative] = str(error)
                    if isinstance(error, ArtifactUploadError):
                        failed_uploads[relative] = error.upload_id
                raise

        def file_completed(result: tuple[str, dict[str, Any]]) -> None:
            relative, artifact = result
            uploaded_files[relative] = artifact

        try:
            check_cancelled(cancel)
            run_uploads(file_entries, upload_file, file_completed, self.cfg.upload_concurrency)
            check_cancelled(cancel)
        except Exception as error:
            raise ArtifactTreeUploadError(
                list(dict.fromkeys(uploaded_files[path]["id"] for path in sorted(uploaded_files))),
                upload_ids=failed_uploads,
                failures=failures,
            ) from error
        entries: list[dict[str, Any]] = []
        for relative, entry_type, _, child_metadata in local_entries:
            if entry_type == "directory":
                entries.append({"path": relative, "type": "directory"})
            else:
                entries.append(
                    {
                        "path": relative,
                        "type": "file",
                        "artifact_id": uploaded_files[relative]["id"],
                        "executable": bool(child_metadata.st_mode & 0o111),
                    }
                )
        child_artifact_ids = list(
            dict.fromkeys(entry["artifact_id"] for entry in entries if entry["type"] == "file")
        )

        def report_tree(phase: ArtifactUploadPhase, artifact_id: str | None = None) -> None:
            if progress is not None:
                progress(
                    ArtifactUploadProgress(
                        path=source,
                        completed_bytes=totals[0],
                        total_bytes=total_bytes,
                        transferred_bytes=totals[1],
                        reused_bytes=totals[2],
                        completed_files=totals[3],
                        total_files=total_files,
                        phase=phase,
                        artifact_id=artifact_id,
                    )
                )

        try:
            report_tree("publishing")
            tree = self._upload_request(
                "POST",
                "/v1/artifact-trees",
                operation="Artifact tree creation",
                cancel=cancel,
                json={
                    "kind": kind,
                    "filename": filename if filename is not None else source.name,
                    "entries": entries,
                    "retention": retention,
                },
            ).json()
        except Exception as error:
            raise ArtifactTreeUploadError(child_artifact_ids) from error
        tree["child_artifact_ids"] = child_artifact_ids
        report_tree("published", tree["id"])
        return tree

    def preview_artifact_cleanup(
        self,
        artifact_ids: Sequence[str],
        *,
        maximum_artifacts: int = 100,
        maximum_bytes: int = 64 * 1024**3,
    ) -> dict[str, Any]:
        """Preview deletion of selected Artifacts without changing their references or bytes."""
        return self._artifact_cleanup(artifact_ids, False, maximum_artifacts, maximum_bytes)

    def cleanup_artifacts(
        self,
        artifact_ids: Sequence[str],
        *,
        maximum_artifacts: int = 100,
        maximum_bytes: int = 64 * 1024**3,
    ) -> dict[str, Any]:
        """Delete a bounded selection and report blockers or pending storage cleanup per ID."""
        return self._artifact_cleanup(artifact_ids, True, maximum_artifacts, maximum_bytes)

    def _artifact_cleanup(
        self,
        artifact_ids: Sequence[str],
        apply: bool,
        maximum_artifacts: int,
        maximum_bytes: int,
    ) -> dict[str, Any]:
        if isinstance(artifact_ids, (str, bytes)) or not 1 <= len(artifact_ids) <= 1000:
            raise ValueError("select 1 to 1000 Artifact IDs")
        if any(
            not isinstance(id, str) or not re.fullmatch(r"art_[A-Za-z0-9_-]{1,128}", id)
            for id in artifact_ids
        ):
            raise ValueError("invalid Artifact ID in cleanup selection")
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("Artifact IDs must be unique")
        if (
            isinstance(maximum_artifacts, bool)
            or not isinstance(maximum_artifacts, int)
            or not 1 <= maximum_artifacts <= 1000
        ):
            raise ValueError("maximum_artifacts must be an integer from 1 to 1000")
        if (
            isinstance(maximum_bytes, bool)
            or not isinstance(maximum_bytes, int)
            or not 1 <= maximum_bytes <= 2**63 - 1
        ):
            raise ValueError(
                "maximum_bytes must be a positive integer no greater than 9223372036854775807"
            )
        return self._request(
            "POST",
            "/v1/artifact-cleanup",
            operation="Artifact cleanup" if apply else "Artifact cleanup preview",
            json={
                "artifact_ids": list(artifact_ids),
                "apply": apply,
                "maximum_artifacts": maximum_artifacts,
                "maximum_bytes": maximum_bytes,
            },
        ).json()

    def set_artifact_retention(
        self, artifact_id: str, *, ttl_seconds: int | None = None
    ) -> dict[str, Any]:
        """Keep an Artifact, or make it eligible for collection after a duration."""
        return self._request(
            "PUT",
            f"/v1/artifacts/{artifact_id}/retention",
            operation="Artifact retention update",
            json=_artifact_retention(ttl_seconds),
        ).json()

    def get_artifact(self, artifact_id: str) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/v1/artifacts/{artifact_id}",
            operation="artifact lookup",
        ).json()

    def list_artifacts(
        self,
        *,
        after: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Return one page of owner-scoped Artifact metadata.

        Pass the ``next_cursor`` from a page as ``after`` to request the next one, or use
        :meth:`iter_artifacts` to walk every page.
        """
        params: dict[str, Any] = {"limit": limit}
        if after is not None:
            params["after"] = after
        return self._request(
            "GET",
            "/v1/artifacts",
            operation="artifact listing",
            params=params,
        ).json()

    def iter_artifacts(self, *, limit: int = 100) -> Iterator[dict[str, Any]]:
        """Yield every owner-scoped Artifact, following the cursor across pages."""
        cursor: str | None = None
        while True:
            page = self.list_artifacts(after=cursor, limit=limit)
            yield from page.get("items", [])
            cursor = page.get("next_cursor")
            if not cursor:
                return

    def download_artifact(self, artifact_id: str) -> tuple[bytes, str]:
        with tempfile.TemporaryDirectory(prefix="vfunc-download-") as directory:
            path, media = self.download_artifact_file(artifact_id, Path(directory) / "body")
            return path.read_bytes(), media

    def download_artifact_file(
        self,
        artifact_id: str,
        destination: Path | str,
    ) -> tuple[Path, str]:
        """Download an Artifact to a new file without buffering it in memory."""
        return self._transfers.file(artifact_id, Path(destination))

    def download_artifact_directory(
        self,
        artifact_id: str,
        destination: Path | str,
    ) -> Path:
        """Download a tree Artifact into a new directory."""
        metadata = self.get_artifact(artifact_id)
        if (
            metadata.get("layout") != "tree"
            or metadata.get("media_type") != TREE_ARTIFACT_MEDIA_TYPE
        ):
            raise GfaasError(f"Artifact {artifact_id} is not a tree Artifact")
        target = Path(destination)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Artifact tree destination already exists: {target}")
        with self._transfers.workspace(f"tree:{artifact_id}") as workspace:
            manifest_path = workspace / "manifest.json"
            manifest_path.unlink(missing_ok=True)
            _, media_type = self.download_artifact_file(artifact_id, manifest_path)
            if media_type != TREE_ARTIFACT_MEDIA_TYPE:
                raise GfaasError("Artifact tree download returned the wrong media type")
            try:
                manifest = json.loads(manifest_path.read_bytes())
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise GfaasError("Artifact tree manifest is invalid JSON") from error
            entries = _validated_tree_entries(manifest)
            contents = workspace / "contents"
            contents.mkdir(exist_ok=True, mode=0o700)
            files = []
            for entry in entries:
                path = contents.joinpath(*Path(entry["path"]).parts)
                if path.is_symlink():
                    raise GfaasError("Artifact tree cache contains a symbolic link")
                if entry["type"] == "directory":
                    path.mkdir(exist_ok=True)
                else:
                    files.append(entry)

            def receive(entry: dict[str, Any]) -> None:
                path = contents.joinpath(*Path(entry["path"]).parts)
                if path.exists():
                    size, digest = _file_identity(path)
                    if size == entry["size_bytes"] and digest == entry["digest"]:
                        return
                    path.unlink()
                self.download_artifact_file(str(entry["artifact_id"]), path)
                size, digest = _file_identity(path)
                if size != entry["size_bytes"] or digest != entry["digest"]:
                    path.unlink()
                    raise GfaasError("Artifact tree entry failed manifest verification")

            run_uploads(files, receive, lambda _: None, self.cfg.download_concurrency)
            temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.part")
            try:
                shutil.copytree(contents, temporary)
                for entry in entries:
                    path = temporary.joinpath(*Path(entry["path"]).parts)
                    if entry["type"] == "file":
                        path.chmod(int(entry["mode"]))
                for entry in reversed(entries):
                    if entry["type"] == "directory":
                        temporary.joinpath(*Path(entry["path"]).parts).chmod(0o555)
                temporary.chmod(0o555)
                if target.exists() or target.is_symlink():
                    raise FileExistsError(target)
                temporary.rename(target)
            except BaseException:
                if temporary.exists():
                    _remove_tree(temporary)
                raise
            self._transfers.finish(workspace)
            return target

    def delete_artifact(self, artifact_id: str) -> None:
        self._request(
            "DELETE",
            f"/v1/artifacts/{artifact_id}",
            operation="artifact deletion",
        )

    def preview_retirement(
        self,
        resource_kind: Literal["call", "function", "environment"],
        resource_id: str,
    ) -> dict[str, Any]:
        """Show retirement blockers and Artifact references without changing the resource."""
        return self._resource_retirement("GET", resource_kind, resource_id)

    def retire_resource(
        self,
        resource_kind: Literal["call", "function", "environment"],
        resource_id: str,
    ) -> dict[str, Any]:
        """Release live references while keeping history and independently retained Artifacts.

        Retirement is permanent. Active Calls and resources with active consumers are rejected.
        Repeat the request to retrieve its original receipt after an ambiguous response.
        """
        return self._resource_retirement("POST", resource_kind, resource_id)

    def _resource_retirement(
        self, method: str, resource_kind: str, resource_id: str
    ) -> dict[str, Any]:
        prefixes = {"call": "call_", "function": "fn_", "environment": "env_"}
        prefix = prefixes.get(resource_kind)
        if prefix is None or not re.fullmatch(re.escape(prefix) + r"[A-Za-z0-9_-]+", resource_id):
            raise ValueError(
                "resource kind and ID must identify one Call, Function, or Environment"
            )
        return self._request(
            method,
            f"/v1/{resource_kind}s/{resource_id}/retirement",
            operation=f"{resource_kind} retirement",
        ).json()

    def create_environment(
        self,
        definition: dict[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        return self._post_resource(
            "/v1/environments", definition, "environment creation", idempotency_key
        )

    def get_environment(self, environment_id: str) -> dict[str, Any]:
        return self._request(
            "GET", f"/v1/environments/{environment_id}", operation="environment lookup"
        ).json()

    def create_function(
        self,
        definition: dict[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        return self._post_resource(
            "/v1/functions", definition, "function creation", idempotency_key
        )

    def get_function(self, function_id: str) -> dict[str, Any]:
        return self._request(
            "GET", f"/v1/functions/{function_id}", operation="function lookup"
        ).json()

    def create_call(
        self,
        request: dict[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        key = idempotency_key or uuid.uuid4().hex
        return self._post_resource("/v1/calls", request, "call creation", key)

    def get_call(self, call_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/calls/{call_id}", operation="call status").json()

    def cancel_call(self, call_id: str, *, reason: str | None = None) -> dict[str, Any]:
        body = {"reason": reason} if reason is not None else {}
        return self._request(
            "POST",
            f"/v1/calls/{call_id}/cancellation",
            operation="call cancellation",
            json=body,
        ).json()

    def get_call_result(self, call_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/calls/{call_id}/result", operation="call result").json()

    def list_attempts(self, call_id: str) -> dict[str, Any]:
        return self._request(
            "GET", f"/v1/calls/{call_id}/attempts", operation="attempt listing"
        ).json()

    def list_events(
        self,
        call_id: str,
        *,
        after: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit, "follow": "false"}
        if after is not None:
            params["after"] = after
        return self._request(
            "GET",
            f"/v1/calls/{call_id}/events",
            operation="event listing",
            params=params,
        ).json()

    def iter_events(
        self,
        call_id: str,
        *,
        after: str | None = None,
        follow: bool = True,
        timeout_s: float | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield durable events, with bounded recovery from temporary disconnects."""
        yield from EventWatcher(
            self._http,
            call_id,
            after=after,
            follow=follow,
            timeout_s=timeout_s,
            request_timeout_s=self.cfg.request_timeout_s,
            poll_interval_s=self.cfg.poll_interval_s,
        )

    def get_call_logs(self, call_id: str) -> dict[str, Any]:
        """Collect all retained output events for a completed or running Call."""
        output: dict[str, list[str]] = {"stdout": [], "stderr": []}
        after: str | None = None
        truncated = False
        while True:
            page = self.list_events(call_id, after=after, limit=1000)
            for event in page.get("items", []):
                event_type = event.get("type")
                if event_type in output:
                    output[event_type].append(str(event.get("stream_data", "")))
                elif event_type == "retention.truncated":
                    truncated = True
            after = page.get("next_cursor")
            if after is None:
                break
        return {
            "stdout": "".join(output["stdout"]),
            "stderr": "".join(output["stderr"]),
            "truncated": truncated,
        }

    def list_call_artifacts(self, call_id: str) -> dict[str, Any]:
        return self._request(
            "GET", f"/v1/calls/{call_id}/artifacts", operation="call artifact listing"
        ).json()

    def _post_resource(
        self,
        path: str,
        body: dict[str, Any],
        operation: str,
        idempotency_key: str | None,
    ) -> dict[str, Any]:
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else None
        return self._request("POST", path, operation=operation, json=body, headers=headers).json()

    # --- Python convenience ----------------------------------------------

    def submit(
        self,
        *,
        image: Image | str,
        compatibility: Compatibility | None = None,
        function: tuple[str, str] | Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        gpu: str | None = None,
        gpu_count: int | None = None,
        gpu_type: str = "any",
        app_name: str = "gfaas",
        timeout_s: int = 300,
        capacity_wait_s: int | None = None,
        cpu_millicores: int | None = None,
        memory_bytes: int | None = None,
        ephemeral_storage_bytes: int | None = None,
        shared_memory_bytes: int | None = None,
        max_log_bytes: int | None = None,
        max_output_bytes: int | None = None,
        env: dict[str, str] | None = None,
        source_file: Path | str | None = None,
        idempotency_key: str | None = None,
        qualification_image_digest: str | None = None,
        outputs: tuple[ArtifactOutput | ArtifactCheckpoint, ...] = (),
        stages: tuple[CallStage, ...] = (),
    ) -> RemoteResult:
        """Package a Python callable and create an asynchronous Call."""
        image_name = _image_name(image)
        if _image_spec(image) is not None:
            raise GfaasError(
                "container build images are not enabled by the current gfaas service; "
                "use a registered image"
            )
        if qualification_image_digest is not None and not re.fullmatch(
            r"sha256:[0-9a-f]{64}", qualification_image_digest
        ):
            raise GfaasError("qualification_image_digest must be a lowercase SHA-256 digest")
        remote_source = _image_remote_source(image)
        if (
            qualification_image_digest is not None
            and remote_source is not None
            and remote_source.get("image_digest") != qualification_image_digest
        ):
            raise GfaasError(
                "qualification_image_digest does not match the Image remote descriptor"
            )
        variants = image.registry_variants if isinstance(image, Image) else None
        requirements = compatibility.request() if compatibility is not None else {}
        if variants is not None and compatibility is not None:
            required = compatibility.cpu_architecture
            if required is not None and required not in variants:
                raise GfaasError(f"image has no variant for required CPU architecture {required}")
        if requirements or variants is not None:
            supported = self.get_capabilities().get("environment_compatibility")
            if not isinstance(supported, dict):
                raise GfaasError("service does not advertise Environment hardware requirements")
            if variants is not None and supported.get("image_variants") is not True:
                raise GfaasError("service does not support native image variants")
            if requirements.get("cpu_architecture") is not None and requirements[
                "cpu_architecture"
            ] not in supported.get("cpu_architectures", []):
                raise GfaasError("service does not support the requested CPU architecture")
            if (
                requirements.get("gpu_compute_capabilities")
                and supported.get("gpu_compute_capabilities") != "exact"
            ):
                raise GfaasError(
                    "service does not support exact GPU compute capability requirements"
                )
        gpu_request = _gpu_request(gpu, gpu_count, gpu_type)
        kwargs = dict(kwargs or {})
        if isinstance(function, tuple) and source_file is None:
            module_name, qualname = function
            bundle = empty_bundle()
            filename = None
        elif isinstance(function, tuple):
            assert source_file is not None
            file = Path(source_file)
            bundle = package_single_file(file)
            if function[0] != bundle.module_name:
                raise GfaasError(
                    f"callable module {function[0]!r} does not match source module "
                    f"{bundle.module_name!r}"
                )
            module_name = bundle.module_name
            qualname = function[1]
            filename = file.name
        else:
            file = Path(source_file) if source_file else Path(_resolve_source_file(function))
            bundle = package_single_file(file)
            module_name = bundle.module_name
            qualname = function.__name__
            filename = file.name

        source = self.upload_artifact(
            bundle.data,
            kind="source",
            media_type="application/gzip",
            filename=filename,
        )
        environment_source: dict[str, Any] = {
            "kind": "registered_image",
            "name": qualification_image_digest or image_name,
        }
        if remote_source is not None:
            environment_source["remote"] = remote_source
        if variants is not None:
            environment_source = {"kind": "registered_images", "images": dict(variants)}
        environment_definition = {
            "name": image_name,
            "source": environment_source,
            "variables": dict(env or {}),
        }
        if requirements:
            environment_definition["compatibility"] = requirements
        environment = self.create_environment(environment_definition)
        resources = {
            "gpu": gpu_request,
            "timeout_seconds": timeout_s,
        }
        if cpu_millicores is not None:
            resources["cpu_millicores"] = cpu_millicores
        if memory_bytes is not None:
            resources["memory_bytes"] = memory_bytes
        if ephemeral_storage_bytes is not None:
            resources["ephemeral_storage_bytes"] = ephemeral_storage_bytes
        if shared_memory_bytes is not None:
            resources["shared_memory_bytes"] = shared_memory_bytes
        if max_log_bytes is not None:
            resources["max_log_bytes"] = max_log_bytes
        if max_output_bytes is not None:
            resources["max_output_bytes"] = max_output_bytes
        function_resource = self.create_function(
            {
                "name": app_name,
                "environment_id": environment["id"],
                "executable": {
                    "kind": "python_callable",
                    "module": module_name,
                    "qualname": qualname,
                    "source_artifact_id": source["id"],
                },
                "default_resources": resources,
            }
        )

        call_request: dict[str, Any] = {"function_id": function_resource["id"]}
        if qualification_image_digest is not None:
            call_request["qualification"] = {"image_digest": qualification_image_digest}
        if capacity_wait_s is not None:
            call_request["capacity_wait_seconds"] = capacity_wait_s
        if outputs:
            call_request["outputs"] = [output.request() for output in outputs]
        if stages:
            call_request["stages"] = [stage.request(default_module=module_name) for stage in stages]
        artifact_ids = collect_artifact_ids(args, kwargs)
        if artifact_ids:
            call_request["artifacts"] = [
                {"artifact_id": artifact_id} for artifact_id in artifact_ids
            ]
        if args or kwargs:
            payload, _codec = encode_args(args, kwargs, codec=CLOUDPICKLE_CODEC)
            input_artifact = self.upload_artifact(
                payload,
                kind="input",
                media_type=CLOUDPICKLE_MEDIA_TYPE,
            )
            call_request["input"] = {
                "storage": "artifact",
                "media_type": CLOUDPICKLE_MEDIA_TYPE,
                "artifact_id": input_artifact["id"],
            }
        call = self.create_call(call_request, idempotency_key=idempotency_key)
        return RemoteResult(
            client=self,
            job_id=call["id"],
            gpu_type=gpu_type,
            environment_id=environment["id"],
            function_id=function_resource["id"],
        )

    def wait_for_result(self, call_id: str, *, timeout_s: float | None = None) -> Any:
        deadline = time.monotonic() + timeout_s if timeout_s is not None else None
        while True:
            call = self.get_call(call_id)
            state = call.get("state", "queued")
            if state in TERMINAL_CALL_STATES:
                result = self.get_call_result(call_id)
                if result.get("outcome") != "succeeded":
                    raise GfaasError(_result_error(result, state))
                output = result.get("output")
                if output is None:
                    return None
                if output.get("storage") == "inline":
                    return output.get("value")
                with tempfile.TemporaryDirectory(prefix="vfunc-result-") as directory:
                    path, media_type = self.download_artifact_file(
                        output["artifact_id"],
                        Path(directory) / "result",
                    )
                    declared_type = output.get("media_type") or media_type
                    with path.open("rb") as source:
                        if declared_type == "application/json":
                            return json.load(source)
                        if declared_type == CLOUDPICKLE_MEDIA_TYPE:
                            codec = CLOUDPICKLE_CODEC
                        elif declared_type in {PICKLE_MEDIA_TYPE, "application/octet-stream"}:
                            codec = PICKLE_CODEC
                        else:
                            codec = declared_type
                        return decode_result_file(source, codec=codec)
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"call {call_id} not finished within {timeout_s}s")
            time.sleep(self.cfg.poll_interval_s)


def _artifact_expired(artifact: dict[str, Any]) -> bool:
    expiry = artifact.get("expires_at")
    return (
        expiry is not None
        and datetime.fromisoformat(str(expiry)).timestamp() <= datetime.now(UTC).timestamp()
    )


def _upload_part_size(session: dict[str, Any]) -> int:
    size = int(session["part_size_bytes"])
    if not 0 < size <= MAXIMUM_UPLOAD_PART_BYTES:
        raise GfaasError("Artifact upload part size must be between 1 byte and 64 MiB")
    return size


def _problem_body(response: httpx.Response) -> dict[str, Any] | None:
    try:
        body = response.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


def _response_detail(
    response: httpx.Response,
    body: dict[str, Any] | None = None,
) -> str:
    if body is None:
        body = _problem_body(response)
    if body is not None:
        for field in ("detail", "message", "title"):
            value = body.get(field)
            if isinstance(value, str) and value:
                return value[:1000]
    return response.text.strip()[:1000]


def _content_digest(data: bytes) -> str:
    return _format_digest(hashlib.sha256(data).digest())


def _format_digest(digest: bytes) -> str:
    return f"sha-256=:{base64.b64encode(digest).decode('ascii')}:"


def _file_identity(path: Path) -> tuple[int, str]:
    hasher = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
            size_bytes += len(chunk)
    return size_bytes, _format_digest(hasher.digest())


def _validate_local_tree_path(path: str) -> None:
    if (
        not path
        or len(path.encode()) > 16_384
        or path.startswith("/")
        or "\\" in path
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        raise ValueError(f"Artifact tree path is not canonical: {path!r}")


def _validated_tree_entries(manifest: Any) -> list[dict[str, Any]]:
    if not isinstance(manifest, dict) or set(manifest) != {"schema", "entries"}:
        raise GfaasError("Artifact tree manifest has invalid fields")
    if manifest.get("schema") != "gfaas.artifact-tree/v1":
        raise GfaasError("Artifact tree manifest has an unsupported schema")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) > 1_000_000:
        raise GfaasError("Artifact tree manifest has too many entries")
    previous: str | None = None
    directories: set[str] = set()
    validated: list[dict[str, Any]] = []
    for raw in entries:
        if not isinstance(raw, dict):
            raise GfaasError("Artifact tree manifest entry is not an object")
        path = raw.get("path")
        if not isinstance(path, str):
            raise GfaasError("Artifact tree manifest path is not text")
        parsed = PurePosixPath(path)
        if (
            not path
            or len(path.encode()) > 16_384
            or path.startswith("/")
            or "\\" in path
            or any(part in {"", ".", ".."} for part in path.split("/"))
            or any(part in {"", ".", ".."} for part in parsed.parts)
            or previous is not None
            and previous >= path
        ):
            raise GfaasError("Artifact tree manifest path is not canonical")
        parent = parsed.parent.as_posix()
        if parent != "." and parent not in directories:
            raise GfaasError("Artifact tree manifest omits a parent directory")
        entry_type = raw.get("type")
        mode = raw.get("mode")
        if entry_type == "directory":
            if set(raw) != {"path", "type", "mode"} or mode != 0o555:
                raise GfaasError("Artifact tree directory entry is invalid")
            directories.add(path)
        elif entry_type == "file":
            if set(raw) != {"path", "type", "artifact_id", "size_bytes", "digest", "mode"}:
                raise GfaasError("Artifact tree file entry has invalid fields")
            artifact_id = raw.get("artifact_id")
            size_bytes = raw.get("size_bytes")
            digest = raw.get("digest")
            if (
                not isinstance(artifact_id, str)
                or not re.fullmatch(r"art_[A-Za-z0-9_-]{1,128}", artifact_id)
                or not isinstance(size_bytes, int)
                or isinstance(size_bytes, bool)
                or size_bytes < 0
                or not isinstance(digest, str)
                or not _valid_content_digest(digest)
                or mode not in {0o444, 0o555}
            ):
                raise GfaasError("Artifact tree file entry is invalid")
        else:
            raise GfaasError("Artifact tree manifest entry type is invalid")
        validated.append(raw)
        previous = path
    return validated


def _valid_content_digest(value: str) -> bool:
    if not value.startswith("sha-256=:") or not value.endswith(":"):
        return False
    try:
        digest = base64.b64decode(value[9:-1], validate=True)
    except ValueError:
        return False
    return len(digest) == 32


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return
    for root, directories, files in os.walk(path, topdown=False):
        for name in files:
            (Path(root) / name).chmod(0o600)
        for name in directories:
            (Path(root) / name).chmod(0o700)
    path.chmod(0o700)
    shutil.rmtree(path)


def _result_error(result: dict[str, Any], state: str) -> str:
    error = result.get("error")
    if not isinstance(error, dict):
        return f"call ended in state {state!r}"
    message = str(error.get("message") or f"call ended in state {state!r}")
    details = error.get("details")
    journey = details.get("journey") if isinstance(details, dict) else None
    if not isinstance(journey, list) or not journey:
        return message
    lines = [message, "", "journey:"]
    visible: list[Any] = journey
    omitted = 0
    if len(journey) > MAX_ERROR_JOURNEY_EVENTS:
        half = MAX_ERROR_JOURNEY_EVENTS // 2
        visible = journey[:half] + [None] + journey[-half:]
        omitted = len(journey) - MAX_ERROR_JOURNEY_EVENTS
    for event in visible:
        if event is None:
            lines.append(f"- {omitted} journey events omitted")
            continue
        if not isinstance(event, dict):
            continue
        component = event.get("component", "unknown")
        stage = event.get("stage", "unknown")
        detail = ", ".join(
            f"{key}={value}"
            for key, value in event.items()
            if key not in {"component", "stage", "ts"} and value is not None
        )
        lines.append(f"- {component}.{stage}" + (f": {detail}" if detail else ""))
    return "\n".join(lines)


def _gpu_request(
    gpu: str | None,
    gpu_count: int | None,
    gpu_type: str,
) -> dict[str, Any]:
    if gpu_count is not None:
        if gpu is not None:
            raise GfaasError("gpu and gpu_count cannot be used together")
        if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count < 0:
            raise GfaasError("gpu_count must be a non-negative integer")
        explicit_request: dict[str, Any] = {"count": gpu_count}
        if gpu_type != "any":
            explicit_request["models"] = [gpu_type]
        return explicit_request
    if gpu is None:
        return {"count": 0}
    selector = gpu.strip()
    if not selector:
        raise GfaasError("GPU selector must not be empty")
    if selector.lower() == "all":
        raise GfaasError("gpu='all' is ambiguous; request an explicit GPU count")
    tokens = [token.strip() for token in selector.split(",")]
    count = len(tokens) if all(token.isdigit() for token in tokens) else 1
    request: dict[str, Any] = {"count": count}
    if gpu_type != "any":
        request["models"] = [gpu_type]
    return request


def _resolve_source_file(func: Any) -> str:
    import inspect

    source = inspect.getsourcefile(func)
    if source is None:
        raise GfaasError(f"cannot resolve source file for {func!r}")
    return source


def _image_name(image: Image | str) -> str:
    if isinstance(image, str):
        return image
    if isinstance(image, Image):
        return image.name
    raise GfaasError(f"unsupported image reference: {image!r}")


def _image_spec(image: Image | str) -> dict[str, Any] | None:
    if isinstance(image, str):
        return None
    if isinstance(image, Image):
        return image.build_spec
    raise GfaasError(f"unsupported image reference: {image!r}")


def _image_remote_source(image: Image | str) -> dict[str, Any] | None:
    if isinstance(image, str):
        return None
    if isinstance(image, Image):
        return dict(image.remote_source) if image.remote_source is not None else None
    raise GfaasError(f"unsupported image reference: {image!r}")


def _artifact_retention(ttl_seconds: int | None) -> dict[str, Any]:
    if ttl_seconds is None:
        return {"mode": "keep"}
    if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int) or ttl_seconds <= 0:
        raise ValueError("ttl_seconds must be a positive integer")
    return {"mode": "expire", "ttl_seconds": ttl_seconds}
