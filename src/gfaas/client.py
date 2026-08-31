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
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Self

import httpx

from .artifacts import (
    TREE_ARTIFACT_MEDIA_TYPE,
    ArtifactCheckpoint,
    ArtifactOutput,
    ArtifactUploadProgress,
    collect_artifact_ids,
)
from .bundle import empty_bundle, package_single_file
from .config import ClientConfig
from .errors import ArtifactTreeUploadError, GfaasError, UnsupportedGpuPoolError
from .image import Image
from .serialization import CLOUDPICKLE_CODEC, PICKLE_CODEC, decode_result, encode_args

TERMINAL_CALL_STATES = {"succeeded", "failed", "timed_out", "cancelled"}
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
        headers = {"X-API-Key": self.cfg.api_key} if self.cfg.api_key else {}
        self._http = httpx.Client(
            base_url=self.cfg.api_base,
            headers=headers,
            timeout=self.cfg.request_timeout_s,
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
        raise GfaasError(f"{operation} failed: {response.status_code}{suffix}")

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
    ) -> dict[str, Any]:
        digest = base64.b64encode(hashlib.sha256(data).digest()).decode("ascii")
        headers = {
            "Content-Type": media_type,
            "Content-Digest": f"sha-256=:{digest}:",
            "X-Gfunc-Artifact-Kind": kind,
        }
        if filename is not None:
            headers["X-Gfunc-Filename"] = filename
        response = self._request(
            "POST",
            "/v1/artifacts",
            operation="artifact upload",
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
        upload_id: str | None = None,
        progress: Callable[[ArtifactUploadProgress], None] | None = None,
    ) -> dict[str, Any]:
        """Upload a file with resumable, bounded-memory transfer requests."""
        source = Path(path)
        size_bytes, digest = _file_identity(source)
        expected: dict[str, Any] = {
            "kind": kind,
            "media_type": media_type,
            "filename": filename if filename is not None else source.name,
            "digest": digest,
            "size_bytes": size_bytes,
        }
        if upload_id is None:
            session = self.create_artifact_file_upload(
                source,
                kind=kind,
                media_type=media_type,
                filename=expected["filename"],
            )
        else:
            session = self._request(
                "GET",
                f"/v1/artifact-uploads/{upload_id}",
                operation="Artifact upload session lookup",
            ).json()
            for field, value in expected.items():
                if session.get(field) != value:
                    raise GfaasError(f"Artifact upload session {upload_id} does not match {field}")
        session_id = str(session["id"])
        part_size = int(session["part_size_bytes"])
        uploaded = {int(part["part_number"]): part for part in session.get("parts", [])}
        completed_bytes = 0
        transferred_bytes = 0
        reused_bytes = 0
        with source.open("rb") as stream:
            part_number = 0
            while True:
                chunk = stream.read(part_size)
                if not chunk:
                    break
                chunk_digest = _content_digest(chunk)
                existing = uploaded.get(part_number)
                if existing is None:
                    self._request(
                        "PUT",
                        f"/v1/artifact-uploads/{session_id}/parts/{part_number}",
                        operation="Artifact upload part transfer",
                        content=chunk,
                        headers={
                            "Content-Type": "application/octet-stream",
                            "Content-Digest": chunk_digest,
                        },
                    )
                    transferred_bytes += len(chunk)
                elif (
                    int(existing.get("size_bytes", -1)) != len(chunk)
                    or existing.get("digest") != chunk_digest
                ):
                    raise GfaasError(
                        f"Artifact upload session {session_id} has a conflicting part {part_number}"
                    )
                else:
                    reused_bytes += len(chunk)
                completed_bytes += len(chunk)
                if progress is not None:
                    progress(
                        ArtifactUploadProgress(
                            path=source,
                            completed_bytes=completed_bytes,
                            total_bytes=size_bytes,
                            transferred_bytes=transferred_bytes,
                            reused_bytes=reused_bytes,
                            completed_files=int(completed_bytes == size_bytes),
                        )
                    )
                part_number += 1
        artifact = self._request(
            "POST",
            f"/v1/artifact-uploads/{session_id}/completion",
            operation="Artifact upload completion",
        ).json()
        if progress is not None and size_bytes == 0:
            progress(
                ArtifactUploadProgress(
                    path=source,
                    completed_bytes=0,
                    total_bytes=0,
                    transferred_bytes=0,
                    reused_bytes=0,
                    completed_files=1,
                )
            )
        return artifact

    def create_artifact_file_upload(
        self,
        path: Path | str,
        *,
        kind: str = "other",
        media_type: str = "application/octet-stream",
        filename: str | None = None,
    ) -> dict[str, Any]:
        """Create a durable upload session before a resumable file transfer."""
        source = Path(path)
        size_bytes, digest = _file_identity(source)
        return self._request(
            "POST",
            "/v1/artifact-uploads",
            operation="Artifact upload session creation",
            json={
                "kind": kind,
                "media_type": media_type,
                "filename": filename if filename is not None else source.name,
                "digest": digest,
                "size_bytes": size_bytes,
            },
        ).json()

    def upload_artifact_directory(
        self,
        path: Path | str,
        *,
        kind: str = "input",
        filename: str | None = None,
        progress: Callable[[ArtifactUploadProgress], None] | None = None,
    ) -> dict[str, Any]:
        """Upload a directory as one immutable tree Artifact."""
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
        completed_bytes = 0
        transferred_bytes = 0
        reused_bytes = 0
        completed_files = 0
        entries: list[dict[str, Any]] = []
        for relative, entry_type, child, child_metadata in local_entries:
            if entry_type == "directory":
                entries.append({"path": relative, "type": "directory"})
            else:
                file_progress: ArtifactUploadProgress | None = None

                def report_file(
                    item: ArtifactUploadProgress,
                    *,
                    current_path: Path = child,
                    prior_completed_bytes: int = completed_bytes,
                    prior_transferred_bytes: int = transferred_bytes,
                    prior_reused_bytes: int = reused_bytes,
                    prior_completed_files: int = completed_files,
                ) -> None:
                    nonlocal file_progress
                    file_progress = item
                    if progress is None:
                        return
                    progress(
                        ArtifactUploadProgress(
                            path=current_path,
                            completed_bytes=prior_completed_bytes + item.completed_bytes,
                            total_bytes=total_bytes,
                            transferred_bytes=prior_transferred_bytes + item.transferred_bytes,
                            reused_bytes=prior_reused_bytes + item.reused_bytes,
                            completed_files=prior_completed_files + item.completed_files,
                            total_files=total_files,
                        )
                    )

                uploaded = self.upload_artifact_file(
                    child,
                    kind=kind,
                    filename=child.name,
                    progress=report_file if progress is not None else None,
                )
                completed_bytes += child_metadata.st_size
                completed_files += 1
                if file_progress is not None:
                    transferred_bytes += file_progress.transferred_bytes
                    reused_bytes += file_progress.reused_bytes
                entries.append(
                    {
                        "path": relative,
                        "type": "file",
                        "artifact_id": uploaded["id"],
                        "executable": bool(child_metadata.st_mode & 0o111),
                    }
                )
        child_artifact_ids = list(
            dict.fromkeys(entry["artifact_id"] for entry in entries if entry["type"] == "file")
        )
        try:
            tree = self._request(
                "POST",
                "/v1/artifact-trees",
                operation="Artifact tree creation",
                json={
                    "kind": kind,
                    "filename": filename if filename is not None else source.name,
                    "entries": entries,
                },
            ).json()
        except Exception as error:
            raise ArtifactTreeUploadError(child_artifact_ids) from error
        tree["child_artifact_ids"] = child_artifact_ids
        return tree

    def get_artifact(self, artifact_id: str) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/v1/artifacts/{artifact_id}",
            operation="artifact lookup",
        ).json()

    def download_artifact(self, artifact_id: str) -> tuple[bytes, str]:
        response = self._request(
            "GET",
            f"/v1/artifacts/{artifact_id}/content",
            operation="artifact download",
        )
        return response.content, response.headers.get("Content-Type", "application/octet-stream")

    def download_artifact_file(
        self,
        artifact_id: str,
        destination: Path | str,
    ) -> tuple[Path, str]:
        """Download an Artifact to a new file without buffering it in memory."""
        target = Path(destination)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.part")
        hasher = hashlib.sha256()
        written = 0
        try:
            with self._http.stream(
                "GET",
                f"/v1/artifacts/{artifact_id}/content",
            ) as response:
                if not response.is_success:
                    detail = _response_detail(response)
                    suffix = f": {detail}" if detail else ""
                    raise GfaasError(f"Artifact download failed: {response.status_code}{suffix}")
                expected_size = int(response.headers["Content-Length"])
                expected_digest = response.headers["Content-Digest"]
                media_type = response.headers.get("Content-Type", "application/octet-stream")
                with temporary.open("xb") as output:
                    for chunk in response.iter_bytes():
                        output.write(chunk)
                        hasher.update(chunk)
                        written += len(chunk)
                    output.flush()
                if written != expected_size or _format_digest(hasher.digest()) != expected_digest:
                    raise GfaasError("Artifact download failed integrity verification")
            temporary.replace(target)
            return target, media_type
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

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
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.part")
        manifest_path = temporary / ".gfaas-tree-manifest.json"
        temporary.mkdir()
        try:
            _, media_type = self.download_artifact_file(artifact_id, manifest_path)
            if media_type != TREE_ARTIFACT_MEDIA_TYPE:
                raise GfaasError("Artifact tree download returned the wrong media type")
            try:
                manifest = json.loads(manifest_path.read_bytes())
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise GfaasError("Artifact tree manifest is invalid JSON") from error
            manifest_path.unlink()
            entries = _validated_tree_entries(manifest)
            directories = [temporary]
            for entry in entries:
                path = temporary.joinpath(*Path(entry["path"]).parts)
                if entry["type"] == "directory":
                    path.mkdir()
                    directories.append(path)
                    continue
                self.download_artifact_file(str(entry["artifact_id"]), path)
                size_bytes, digest = _file_identity(path)
                if size_bytes != entry["size_bytes"] or digest != entry["digest"]:
                    raise GfaasError("Artifact tree entry failed manifest verification")
                path.chmod(int(entry["mode"]))
            for directory in reversed(directories):
                directory.chmod(0o555)
            temporary.replace(target)
            return target
        except Exception:
            _remove_tree(temporary)
            raise

    def delete_artifact(self, artifact_id: str) -> None:
        self._request(
            "DELETE",
            f"/v1/artifacts/{artifact_id}",
            operation="artifact deletion",
        )

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
        """Yield Call events from the public resumable SSE stream."""
        if timeout_s is not None:
            yield from self._iter_events_with_timeout(
                call_id,
                after=after,
                follow=follow,
                timeout_s=timeout_s,
            )
            return
        params: dict[str, str] = {"follow": "true" if follow else "false"}
        if after is not None:
            params["after"] = after
        with self._http.stream(
            "GET",
            f"/v1/calls/{call_id}/events",
            params=params,
            headers={"accept": "text/event-stream"},
        ) as response:
            if not response.is_success:
                detail = _response_detail(response)
                suffix = f": {detail}" if detail else ""
                raise GfaasError(f"event stream failed: {response.status_code}{suffix}")
            yield from _decode_event_stream(response.iter_lines())

    def _iter_events_with_timeout(
        self,
        call_id: str,
        *,
        after: str | None,
        follow: bool,
        timeout_s: float,
    ) -> Iterator[dict[str, Any]]:
        """Poll durable events when the caller needs a total deadline."""
        deadline = time.monotonic() + timeout_s
        cursor = after
        while True:
            page = self.list_events(call_id, after=cursor, limit=1000)
            terminal = False
            for event in page.get("items", []):
                event_cursor = event.get("cursor")
                if event_cursor is not None:
                    cursor = str(event_cursor)
                yield event
                if event.get("type") == "state" and event.get("state") in TERMINAL_CALL_STATES:
                    terminal = True
            if terminal or not follow:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"call {call_id} event stream exceeded {timeout_s}s")
            time.sleep(min(self.cfg.poll_interval_s, remaining))

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
        outputs: tuple[ArtifactOutput | ArtifactCheckpoint, ...] = (),
    ) -> RemoteResult:
        """Package a Python callable and create an asynchronous Call."""
        image_name = _image_name(image)
        if _image_spec(image) is not None:
            raise GfaasError(
                "container build images are not enabled by the current gfaas service; "
                "use a registered image"
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
            "name": image_name,
        }
        remote_source = _image_remote_source(image)
        if remote_source is not None:
            environment_source["remote"] = remote_source
        environment = self.create_environment(
            {
                "name": image_name,
                "source": environment_source,
                "variables": dict(env or {}),
            }
        )
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
        if capacity_wait_s is not None:
            call_request["capacity_wait_seconds"] = capacity_wait_s
        if outputs:
            call_request["outputs"] = [output.request() for output in outputs]
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
                payload, media_type = self.download_artifact(output["artifact_id"])
                declared_type = output.get("media_type") or media_type
                if declared_type == "application/json":
                    return json.loads(payload)
                if declared_type == CLOUDPICKLE_MEDIA_TYPE:
                    codec = CLOUDPICKLE_CODEC
                elif declared_type in {PICKLE_MEDIA_TYPE, "application/octet-stream"}:
                    codec = PICKLE_CODEC
                else:
                    codec = declared_type
                return decode_result(payload, codec=codec)
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"call {call_id} not finished within {timeout_s}s")
            time.sleep(self.cfg.poll_interval_s)


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


def _decode_event_stream(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    data_lines: list[str] = []
    for line in lines:
        if line == "":
            if data_lines:
                try:
                    event = json.loads("\n".join(data_lines))
                except json.JSONDecodeError as error:
                    raise GfaasError("event stream returned invalid JSON") from error
                if not isinstance(event, dict):
                    raise GfaasError("event stream payload must be an object")
                yield event
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "data":
            data_lines.append(value)


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
        if gpu_count > 0 and gpu_type != "any":
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
