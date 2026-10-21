"""Transfer recovery shared by buffered results, files, and directory downloads.

Partial bytes live outside the caller's staging directory. A private, locked workspace
binds them to an API credential scope and an Artifact identity. Successful transfers
remove their workspace; interrupted transfers can resume with a new Client or process.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import BoundedSemaphore, Event
from typing import Any

import httpx
from filelock import FileLock
from filelock import Timeout as LockTimeout

from .errors import GfaasError

ATTEMPTS = 4
CHUNK_BYTES = 8 * 1024 * 1024
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504, 520, 522, 523, 524}
TRANSIENT_ERRORS = (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)


def pause(attempt: int, cancel: Event | None = None) -> None:
    delay = 0.25 * 2**attempt
    if cancel is None:
        time.sleep(delay)
    elif cancel.wait(delay):
        raise GfaasError("Artifact transfer cancelled")


def retry_request(action: Callable[[], httpx.Response], cancel: Event | None) -> httpx.Response:
    """Retry only idempotent operations; creation and completion have separate reconciliation."""
    for attempt in range(ATTEMPTS):
        if cancel is not None and cancel.is_set():
            raise GfaasError("Artifact transfer cancelled")
        try:
            return action()
        except GfaasError as error:
            if error.status_code not in RETRYABLE_STATUS or attempt == ATTEMPTS - 1:
                raise
        except TRANSIENT_ERRORS:
            if attempt == ATTEMPTS - 1:
                raise
        pause(attempt, cancel)
    raise AssertionError("unreachable")


class TransferManager:
    def __init__(
        self, http: Callable[[], httpx.Client], scope: str, cache: Path | None, concurrency: int
    ) -> None:
        self.http = http
        self._download_slots = BoundedSemaphore(concurrency)
        root = cache or Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "vfunc"
        self.root = root / "transfers" / hashlib.sha256(scope.encode()).hexdigest()
        self._last_cleanup: dict[Path, float] = {}

    @contextmanager
    def workspace(self, identity: str) -> Iterator[Path]:
        kind = identity.partition(":")[0]
        if kind not in {"download", "upload", "tree", "tree-upload"}:
            raise ValueError("Unknown transfer workspace kind")
        root = self.root / kind
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if root.is_symlink() or (os.name == "posix" and root.stat().st_mode & 0o077):
            raise GfaasError("Transfer cache must be a private directory")
        self._expire_abandoned(root)
        name = hashlib.sha256(identity.encode()).hexdigest()
        # Lock files remain in place: unlinking one could let a third process bypass a waiter.
        lock = FileLock(
            root / f"{name[:2]}.lock",
            mode=0o600,
            timeout=60,
            fallback_to_soft=False,
            preserve_lock_file=True,
        )
        with lock:
            directory = root / name
            directory.mkdir(mode=0o700, exist_ok=True)
            if directory.is_symlink() or (os.name == "posix" and directory.stat().st_mode & 0o077):
                raise GfaasError("Unsafe transfer workspace")
            os.utime(directory, None)
            yield directory

    @staticmethod
    def finish(directory: Path) -> None:
        """Cleanup cannot turn a completed transfer into a reported publication failure."""
        try:
            shutil.rmtree(directory)
        except OSError as error:
            logging.getLogger(__name__).warning(
                "Transfer completed; local cache cleanup failed: %s", error
            )

    def _expire_abandoned(self, root: Path) -> None:
        now = time.time()
        if now - self._last_cleanup.get(root, 0.0) < 3600:
            return
        self._last_cleanup[root] = now
        for directory in root.iterdir():
            if (
                not re.fullmatch(r"[0-9a-f]{64}", directory.name)
                or directory.is_symlink()
                or not directory.is_dir()
            ):
                continue
            try:
                if now - directory.stat().st_mtime < 7 * 86400:
                    continue
            except FileNotFoundError:
                continue
            lock = FileLock(
                root / f"{directory.name[:2]}.lock",
                mode=0o600,
                timeout=0,
                fallback_to_soft=False,
                preserve_lock_file=True,
            )
            try:
                with lock:
                    if directory.exists() and now - directory.stat().st_mtime >= 7 * 86400:
                        shutil.rmtree(directory)
            except LockTimeout:
                continue

    def file(self, artifact_id: str, target: Path) -> tuple[Path, str]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", artifact_id):
            raise ValueError("Invalid Artifact ID")
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        with self.workspace(f"download:{artifact_id}") as workspace:
            media_type = self._receive(artifact_id, workspace)
            # Publish without replacing an existing file. Copy to the target filesystem first;
            # linking the cache itself would let a caller modify bytes used by another download.
            fd, name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".part", dir=target.parent
            )
            try:
                with os.fdopen(fd, "wb") as output, (workspace / "body").open("rb") as source:
                    shutil.copyfileobj(source, output, 1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
                os.link(name, target)
            finally:
                Path(name).unlink(missing_ok=True)
            self.finish(workspace)
            return target, media_type

    def _receive(self, artifact_id: str, directory: Path) -> str:
        body = directory / "body"
        metadata = directory / "metadata.json"
        if body.is_symlink() or metadata.is_symlink():
            raise GfaasError("Unsafe transfer state")
        state: dict[str, Any] | None = (
            json.loads(metadata.read_bytes()) if metadata.exists() else None
        )
        if state is None:
            body.unlink(missing_ok=True)
        failures = 0
        while True:
            offset = body.stat().st_size if body.exists() else 0
            if state is not None and offset == state["size"]:
                with body.open("rb") as source:
                    actual = hashlib.file_digest(source, "sha256").digest()
                if base64.b64encode(actual).decode() != state["sha256"]:
                    body.unlink(missing_ok=True)
                    metadata.unlink(missing_ok=True)
                    raise GfaasError(
                        "Artifact download failed integrity verification; partial removed"
                    )
                return str(state["media_type"])
            if state is not None and offset > state["size"]:
                raise GfaasError("Partial Artifact exceeds its declared size")
            headers = {
                "Accept-Encoding": "identity",
                "Range": f"bytes={offset}-{offset + CHUNK_BYTES - 1}",
            }
            if state is not None and state.get("etag"):
                headers["If-Match"] = state["etag"]
            try:
                with (
                    self._download_slots,
                    self.http().stream(
                        "GET", f"/v1/artifacts/{artifact_id}/content", headers=headers
                    ) as response,
                ):
                    if response.status_code in RETRYABLE_STATUS:
                        raise GfaasError(
                            "Temporary Artifact download failure", status_code=response.status_code
                        )
                    if response.status_code in {401, 403}:
                        raise GfaasError(
                            f"artifact download failed: authentication rejected ({response.status_code})",
                            status_code=response.status_code,
                        )
                    if response.status_code not in {200, 206}:
                        raise GfaasError(
                            f"Artifact download failed: {response.status_code}",
                            status_code=response.status_code,
                        )
                    if response.headers.get("content-encoding", "identity") != "identity":
                        raise GfaasError("Artifact response must not use content encoding")
                    length = int(response.headers["Content-Length"])
                    if length < 0:
                        raise GfaasError("Invalid Artifact size")
                    if response.status_code == 206:
                        match = re.fullmatch(
                            r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
                        )
                        if match is None:
                            raise GfaasError("Artifact response omitted a valid Content-Range")
                        start, end, total = map(int, match.groups())
                        if (
                            start != offset
                            or end < start
                            or end >= total
                            or end - start + 1 != length
                            or length > CHUNK_BYTES
                        ):
                            raise GfaasError("Artifact response does not match the requested range")
                        digest = response.headers.get("Repr-Digest", "")
                    else:
                        # An older server may ignore Range. Restart this file only after checking identity.
                        total = length
                        digest = response.headers.get("Content-Digest", "")
                    match_digest = re.fullmatch(r"sha-256=:([A-Za-z0-9+/]+={0,2}):", digest)
                    if (
                        match_digest is None
                        or len(base64.b64decode(match_digest[1], validate=True)) != 32
                    ):
                        raise GfaasError("Artifact response omitted a valid SHA-256 digest")
                    incoming = {
                        "size": total,
                        "sha256": match_digest[1],
                        "media_type": response.headers.get(
                            "Content-Type", "application/octet-stream"
                        ),
                        "etag": response.headers.get("ETag"),
                    }
                    if state is not None and (
                        any(
                            incoming[field] != state[field]
                            for field in ("size", "sha256", "media_type")
                        )
                        or (state.get("etag") is not None and incoming["etag"] != state["etag"])
                    ):
                        raise GfaasError(
                            "Artifact identity changed during download; refusing to combine bytes"
                        )
                    state = incoming
                    staging = metadata.with_suffix(".new")
                    with staging.open("w") as output:
                        json.dump(state, output)
                        output.flush()
                        os.fsync(output.fileno())
                    staging.replace(metadata)
                    received = 0
                    with body.open("wb" if response.status_code == 200 else "ab") as output:
                        try:
                            for chunk in response.iter_bytes(chunk_size=64 * 1024):
                                if received + len(chunk) > length:
                                    raise GfaasError("Artifact response exceeds its declared size")
                                output.write(chunk)
                                received += len(chunk)
                        finally:
                            output.flush()
                            os.fsync(output.fileno())
                    if received != length:
                        raise httpx.RemoteProtocolError(
                            "Artifact response ended before its declared size"
                        )
            except (GfaasError, *TRANSIENT_ERRORS) as error:
                if isinstance(error, GfaasError) and error.status_code not in RETRYABLE_STATUS:
                    raise
                failures += 1
                if failures >= ATTEMPTS:
                    raise GfaasError(
                        f"Artifact download ended before its declared size after {ATTEMPTS} attempts; retry to resume from {directory}"
                    ) from error
                pause(failures - 1)
