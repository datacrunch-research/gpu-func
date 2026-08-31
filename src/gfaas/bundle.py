"""Deterministic Python source bundles submitted through gfaas."""

from __future__ import annotations

import gzip
import hashlib
import io
import tarfile
from dataclasses import dataclass
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Bundle:
    """A content-addressed bundle and its importable top-level module."""

    data: bytes
    sha256: str
    module_name: str


def _add_bytes(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    tar.addfile(info, io.BytesIO(payload))


def _package_bytes(files: list[tuple[str, bytes]]) -> bytes:
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name, payload in files:
            _add_bytes(tar, name, payload)

    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode="wb", compresslevel=6, mtime=0) as output:
        output.write(tar_buffer.getvalue())
    return compressed.getvalue()


def _package_files(package_root: Path) -> list[tuple[str, bytes]]:
    files: list[tuple[str, bytes]] = []
    for path in sorted(package_root.rglob("*.py")):
        relative = path.relative_to(package_root.parent).as_posix()
        files.append((relative, path.read_bytes()))
    return files


def package_single_file(
    source_file: Path,
    *,
    include_gfaas: bool = True,
    package_root: Path | None = None,
) -> Bundle:
    """Package one user module and the gfaas decorator surface it may import."""
    source_file = source_file.resolve()
    files = [(source_file.name, source_file.read_bytes())]
    if include_gfaas:
        files.extend(_package_files(package_root or _PACKAGE_ROOT))
    data = _package_bytes(files)
    return Bundle(
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
        module_name=source_file.stem,
    )


def empty_bundle() -> Bundle:
    """Return the placeholder bundle used for modules already in an image."""
    data = _package_bytes([(".gfaas-empty", b"")])
    return Bundle(data=data, sha256=hashlib.sha256(data).hexdigest(), module_name="")
