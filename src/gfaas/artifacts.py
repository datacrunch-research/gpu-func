"""Credential-free references to Artifacts staged for a function invocation."""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, TextIO, cast

from .errors import SerializationError

_ARTIFACT_ID = re.compile(r"^art_[A-Za-z0-9_-]{1,128}$")
_ARTIFACT_ROOT_ENV = "GFAAS_ARTIFACT_ROOT"
_OUTPUT_ROOT_ENV = "GFAAS_OUTPUT_ROOT"
_SCRATCH_ROOT_ENV = "GFAAS_SCRATCH_ROOT"
_OUTPUT_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_OUTPUT_KINDS = {"output", "log", "profile", "diagnostic", "other"}
_MAX_ARTIFACT_REFS = 32
_MAX_TRAVERSED_VALUES = 10_000
TREE_ARTIFACT_MEDIA_TYPE = "application/vnd.gfaas.artifact-tree.v1+json"


def scratch_path() -> Path:
    """Return the writable scratch directory inside a running gfaas function."""
    root = os.environ.get(_SCRATCH_ROOT_ENV)
    if not root:
        raise RuntimeError("scratch paths are only available inside a running function")
    path = Path(root)
    if path.is_symlink() or not path.is_dir():
        raise FileNotFoundError("the function scratch directory is unavailable")
    return path


@dataclass(frozen=True)
class ArtifactRef:
    """Reference to an Artifact made available as a read-only workload file."""

    artifact_id: str

    def __post_init__(self) -> None:
        if not _ARTIFACT_ID.fullmatch(self.artifact_id):
            raise ValueError(f"invalid Artifact identity {self.artifact_id!r}")

    @property
    def path(self) -> Path:
        """Return the staged path inside a running gfaas function."""
        root = os.environ.get(_ARTIFACT_ROOT_ENV)
        if not root:
            raise RuntimeError("ArtifactRef paths are only available inside a running function")
        path = Path(root) / self.artifact_id
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise FileNotFoundError(f"Artifact {self.artifact_id} was not staged for this function")
        return path

    def __fspath__(self) -> str:
        return os.fspath(self.path)

    def open(self, mode: str = "rb", *, encoding: str | None = None) -> BinaryIO | TextIO:
        """Open the staged Artifact using ordinary file semantics."""
        return cast(BinaryIO | TextIO, self.path.open(mode, encoding=encoding))

    def read_bytes(self) -> bytes:
        return self.path.read_bytes()

    def read_text(self, encoding: str = "utf-8") -> str:
        return self.path.read_text(encoding=encoding)


@dataclass(frozen=True)
class ArtifactUploadProgress:
    """Progress for one file upload or one file within a tree upload."""

    path: Path
    completed_bytes: int
    total_bytes: int
    transferred_bytes: int
    reused_bytes: int
    completed_files: int = 0
    total_files: int = 1


@dataclass(frozen=True)
class ArtifactOutput:
    """Declare one file or directory that a function publishes at completion."""

    name: str
    relative_path: str
    kind: str = "output"
    media_type: str = "application/octet-stream"
    required: bool = True
    publish_on_failure: bool = True
    layout: str = "blob"

    def __post_init__(self) -> None:
        _validate_output_identity(self.name, self.relative_path, self.kind)
        if not self.media_type or len(self.media_type) > 255:
            raise ValueError(f"invalid Artifact output media type {self.media_type!r}")
        if self.layout not in {"blob", "tree"}:
            raise ValueError(f"invalid Artifact output layout {self.layout!r}")
        if self.layout == "tree" and self.media_type != TREE_ARTIFACT_MEDIA_TYPE:
            raise ValueError("directory outputs require the Artifact tree media type")

    @classmethod
    def directory(
        cls,
        name: str,
        relative_path: str,
        *,
        kind: str = "output",
        required: bool = True,
        publish_on_failure: bool = True,
    ) -> ArtifactOutput:
        """Declare one directory that the function publishes at completion."""
        return cls(
            name,
            relative_path,
            kind=kind,
            media_type=TREE_ARTIFACT_MEDIA_TYPE,
            required=required,
            publish_on_failure=publish_on_failure,
            layout="tree",
        )

    @property
    def path(self) -> Path:
        """Return the writable path inside a running gfaas function."""
        root = os.environ.get(_OUTPUT_ROOT_ENV)
        if not root:
            raise RuntimeError("Artifact output paths are only available inside a running function")
        return Path(root).joinpath(*PurePosixPath(self.relative_path).parts)

    def open(self, mode: str = "wb", *, encoding: str | None = None) -> BinaryIO | TextIO:
        """Create parent directories and open this output file."""
        if self.layout != "blob":
            raise IsADirectoryError("use ArtifactOutput.path for a directory output")
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        return cast(BinaryIO | TextIO, path.open(mode, encoding=encoding))

    def request(self) -> dict[str, Any]:
        """Return the public Call declaration for this output."""
        return {
            "name": self.name,
            "path": self.relative_path,
            "kind": self.kind,
            "media_type": self.media_type,
            "layout": self.layout,
            "publication": "terminal",
            "maximum_versions": 1,
            "required": self.required,
            "publish_on_failure": self.publish_on_failure,
        }


@dataclass(frozen=True)
class ArtifactCheckpoint:
    """Declare bounded immutable directory checkpoints for a running function."""

    name: str
    relative_path: str
    kind: str = "output"
    maximum_versions: int = 8

    def __post_init__(self) -> None:
        _validate_output_identity(self.name, self.relative_path, self.kind)
        if len(self.name) > 119:
            raise ValueError("checkpoint names must be at most 119 characters")
        if not 1 <= self.maximum_versions <= 64:
            raise ValueError("checkpoint maximum_versions must be between 1 and 64")

    @property
    def path(self) -> Path:
        """Return the root below which the function writes immutable versions."""
        root = os.environ.get(_OUTPUT_ROOT_ENV)
        if not root:
            raise RuntimeError("checkpoint paths are only available inside a running function")
        return Path(root).joinpath(*PurePosixPath(self.relative_path).parts)

    def publish(self, relative_directory: str | Path) -> int:
        """Atomically mark one immutable child directory for publication."""
        raw_value = os.fspath(relative_directory)
        relative = PurePosixPath(raw_value)
        value = relative.as_posix()
        if (
            not raw_value
            or len(raw_value.encode()) > 255
            or raw_value.startswith("/")
            or "\\" in raw_value
            or any(part in {"", ".", ".."} for part in raw_value.split("/"))
        ):
            raise ValueError(f"invalid checkpoint directory {raw_value!r}")
        source = self.path.joinpath(*relative.parts)
        if source.is_symlink() or not source.is_dir():
            raise FileNotFoundError(f"checkpoint directory is unavailable: {source}")
        output_root_value = os.environ.get(_OUTPUT_ROOT_ENV)
        if not output_root_value:
            raise RuntimeError("checkpoint paths are only available inside a running function")
        output_root = Path(output_root_value)
        marker_root = output_root / ".gfaas" / "checkpoints" / self.name
        marker_root.mkdir(parents=True, exist_ok=True)
        generations = [
            int(path.stem)
            for path in marker_root.glob("[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].json")
        ]
        generation = max(generations, default=0) + 1
        if generation > self.maximum_versions:
            raise RuntimeError("checkpoint version limit reached")
        destination = marker_root / f"{generation:08}.json"
        temporary = marker_root / f".{generation:08}.{uuid.uuid4().hex}.tmp"
        payload = json.dumps(
            {
                "schema": "gfaas.checkpoint/v1",
                "generation": generation,
                "path": value,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        try:
            with temporary.open("xb") as marker:
                marker.write(payload)
                marker.flush()
                os.fsync(marker.fileno())
            os.replace(temporary, destination)
            directory = os.open(marker_root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)
        return generation

    def request(self) -> dict[str, Any]:
        """Return the public Call declaration for this checkpoint."""
        return {
            "name": self.name,
            "path": self.relative_path,
            "kind": self.kind,
            "media_type": TREE_ARTIFACT_MEDIA_TYPE,
            "layout": "tree",
            "publication": "checkpoint",
            "maximum_versions": self.maximum_versions,
            "required": False,
            "publish_on_failure": False,
        }


def _validate_output_identity(name: str, relative_path: str, kind: str) -> None:
    if not _OUTPUT_NAME.fullmatch(name):
        raise ValueError(f"invalid Artifact output name {name!r}")
    path = PurePosixPath(relative_path)
    if (
        not relative_path
        or len(relative_path) > 255
        or relative_path.startswith("/")
        or "\\" in relative_path
        or any(part in {"", ".", ".."} for part in relative_path.split("/"))
        or not path.parts
        or path.parts[0] == ".gfaas"
    ):
        raise ValueError(f"invalid Artifact output path {relative_path!r}")
    if kind not in _OUTPUT_KINDS:
        raise ValueError(f"invalid Artifact output kind {kind!r}")


def collect_artifact_ids(args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[str]:
    """Collect bounded Artifact references from supported argument containers."""
    stack: list[Any] = [args, kwargs]
    traversed: set[int] = set()
    artifact_ids: list[str] = []
    unique_ids: set[str] = set()
    visited_values = 0

    while stack:
        value = stack.pop()
        visited_values += 1
        if visited_values > _MAX_TRAVERSED_VALUES:
            raise SerializationError("call arguments contain too many values to inspect")
        if isinstance(value, ArtifactRef):
            if value.artifact_id not in unique_ids:
                unique_ids.add(value.artifact_id)
                artifact_ids.append(value.artifact_id)
                if len(artifact_ids) > _MAX_ARTIFACT_REFS:
                    raise SerializationError(
                        f"a call can reference at most {_MAX_ARTIFACT_REFS} Artifacts"
                    )
            continue

        children: list[Any] | None = None
        if isinstance(value, dict):
            children = [*value.keys(), *value.values()]
        elif isinstance(value, (list, tuple, set, frozenset)):
            children = list(value)
        elif is_dataclass(value) and not isinstance(value, type):
            children = [getattr(value, field.name) for field in fields(value)]
        if children is None:
            continue
        identity = id(value)
        if identity in traversed:
            continue
        traversed.add(identity)
        stack.extend(children)

    return artifact_ids
