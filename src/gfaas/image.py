"""Image reference for gfaas.

An image can name one registered filesystem, select native registered variants,
or describe a container build. Registered variants use CPU architecture keys.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


def _normalized_build_spec(
    *,
    base_image: str,
    python_version: str = "3.11",
    apt_packages: tuple[str, ...] | list[str] = (),
    pip_packages: tuple[str, ...] | list[str] = (),
    env: Mapping[str, str] | None = None,
    commands: tuple[str, ...] | list[str] = (),
) -> dict[str, Any]:
    return {
        "base_image": base_image,
        "python_version": python_version,
        "apt_packages": list(apt_packages),
        "pip_packages": list(pip_packages),
        "env": dict(env or {}),
        "commands": list(commands),
    }


DEFAULT_NAME_PREFIX = "img"
# The coordinator accepts names of at most 128 characters; the suffix takes 7.
_NAME_PREFIX = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,119}[A-Za-z0-9])?$")


def _spec_digest(spec: dict[str, Any]) -> bytes:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode("utf-8")).digest()


def build_spec_hash(spec: dict[str, Any]) -> str:
    return _spec_digest(spec).hex()[:24]


def build_spec_suffix(spec: dict[str, Any]) -> str:
    """Six lowercase base32 characters (30 bits) that identify a build specification."""
    return base64.b32encode(_spec_digest(spec)).decode("ascii").lower()[:6]


def build_spec_name(spec: dict[str, Any], prefix: str | None = None) -> str:
    """The published name of a container image: ``PREFIX-xxxxxx``.

    The suffix changes whenever any build field changes, so one name never stands for two
    specifications.
    """
    prefix = prefix or DEFAULT_NAME_PREFIX
    if _NAME_PREFIX.fullmatch(prefix) is None:
        raise ValueError(
            "image name must have 1 to 121 letters, digits, '.', '_', or '-', "
            "and start and end with a letter or digit"
        )
    return f"{prefix}-{build_spec_suffix(spec)}"


@dataclass(frozen=True)
class Image:
    name: str
    build_spec: dict[str, Any] | None = None
    remote_source: dict[str, Any] | None = None
    registry_variants: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if self.registry_variants is None:
            return
        if self.build_spec is not None or self.remote_source is not None:
            raise ValueError(
                "registry variants cannot be combined with a build or remote descriptor"
            )
        variants = dict(self.registry_variants)
        if not variants or any(arch not in {"amd64", "arm64"} for arch in variants):
            raise ValueError("registry variants must map amd64 and/or arm64 to image references")
        if any(
            not isinstance(ref, str) or not ref or len(ref.encode()) > 255
            for ref in variants.values()
        ):
            raise ValueError("image references must contain 1 to 255 UTF-8 bytes")
        object.__setattr__(self, "registry_variants", MappingProxyType(variants))

    @classmethod
    def from_registry_variants(cls, name: str, images: Mapping[str, str]) -> Image:
        """Select a native registered name or digest for each supported CPU architecture."""
        return cls(name=name, registry_variants=images)

    @classmethod
    def from_registry(cls, name: str) -> Image:
        return cls(name=name)

    @classmethod
    def from_remote(cls, name: str, descriptor: Mapping[str, Any]) -> Image:
        """Reference an immutable image published to the configured object origin."""
        return cls(name=name, remote_source=dict(descriptor))

    @classmethod
    def from_container(
        cls,
        base_image: str,
        *,
        python_version: str = "3.11",
        apt_packages: tuple[str, ...] | list[str] = (),
        pip_packages: tuple[str, ...] | list[str] = (),
        env: Mapping[str, str] | None = None,
        commands: tuple[str, ...] | list[str] = (),
        name: str | None = None,
    ) -> Image:
        """Describe an image built from ``base_image``.

        Its published name is ``name`` (default ``img``) followed by a hash of the build fields,
        such as ``flashinfer-cu13-k3x9qa``.
        """
        spec = _normalized_build_spec(
            base_image=base_image,
            python_version=python_version,
            apt_packages=apt_packages,
            pip_packages=pip_packages,
            env=env,
            commands=commands,
        )
        return cls(name=build_spec_name(spec, name), build_spec=spec)

    def __str__(self) -> str:
        return self.name


__all__ = ["Image", "build_spec_hash", "build_spec_name", "build_spec_suffix"]
