"""Image reference for gfaas.

There are two image description modes:

1. A registered image uses a logical name that the gfaas service provides.
2. A container image uses a portable build description.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
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


def build_spec_hash(spec: dict[str, Any]) -> str:
    payload = json.dumps(spec, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def build_spec_name(spec: dict[str, Any]) -> str:
    return f"dyn_{build_spec_hash(spec)}"


@dataclass(frozen=True)
class Image:
    name: str
    build_spec: dict[str, Any] | None = None
    remote_source: dict[str, Any] | None = None

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
        spec = _normalized_build_spec(
            base_image=base_image,
            python_version=python_version,
            apt_packages=apt_packages,
            pip_packages=pip_packages,
            env=env,
            commands=commands,
        )
        return cls(name=name or build_spec_name(spec), build_spec=spec)

    def __str__(self) -> str:
        return self.name


__all__ = ["Image", "build_spec_hash", "build_spec_name"]
