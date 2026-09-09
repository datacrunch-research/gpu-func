"""Durable sequential execution-stage declarations for one remote Call."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .artifacts import ArtifactCheckpoint, ArtifactOutput


@dataclass(frozen=True)
class StageArtifactBinding:
    """Mount a prior stage output and expose its Artifact ID to a later stage."""

    from_stage: str
    output: str
    env: str

    def request(self) -> dict[str, str]:
        return {
            "from_stage": self.from_stage,
            "output": self.output,
            "env": self.env,
        }


@dataclass(frozen=True)
class CallStage:
    """One sequential callable and resource envelope within a remote Call."""

    name: str
    qualname: str
    module: str | None = None
    resources: Mapping[str, Any] | None = None
    outputs: tuple[ArtifactOutput | ArtifactCheckpoint, ...] = field(default_factory=tuple)
    artifacts: tuple[StageArtifactBinding, ...] = field(default_factory=tuple)

    def request(self, *, default_module: str) -> dict[str, Any]:
        value: dict[str, Any] = {
            "name": self.name,
            "executable": {
                "module": self.module or default_module,
                "qualname": self.qualname,
            },
        }
        if self.resources is not None:
            value["resources"] = dict(self.resources)
        if self.outputs:
            value["outputs"] = [output.request() for output in self.outputs]
        if self.artifacts:
            value["artifacts"] = [artifact.request() for artifact in self.artifacts]
        return value
