from __future__ import annotations

from dataclasses import dataclass

import cloudpickle
import pytest

from gfaas import ArtifactRef
from gfaas.artifacts import collect_artifact_ids
from gfaas.errors import SerializationError
from gfaas.serialization import encode_args


def test_artifact_ref_reads_its_staged_file(tmp_path, monkeypatch) -> None:
    staged = tmp_path / "art_input"
    staged.write_text("payload\n", encoding="utf-8")
    monkeypatch.setenv("GFAAS_ARTIFACT_ROOT", str(tmp_path))

    reference = ArtifactRef("art_input")

    assert reference.path == staged
    assert reference.read_bytes() == b"payload\n"
    assert reference.read_text() == "payload\n"


def test_artifact_ref_can_resolve_a_staged_directory(tmp_path, monkeypatch) -> None:
    staged = tmp_path / "art_tree"
    staged.mkdir()
    (staged / "data.bin").write_bytes(b"tree payload")
    monkeypatch.setenv("GFAAS_ARTIFACT_ROOT", str(tmp_path))

    reference = ArtifactRef("art_tree")

    assert reference.path == staged
    assert (reference.path / "data.bin").read_bytes() == b"tree payload"


def test_artifact_ref_is_unavailable_outside_a_function(monkeypatch) -> None:
    monkeypatch.delenv("GFAAS_ARTIFACT_ROOT", raising=False)

    with pytest.raises(RuntimeError, match="only available inside"):
        _ = ArtifactRef("art_input").path


def test_artifact_ref_survives_argument_serialization(tmp_path, monkeypatch) -> None:
    staged = tmp_path / "art_input"
    staged.write_bytes(b"serialized payload")
    monkeypatch.setenv("GFAAS_ARTIFACT_ROOT", str(tmp_path))

    payload, _ = encode_args((ArtifactRef("art_input"),), {})
    decoded = cloudpickle.loads(payload)

    assert decoded["args"][0].read_bytes() == b"serialized payload"


@dataclass
class Inputs:
    primary: ArtifactRef
    nested: list[object]


def test_collects_unique_refs_from_nested_and_cyclic_arguments() -> None:
    nested: list[object] = []
    nested.append(nested)
    nested.extend([ArtifactRef("art_second"), ArtifactRef("art_first")])
    value = Inputs(primary=ArtifactRef("art_first"), nested=nested)

    assert collect_artifact_ids((value,), {}) == ["art_first", "art_second"]


def test_rejects_more_than_the_per_call_artifact_limit() -> None:
    references = [ArtifactRef(f"art_{index}") for index in range(33)]

    with pytest.raises(SerializationError, match="at most 32"):
        collect_artifact_ids((references,), {})
