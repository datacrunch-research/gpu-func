from __future__ import annotations

from pathlib import Path

import pytest

from gfaas.artifacts import (
    TREE_ARTIFACT_MEDIA_TYPE,
    ArtifactCheckpoint,
    ArtifactOutput,
    scratch_path,
)


def test_scratch_path_resolves_inside_a_function(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GFAAS_SCRATCH_ROOT", str(tmp_path))

    assert scratch_path() == tmp_path


def test_scratch_path_is_only_available_inside_a_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GFAAS_SCRATCH_ROOT", raising=False)

    with pytest.raises(RuntimeError, match="inside a running function"):
        scratch_path()


def test_artifact_output_resolves_below_the_workload_output_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GFAAS_OUTPUT_ROOT", str(tmp_path))
    output = ArtifactOutput(
        "kernel-profile",
        "profiles/kernel.json",
        kind="profile",
        media_type="application/json",
    )

    with output.open("w", encoding="utf-8") as file:
        file.write('{"duration_ms":12}')

    assert output.path == tmp_path / "profiles/kernel.json"
    assert output.path.read_text() == '{"duration_ms":12}'
    assert output.request() == {
        "name": "kernel-profile",
        "path": "profiles/kernel.json",
        "kind": "profile",
        "media_type": "application/json",
        "layout": "blob",
        "publication": "terminal",
        "maximum_versions": 1,
        "required": True,
        "publish_on_failure": True,
    }


@pytest.mark.parametrize(
    "relative_path",
    ["", ".", "/absolute", "../escape", "nested/../escape", "nested\\escape"],
)
def test_artifact_output_rejects_unsafe_paths(relative_path: str) -> None:
    with pytest.raises(ValueError, match="output path"):
        ArtifactOutput("result", relative_path)


def test_artifact_output_path_is_only_available_inside_a_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GFAAS_OUTPUT_ROOT", raising=False)

    with pytest.raises(RuntimeError, match="inside a running function"):
        _ = ArtifactOutput("result", "result.bin").path


@pytest.mark.parametrize("path", ["reports//result.json", "reports/./result.json", "reports/"])
def test_artifact_output_rejects_noncanonical_paths(path: str) -> None:
    with pytest.raises(ValueError, match="invalid Artifact output path"):
        ArtifactOutput("result", path)


def test_directory_output_uses_the_tree_contract() -> None:
    output = ArtifactOutput.directory("weights", "weights")

    assert output.layout == "tree"
    assert output.media_type == TREE_ARTIFACT_MEDIA_TYPE
    assert output.request()["publication"] == "terminal"


def test_checkpoint_publish_writes_contiguous_atomic_markers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GFAAS_OUTPUT_ROOT", str(tmp_path))
    checkpoint = ArtifactCheckpoint("weights", "checkpoints", maximum_versions=2)
    first = checkpoint.path / "step-0001"
    second = checkpoint.path / "step-0002"
    first.mkdir(parents=True)
    second.mkdir()

    assert checkpoint.publish("step-0001") == 1
    assert checkpoint.publish(Path("step-0002")) == 2
    assert checkpoint.request() == {
        "name": "weights",
        "path": "checkpoints",
        "kind": "output",
        "media_type": TREE_ARTIFACT_MEDIA_TYPE,
        "layout": "tree",
        "publication": "checkpoint",
        "maximum_versions": 2,
        "required": False,
        "publish_on_failure": False,
    }
    markers = tmp_path / ".gfaas/checkpoints/weights"
    assert sorted(path.name for path in markers.glob("*.json")) == [
        "00000001.json",
        "00000002.json",
    ]
    with pytest.raises(RuntimeError, match="version limit"):
        checkpoint.publish("step-0002")


def test_checkpoint_rejects_a_name_that_cannot_fit_the_generation_suffix() -> None:
    with pytest.raises(ValueError, match="at most 119"):
        ArtifactCheckpoint("x" * 120, "checkpoints")


def test_checkpoint_rejects_an_overlong_version_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GFAAS_OUTPUT_ROOT", str(tmp_path))
    checkpoint = ArtifactCheckpoint("weights", "checkpoints")

    with pytest.raises(ValueError, match="invalid checkpoint directory"):
        checkpoint.publish("x" * 256)
