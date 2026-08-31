"""Tests for self-contained gfaas source bundles."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

from gfaas.bundle import empty_bundle, package_single_file


def test_bundle_is_deterministic_and_vendors_gfaas(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = first_dir / "kernel.py"
    second = second_dir / "kernel.py"
    first.write_text("def run():\n    return 1\n")
    second.write_text("def run():\n    return 1\n")

    first_bundle = package_single_file(first)
    second_bundle = package_single_file(second)

    assert first_bundle.data == second_bundle.data
    assert first_bundle.sha256 == second_bundle.sha256
    assert first_bundle.module_name == "kernel"
    with tarfile.open(fileobj=io.BytesIO(first_bundle.data), mode="r:gz") as archive:
        names = set(archive.getnames())
    assert "kernel.py" in names
    assert "gfaas/__init__.py" in names
    assert "gfaas/app.py" in names
    assert not any(name.startswith("fast_containers/") for name in names)


def test_empty_bundle_is_deterministic() -> None:
    first = empty_bundle()
    second = empty_bundle()

    assert first == second
    with tarfile.open(fileobj=io.BytesIO(first.data), mode="r:gz") as archive:
        assert archive.getnames() == [".gfaas-empty"]
