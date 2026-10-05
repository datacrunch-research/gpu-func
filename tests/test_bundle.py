"""Tests for self-contained gfaas source bundles."""

from __future__ import annotations

import importlib.util
import io
import sys
import tarfile
from pathlib import Path

import pytest

import gfaas.bundle as sdk_bundle
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


def test_bundle_from_projected_sdk_keeps_importable_package_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "gfaas"
    package.mkdir()
    revision = package / "..2026_10_05"
    revision.mkdir()
    (revision / "bundle.py").write_text(Path(sdk_bundle.__file__).read_text())
    (revision / "__init__.py").write_text("")
    (revision / "app.py").write_text("class App: pass\n")
    for name in ("bundle.py", "__init__.py", "app.py"):
        (package / name).symlink_to(Path("..2026_10_05") / name)
    spec = importlib.util.spec_from_file_location("projected_bundle", package / "bundle.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    source = tmp_path / "kernel.py"
    source.write_text("def run(): return 1\n")
    bundle = module.package_single_file(source)
    with tarfile.open(fileobj=io.BytesIO(bundle.data), mode="r:gz") as archive:
        names = set(archive.getnames())
    assert "gfaas/__init__.py" in names
    assert "gfaas/app.py" in names
    assert not any("..2026_10_05" in name for name in names)
