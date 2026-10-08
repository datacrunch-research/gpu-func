from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from gfaas import artifacts, python_runner
from gfaas.bundle import package_single_file


def test_python_runner_imports_as_the_top_level_module_of_its_bundle(tmp_path: Path) -> None:
    bundle = package_single_file(Path(python_runner.__file__))
    root = tmp_path / "bundle"
    with tarfile.open(fileobj=io.BytesIO(bundle.data), mode="r:gz") as archive:
        for member in archive.getmembers():
            source = archive.extractfile(member)
            assert source is not None
            target = root / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read())
    scratch = tmp_path / "scratch"
    outputs = tmp_path / "outputs"
    scratch.mkdir()
    outputs.mkdir()
    # As the worker's runner does: the bundle first on the path, the module imported by name. -S
    # leaves out installed packages, which an image need not contain.
    program = "\n".join(
        [
            "import importlib, json, sys",
            f"sys.path.insert(0, {str(root)!r})",
            f"module = importlib.import_module({bundle.module_name!r})",
            "result = module.run_script(source='print(\"ran\")', filename='hello.py')",
            "print(json.dumps(result))",
        ]
    )
    environment = {name: value for name, value in os.environ.items() if name != "PYTHONPATH"}
    environment[artifacts._SCRATCH_ROOT_ENV] = str(scratch)
    environment["GFAAS_OUTPUT_ROOT"] = str(outputs)

    completed = subprocess.run(
        [sys.executable, "-S", "-c", program],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.splitlines()
    assert lines[0] == "ran"
    assert json.loads(lines[-1])["returncode"] == 0


def test_python_runner_executes_script_in_the_output_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scratch = tmp_path / "scratch"
    outputs = tmp_path / "outputs"
    scratch.mkdir()
    outputs.mkdir()
    monkeypatch.setenv("GFAAS_SCRATCH_ROOT", str(scratch))
    monkeypatch.setenv("GFAAS_OUTPUT_ROOT", str(outputs))
    calls: list[dict[str, object]] = []

    def run(command, **kwargs):
        script = Path(command[2])
        calls.append({"command": command, "source": script.read_text(), **kwargs})
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(python_runner.subprocess, "run", run)

    result = python_runner.run_script(
        source="print('hello')\n",
        filename="experiment.py",
        program_args=["--steps", "10"],
    )

    assert result["phase"] == "run"
    assert result["returncode"] == 0
    assert calls[0]["command"][1] == "-u"
    assert calls[0]["command"][-2:] == ["--steps", "10"]
    assert calls[0]["source"] == "print('hello')\n"
    assert calls[0]["cwd"] == str(outputs)
    assert calls[0]["check"] is False


def test_python_runner_fails_the_call_for_a_nonzero_script_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("GFAAS_SCRATCH_ROOT", str(tmp_path))
    monkeypatch.setenv("GFAAS_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setattr(
        python_runner.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 7),
    )

    with pytest.raises(RuntimeError, match="stopped with status 7"):
        python_runner.run_script(source="raise SystemExit(7)\n", filename="experiment.py")


@pytest.mark.parametrize("filename", ["../experiment.py", "/tmp/experiment.py", ""])
def test_python_runner_rejects_unsafe_filenames(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    filename: str,
) -> None:
    monkeypatch.setenv("GFAAS_SCRATCH_ROOT", str(tmp_path))
    monkeypatch.setenv("GFAAS_OUTPUT_ROOT", str(tmp_path))

    with pytest.raises(ValueError, match="filename"):
        python_runner.run_script(source="", filename=filename)
