from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from gfaas import python_runner


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
