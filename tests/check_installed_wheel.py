"""Offline CLI checks: run with the wheel environment's Python and -I, without pytest."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
import sysconfig
import tempfile
import unittest
from pathlib import Path


class InstalledWheelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not sys.flags.isolated or sys.prefix == sys.base_prefix:
            raise RuntimeError("Run with -I in a fresh virtual environment containing the wheel")
        site_packages = Path(sysconfig.get_path("purelib")).resolve()
        for name in ("gfaas", "gfaas_cli", "gfaas_cli.main"):
            module = importlib.import_module(name)
            if not Path(module.__file__).resolve().is_relative_to(site_packages):
                raise RuntimeError(
                    f"{name} did not load from the installed wheel: {module.__file__}"
                )
        distribution = importlib.metadata.distribution("gfaas")
        direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
        if direct_url.get("dir_info", {}).get("editable"):
            raise RuntimeError("An editable installation cannot qualify the wheel")
        cls.cli = Path(sysconfig.get_path("scripts")) / "vfunc"
        if not cls.cli.is_file():
            raise RuntimeError(f"Installed console script is missing: {cls.cli}")

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="vfunc-wheel-check-")
        self.addCleanup(temporary.cleanup)
        self.cwd = Path(temporary.name)
        self.environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("PYTHON", "GFAAS_", "CUDA_", "_ARGCOMPLETE", "COMP_"))
        }
        self.environment.update(
            PYTHONNOUSERSITE="1",
            GFAAS_API_BASE="http://127.0.0.1:1/api",
        )

    def run_cli(self, *arguments: str, extra_env: dict[str, str] | None = None) -> str:
        result = subprocess.run(
            [str(self.cli), *arguments],
            cwd=self.cwd,
            env=self.environment | (extra_env or {}),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stderr, "")
        return result.stdout

    def test_public_sdk_exports_and_console_entrypoint(self) -> None:
        import gfaas

        for name in ("App", "Client", "ArtifactRef", "ArtifactOutput", "ArtifactCheckpoint"):
            self.assertIsNotNone(getattr(gfaas, name))
        entrypoints = {
            entry.name: entry.value
            for entry in importlib.metadata.distribution("gfaas").entry_points
            if entry.group == "console_scripts"
        }
        self.assertEqual(entrypoints, {"vfunc": "gfaas_cli.main:entrypoint"})

    def test_command_family_help(self) -> None:
        commands = [
            (),
            ("run",),
            ("local", "info"),
            ("local", "run"),
            ("call", "show"),
            ("call", "watch"),
            ("call", "logs"),
            ("call", "cancel"),
            ("call", "artifacts"),
            ("artifact", "download"),
            ("pool", "list"),
            ("workers",),
            ("pools",),
            ("custom",),
            ("exercise",),
            ("compile",),
            ("test",),
            ("benchmark",),
            ("sanitizer",),
            ("profile",),
            ("grade",),
            ("report", "summary"),
            ("report", "feedback"),
            ("completion",),
        ]
        for command in commands:
            with self.subTest(command=command):
                output = self.run_cli(*command, "--help")
                self.assertIn("usage: vfunc", output)
                self.assertIn("--help", output)

    def test_shell_completion_scripts(self) -> None:
        for shell, marker in {
            "bash": "_python_argcomplete",
            "zsh": "_python_argcomplete",
            "fish": "__fish_vfunc_complete",
            "powershell": "Register-ArgumentCompleter",
        }.items():
            with self.subTest(shell=shell):
                output = self.run_cli("completion", shell)
                self.assertIn(marker, output)
                self.assertIn("vfunc", output)

    def test_tab_completion_candidates(self) -> None:
        cases = {
            "vfunc ": {"run", "call", "artifact", "custom", "exercise", "report", "completion"},
            "vfunc call ": {"show", "watch", "logs", "cancel", "artifacts"},
            "vfunc artifact ": {"download"},
            "vfunc completion ": {"bash", "zsh", "fish", "powershell"},
        }
        for command_line, expected in cases.items():
            with self.subTest(command_line=command_line):
                output_path = self.cwd / "completions"
                self.run_cli(
                    extra_env={
                        "_ARGCOMPLETE": "1",
                        "_ARGCOMPLETE_IFS": "\n",
                        "_ARGCOMPLETE_STDOUT_FILENAME": str(output_path),
                        "COMP_LINE": command_line,
                        "COMP_POINT": str(len(command_line)),
                        "COMP_TYPE": "9",
                    }
                )
                candidates = {line.strip() for line in output_path.read_text().splitlines()}
                self.assertTrue(expected <= candidates, (expected, candidates))


if __name__ == "__main__":
    unittest.main(verbosity=2)
