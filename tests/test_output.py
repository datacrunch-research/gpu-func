import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from gpu_func_cli.constants import RC_OK, RC_SETUP
from gpu_func_cli.output import _print_custom_result


class OutputTests(unittest.TestCase):
    def test_print_custom_result_reports_downloaded_profile_artifact(self):
        result = {
            "status": "passed",
            "compile": {"returncode": 0, "args": ["nvcc"], "stdout": ""},
            "run": {
                "returncode": 0,
                "args": ["ncu"],
                "stdout": "ok\n",
                "stderr": "",
                "timed_out": False,
            },
            "artifacts": {"profiles": [{"filename": "custom.ncu-rep", "output_name": "profiles"}]},
        }
        stdout = io.StringIO()
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(verbose=False, artifact_dir=tmp, custom_command="profile")
            report_path = Path(tmp) / "custom.ncu-rep"
            report_path.write_bytes(b"report bytes")
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = _print_custom_result(
                    result,
                    args,
                    downloaded_profiles=[report_path],
                )
            self.assertEqual(report_path.read_bytes(), b"report bytes")

        self.assertEqual(code, RC_OK)
        self.assertIn("Custom profile passed", stdout.getvalue())
        self.assertIn("profile report", stderr.getvalue())

    def test_custom_setup_error_is_visible(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = _print_custom_result(
                {"status": "setup_error", "error": "nvcc is unavailable"},
                argparse.Namespace(custom_command="run", verbose=False),
            )

        self.assertEqual(code, RC_SETUP)
        self.assertIn("nvcc is unavailable", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
