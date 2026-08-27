import argparse
import tempfile
import unittest
from pathlib import Path

from gpu_func_cli.errors import CliError
from gpu_func_cli.reports import _cmd_report_feedback


class ReportTests(unittest.TestCase):
    def test_feedback_requires_explicit_trust_before_loading_course_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary, "profile.ncu-rep")
            report.write_bytes(b"report")
            args = argparse.Namespace(
                report=str(report),
                trust_course_code=False,
                course_dir=temporary,
            )

            with self.assertRaisesRegex(CliError, "--trust-course-code"):
                _cmd_report_feedback(args)


if __name__ == "__main__":
    unittest.main()
