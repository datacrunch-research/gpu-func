import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from gfaas import ArtifactOutput, ArtifactRef

from gpu_func_cli.worker_job import _detect_cuda_arch, _run_process, run


class WorkerJobTests(unittest.TestCase):
    def test_course_job_uses_staged_tree_and_cleans_scratch_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact_root = root / "artifacts"
            workspace = artifact_root / "art_workspace"
            output_root = root / "outputs"
            scratch_root = root / "scratch"
            workspace.mkdir(parents=True)
            output_root.mkdir()
            scratch_root.mkdir()
            source = workspace / "run.py"
            source.write_text(
                "import json\n"
                "from pathlib import Path\n"
                "Path('report.json').write_text(json.dumps({'passed': True}))\n",
                encoding="utf-8",
            )
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            job = {
                "remote": {"timeout_s": 10},
                "target": {"kind": "exercise"},
                "hashes": {"run.py": digest},
                "course_runner": {
                    "enabled": True,
                    "cwd": ".",
                    "command": [sys.executable, "run.py"],
                    "json_out": "report.json",
                    "artifact_globs": ["report.json"],
                },
            }
            output = ArtifactOutput.directory(
                "profiles",
                "profiles",
                kind="profile",
                required=False,
            )
            environment = {
                "GFAAS_ARTIFACT_ROOT": str(artifact_root),
                "GFAAS_OUTPUT_ROOT": str(output_root),
                "GFAAS_SCRATCH_ROOT": str(scratch_root),
            }
            with mock.patch.dict(os.environ, environment):
                result = run(
                    job=job,
                    workspace=ArtifactRef("art_workspace"),
                    profile_output=output,
                )

            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["report_json"], {"passed": True})
            self.assertEqual(list(scratch_root.iterdir()), [])

    def test_process_timeout_terminates_the_process_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = _run_process(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                Path(temporary),
                0.01,
            )

        self.assertTrue(result["timed_out"])
        self.assertIsNone(result["returncode"])

    @mock.patch("gpu_func_cli.worker_job.subprocess.run")
    def test_cuda_arch_detection_uses_compute_capability(self, run_process):
        run_process.return_value = mock.Mock(stdout="10.3\n")
        self.assertEqual(_detect_cuda_arch(), "sm_103")


if __name__ == "__main__":
    unittest.main()
