import hashlib
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from gfaas import ArtifactOutput, ArtifactRef
from gfaas_cli.worker_job import (
    COMPILED_CUSTOM_ARTIFACT_ENV,
    _detect_cuda_arch,
    _run_process,
    compile_custom_stage,
    execute_custom_stage,
    run,
)


class WorkerJobTests(unittest.TestCase):
    def test_custom_stages_publish_and_execute_the_compiled_workspace(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact_root = root / "artifacts"
            workspace = artifact_root / "art_workspace"
            output_root = root / "outputs"
            scratch_root = root / "scratch"
            workspace.mkdir(parents=True)
            output_root.mkdir()
            scratch_root.mkdir()
            source = workspace / "kernel.cu"
            source.write_text("int main() {}\n", encoding="utf-8")
            job = {
                "remote": {
                    "timeout_s": 10,
                    "image": "cuda-nvcc",
                    "gpu_type": "gb300",
                },
                "target": {"kind": "custom"},
                "hashes": {"kernel.cu": hashlib.sha256(source.read_bytes()).hexdigest()},
                "custom": {
                    "command": "run",
                    "sources": ["kernel.cu"],
                    "flags": ["-arch=sm_103"],
                    "output": "kernel",
                    "program_args": [],
                },
            }
            environment = {
                "GFAAS_ARTIFACT_ROOT": str(artifact_root),
                "GFAAS_OUTPUT_ROOT": str(output_root),
                "GFAAS_SCRATCH_ROOT": str(scratch_root),
            }

            def compile_success(workdir, _job, _deadline):
                binary = workdir / "kernel"
                binary.write_text("compiled", encoding="utf-8")
                binary.chmod(0o755)
                return {
                    "args": ["nvcc", "-arch=sm_103", "kernel.cu", "-o", "kernel"],
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "ms": 2,
                    "timed_out": False,
                }

            with (
                mock.patch.dict(os.environ, environment),
                mock.patch(
                    "gfaas_cli.worker_job._compile_custom_job",
                    side_effect=compile_success,
                ),
            ):
                compile_result = compile_custom_stage(
                    job=job,
                    workspace=ArtifactRef("art_workspace"),
                )

            self.assertEqual(compile_result["returncode"], 0)
            compiled_artifact = artifact_root / "art_compiled"
            shutil.copytree(output_root / "compiled-custom", compiled_artifact)
            environment[COMPILED_CUSTOM_ARTIFACT_ENV] = "art_compiled"
            run_result = {
                "args": ["./kernel"],
                "returncode": 0,
                "stdout": "ran\n",
                "stderr": "",
                "ms": 1,
                "timed_out": False,
            }
            with (
                mock.patch.dict(os.environ, environment),
                mock.patch("gfaas_cli.worker_job._run_process", return_value=run_result),
            ):
                result = execute_custom_stage(
                    job=job,
                    workspace=ArtifactRef("art_workspace"),
                )

            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["compile"], compile_result)
            self.assertEqual(result["run"]["stdout"], "ran\n")
            self.assertEqual(list(scratch_root.iterdir()), [])

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

    @mock.patch("gfaas_cli.worker_job.subprocess.run")
    def test_cuda_arch_detection_uses_compute_capability(self, run_process):
        run_process.return_value = mock.Mock(stdout="10.3\n")
        self.assertEqual(_detect_cuda_arch(), "sm_103")


if __name__ == "__main__":
    unittest.main()
