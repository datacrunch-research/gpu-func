import argparse
import hashlib
import tempfile
import unittest
from pathlib import Path

from gfaas_cli.errors import CliError
from gfaas_cli.payloads import (
    _build_checkout_payload,
    _build_custom_payload,
    _clean_payload_path,
    _resolve_gpu,
    _walk_checkout,
)


def custom_args(source, **overrides):
    values = {
        "source": str(source),
        "harness": None,
        "report_name": None,
        "nvcc_flags": "-std=c++20 -O3 -lineinfo",
        "custom_command": "profile",
        "output": "custom_kernel",
        "gpu": "B200",
        "image": "cuda-nvcc",
        "timeout": 600,
        "arg": ["--n", "4"],
        "ncu_args": "--set full",
        "no_nvtx_filter": False,
        "nvtx_range": "profile_kernel",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class PayloadTests(unittest.TestCase):
    def test_build_custom_payload_hashes_sources_and_sanitizes_report_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "kernel.cu"
            source.write_text("__global__ void kernel() {}\n", encoding="utf-8")
            harness = tmp_path / "harness.cu"
            harness.write_text("int main() { return 0; }\n", encoding="utf-8")

            payload = _build_custom_payload(
                custom_args(source, harness=str(harness), report_name="bad/name with spaces"),
                gpu_type="b200",
                arch="sm_100a",
            )

            self.assertEqual(payload["target"]["kind"], "custom")
            self.assertEqual(payload["target"]["source"], "kernel.cu")
            self.assertEqual(payload["target"]["harness"], "harness.cu")
            self.assertNotIn(str(tmp_path), repr(payload["target"]))
            self.assertEqual(payload["custom"]["sources"], ["kernel.cu", "harness.cu"])
            self.assertEqual(payload["custom"]["report_name"], "bad_name_with_spaces")
            self.assertEqual(payload["custom"]["flags"][-1], "-arch=sm_100a")
            self.assertEqual(payload["custom"]["ncu_args"], ["--set", "full"])
            self.assertEqual(payload["custom"]["nvtx_range"], "profile_kernel")
            self.assertEqual(payload["files"]["kernel.cu"], source.read_bytes())
            self.assertEqual(
                payload["hashes"]["kernel.cu"], hashlib.sha256(source.read_bytes()).hexdigest()
            )

    def test_build_custom_payload_can_disable_nvtx_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "kernel.cu"
            source.write_text("int main() { return 0; }\n", encoding="utf-8")

            payload = _build_custom_payload(custom_args(source, no_nvtx_filter=True), "b200", "")

            self.assertEqual(payload["custom"]["nvtx_range"], "")
            self.assertEqual(payload["custom"]["flags"], ["-std=c++20", "-O3", "-lineinfo"])

    def test_build_custom_payload_rejects_output_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "kernel.cu"
            source.write_text("int main() { return 0; }\n", encoding="utf-8")
            with self.assertRaisesRegex(CliError, "--output must be a file name"):
                _build_custom_payload(
                    custom_args(source, output="../outside"),
                    "gb300",
                    "sm_103",
                )

    def test_walk_checkout_rejects_symbolic_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "source.cu").write_text("int main() {}\n", encoding="utf-8")
            (root / "linked.cu").symlink_to(root / "source.cu")
            with self.assertRaisesRegex(CliError, "symbolic link"):
                _walk_checkout({}, {}, root, root)

    def test_clean_payload_path_rejects_absolute_and_parent_paths(self):
        self.assertEqual(_clean_payload_path("runner/include/tester.h"), "runner/include/tester.h")

        with self.assertRaises(CliError):
            _clean_payload_path("/tmp/file.cu")
        with self.assertRaises(CliError):
            _clean_payload_path("../file.cu")
        with self.assertRaises(CliError):
            _clean_payload_path("")

    def test_resolve_gpu_uses_defaults_and_overrides(self):
        self.assertEqual(_resolve_gpu("B200", None, None), ("b200", "sm_100a"))
        self.assertEqual(_resolve_gpu("unknown", None, None), ("unknown", ""))
        self.assertEqual(
            _resolve_gpu("B200", "manual-type", "manual-arch"), ("manual-type", "manual-arch")
        )

    def test_build_checkout_payload_skips_solutions_and_adds_external_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            course = tmp_path / "course"
            runner = course / "runner"
            exercise = course / "exercises" / "99-demo"
            (runner / "include").mkdir(parents=True)
            (exercise / "solutions").mkdir(parents=True)
            (runner / "cli.py").write_text("print('runner')\n", encoding="utf-8")
            (runner / "include" / "tester.h").write_text("// tester\n", encoding="utf-8")
            (runner / "include" / "reporter.h").write_text("// reporter\n", encoding="utf-8")
            (exercise / "run.py").write_text("print('run')\n", encoding="utf-8")
            (exercise / "starter.cu").write_text("// starter\n", encoding="utf-8")
            (exercise / "solutions" / "skip.cu").write_text("// skip\n", encoding="utf-8")
            source = tmp_path / "solution.cu"
            source.write_text("// submitted\n", encoding="utf-8")

            payload = _build_checkout_payload(
                course_root=course,
                exercise_id="99-demo",
                mode="test",
                source_file=source,
                specs=["tests/example.txt"],
                gpu="B200",
                gpu_type="b200",
                arch="sm_100a",
                image="cuda-nvcc",
                timeout_s=600,
                verbose=True,
            )

        self.assertNotIn("exercises/99-demo/solutions/skip.cu", payload["files"])
        self.assertEqual(
            payload["files"]["exercises/99-demo/__submitted__.cu"],
            b"// submitted\n",
        )
        self.assertEqual(
            payload["course_runner"]["command"],
            [
                "python3",
                "run.py",
                "--json",
                "_gfaas_cli.json",
                "--file",
                "__submitted__.cu",
                "-v",
                "--arch",
                "sm_100a",
                "test",
                "tests/example.txt",
            ],
        )
        self.assertEqual(payload["remote"]["arch"], "sm_100a")


if __name__ == "__main__":
    unittest.main()
