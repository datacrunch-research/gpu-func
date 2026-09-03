import argparse
import tempfile
import unittest
from pathlib import Path

from gfaas_cli.commands import _submit_payload, _write_result_json
from gfaas_cli.errors import CliError


class FakeRemote:
    call_id = "call_test"


class FakeClient:
    def submit_job(self, *, job, workspace, args, app_name):
        self.job = job
        self.workspace_bytes = (workspace / "kernel.cu").read_bytes()
        self.app_name = app_name
        return FakeRemote()

    def wait_for_result(self, remote, *, timeout_s, json_events):
        return {"status": "passed"}

    def profile_publications(self, call_id):
        return []

    def download_profiles(self, call_id, destination):
        raise AssertionError("profile download was not requested")


def args(**overrides):
    values = {
        "gpu_type": "gb300",
        "gpu_count": 1,
        "image": "cuda-nvcc",
        "detach": False,
        "wait_timeout": 30,
        "json_events": False,
        "artifact_dir": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class CommandTests(unittest.TestCase):
    def test_submit_materializes_workspace_then_uses_durable_call(self):
        client = FakeClient()
        payload = {
            "target": {"kind": "custom"},
            "hashes": {"kernel.cu": "digest"},
            "files": {"kernel.cu": b"int main() {}\n"},
        }

        result, call_id, downloaded = _submit_payload(
            client,
            args(),
            payload,
            app_name="gfaas-custom",
            label="custom run",
            arch="sm_103",
        )

        self.assertEqual(result, {"status": "passed"})
        self.assertEqual(call_id, "call_test")
        self.assertEqual(downloaded, [])
        self.assertEqual(client.workspace_bytes, b"int main() {}\n")
        self.assertNotIn("files", client.job)

    def test_detach_returns_after_submission(self):
        client = FakeClient()
        result, call_id, downloaded = _submit_payload(
            client,
            args(detach=True),
            {
                "target": {"kind": "custom"},
                "hashes": {"kernel.cu": "digest"},
                "files": {"kernel.cu": b"int main() {}\n"},
            },
            app_name="gfaas-custom",
            label="custom run",
            arch="worker-detected",
        )

        self.assertIsNone(result)
        self.assertEqual(call_id, "call_test")
        self.assertEqual(downloaded, [])

    def test_json_result_refuses_to_replace_existing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary, "result.json")
            _write_result_json(path, {"status": "passed"})
            with self.assertRaisesRegex(CliError, "refusing to replace"):
                _write_result_json(path, {"status": "passed"})


if __name__ == "__main__":
    unittest.main()
