import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from gfaas_cli.client import GfaasClient
from gfaas_cli.errors import CliError


class FakeRemote:
    call_id = "call_test"

    def __init__(self) -> None:
        self.cancelled = False

    def wait(self, *, timeout_s=None):
        return {"status": "passed", "timeout": timeout_s}

    def cancel(self, *, reason=None):
        self.cancelled = True
        return {"state": "cancelling", "reason": reason}


class FakeSdkClient:
    def __init__(self) -> None:
        self.remote = FakeRemote()
        self.submission = None
        self.event_pages = [
            {"items": [{"cursor": "1", "type": "state", "state": "queued"}]},
            {"items": [{"cursor": "2", "type": "state", "state": "succeeded"}]},
        ]

    def close(self):
        pass

    def get_capabilities(self):
        return {
            "gpu_pools": [
                {
                    "name": "gb300",
                    "status": "available",
                    "connected_workers": 2,
                    "available_workers": 1,
                }
            ]
        }

    def upload_artifact_directory(self, path, **kwargs):
        self.uploaded_path = Path(path)
        return {"id": "art_workspace", "child_artifact_ids": ["art_child"]}

    def submit(self, **kwargs):
        self.submission = kwargs
        return self.remote

    def list_events(self, call_id, *, after, limit):
        self.last_after = after
        return self.event_pages.pop(0)

    def get_call(self, call_id):
        return {"id": call_id, "state": "queued"}

    def list_call_artifacts(self, call_id):
        return {
            "items": [
                {
                    "name": "profiles",
                    "artifact": {"id": "art_profiles", "layout": "tree"},
                }
            ]
        }

    def download_artifact_directory(self, artifact_id, destination):
        destination = Path(destination)
        destination.mkdir()
        (destination / "kernel.ncu-rep").write_bytes(b"profile")
        return destination


def args(**overrides):
    values = {
        "image": "cuda-nvcc",
        "gpu_count": 1,
        "gpu_type": "gb300",
        "timeout": 600,
        "capacity_wait": 30,
        "cpu_millicores": None,
        "memory": None,
        "storage": None,
        "shared_memory": None,
        "max_log": None,
        "max_output": None,
        "env": [],
        "idempotency_key": "request-1",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class ClientTests(unittest.TestCase):
    def test_resolve_gpu_type_uses_only_configured_pool(self):
        client = GfaasClient(FakeSdkClient(), poll_interval=0)
        self.assertEqual(client.resolve_gpu_type(None), "gb300")
        self.assertEqual(client.resolve_gpu_type("b200"), "b200")

    def test_resolve_gpu_type_requires_selection_when_multiple_pools_exist(self):
        sdk = FakeSdkClient()
        sdk.get_capabilities = lambda: {"gpu_pools": [{"name": "gb300"}, {"name": "h200"}]}
        client = GfaasClient(sdk, poll_interval=0)
        with self.assertRaisesRegex(CliError, "more than one GPU pool"):
            client.resolve_gpu_type(None)

    def test_submit_uses_tree_artifact_durable_call_and_declared_profile_output(self):
        sdk = FakeSdkClient()
        client = GfaasClient(sdk, poll_interval=0)
        with tempfile.TemporaryDirectory() as temporary:
            remote = client.submit_job(
                job={"target": {"kind": "custom"}},
                workspace=Path(temporary),
                args=args(),
                app_name="gfaas-custom",
            )

        self.assertIs(remote, sdk.remote)
        self.assertEqual(sdk.submission["gpu_type"], "gb300")
        self.assertEqual(sdk.submission["gpu_count"], 1)
        self.assertEqual(sdk.submission["idempotency_key"], "request-1")
        self.assertEqual(sdk.submission["kwargs"]["workspace"].artifact_id, "art_workspace")
        self.assertEqual(sdk.submission["outputs"][0].name, "profiles")

    def test_wait_resumes_events_by_cursor_and_fetches_result(self):
        sdk = FakeSdkClient()
        client = GfaasClient(sdk, poll_interval=0)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            result = client.wait_for_result(sdk.remote, timeout_s=1)

        self.assertEqual(result["status"], "passed")
        self.assertEqual(sdk.last_after, "1")
        self.assertIn("state=queued", stderr.getvalue())
        self.assertIn("state=succeeded", stderr.getvalue())

    def test_profile_download_refuses_to_replace_existing_file(self):
        sdk = FakeSdkClient()
        client = GfaasClient(sdk, poll_interval=0)
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary, "profiles")
            destination.mkdir()
            (destination / "kernel.ncu-rep").write_bytes(b"existing")
            with self.assertRaisesRegex(CliError, "refusing to replace"):
                client.download_profiles("call_test", destination)


if __name__ == "__main__":
    unittest.main()
