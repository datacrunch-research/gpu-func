import tempfile
import unittest
from pathlib import Path

from gfaas_cli.payloads import _materialize_workspace


class WorkspaceTests(unittest.TestCase):
    def test_materialize_workspace_preserves_binary_files_and_removes_inline_data(self):
        payload = {
            "target": {"kind": "custom"},
            "files": {"kernel.cu": b"int main() {}\n", "fixtures/input.bin": b"\x00\xff"},
            "hashes": {"kernel.cu": "digest", "fixtures/input.bin": "binary-digest"},
        }
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary, "workspace")
            job = _materialize_workspace(payload, destination)

            self.assertEqual((destination / "kernel.cu").read_bytes(), b"int main() {}\n")
            self.assertEqual((destination / "fixtures/input.bin").read_bytes(), b"\x00\xff")
            self.assertNotIn("files", job)
            self.assertEqual(job["hashes"], payload["hashes"])


if __name__ == "__main__":
    unittest.main()
