"""Exercise a real truncated HTTP response, then recovery with a fresh client."""

from __future__ import annotations

import base64
import hashlib
import os
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from gfaas.client import Client
from gfaas.config import ClientConfig
from gfaas.errors import GfaasError


def test_download_resumes_after_retry_exhaustion_and_new_client(tmp_path: Path) -> None:
    payload = bytes(range(256)) * 4096
    digest = "sha-256=:" + base64.b64encode(hashlib.sha256(payload).digest()).decode() + ":"
    offsets: list[int] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            match = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers["Range"])
            assert match is not None
            offset, end = map(int, match.groups())
            end = min(end, len(payload) - 1)
            offsets.append(offset)
            if offset:
                assert self.headers["If-Match"] == '"fixed"'
            self.send_response(206)
            self.send_header("Content-Length", str(end - offset + 1))
            self.send_header("Content-Range", f"bytes {offset}-{end}/{len(payload)}")
            self.send_header("Repr-Digest", digest)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("ETag", '"fixed"')
            self.end_headers()
            # Each of the first four connections delivers a valid prefix, then closes early.
            self.wfile.write(
                payload[offset : offset + 65536] if len(offsets) <= 4 else payload[offset : end + 1]
            )
            self.wfile.flush()
            self.close_connection = True

        def log_message(self, *_: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = ClientConfig(
        f"http://127.0.0.1:{server.server_port}",
        "test",
        0.01,
        5,
        transfer_cache_dir=tmp_path / "cache",
    )
    target = tmp_path / "result"
    try:
        with Client(config) as client, pytest.raises(GfaasError, match="retry to resume"):
            client.download_artifact_file("art_resume", target)
        assert not target.exists()
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from gfaas import Client, ClientConfig; from pathlib import Path; import sys; "
                "c=Client(ClientConfig(sys.argv[1], 'test', 0.01, 5, transfer_cache_dir=Path(sys.argv[2]))); "
                "c.download_artifact_file('art_resume', sys.argv[3]); c.close()",
                config.api_base,
                str(tmp_path / "cache"),
                str(target),
            ],
            check=True,
            timeout=15,
            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
        )
        assert target.read_bytes() == payload
        assert offsets == [0, 65536, 131072, 196608, 262144]
        assert not list((tmp_path / "cache").rglob("body"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("fault", ["wrong_range", "auth", "corrupt", "existing"])
def test_download_never_exposes_unverified_or_overwritten_bytes(tmp_path: Path, fault: str) -> None:
    payload = b"content"
    digest = "sha-256=:" + base64.b64encode(hashlib.sha256(payload).digest()).decode() + ":"
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if fault == "auth":
            return httpx.Response(403)
        return httpx.Response(
            206,
            content=b"damaged" if fault == "corrupt" else payload,
            headers={
                "Content-Range": "bytes 1-7/8" if fault == "wrong_range" else "bytes 0-6/7",
                "Content-Length": "7",
                "Repr-Digest": digest,
            },
        )

    with Client(ClientConfig("https://example.invalid", None, 0.01, 5)) as client:
        client._http.close()
        client._http = httpx.Client(
            base_url="https://example.invalid", transport=httpx.MockTransport(handler)
        )
        target = tmp_path / "result"
        if fault == "existing":
            target.write_bytes(b"keep")
            with pytest.raises(FileExistsError):
                client.download_artifact_file("art_test", target)
            assert target.read_bytes() == b"keep"
            assert not requests
        else:
            with pytest.raises(GfaasError):
                client.download_artifact_file("art_test", target)
            assert not target.exists()
            assert len(requests) == 1
