"""Client-side configuration for the gfaas SDK."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ClientConfig:
    api_base: str
    api_key: str | None
    poll_interval_s: float
    request_timeout_s: float

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> ClientConfig:
        env = env if env is not None else dict(os.environ)
        return cls(
            api_base=env.get("GFAAS_API_BASE", "http://127.0.0.1:8000").rstrip("/"),
            api_key=env.get("GFAAS_API_KEY") or None,
            poll_interval_s=float(env.get("GFAAS_POLL_INTERVAL", "0.5")),
            request_timeout_s=float(env.get("GFAAS_REQUEST_TIMEOUT", "60")),
        )
