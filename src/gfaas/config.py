"""Client-side configuration for the gfaas SDK."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ClientConfig:
    api_base: str
    api_key: str | None
    poll_interval_s: float
    request_timeout_s: float
    upload_concurrency: int = 4
    upload_completion_timeout_s: float = 600

    def __post_init__(self) -> None:
        if type(self.upload_concurrency) is not int or not 1 <= self.upload_concurrency <= 16:
            raise ValueError("upload_concurrency must be an integer from 1 to 16")
        if (
            isinstance(self.upload_completion_timeout_s, bool)
            or not math.isfinite(self.upload_completion_timeout_s)
            or self.upload_completion_timeout_s <= 0
        ):
            raise ValueError("upload_completion_timeout_s must be positive and finite")

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> ClientConfig:
        env = env if env is not None else dict(os.environ)
        return cls(
            api_base=env.get("GFAAS_API_BASE", "http://127.0.0.1:8000/api").rstrip("/"),
            api_key=env.get("GFAAS_API_KEY") or None,
            poll_interval_s=float(env.get("GFAAS_POLL_INTERVAL", "0.5")),
            request_timeout_s=float(env.get("GFAAS_REQUEST_TIMEOUT", "60")),
            upload_concurrency=int(env.get("GFAAS_UPLOAD_CONCURRENCY", "4")),
            upload_completion_timeout_s=float(env.get("GFAAS_UPLOAD_COMPLETION_TIMEOUT", "600")),
        )
