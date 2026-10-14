"""Client-side configuration for the gfaas SDK."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from . import credentials

DEFAULT_API_BASE = "http://127.0.0.1:8000/api"


@dataclass(frozen=True)
class ClientConfig:
    api_base: str
    api_key: str | None
    poll_interval_s: float
    request_timeout_s: float
    upload_concurrency: int = 4
    upload_completion_timeout_s: float = 600
    transfer_cache_dir: Path | None = None
    download_concurrency: int = 4

    def __post_init__(self) -> None:
        if type(self.download_concurrency) is not int or not 1 <= self.download_concurrency <= 16:
            raise ValueError("download_concurrency must be an integer from 1 to 16")
        if type(self.upload_concurrency) is not int or not 1 <= self.upload_concurrency <= 16:
            raise ValueError("upload_concurrency must be an integer from 1 to 16")
        if (
            isinstance(self.upload_completion_timeout_s, bool)
            or not math.isfinite(self.upload_completion_timeout_s)
            or self.upload_completion_timeout_s <= 0
        ):
            raise ValueError("upload_completion_timeout_s must be positive and finite")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ClientConfig:
        """Read GFAAS_* variables, then any `vfunc login` profile for the same API base.

        GFAAS_API_KEY and GFAAS_API_BASE take precedence. A stored key is used only when its
        profile's API base is the selected one.
        """
        env = env if env is not None else dict(os.environ)
        requested_base = env.get("GFAAS_API_BASE")
        requested_base = requested_base.rstrip("/") if requested_base else None
        api_base = requested_base or DEFAULT_API_BASE
        api_key = env.get("GFAAS_API_KEY") or None
        if api_key is None:
            profile = credentials.select(env, requested_base)
            if profile is not None and profile.api_base == (requested_base or profile.api_base):
                api_base = profile.api_base
                api_key = profile.api_key
        return cls(
            api_base=api_base,
            api_key=api_key,
            poll_interval_s=float(env.get("GFAAS_POLL_INTERVAL", "0.5")),
            request_timeout_s=float(env.get("GFAAS_REQUEST_TIMEOUT", "60")),
            transfer_cache_dir=Path(env["GFAAS_TRANSFER_CACHE_DIR"])
            if env.get("GFAAS_TRANSFER_CACHE_DIR")
            else None,
            download_concurrency=int(env.get("GFAAS_DOWNLOAD_CONCURRENCY", "4")),
            upload_concurrency=int(env.get("GFAAS_UPLOAD_CONCURRENCY", "4")),
            upload_completion_timeout_s=float(env.get("GFAAS_UPLOAD_COMPLETION_TIMEOUT", "600")),
        )
