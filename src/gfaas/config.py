"""Client-side configuration for the gfaas SDK."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from . import credentials

DEFAULT_API_BASE = "http://127.0.0.1:8000/api"


@dataclass(frozen=True)
class ClientConfig:
    api_base: str
    api_key: str | None
    poll_interval_s: float
    request_timeout_s: float

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
        )
