"""Argument and result serialization for gfaas blobs."""

from __future__ import annotations

import pickle
from typing import Any

import cloudpickle

from .errors import SerializationError

CLOUDPICKLE_CODEC = "cloudpickle-v1"
PICKLE_CODEC = "pickle-v1"


def _codec(codec: str | None) -> Any:
    if codec is None or codec.startswith("cloudpickle"):
        return cloudpickle
    if codec.startswith("pickle"):
        return pickle
    raise SerializationError(f"unsupported serialization codec {codec!r}")


def encode_args(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    codec: str = CLOUDPICKLE_CODEC,
) -> tuple[bytes, str]:
    """Encode call arguments using the worker's explicit argument envelope."""
    try:
        payload = _codec(codec).dumps({"args": list(args), "kwargs": dict(kwargs)})
    except Exception as error:
        raise SerializationError(f"could not serialize call arguments: {error}") from error
    return payload, codec


def decode_result(payload: bytes, *, codec: str | None) -> Any:
    """Decode a result blob according to its declared codec."""
    try:
        return _codec(codec).loads(payload)
    except SerializationError:
        raise
    except Exception as error:
        raise SerializationError(
            f"could not deserialize result blob with codec {codec!r}: {error}"
        ) from error
