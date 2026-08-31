"""Tests for the gfaas argument and result codecs."""

from __future__ import annotations

import pickle

import cloudpickle
import pytest

from gfaas.errors import SerializationError
from gfaas.serialization import (
    CLOUDPICKLE_CODEC,
    PICKLE_CODEC,
    decode_result,
    encode_args,
)


def test_argument_envelope_round_trips() -> None:
    payload, codec = encode_args((1, "two"), {"three": 3})

    assert codec == CLOUDPICKLE_CODEC
    assert pickle.loads(payload) == {
        "args": [1, "two"],
        "kwargs": {"three": 3},
    }


@pytest.mark.parametrize(
    ("codec", "serializer"),
    [
        (CLOUDPICKLE_CODEC, cloudpickle),
        (PICKLE_CODEC, pickle),
    ],
)
def test_result_round_trips_declared_codec(codec, serializer) -> None:
    value = {"values": [1, 2, 3], "ok": True}
    assert decode_result(serializer.dumps(value), codec=codec) == value


def test_unknown_codec_is_rejected() -> None:
    with pytest.raises(SerializationError, match="unsupported serialization codec"):
        decode_result(pickle.dumps("value"), codec="invented-v1")


def test_invalid_payload_is_a_serialization_error() -> None:
    with pytest.raises(SerializationError, match="could not deserialize"):
        decode_result(b"not a pickle", codec=CLOUDPICKLE_CODEC)
