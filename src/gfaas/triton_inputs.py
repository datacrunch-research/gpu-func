"""JSON metadata contract supplied to TritonKernel input generators."""

from __future__ import annotations

from typing import Any, TypedDict


class TritonInputMetadata(TypedDict):
    args: list[dict[str, Any]]
    kwargs: dict[str, dict[str, Any]]


def capture_inputs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> TritonInputMetadata:
    groups: dict[int, int] = {}

    def describe(value: Any) -> dict[str, Any]:
        if hasattr(value, "untyped_storage") and hasattr(value, "dtype"):
            if str(value.layout) != "torch.strided":
                raise ValueError("Input generation currently supports strided tensors")
            storage = value.untyped_storage()
            identity = int(storage.data_ptr())
            group = groups.setdefault(identity, len(groups))
            return {
                "kind": "tensor",
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "dtype": str(value.dtype).removeprefix("torch."),
                "storage_offset": value.storage_offset(),
                "storage_group": group,
            }
        if value is None or type(value) in (int, float, bool, str):
            return {"kind": "value", "value": value}
        raise TypeError("Kernel inputs must be tensors or scalar values")

    return {
        "args": [describe(v) for v in args],
        "kwargs": {k: describe(v) for k, v in kwargs.items()},
    }
