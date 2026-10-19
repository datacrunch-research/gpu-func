"""Transport tensor return values together with input writes and storage aliases."""

from __future__ import annotations

from typing import Any

from .triton_inputs import _storage_tensor, apply_writes, capture_inputs, snapshot_inputs


def snapshot_result(value: Any, args: tuple[Any, ...]) -> dict[str, Any]:
    """Preserve nested return containers and aliases to supplied input tensors."""
    leaves: list[Any] = []

    def encode(item: Any) -> dict[str, Any]:
        if hasattr(item, "untyped_storage"):
            index = len(args) + len(leaves)
            leaves.append(item)
            return {"kind": "tensor", "index": index}
        if item is None or type(item) in (bool, int, float, str):
            return {"kind": "value", "value": item}
        if type(item) in (list, tuple):
            return {"kind": type(item).__name__, "items": [encode(v) for v in item]}
        if type(item) is dict and all(type(k) in (str, int) for k in item):
            return {"kind": "dict", "items": [(k, encode(v)) for k, v in item.items()]}
        raise TypeError(
            "Kernel returns must contain tensors, scalar values, lists, tuples or dicts"
        )

    tree = encode(value)
    return {
        "snapshot": snapshot_inputs((*args, *leaves), {}),
        "return": tree,
        "input_count": len(args),
    }


def apply_result(payload: dict[str, Any], args: tuple[Any, ...]) -> Any:
    """Validate returned views, apply input writes, then reconstruct the result."""
    import torch  # type: ignore[import-not-found]

    if payload["input_count"] != len(args):
        raise ValueError("Returned input count differs from the invocation")
    snapshot = payload["snapshot"]
    specs = snapshot["metadata"]["args"]
    original = capture_inputs(args, {})
    if specs[: len(args)] != original["args"] or snapshot["metadata"]["kwargs"]:
        raise ValueError("Returned input metadata differs from the invocation")
    buffers = {
        group: torch.frombuffer(bytearray(data), dtype=torch.uint8).clone()
        if data
        else torch.empty(0, dtype=torch.uint8)
        for group, data in snapshot["storages"].items()
    }
    for spec in specs:
        if spec["kind"] != "tensor":
            continue
        shape, stride, offset = spec["shape"], spec["stride"], spec["storage_offset"]
        if len(shape) != len(stride) or offset < 0 or any(v < 0 for v in (*shape, *stride)):
            raise ValueError("Invalid returned tensor view")
        dtype = getattr(torch, spec["dtype"])
        element_size = torch.empty(0, dtype=dtype).element_size()
        extent = (
            offset
            if 0 in shape
            else offset + 1 + sum((d - 1) * s for d, s in zip(shape, stride, strict=True))
        )
        if extent * element_size > buffers[spec["storage_group"]].numel():
            raise ValueError("Returned tensor view exceeds its storage")
    device = next((v.device for v in args if isinstance(v, torch.Tensor)), torch.device("cpu"))
    input_groups: dict[int, Any] = {}
    for value, spec in zip(args, original["args"], strict=True):
        if spec["kind"] == "tensor":
            input_groups.setdefault(spec["storage_group"], _storage_tensor(value, torch))
    output_buffers = {
        group: buffer.to(device) for group, buffer in buffers.items() if group not in input_groups
    }
    all_buffers = {**output_buffers, **input_groups}

    def decode(node: dict[str, Any]) -> Any:
        kind = node["kind"]
        if kind == "value":
            return node["value"]
        if kind == "tensor":
            index = node["index"]
            if type(index) is not int or not len(args) <= index < len(specs):
                raise ValueError("Invalid returned tensor index")
            spec = specs[index]
            for value, original_spec in zip(args, original["args"], strict=True):
                if original_spec == spec:
                    return value
            buffer = all_buffers[spec["storage_group"]]
            return torch.empty(0, dtype=getattr(torch, spec["dtype"]), device=buffer.device).set_(
                buffer.untyped_storage(),
                spec["storage_offset"],
                tuple(spec["shape"]),
                tuple(spec["stride"]),
            )
        if kind in ("list", "tuple"):
            values = [decode(n) for n in node["items"]]
            return values if kind == "list" else tuple(values)
        if kind == "dict":
            return {k: decode(n) for k, n in node["items"]}
        raise ValueError("Invalid returned container kind")

    result = decode(payload["return"])
    apply_writes(
        {"metadata": original, "storages": {g: snapshot["storages"][g] for g in input_groups}},
        args,
        {},
    )
    return result
