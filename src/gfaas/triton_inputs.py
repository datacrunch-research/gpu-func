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
            identity = int(getattr(storage, "_cdata", storage.data_ptr()))
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


def _storage_tensor(value: Any, torch: Any) -> Any:
    storage = value.untyped_storage()
    return torch.empty(0, dtype=torch.uint8, device=value.device).set_(
        storage, 0, (storage.nbytes(),), (1,)
    )


def snapshot_inputs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Copy complete storages to host bytes, preserving views and storage aliases."""
    import ctypes

    metadata = capture_inputs(args, kwargs)
    tensors = [v for v in (*args, *kwargs.values()) if hasattr(v, "untyped_storage")]
    storages: dict[int, bytes] = {}
    if tensors:
        import torch  # type: ignore[import-not-found]

        for value in tensors:
            if value.requires_grad or value.is_conj() or value.is_neg():
                raise ValueError(
                    "Triton inputs must be ordinary tensors without gradients or view bits"
                )
        for value, spec in zip(
            (*args, *kwargs.values()),
            (*metadata["args"], *metadata["kwargs"].values()),
            strict=True,
        ):
            if spec["kind"] == "tensor" and spec["storage_group"] not in storages:
                host = _storage_tensor(value, torch).cpu()
                storages[spec["storage_group"]] = ctypes.string_at(host.data_ptr(), host.numel())
    return {"metadata": metadata, "storages": storages}


class SnapshotInputs:
    """Clone supplied storage on the assigned GPU; restore from a host snapshot."""

    def __init__(
        self, snapshot: dict[str, Any], reset: list[str], restore: list[str], names: list[str]
    ):
        import torch

        self.torch, self.snapshot = torch, snapshot
        self.reset_names, self.restore_names, self.names = reset, restore, names
        self.host = {
            group: torch.frombuffer(bytearray(data), dtype=torch.uint8)
            if data
            else torch.empty(0, dtype=torch.uint8)
            for group, data in snapshot["storages"].items()
        }

    def __call__(self, metadata: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
        torch = self.torch
        buffers = {group: host.to(device="cuda", copy=True) for group, host in self.host.items()}

        def rebuild(spec: dict[str, Any]) -> Any:
            if spec["kind"] == "value":
                return spec["value"]
            return torch.empty(0, dtype=getattr(torch, spec["dtype"]), device="cuda").set_(
                buffers[spec["storage_group"]].untyped_storage(),
                spec["storage_offset"],
                tuple(spec["shape"]),
                tuple(spec["stride"]),
            )

        return tuple(rebuild(s) for s in metadata["args"]), {
            name: rebuild(s) for name, s in metadata["kwargs"].items()
        }

    def reset(self, *args: Any, **kwargs: Any) -> None:
        values = {**dict(zip(self.names, args, strict=False)), **kwargs}
        metadata = self.snapshot["metadata"]
        specs = {**dict(zip(self.names, metadata["args"], strict=False)), **metadata["kwargs"]}
        restored = set()
        for name in self.restore_names:
            value = values.get(name)
            if value is None:
                continue
            group = specs[name]["storage_group"]
            if group not in restored:
                _storage_tensor(value, self.torch).copy_(self.host[group])
                restored.add(group)
        for name in self.reset_names:
            value = values.get(name)
            if value is not None:
                value.zero_()


def apply_writes(snapshot: dict[str, Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    """Write each returned storage once into the original client tensor objects."""
    if not snapshot["storages"]:
        if capture_inputs(args, kwargs) != snapshot["metadata"]:
            raise ValueError("Returned input metadata differs from the invocation")
        return
    import torch

    metadata = capture_inputs(args, kwargs)
    if metadata != snapshot["metadata"]:
        raise ValueError("Returned input metadata differs from the invocation")
    destinations: dict[int, Any] = {}
    for value, spec in zip(
        (*args, *kwargs.values()), (*metadata["args"], *metadata["kwargs"].values()), strict=True
    ):
        if spec["kind"] == "tensor":
            destinations.setdefault(spec["storage_group"], _storage_tensor(value, torch))
    # Validate the complete response before modifying any client tensors.
    if destinations.keys() != snapshot["storages"].keys() or any(
        dst.numel() != len(snapshot["storages"][group]) for group, dst in destinations.items()
    ):
        raise ValueError("Returned storage sizes differ from the invocation")
    with torch.no_grad():
        for group, dst in destinations.items():
            data = snapshot["storages"][group]
            if data:
                dst.copy_(torch.frombuffer(bytearray(data), dtype=torch.uint8))
