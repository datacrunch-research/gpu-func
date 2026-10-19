"""Real storage/view tests; optional PyTorch is supplied by live Triton environments."""

import ctypes

import pytest

torch = pytest.importorskip("torch")

from gfaas.triton_inputs import apply_writes, capture_inputs, snapshot_inputs  # noqa: E402


def test_snapshot_preserves_overlapping_views_strides_offsets_and_mixed_dtypes():
    base = torch.arange(32, dtype=torch.float32)
    a, b = base[2:18:2], base[4:20:2]
    raw = base.view(torch.uint8)
    snapshot = snapshot_inputs((a, b, raw), {"N": 8})
    assert len(snapshot["storages"]) == 1
    assert snapshot["metadata"]["args"][0]["stride"] == [2]
    assert snapshot["metadata"]["args"][0]["storage_offset"] == 2
    assert snapshot["metadata"]["args"][2]["dtype"] == "uint8"
    base.fill_(0)
    assert snapshot["storages"][0] != ctypes.string_at(
        base.data_ptr(), base.numel() * base.element_size()
    )
    apply_writes(snapshot, (a, b, raw), {"N": 8})
    torch.testing.assert_close(base, torch.arange(32, dtype=torch.float32))
    assert a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


def test_writeback_validates_all_storages_before_mutating_any():
    a, b = torch.ones(4), torch.zeros(5)
    snapshot = snapshot_inputs((a, b), {})
    snapshot["storages"][0] = bytes(16)
    snapshot["storages"][1] = bytes(1)
    with pytest.raises(ValueError, match="sizes"):
        apply_writes(snapshot, (a, b), {})
    assert a.tolist() == [1] * 4


def test_empty_storages_remain_distinct_and_scalar_only_transport():
    metadata = capture_inputs((torch.empty(0), torch.empty(0)), {})
    assert metadata["args"][0]["storage_group"] != metadata["args"][1]["storage_group"]
    apply_writes(snapshot_inputs((None, 12), {"flag": True}), (None, 12), {"flag": True})


def test_gradients_and_conjugate_views_are_rejected():
    with pytest.raises(ValueError, match="gradients"):
        snapshot_inputs((torch.ones(3, requires_grad=True),), {})
    with pytest.raises(ValueError, match="view bits"):
        snapshot_inputs((torch.ones(3, dtype=torch.complex64).conj(),), {})
