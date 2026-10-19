import pytest

from gfaas.kernel_results import apply_result, snapshot_result

torch = pytest.importorskip("torch")


def test_returned_outputs_and_writes_preserve_storage_aliases():
    client = torch.arange(12, dtype=torch.float32)
    remote = client.clone()
    remote.add_(3)
    output = torch.arange(8, dtype=torch.float32)
    result = (remote, {"view": remote[2:10:2], "new": [output, output[1:6:2]], "scalar": 7})
    payload = snapshot_result(result, (remote, 12))
    returned = apply_result(payload, (client, 12))
    assert returned[0] is client
    assert torch.equal(client, remote)
    view = returned[1]["view"]
    assert view.untyped_storage()._cdata == client.untyped_storage()._cdata
    assert view.stride() == (2,) and view.storage_offset() == 2
    assert torch.equal(view, remote[2:10:2])
    new, other_view = returned[1]["new"]
    assert new.untyped_storage()._cdata == other_view.untyped_storage()._cdata
    assert torch.equal(other_view, output[1:6:2])
    assert returned[1]["scalar"] == 7


def test_invalid_result_is_rejected_before_writing_client_tensors():
    client = torch.zeros(4)
    payload = snapshot_result(torch.ones(3), (torch.ones(4),))
    payload["snapshot"]["metadata"]["args"][-1]["shape"] = [1000]
    with pytest.raises(ValueError, match="exceeds"):
        apply_result(payload, (client,))
    assert torch.count_nonzero(client) == 0


def test_empty_return_and_empty_tensor():
    client = torch.empty(0)
    assert apply_result(snapshot_result(None, (client,)), (client,)) is None
    out = apply_result(snapshot_result(torch.empty(0), (client,)), (client,))
    assert out.shape == (0,)
