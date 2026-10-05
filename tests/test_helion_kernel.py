from __future__ import annotations

import hashlib
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

import gfaas
from gfaas.kernel_results import snapshot_result

torch = pytest.importorskip("torch")


def add(x, y):
    return x + y


class Settings:
    backend = "triton"

    def to_dict(self):
        return {"backend": self.backend}


class Native:
    def __init__(self, fn, configs=None):
        self.fn, self.configs = fn, configs or [{"block_sizes": [64]}]
        self.settings = Settings()
        self._key_fn = None

    def bind(self, args):
        pass


@pytest.fixture
def native(monkeypatch):
    module = ModuleType("helion")
    module.Kernel = Native
    monkeypatch.setitem(sys.modules, "helion", module)
    monkeypatch.setattr("gfaas.helion_kernel.installed_version", lambda: "1.4.0")
    return Native(add)


class Client:
    def __init__(self):
        self.phases = []
        self.sequence = 0

    def submit(self, **request):
        handler = request["function"].__name__
        data = request["kwargs"]
        self.phases.append(handler)
        self.sequence += 1
        identity = f"call_{self.sequence}"
        if handler == "prepare":
            variants = [
                {
                    "id": hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest(),
                    "configuration": c,
                    "status": "prepared",
                }
                for c in data["configurations"]
            ]
            output = {
                "variants": variants,
                "target": {"backend": "cuda", "arch": 103, "warp_size": 32},
                "triton_version": "3.8.0",
            }
        elif handler == "compile_batch":
            output = {"results": [{"id": v["id"], "status": "compiled"} for v in data["variants"]]}
        elif handler in ("benchmark_cycle", "benchmark_replicas"):
            variants = json.loads(data["variants"])

            def measured(gpu, runtime):
                return {
                    "status": "passed",
                    "gpu_uuid": f"gpu{gpu}",
                    "results": [
                        {
                            "id": v["id"],
                            "status": "measured",
                            "runtime_us": runtime,
                            "refined_us": runtime,
                            "evaluation": "passed",
                        }
                        for v in variants
                    ],
                }

            if handler == "benchmark_cycle":
                output = measured(0, 10)
            else:
                output = {
                    "replica_reports": [
                        {"status": "duplicate_gpu", "gpu_uuid": f"gpu{i}"}
                        if f"gpu{i}" in data["excluded_gpu_uuids"]
                        else measured(i, 10 + i)
                        for i in range(data["device_count"])
                    ]
                }
        elif handler == "execute_winner":
            snapshot = data["inputs"]
            args = []
            for spec in snapshot["metadata"]["args"]:
                buffer = torch.frombuffer(
                    bytearray(snapshot["storages"][spec["storage_group"]]), dtype=torch.uint8
                )
                args.append(
                    torch.empty(0, dtype=getattr(torch, spec["dtype"])).set_(
                        buffer.untyped_storage(),
                        spec["storage_offset"],
                        tuple(spec["shape"]),
                        tuple(spec["stride"]),
                    )
                )
            output = snapshot_result(args[0] + args[1], tuple(args))
        else:
            raise AssertionError(handler)
        return SimpleNamespace(call_id=identity, wait=lambda: output)

    def get_call_result(self, identity):
        return {"artifacts": [{"name": "compiled-helion", "artifact_id": "art_" + identity}]}


def test_helion_call_returns_tensors_caches_specialization_and_benchmarks_one_job(native):
    client = Client()
    kernel = gfaas.HelionKernel(native)
    app = gfaas.App("helion", image=gfaas.Image("compiler"), client=client)
    x, y = torch.arange(8, dtype=torch.float32), torch.ones(8)
    with app.function(gpu="gb300"):
        assert torch.equal(kernel(x, y), x + y)
        assert torch.equal(kernel(y=y, x=x), x + y)
        before = dict(kernel.tuning_results)
        first = gfaas.benchmark(kernel, x, y)
        second = gfaas.benchmark(kernel, x, y)
    assert client.phases.count("prepare") == 1
    assert client.phases.count("compile_batch") == 1
    assert client.phases.count("benchmark_cycle") == 1
    assert client.phases.count("benchmark_replicas") == 3  # Tune, benchmark, benchmark.
    assert client.phases.count("execute_winner") == 2
    assert first["runtime_us"] == second["runtime_us"] == 11
    assert first["reused_specialization"] and not first["autotuned"]
    assert len({r["gpu_uuid"] for r in first["replicas"]}) == 3
    assert dict(kernel.tuning_results) == before
    with pytest.raises(TypeError):
        first["runtime_us"] = 1


def test_helion_cold_benchmark_tunes_without_execution_then_shape_change_retunes(native):
    client = Client()
    kernel = gfaas.HelionKernel(native)
    app = gfaas.App("helion", image=gfaas.Image("compiler"), client=client)
    with app.function(gpu="gb300"):
        first = gfaas.benchmark(kernel, torch.ones(8), torch.ones(8))
        second = gfaas.benchmark(kernel, torch.ones(16), torch.ones(16))
    assert first["autotuned"] and second["autotuned"]
    assert len(kernel.tuning_results) == 2
    assert "execute_winner" not in client.phases
    assert client.phases.count("prepare") == 2


def test_helion_configuration_overrides_are_deduplicated(native):
    kernel = gfaas.HelionKernel(native, configs=[{"block_sizes": [128]}, {"block_sizes": [128]}])
    assert kernel.configurations == [{"block_sizes": [128]}]
    with pytest.raises(ValueError, match="at least one"):
        gfaas.HelionKernel(native, configs=[])
    with pytest.raises(ValueError, match="launch grids"):
        kernel[(1,)](torch.ones(8), torch.ones(8))


def test_helion_backend_and_specialization_callbacks_are_rejected(native):
    native.settings.backend = "cute"
    with pytest.raises(gfaas.UnsupportedHelionKernelError, match="Triton backend"):
        gfaas.HelionKernel(native)
    native.settings.backend = "triton"
    native._key_fn = lambda x: 1
    with pytest.raises(gfaas.UnsupportedHelionKernelError, match="specialization"):
        gfaas.HelionKernel(native)
