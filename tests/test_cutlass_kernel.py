import json
from types import SimpleNamespace

import pytest

from gfaas import App, CutlassKernel, CutlassTuning, Image, KernelBenchmark, benchmark


@pytest.mark.parametrize(
    "options",
    [
        {"argument_names": []},
        {"argument_names": ["x", "x"]},
        {"argument_names": ["x"], "configurations": []},
        {"argument_names": ["x"], "configurations": [{"A;bad": 1}]},
        {"argument_names": ["x"], "configurations": [{"A": float("nan")}]},
        {"argument_names": ["x"], "reset_to_zero": ["missing"]},
        {"argument_names": ["x"], "reset_to_zero": ["x"], "restore_value": ["x"]},
        {"argument_names": ["x"], "max_concurrent_jobs": 0},
        {"argument_names": ["x"], "architecture": "bad"},
    ],
)
def test_invalid_cutlass_options(options):
    with pytest.raises((ValueError, TypeError)):
        CutlassKernel("source", **options)


class Client:
    def __init__(self):
        self.calls = []

    def submit(self, **request):
        name = request["function"].__name__
        data = request["kwargs"]
        call_id = f"call_{len(self.calls)}"
        self.calls.append((name, request))
        if name == "probe_target":
            result = {"backend": "cuda", "arch": 103, "warp_size": 32}
        elif name == "compile_batch":
            result = {
                "results": [
                    {"id": v["id"], "status": "compiled"} for v in json.loads(data["variants"])
                ]
            }
        elif name == "execute_winner":
            result = data["inputs"]
        else:

            def report(gpu, runtime):
                return {
                    "status": "passed",
                    "gpu_uuid": gpu,
                    "results": [
                        {
                            "id": v["id"],
                            "status": "measured",
                            "runtime_us": runtime,
                            "refined_us": runtime,
                            "evaluation": "passed",
                        }
                        for v in json.loads(data["variants"])
                    ],
                }

            result = (
                {"replica_reports": [report("gpu0", 12), report("gpu1", 14)]}
                if name == "benchmark_selected_replicas"
                else report("gpu0", 10)
            )
        return SimpleNamespace(call_id=call_id, wait=lambda: result)

    def get_call_result(self, call):
        return {"artifacts": [{"name": "compiled-cutlass", "artifact_id": "art_" + call}]}


def test_cold_tuning_cached_single_job_replication_and_execution():
    client = Client()
    app = App("cutlass", image=Image("test"), client=client)
    kernel = CutlassKernel(
        "source",
        argument_names=("x", "n"),
        configurations=[{"BLOCK": 32}, {"BLOCK": 64}],
        tuning=CutlassTuning(replication_factor=1),
    )
    with app.function(gpu="gb300"):
        cold = benchmark(kernel, None, n=64, options=KernelBenchmark(replication_factor=2))
        original = next(iter(kernel.tuning_results.values()))
        warm = benchmark(kernel, None, n=64, options=KernelBenchmark(replication_factor=2))
        assert kernel(None, n=64) is None
    names = [name for name, _ in client.calls]
    assert names.count("compile_batch") == 2
    assert names.count("benchmark_cycle") == 1
    assert names.count("benchmark_selected_replicas") == 2
    assert names.count("execute_winner") == 1
    assert cold["autotuned"] and not cold["reused_specialization"]
    assert warm["reused_specialization"] and not warm["autotuned"]
    assert cold["runtime_us"] == warm["runtime_us"] == 13
    assert len(cold["benchmark_call_ids"]) == 1
    assert next(iter(kernel.tuning_results.values())) is original
    for name, call in client.calls:
        if name == "benchmark_selected_replicas":
            assert call["gpu_count"] == 2
            assert len(call["kwargs"]["artifacts"]) == 1
    with pytest.raises(TypeError):
        warm["runtime_us"] = 1


def test_argument_and_context_errors_precede_submissions():
    kernel = CutlassKernel("source", argument_names=("x", "n"))
    with pytest.raises(TypeError, match="Missing"):
        kernel(None)
    with pytest.raises(TypeError, match="Invalid"):
        kernel(None, 1, x=None)
    with pytest.raises(RuntimeError):
        kernel(None, 1)


def test_environment_change_requires_recompilation():
    client = Client()
    app = App("cutlass", image=Image("test"), client=client)
    kernel = CutlassKernel("source", argument_names=("n",))
    for value in ("a", "a", "b"):
        with app.function(gpu="gb300", env={"COMPILER_SETTING": value}):
            kernel(64)
    names = [name for name, _ in client.calls]
    assert names.count("compile_batch") == 2
    assert names.count("execute_winner") == 3
    assert len(kernel.tuning_results) == 2
