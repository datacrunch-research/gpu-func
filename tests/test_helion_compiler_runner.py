import json

from gfaas import helion_compiler_runner


def test_cpu_compilation_fans_out_all_launches_and_preserves_configuration_failure(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GFAAS_OUTPUT_ROOT", str(tmp_path))
    requests = []

    class Pool:
        def __init__(self, **kwargs):
            assert kwargs["max_workers"] == 3
            assert kwargs["mp_context"].get_start_method() == "spawn"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def map(self, fn, jobs):
            for request in jobs:
                requests.append(request)
                launch = request["variants"][0]
                yield {
                    "results": [
                        {
                            "id": launch["id"],
                            "status": "failed" if launch["id"] == "bad" else "compiled",
                        }
                    ]
                }

    monkeypatch.setattr(helion_compiler_runner, "ProcessPoolExecutor", Pool)

    def unit(identity):
        return {
            "id": identity,
            "kernel_name": "device",
            "signature": {"x": "*fp32"},
            "constants": {},
            "options": {},
        }

    variants = [
        {
            "id": "a",
            "configuration": {"block_sizes": [64]},
            "status": "prepared",
            "source": "source",
            "launches": [unit("one"), unit("two")],
        },
        {
            "id": "b",
            "configuration": {"block_sizes": [128]},
            "status": "prepared",
            "source": "source",
            "launches": [unit("bad")],
        },
    ]
    report = helion_compiler_runner.compile_batch(
        variants=variants,
        target={"arch": 103},
        triton_version="3.8.0",
        workers=16,
        compression_level=1,
    )
    assert len(requests) == 3
    assert len({r["output_root"] for r in requests}) == 3
    assert all(r["workers"] == 1 for r in requests)
    assert [r["status"] for r in report["results"]] == ["compiled", "failed"]
    assert len(report["results"][0]["units"]) == 2
    assert json.loads((tmp_path / "compiled-helion/a/variant.json").read_text()) == variants[0]
    assert json.loads((tmp_path / "compiled-helion/manifest.json").read_text()) == report
