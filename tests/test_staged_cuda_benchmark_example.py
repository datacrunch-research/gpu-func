import importlib.util
import inspect
from pathlib import Path

from gfaas import cuda_runner

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "staged_cuda_benchmark.py"


def test_the_baseline_submission_passes_arguments_the_cuda_runner_accepts():
    spec = importlib.util.spec_from_file_location("staged_cuda_benchmark", EXAMPLE)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    submitted = {}

    class Client:
        def submit(self, **arguments):
            submitted.update(arguments)

    example._submit(Client(), staged=False, gpu_type="gb300", image="cuda-nvcc", hold_ms=5)

    assert submitted["function"] is cuda_runner.run
    inspect.signature(cuda_runner.run).bind(**submitted["kwargs"])
    assert submitted["kwargs"]["program_args"] == ["5"]
