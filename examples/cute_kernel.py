"""Tune, benchmark and execute a CuTe DSL kernel through normal vFunc settings."""

import argparse

import torch
from cute_kernels import add_one, evaluate_add

import gfaas as vfunc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Image with matching CuTe DSL and PyTorch")
    args = parser.parse_args()
    kernel = vfunc.CuteDSLKernel(
        add_one,
        configurations=[{"BLOCK": n} for n in (64, 128, 256, 512)],
        tuning=vfunc.KernelTuning(
            evaluate=evaluate_add,
            pilot_pruning=vfunc.KernelPruning(relative_delta=0.25, absolute_us=1.0),
            refined_pruning=vfunc.KernelPruning(relative_delta=0.05, absolute_us=0.1),
            replication_factor=3,
            benchmark=vfunc.KernelTiming(pilot_trials=3, refinement_duration_ms=1.0),
        ),
        variants_per_job=2,
        max_concurrent_jobs=2,
    )
    a = torch.arange(4097, dtype=torch.float32)
    b = torch.full_like(a, -1)
    app = vfunc.App("cute-add", image=vfunc.Image(args.image))
    with app.function(gpu="gb300", timeout=600, capacity_wait=300):
        # Autotune if needed, then collect new R=3 timings. Caller tensors remain unchanged.
        first = vfunc.benchmark(kernel, a, b, options=vfunc.KernelBenchmark(replication_factor=3))
        cached = vfunc.benchmark(kernel, a, b)
        assert first["autotuned"] and cached["reused_specialization"]
        assert torch.all(b == -1)
        # Reuse the selected object, execute once, and preserve output writes on the client.
        kernel(a, b)
    assert torch.equal(b, a + 1)
    print("Configuration:", cached["configuration"]["constants"])
    print("Runtime:", cached["runtime_us"], "us")
    print("Reports:", list(kernel.tuning_results))


if __name__ == "__main__":
    main()
