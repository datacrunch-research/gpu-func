"""Tune 1,024 BF16 matmul configurations, then benchmark across three GPUs.

Set GFAAS_API_BASE / GFAAS_API_KEY through your normal client configuration.
Run with --image matching the local PyTorch/Triton versions. CPU tensors suffice:
vFunc transports their contents and benchmarks independent GPU copies.
"""

from __future__ import annotations

import argparse
import itertools

import torch
import triton
import triton.language as tl

import gfaas as vfunc

# 4 * 4 * 2 * 4 * 2 * 4 = 1,024 distinct configurations.
CONFIGS = [
    triton.Config(
        {"BM": bm, "BN": bn, "BK": bk, "GROUP_M": group},
        num_warps=warps,
        num_stages=stages,
    )
    for bm, bn, bk, group, warps, stages in itertools.product(
        (32, 64, 128, 256),
        (32, 64, 128, 256),
        (32, 64),
        (1, 2, 4, 8),
        (4, 8),
        (1, 2, 3, 4),
    )
]


@triton.jit
def matmul(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m, num_n = tl.cdiv(M, BM), tl.cdiv(N, BN)
    group_size = GROUP_M * num_n
    first_m = (pid // group_size) * GROUP_M
    actual_m = tl.minimum(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group_size) % actual_m
    pid_n = (pid % group_size) // actual_m
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    accumulator = tl.full((BM, BN), 0, tl.float32)
    for block in range(tl.cdiv(K, BK)):
        k = block * BK + ks
        a = tl.load(
            A + rows[:, None] * K + k[None, :], (rows[:, None] < M) & (k[None, :] < K), other=0
        )
        b = tl.load(
            B + k[:, None] * N + cols[None, :], (k[:, None] < K) & (cols[None, :] < N), other=0
        )
        accumulator += tl.dot(a, b)
    tl.store(
        C + rows[:, None] * N + cols[None, :],
        accumulator,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


def grid(meta):
    return (triton.cdiv(meta["M"], meta["BM"]) * triton.cdiv(meta["N"], meta["BN"]),)


def evaluate(candidate, a, b, c, *, M, N, K):
    candidate(a, b, c, M=M, N=N, K=K)
    reference = a.float() @ b.float()
    return bool(torch.allclose(c.float(), reference, atol=0.03, rtol=0.02))


def make_kernel(configs=CONFIGS):
    native = triton.autotune(configs=configs, key=["M", "N", "K"])(matmul)
    return vfunc.TritonKernel(
        native,
        variants_per_job=None,
        max_concurrent_jobs=8,
        cache_compression_level=1,
        tuning=vfunc.TritonTuning(
            pilot_pruning=vfunc.TritonPruning(relative_delta=0.25, absolute_us=1.0),
            refined_pruning=vfunc.TritonPruning(relative_delta=0.05, absolute_us=0.1),
            evaluate=evaluate,
            quick_benchmark_group_size=8,
            quick_benchmark_variants_per_job=256,
            quick_benchmark_max_concurrent_jobs=4,
            replication_factor=3,
            replication_max_attempts=8,
            max_input_sets=65_536,
            max_ring_bytes=8 * 1024**3,
            benchmark=vfunc.TritonBenchmark(
                pilot_trials=3,
                refinement_duration_ms=1.0,
                min_refinement_trials=10,
                max_refinement_trials=250,
                final_duration_ms=25.0,
                min_final_trials=25,
                max_final_trials=1_000,
                graph_duration_ms=1.0,
                min_calls_per_graph=10,
                max_calls_per_graph=100,
                l2_flush_iterations=100,
            ),
        ),
    )


# Every standalone benchmark control, with its default.
BENCHMARK = vfunc.KernelBenchmark(
    estimate_trials=3,
    final_duration_ms=25.0,
    min_final_trials=25,
    max_final_trials=1_000,
    graph_duration_ms=1.0,
    min_calls_per_graph=10,
    max_calls_per_graph=100,
    l2_flush_iterations=100,
    replication_factor=3,
    replication_max_attempts=8,
    max_concurrent_jobs=4,
    max_input_sets=65_536,
    max_ring_bytes=8 * 1024**3,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--size", type=int, default=2048, help="M and N dimensions")
    parser.add_argument(
        "--k", type=int, default=3072, help="reduction dimension; targets roughly 100 us on GB300"
    )
    args = parser.parse_args()
    n, k = args.size, args.k
    generator = torch.Generator().manual_seed(0)
    a = torch.randn((n, k), dtype=torch.bfloat16, generator=generator) * 0.1
    b = torch.randn((k, n), dtype=torch.bfloat16, generator=generator) * 0.1
    c = torch.full((n, n), -11, dtype=torch.bfloat16)
    kernel = make_kernel()
    app = vfunc.App("matmul-benchmark", image=vfunc.Image(args.image))
    with app.function(gpu="gb300", timeout=600, capacity_wait=120):
        # Case 1: this first benchmark compiles and autotunes before measuring.
        cold = vfunc.benchmark(kernel[grid], a, b, c, M=n, N=n, K=k, options=BENCHMARK)
        assert cold["autotuned"] and torch.all(c == -11)
        # Case 2: same specialization, fresh measurements of the cached winner.
        warm = vfunc.benchmark(kernel[grid], a, b, c, M=n, N=n, K=k, options=BENCHMARK)
        assert warm["reused_specialization"] and torch.all(c == -11)
        # Ordinary execution also reuses the winner and writes to c.
        kernel[grid](a, b, c, M=n, N=n, K=k)
    print(
        "Autotune:",
        kernel.tuning_results[cold["specialization"]]["benchmark"]["best_runtime_us"],
        "us",
    )
    print("Cold benchmark:", cold["runtime_us"], "us")
    print("Cached benchmark:", warm["runtime_us"], "us")
    print("Replicas:", warm["replicas"])
    print("Selected configuration:", warm["configuration"])


if __name__ == "__main__":
    main()
