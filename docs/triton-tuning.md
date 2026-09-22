# Triton candidate tuning

`spawn_triton_tuning` submits one Call with two stages. The first stage uses CPU resources and no GPU.
It compiles each candidate for a specified CUDA SM target and publishes one transient Artifact. The
second stage holds one GPU lease. It loads the prepared variants, checks their results, and measures
them in sequence. The service cleans up the transient Artifact through its normal Call lifecycle.

The helper requires Triton 3.6.0 on both stages. The workload image also needs PyTorch with CUDA
support. The client machine does not need either package. Supply an image qualified for the requested
GPU, and pass its CUDA SM as `target_arch` (for example, `103` for SM 10.3). The GPU stage rejects a
different device architecture or Triton version. Pass the pool name as `gpu`, for example `gb300`;
the helper requests one GPU from that pool.

See [the complete vector-add example](../examples/triton_tuning.py). Run it with a registered image
that has those packages:

```sh
python examples/triton_tuning.py --image YOUR_IMAGE --gpu YOUR_GPU_POOL --target-arch 103
```

The `source` string defines a `@triton.jit` kernel and these Python functions:

- `make_inputs(case)` creates a dictionary of GPU inputs once per tuning key. Its keys match the
  runtime arguments in `signature`.
- `grid(case, candidate)` returns one to three positive launch-grid dimensions.
- `validate(case, inputs)` checks the result after a candidate's first launch. It returns `True`
  only for a correct result.
- `reset_inputs(case, inputs)` is optional. It runs before validation, warmups, and every measured
  trial. Define it when a launch changes inputs or outputs that later trials use.
- `restore_inputs(case, inputs)` is optional. It runs after each candidate, including when its
  validation or benchmark fails. Use it to restore shared input state between candidates.
- `suggest_candidates(case, results)` is optional. It can add a bounded set of candidates after the
  initial candidates have run. These candidates compile during the GPU stage.

Each `TritonCase.key` is a tuning key. Its `params` go to the workload functions. Its `constexprs`
combine with each candidate's `constexprs` to specialize the kernel. The `signature` identifies
runtime pointer arguments and `constexpr` arguments. Candidate names and case keys must be unique.

The result contains a winner and candidate records for each key. A candidate record separates CPU
`compile_ms`, GPU `gpu_prepare_ms`, and GPU event `trial_ms`. `cache_hit` must be true for a prepared
candidate. `gpu_compilations` counts cache misses, including new adaptive compilations. A report
with no valid candidate has `status: "failed"` and retains each error. The helper limits initial
variants to 128, CPU compilation to eight workers, adaptive candidates to 16 per key, and cache
output to 128 MiB.

The result also records the observed GPU UUID, GPU name, driver API version, CUDA version, Triton
version, and target SM. Use these fields when comparing runs.

GPU event times cover the kernel launch. They exclude input reset and validation. Compare full Call
duration and GPU lease occupancy separately before making a performance claim.
