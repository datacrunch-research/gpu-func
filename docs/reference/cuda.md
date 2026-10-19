# CUDA programs and explicit Triton tuning

## compile_and_run and spawn

`compile_and_run(source, ...)` accepts a complete CUDA C++ program and returns a process report.
`gfaas.cuda.spawn(source, ...)` accepts the same options and returns a `RemoteResult`. The `spawn`
helper is imported from `gfaas.cuda`, not the package root.

| Keyword            | Default              | Meaning                                                         |
| ------------------ | -------------------- | --------------------------------------------------------------- |
| `gpu`, `gpu_count` | `None`, `None`       | Older selector or explicit GPU count; set one for GPU execution |
| `gpu_type`         | `"any"`              | Pool selection                                                  |
| `profile`          | `False`              | Run through Nsight Compute                                      |
| `ncu_args`         | `None`               | Profiler arguments                                              |
| `nvcc_flags`       | `None`               | Compiler flags, including an explicit target if needed          |
| `program_args`     | `None`               | Arguments to the compiled executable                            |
| `timeout_s`        | `600`                | Execution timeout                                               |
| `image`            | `Image("cuda-nvcc")` | Registered image with Python, CUDA compiler, and required tools |
| `compatibility`    | `None`               | Hardware placement requirements                                 |
| `client`           | `None`               | Existing Client for connection reuse                            |

Compilation and execution are stages of one Call. The compile stage reserves no GPU. The helper does
not infer a compiler target from `gpu_type`; pass the appropriate `nvcc_flags`.

The report contains `phase`, `stdout`, `stderr`, `returncode`, `compile_ms`, and `run_ms`, plus
`ncu_csv` when profiling produces it. These process times are not isolated kernel times. A staged
compile failure fails the Call. `compile_and_run()` also raises `CudaProcessError` for an
unsuccessful execution report. The asynchronous handle returns the report without that final helper
check.

## CudaSource

`CudaSource(source)` is a fluent wrapper around `compile_and_run()`:

```python
import gfaas

# Fragment: source is a complete CUDA C++ program.
report = (
    gfaas.CudaSource(source)
    .gpu("any", gpu_type="YOUR_POOL")
    .nvcc("-O3", "-arch=sm_103")
    .run()
)
```

`.profile(*ncu_args)` enables profiling. `.run(client=None)` accepts a reusable client. Use
`compile_and_run()` directly for options not exposed by this small builder.

## Explicit Triton candidates

This API takes source text and explicit cases. It is separate from `TritonKernel` and its shared
policies. Read [Triton candidate tuning](../triton-tuning.md) for the source hooks and version
requirements.

`TritonCandidate(name, constexprs={}, num_warps=4, num_stages=2)` describes one launch
configuration. `TritonCase(key, params={}, constexprs={})` describes one input case and tuning key.
Their `request()` methods produce API-ready dictionaries.

`spawn_triton_tuning()` accepts these keyword-only arguments:

| Argument                             | Default       | Meaning                                                     |
| ------------------------------------ | ------------- | ----------------------------------------------------------- |
| `source`, `kernel_name`, `signature` | required      | Python source, JIT entry name, and Triton argument types    |
| `candidates`, `cases`                | required      | Candidate and case lists                                    |
| `target_arch`                        | required      | Numeric CUDA SM, such as `103`                              |
| `image`                              | required      | Registered image with Triton 3.6.0 and CUDA-enabled PyTorch |
| `gpu`                                | required      | Pool name; this helper requests one GPU                     |
| `compatibility`                      | `None`        | Additional hardware requirements                            |
| `compile_workers`                    | `4`           | CPU compiler workers, at most 8                             |
| `compile_cpu_millicores`             | `4000`        | Compile-stage CPU request                                   |
| `compile_memory_bytes`               | `8 * 1024**3` | Compile-stage memory request                                |
| `warmup`, `trials`                   | `2`, `5`      | Warmup and measured trials                                  |
| `max_adaptive_candidates`            | `0`           | Additional candidates per key, at most 16                   |
| `timeout_s`                          | `600`         | Call execution timeout                                      |
| `client`                             | `None`        | Existing Client                                             |

The method returns a `RemoteResult`. `tune_triton(**kwargs)` uses the same arguments and waits for
the report. Limits are 64 candidates, 16 cases, and 128 initial candidate/case variants. The
prepared cache has a 128 MiB bound specific to this helper; this is not a general Artifact download
limit.

The report records candidates, winners, errors, target identity, and phase timings. Inspect the
report's success status and candidate validation results, not only the Call's terminal state.
