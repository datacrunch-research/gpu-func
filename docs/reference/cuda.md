# CUDA programs and explicit Triton tuning

## compile_and_run and spawn

`compile_and_run(source, ...)` accepts a complete CUDA C++ program and returns a process report.
`gfaas.cuda.spawn(source, ...)` accepts the same options and returns a `RemoteResult`. The `spawn`
helper is imported from `gfaas.cuda`, not the package root.

| Keyword            | Default              | Meaning                                                         |
| ------------------ | -------------------- | --------------------------------------------------------------- |
| `gpu`, `gpu_count` | `None`, `None`       | Older selector or explicit GPU count; set one for GPU execution |
| `gpu_type`         | `"any"`              | Pool selection                                                  |
| `profile`          | `False`              | `True` or a `ProfilingRequest`: run through Nsight Compute      |
| `ncu_args`         | `None`               | Profiler arguments, with `profile=True`                         |
| `nvcc_flags`       | `None`               | Compiler flags, including an explicit target if needed          |
| `program_args`     | `None`               | Arguments to the compiled executable                            |
| `timeout_s`        | `600`                | Execution timeout                                               |
| `image`            | `Image("cuda-nvcc")` | Registered image with Python, CUDA compiler, and required tools |
| `compatibility`    | `None`               | Hardware placement requirements                                 |
| `client`           | `None`               | Existing Client for connection reuse                            |

Compilation and execution are stages of one Call. The compile stage reserves no GPU. The helper does
not infer a compiler target from `gpu_type`; pass the appropriate `nvcc_flags`.

The report contains `phase`, `stdout`, `stderr`, `returncode`, `compile_ms`, and `run_ms`. A
profiled run adds `ncu_csv` and `profiling`. These process times are not isolated kernel times. A
staged compile failure fails the Call. `compile_and_run()` also raises `CudaProcessError` when the
program fails. The asynchronous handle returns the report without that final helper check.

## Profiling

A profiled run has two results: what the program did, and what the profiler captured. The program
can exit 0 while Nsight Compute captures no kernel, is denied the GPU counters, fails, or runs out
of its own time. The report's `profiling` entry is that second result.

```python
import gfaas
from gfaas.cuda import profiling_outcome

# Fragment: source is a complete CUDA C++ program.
request = gfaas.ProfilingRequest(metric_set="basic", timeout_s=120, launch_count=8)
report = gfaas.compile_and_run(source, gpu_count=1, gpu_type="YOUR_POOL", profile=request)
outcome = profiling_outcome(report)
if outcome.ok:
    print(outcome.kernels, outcome.passes, outcome.report("ncu-rep").artifact_id)
else:
    print(outcome.status, outcome.detail)
```

`ProfilingRequest` fields:

| Field              | Default  | Meaning                                                            |
| ------------------ | -------- | ------------------------------------------------------------------ |
| `metric_set`       | `"full"` | `"basic"`, `"full"`, or `None` when `arguments` select the metrics |
| `arguments`        | `()`     | At most 32 more `ncu` options, each at most 256 characters         |
| `timeout_s`        | `300`    | Time budget of the profiled run alone, apart from the Call timeout |
| `launch_count`     | `None`   | Profile at most this many kernel launches                          |
| `max_report_bytes` | `64 MiB` | Largest report to publish                                          |
| `baseline`         | `False`  | Also run the program without the profiler, to measure the slowdown |

The run chooses the report's location and format. A request that sets `--export`, `--import`,
`--log-file`, `--csv`, or `--force-overwrite` is rejected. `profile=True` is the default request,
and `ncu_args` replace its metric set when they choose one.

`ProfilingOutcome.status` is one of:

| Status                 | Meaning                                                             |
| ---------------------- | ------------------------------------------------------------------- |
| `profiled`             | At least one kernel launch was captured and a report exists         |
| `no_kernels`           | The program ran, and the profiler captured no kernel launch         |
| `permission_denied`    | The driver refused the profiler access to the GPU counters          |
| `profiler_failed`      | Nsight Compute itself failed, for example on an invalid option      |
| `profiler_timed_out`   | The profiled run exceeded `timeout_s`, or the Call's remaining time |
| `profiler_unavailable` | The image or machine has no `ncu`; the program was not run          |
| `program_failed`       | The program failed and nothing was captured                         |
| `report_too_large`     | The report exceeded `max_report_bytes` and was not published        |

The outcome also carries:

- `program_returncode`: the program's own exit status, or `None` when the run does not establish it.
  `compile_and_run()` raises `CudaProcessError` only when this is nonzero. Any other outcome is
  returned; call `outcome.raise_for_status()` to turn it into a `ProfilingError`.
- `kernels` and `passes`: the kernel launches captured and the times the profiler replayed them.
  With `baseline=True`, `slowdown` is the profiled run time over the unprofiled run time.
- `warnings` and `errors`: the profiler's own messages. They are removed from `stdout`.
- `reports`: one `ProfilingReport` per file, with `format` (`ncu-rep` or `csv`), `size_bytes`,
  `sha256`, and the `artifact_id` of the published profile output.
- `provenance`: the profiler version, each GPU's name, UUID and driver version, the compiler, the
  image, the request, and the exact profiler arguments.
- `call_id`: the Call that produced the profile.

Reports are published to the Call's `profiles` output with a `profile.provenance.json` record of the
same outcome, so a downloaded report still says what produced it. A CSV report of at most 1 MiB is
also returned as `ncu_csv`. `gfaas.profiling.download_reports(client, outcome, directory)` downloads
the published reports, refuses a total above its `max_bytes`, and checks each file against its
recorded digest.

### Authorization

A profiled Call asks the service for the `gpu_counters` capability. The service then runs it only on
a host whose driver lets workloads read the GPU counters. Your API key needs the `profiling` grant
from your operator. Without it, `compile_and_run()` and `Client.submit(profiling=...)` raise
`ProfilingNotGrantedError` before they upload anything. `ProfilingUnsupportedError` means the
service does not offer the capability. A profiled Call needs at least one GPU.

To profile with your own tool, such as Nsight Systems or CUPTI, pass `profiling="gpu_counters"` to
`Client.submit()` and run the tool inside your function.

This grant is separate from the tools in an image. `profiler_unavailable` means the image has no
`ncu`; `permission_denied` means the host's driver refused. A local run with
`vfunc local run --profile` involves no service authorization at all.

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

`.profile(*ncu_args)` enables profiling; `.profile(request=ProfilingRequest(...))` takes a full
request. `.run(client=None)` accepts a reusable client. Use `compile_and_run()` directly for options
not exposed by this small builder.

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
