# First CUDA job

The repository contains a standalone CUDA program. The program prints device information and a
small vector result.

## Run the example

Set the GPU pool that your service operator provides:

```bash
export GFAAS_GPU_TYPE=gb300
```

Submit the example:

```bash
gfaas run examples/cli/hello_cuda.cu --gpu-type "$GFAAS_GPU_TYPE"
```

The command reports the Call ID and its state changes. It also shows the compiler and program
output. The selected worker can change between Calls.

Use `--json` when another program consumes the result:

```bash
gfaas run examples/cli/hello_cuda.cu \
  --gpu-type "$GFAAS_GPU_TYPE" \
  --json
```

## Write a small job

Create `my_gpu_job.py`:

```python
import os

from gfaas.cuda import spawn

SOURCE = r"""
#include <cstdio>
#include <cuda_runtime.h>

__global__ void square(float *values) {
    int i = threadIdx.x;
    values[i] *= values[i];
}

int main() {
    constexpr int n = 8;
    float *values;
    cudaMallocManaged(&values, n * sizeof(float));
    for (int i = 0; i < n; ++i) values[i] = static_cast<float>(i);

    square<<<1, n>>>(values);
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        return 1;
    }

    for (int i = 0; i < n; ++i) std::printf("%.0f%s", values[i], i + 1 == n ? "\n" : " ");
    cudaFree(values);
    return 0;
}
"""

job = spawn(
    SOURCE,
    gpu="any",
    gpu_type=os.environ["GFAAS_GPU_TYPE"],
    timeout_s=120,
)

print("job:", job.job_id)
print("initial state:", job.status()["state"])

result = job.wait(timeout_s=150)
print("stdout:", result["stdout"].strip())
print("compile/run ms:", result["compile_ms"], result["run_ms"])
print("final state:", job.status()["state"])
```

Run it:

```bash
python my_gpu_job.py
```

The expected output contains `0 1 4 9 16 25 36 49`. Replace `SOURCE` with another self-contained
CUDA program to iterate.

## Inspect the job

`job.status()` returns the current lifecycle state. After assignment, the Attempt records the
selected worker. The terminal state event contains the request journey in its `attributes` object.

Use `job.call_id` as the stable Call identity. `job.job_id` remains available for compatibility with
older client code.

`job.cancel()` requests cancellation of a nonterminal Call. Cancellation is asynchronous, so poll
`job.status()` for the terminal state. A later `job.wait()` raises `GfaasError` when cancellation
succeeds.
