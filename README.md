# gpu_func_cli

Standalone CLI for running **custom CUDA kernels** on a remote GPU through the
GFAAS REST API. The local machine needs no CUDA, `nvcc`, Nsight Compute, or GPU.
The CLI sends a self-contained job to a GFAAS worker, and the worker does the
CUDA work.

Full documentation: [`GUIDE.md`](GUIDE.md). It covers install, configuration,
the walkthrough, custom kernels (with and without a harness), reports and
feedback, command reference, and troubleshooting.

## Quick start

```bash
uv tool install --editable /path/to/gpu_func_cli   # or: pip install .
export GFAAS_API_BASE="https://<hub-host>/api"
export GFAAS_API_KEY="<your-api-key>"
gpu_func_cli workers

# run any self-contained .cu (has its own main()) — nothing else to bring:
gpu_func_cli custom run /path/to/your_kernel.cu --gpu B200

# kernel-only source? add a --harness that supplies main():
gpu_func_cli custom run kernel.cu --harness harness.cu --gpu B200

# profile on the GPU, then read the report locally:
gpu_func_cli custom profile your_kernel.cu --gpu B200 --artifact-dir ./out
gpu_func_cli report summary ./out/your_kernel.ncu-rep --per-kernel
```

For a first run, start with the
[hands-on walkthrough](GUIDE.md#4-hands-on-walkthrough) in `GUIDE.md`. It
creates its own test files, so you don't need to bring a CUDA program.
