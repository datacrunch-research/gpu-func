# Historical kernel adapter measurements

These observations came with the kernel adapter contributions. They describe the named examples and
software versions, not a general compatibility or performance guarantee. The documentation audit did
not rerun them. Choose an image and hardware appropriate for your own workload.

## Gluon kernels

Gluon qualification on GB300with Triton 3.8.0 and PyTorch 2.14.0 passed for all three examples: one
strided-copy configuration, three accumulating-add configurations and three softmax configurations.
Fresh and cached benchmarks used R=3 with verified distinct GPU UUIDs, preserved the caller's
tensors, and were followed by successful execution with tensor writeback. Cached replicated means
were approximately 2.257 us, 1.814 us and 2.134 us respectively. These measurements qualify the
examples, not general performance across Gluon kernels.

## C++ CUTLASS kernels

Qualification used CUTLASSv4.8.0, CUDA 13.0.88 and three GB300 GPUs per benchmark. FP32/BF16 GEMM,
strided vector addition with aliased output, and FP8 vector inputs with a scalar argument passed
compilation, tuning, cold and cached benchmarking, and execution. The strided sample rejected an
intentionally incorrect configuration. These samples establish the launcher and measurement path;
other CUTLASS operators still need compatible C++ launchers.
