#include <cstdio>
#include <cuda_runtime.h>

namespace {

constexpr int kValueCount = 8;

__global__ void square_values(int* values) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < kValueCount) {
        values[index] *= values[index];
    }
}

bool cuda_ok(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }

    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
    return false;
}

}  // namespace

int main() {
    cudaDeviceProp device{};
    if (!cuda_ok(cudaGetDeviceProperties(&device, 0), "read CUDA device properties")) {
        return 1;
    }

    int* values = nullptr;
    if (!cuda_ok(
            cudaMallocManaged(reinterpret_cast<void**>(&values), kValueCount * sizeof(int)),
            "allocate managed memory")) {
        return 1;
    }

    for (int index = 0; index < kValueCount; ++index) {
        values[index] = index;
    }

    square_values<<<1, kValueCount>>>(values);
    if (!cuda_ok(cudaGetLastError(), "launch square_values")) {
        cudaFree(values);
        return 1;
    }
    if (!cuda_ok(cudaDeviceSynchronize(), "wait for square_values")) {
        cudaFree(values);
        return 1;
    }

    std::printf("device: %s\nvalues:", device.name);
    for (int index = 0; index < kValueCount; ++index) {
        std::printf(" %d", values[index]);
    }
    std::printf("\n");

    cudaFree(values);
    return 0;
}
