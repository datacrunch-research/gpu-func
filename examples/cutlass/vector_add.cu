#include <cute/tensor.hpp>
#include <cutlass/float8.h>
#ifndef VF_FP8
#define VF_FP8 0
#endif
#if VF_FP8
using Input = cutlass::float_e4m3_t;
#else
using Input = float;
#endif
#ifndef VF_WRONG
#define VF_WRONG 0
#endif
__global__ void add_vectors(Input* a, Input* b, float* c, int64_t n,
                           int64_t sa, int64_t sb, int64_t sc, float alpha) {
    auto ta = cute::make_tensor(cute::make_gmem_ptr(a), cute::make_layout(cute::make_shape(n), cute::make_stride(sa)));
    auto tb = cute::make_tensor(cute::make_gmem_ptr(b), cute::make_layout(cute::make_shape(n), cute::make_stride(sb)));
    auto tc = cute::make_tensor(cute::make_gmem_ptr(c), cute::make_layout(cute::make_shape(n), cute::make_stride(sc)));
    int64_t i = int64_t(blockIdx.x) * VF_BLOCK + threadIdx.x;
    if (i < n) tc(i) = VF_WRONG ? float(ta(i)) - float(tb(i)) : float(ta(i)) + alpha * float(tb(i));
}
extern "C" int vfunc_launch(const VFuncArgument* a, int64_t count, cudaStream_t stream) {
    if (count != (VF_FP8 ? 4 : 3)) return -1;
    if (VF_FP8 && a[3].kind != 3) return -3;
    for (int i = 0; i < 3; ++i)
        if (a[i].kind != 1 || a[i].dtype != ((VF_FP8 && i < 2) ? 11 : 1) || a[i].ndim != 1 || a[i].shape[0] != a[0].shape[0]) return -2;
    int64_t n = a[0].shape[0];
    add_vectors<<<(n + VF_BLOCK - 1) / VF_BLOCK, VF_BLOCK, 0, stream>>>(
        static_cast<Input*>(a[0].data), static_cast<Input*>(a[1].data), static_cast<float*>(a[2].data),
        n, a[0].strides[0], a[1].strides[0], a[2].strides[0], VF_FP8 ? float(a[3].f64) : 1.0f);
    return int(cudaGetLastError());
}
