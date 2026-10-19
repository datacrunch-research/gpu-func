// FP32 SIMT or VF_BF16 tensor-core GEMM, selected at compilation.
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#ifndef VF_BF16
#define VF_BF16 0
#endif
#if VF_BF16
using Element = cutlass::bfloat16_t;
using LayoutB = cutlass::layout::ColumnMajor;
using Gemm = cutlass::gemm::device::Gemm<
    Element, cutlass::layout::RowMajor, Element, LayoutB,
    Element, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<VF_TILE_M, VF_TILE_N, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<Element, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, VF_STAGES>;
#else
using Element = float;
using LayoutB = cutlass::layout::RowMajor;
using Gemm = cutlass::gemm::device::Gemm<
    Element, cutlass::layout::RowMajor, Element, LayoutB,
    Element, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassSimt, cutlass::arch::Sm50,
    cutlass::gemm::GemmShape<VF_TILE_M, VF_TILE_N, 8>,
    cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<Element, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;
#endif
extern "C" int vfunc_launch(const VFuncArgument* a, int64_t count, cudaStream_t stream) {
    if (count != 3) return -1;
    const int dtype = VF_BF16 ? 3 : 1;
    for (int i = 0; i < 3; ++i)
        if (a[i].kind != 1 || a[i].dtype != dtype || a[i].ndim != 2) return -2;
    int m = a[0].shape[0], k = a[0].shape[1], n = a[1].shape[1];
    if (a[1].shape[0] != k || a[2].shape[0] != m || a[2].shape[1] != n ||
        a[0].strides[1] != 1 || a[2].strides[1] != 1 ||
        (VF_BF16 ? a[1].strides[0] != 1 : a[1].strides[1] != 1)) return -3;
    typename Gemm::Arguments args(
        {m, n, k},
        {static_cast<Element*>(a[0].data), int(a[0].strides[0])},
        {static_cast<Element*>(a[1].data), int(a[1].strides[VF_BF16 ? 1 : 0])},
        {static_cast<Element*>(a[2].data), int(a[2].strides[0])},
        {static_cast<Element*>(a[2].data), int(a[2].strides[0])}, {1.0f, 0.0f});
    Gemm op;
    auto status = op.can_implement(args);
    if (status != cutlass::Status::kSuccess) return int(status);
    return int(op(args, nullptr, stream));
}
