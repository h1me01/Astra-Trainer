#include "unary.h"

namespace kernel {

template <typename Op>
__global__ void unary_fwd_kernel(const float* in, float* out, const int size, Op op) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int vec_idx = idx * 4;

    if (vec_idx >= size)
        return;

    if (vec_idx + 4 <= size) {
        float4 in4 = as_vec<const float4>(in)[idx];
        float4 out4;
        out4.x = op.forward(in4.x);
        out4.y = op.forward(in4.y);
        out4.z = op.forward(in4.z);
        out4.w = op.forward(in4.w);
        as_vec<float4>(out)[idx] = out4;
    } else {
        for (int i = vec_idx; i < size; i++)
            out[i] = op.forward(in[i]);
    }
}

template <typename Op>
__global__ void unary_bwd_kernel(const float* in_d, float* in_g, const float* out_g, const int size, Op op) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int vec_idx = idx * 4;

    if (vec_idx >= size)
        return;

    if (vec_idx + 4 <= size) {
        float4 in_d4 = as_vec<const float4>(in_d)[idx];
        float4 in_g4 = as_vec<float4>(in_g)[idx];
        float4 out_g4 = as_vec<const float4>(out_g)[idx];

        in_g4.x += op.backward(in_d4.x) * out_g4.x;
        in_g4.y += op.backward(in_d4.y) * out_g4.y;
        in_g4.z += op.backward(in_d4.z) * out_g4.z;
        in_g4.w += op.backward(in_d4.w) * out_g4.w;

        as_vec<float4>(in_g)[idx] = in_g4;
    } else {
        for (int i = vec_idx; i < size; i++)
            in_g[i] += op.backward(in_d[i]) * out_g[i];
    }
}

template <typename Op>
void ElemwiseUnary<Op>::forward(const DenseMatrix& in, DenseMatrix& out) {
    CHECK(in.size() == out.size());
    CHECK(in.is_dev_allocated() && out.is_dev_allocated());

    const int grid = cuda::ceil_div(in.size(), 4 * BLOCK_SIZE);
    unary_fwd_kernel<<<grid, BLOCK_SIZE>>>(in.dev_address(), out.dev_address(), in.size(), Op{});
    CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename Op>
void ElemwiseUnary<Op>::backward(Tensor& in, const DenseMatrix& out_g) {
    const auto& in_d = in.data();
    auto& in_g = in.grad();

    CHECK(in_d.size() == out_g.size());
    CHECK(in_d.is_dev_allocated() && out_g.is_dev_allocated());

    const int grid = cuda::ceil_div(in_d.size(), 4 * BLOCK_SIZE);
    unary_bwd_kernel<<<grid, BLOCK_SIZE>>>(
        in_d.dev_address(), in_g.dev_address(), out_g.dev_address(), in_d.size(), Op{}
    );
    CUDA_KERNEL_LAUNCH_CHECK();
}

template struct ElemwiseUnary<Linear>;
template struct ElemwiseUnary<ReLU>;
template struct ElemwiseUnary<ClippedReLU>;
template struct ElemwiseUnary<SqrClippedReLU>;
template struct ElemwiseUnary<Sigmoid>;

} // namespace kernel
