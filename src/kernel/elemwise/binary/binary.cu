#include "binary.h"

namespace kernel {

template <typename Op>
__global__ void fwd_kernel(const float* a, const float* b, float* c, const int n, Op op) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int vec_idx = idx * 4;

    if (vec_idx >= n)
        return;

    if (vec_idx + 4 <= n) {
        float4 a4 = as_vec<const float4>(a)[idx];
        float4 b4 = as_vec<const float4>(b)[idx];
        float4 c4;
        c4.x = op.forward(a4.x, b4.x);
        c4.y = op.forward(a4.y, b4.y);
        c4.z = op.forward(a4.z, b4.z);
        c4.w = op.forward(a4.w, b4.w);
        as_vec<float4>(c)[idx] = c4;
    } else {
        for (int i = vec_idx; i < n; i++)
            c[i] = op.forward(a[i], b[i]);
    }
}

template <typename Op>
__global__ void
bwd_kernel(const float* grad_out, const float* a, const float* b, float* grad_a, float* grad_b, const int n, Op op) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int vec_idx = idx * 4;

    if (vec_idx >= n)
        return;

    if (vec_idx + 4 <= n) {
        float4 go4 = as_vec<const float4>(grad_out)[idx];
        float4 a4 = as_vec<const float4>(a)[idx];
        float4 b4 = as_vec<const float4>(b)[idx];
        float4 ga4 = as_vec<float4>(grad_a)[idx];
        float4 gb4 = as_vec<float4>(grad_b)[idx];

        op.backward(go4.x, a4.x, b4.x, ga4.x, gb4.x);
        op.backward(go4.y, a4.y, b4.y, ga4.y, gb4.y);
        op.backward(go4.z, a4.z, b4.z, ga4.z, gb4.z);
        op.backward(go4.w, a4.w, b4.w, ga4.w, gb4.w);

        as_vec<float4>(grad_a)[idx] = ga4;
        as_vec<float4>(grad_b)[idx] = gb4;
    } else {
        for (int i = vec_idx; i < n; i++)
            op.backward(grad_out[i], a[i], b[i], grad_a[i], grad_b[i]);
    }
}

template <typename Op>
void ElemwiseBinary<Op>::forward(const DenseMatrix& a, const DenseMatrix& b, DenseMatrix& c, Op op) {
    CHECK(a.size() == b.size() && a.size() == c.size());
    CHECK(a.is_dev_allocated() && b.is_dev_allocated() && c.is_dev_allocated());

    const int grid = cuda::ceil_div(a.size(), 4 * BLOCK_SIZE);
    fwd_kernel<<<grid, BLOCK_SIZE>>>(a.dev_address(), b.dev_address(), c.dev_address(), a.size(), op);

    CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename Op>
void ElemwiseBinary<Op>::backward(Tensor& a, Tensor& b, const DenseMatrix& grad_out, Op op) {
    const auto& a_d = a.data();
    auto& a_g = a.grad();

    const auto& b_d = b.data();
    auto& b_g = b.grad();

    CHECK(a_d.size() == b_d.size() && a_d.size() == grad_out.size());

    CHECK(
        a_d.is_dev_allocated()    //
        && a_g.is_dev_allocated() //
        && b_d.is_dev_allocated() //
        && b_g.is_dev_allocated() //
        && grad_out.is_dev_allocated()
    );

    const int grid = cuda::ceil_div(a.size(), 4 * BLOCK_SIZE);
    bwd_kernel<<<grid, BLOCK_SIZE>>>(
        grad_out.dev_address(),
        a_d.dev_address(),
        b_d.dev_address(),
        a_g.dev_address(),
        b_g.dev_address(),
        a_d.size(),
        op
    );

    CUDA_KERNEL_LAUNCH_CHECK();
}

template struct ElemwiseBinary<AddBinary>;
template struct ElemwiseBinary<SubBinary>;
template struct ElemwiseBinary<MulBinary>;
template struct ElemwiseBinary<DivBinary>;

} // namespace kernel
