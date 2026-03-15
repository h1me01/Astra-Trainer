#include "binary.h"

namespace kernel {

template <typename Op>
__global__ void
fwd_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, const int n, Op op) {
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
__global__ void bwd_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ grad_a,
    float* __restrict__ grad_b,
    const int n,
    Op op
) {
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
void ElemwiseBinary<Op>::forward(const DenseMatrix& a, const DenseMatrix& b, DenseMatrix& c) {
    CHECK(a.size() == b.size() && a.size() == c.size());

    const int grid = cuda::ceil_div(a.size(), 4 * BLOCK_SIZE);
    fwd_kernel<<<grid, BLOCK_SIZE>>>(a.dev_address(), b.dev_address(), c.dev_address(), a.size(), Op{});
}

template <typename Op>
void ElemwiseBinary<Op>::backward(const Tensor& a, const Tensor& b, const DenseMatrix& grad_out) {
    CHECK(a.data().size() == b.data().size() && a.data().size() == grad_out.size());

    const int grid = cuda::ceil_div(a.size(), 4 * BLOCK_SIZE);
    bwd_kernel<<<grid, BLOCK_SIZE>>>(
        grad_out.dev_address(),
        a.data().dev_address(),
        b.data().dev_address(),
        a.grad().dev_address(),
        b.grad().dev_address(),
        a.data().size(),
        Op{}
    );
}

template struct ElemwiseBinary<Add>;
template struct ElemwiseBinary<Sub>;
template struct ElemwiseBinary<Mul>;
template struct ElemwiseBinary<Div>;

} // namespace kernel
