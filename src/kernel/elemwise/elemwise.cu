#include "elemwise.h"

namespace kernel {

template <typename Op>
__global__ void fwd_kernel(const float* a, const float* b, float* c, const int n, Op op) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = op.forward(a[idx], b[idx]);
}

template <typename Op>
__global__ void
bwd_kernel(const float* grad_out, const float* a, const float* b, float* grad_a, float* grad_b, const int n, Op op) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        op.backward(grad_out[idx], a[idx], b[idx], grad_a[idx], grad_b[idx]);
}

template <typename Op>
void Elemwise<Op>::forward(const DenseMatrix& a, const DenseMatrix& b, DenseMatrix& c) {
    CHECK(a.size() == b.size() && a.size() == c.size());

    const int grid = cuda::ceil_div(a.size(), BLOCK_SIZE);
    fwd_kernel<<<grid, BLOCK_SIZE>>>(a.dev_address(), b.dev_address(), c.dev_address(), a.size(), Op{});
}

template <typename Op>
void Elemwise<Op>::backward(const Tensor& a, const Tensor& b, const DenseMatrix& grad_out) {
    CHECK(a.data().size() == b.data().size() && a.data().size() == grad_out.size());

    const int grid = cuda::ceil_div(a.data().size(), BLOCK_SIZE);
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

template struct Elemwise<Add>;
template struct Elemwise<Sub>;
template struct Elemwise<Mul>;
template struct Elemwise<Div>;

} // namespace kernel
