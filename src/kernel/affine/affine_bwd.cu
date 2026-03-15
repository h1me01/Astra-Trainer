#include "affine.h"

namespace kernel {

constexpr float alpha = 1.0f;
constexpr float beta = 1.0f;

constexpr int BLOCK_SIZE = 256;

__global__ void biases_bwd_kernel(float* biases_g, const float* out_g, const int r, const int c) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= r * c)
        return;

    const float grad = out_g[idx];
    if (grad != 0)
        atomicAdd(&biases_g[idx % r], grad);
}

void affine_bwd(Tensor& weights, Tensor& biases, Tensor& in, Tensor& out) {
    const auto& in_d = in.data();
    auto& in_g = in.grad();

    const auto& out_d = out.data();
    const auto& out_g = out.grad();

    const auto& weights_d = weights.data();
    auto& weights_g = weights.grad();

    auto& biases_g = biases.grad();

    CHECK(
        biases_g.cols() == 1 &&             //
        out_g.rows() == biases_g.rows() &&  //
        in_g.cols() == out_g.cols() &&      //
        weights_g.rows() == out_g.rows() && //
        weights_g.cols() == in_g.rows()
    );

    CHECK(
        weights_d.is_dev_allocated() && //
        weights_g.is_dev_allocated() && //
        biases_g.is_dev_allocated() &&  //
        in_d.is_dev_allocated() &&      //
        in_g.is_dev_allocated() &&      //
        out_g.is_dev_allocated()
    );

    // update biases gradients
    const int blocks = cuda::ceil_div(out_g.size(), BLOCK_SIZE);
    biases_bwd_kernel<<<blocks, BLOCK_SIZE>>>(biases_g.dev_address(), out_g.dev_address(), out_g.rows(), out_g.cols());

    CUDA_KERNEL_LAUNCH_CHECK();

    // update weights gradient
    cublas::sgemm(
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        weights_g.rows(),
        weights_g.cols(),
        out_g.cols(),
        alpha,
        out_g.dev_address(),
        out_g.rows(),
        in_d.dev_address(),
        in_d.rows(),
        beta,
        weights_g.dev_address(),
        weights_g.rows()
    );

    // calculates delta for the layer before this one as well
    cublas::sgemm(
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        in_g.rows(),
        in_g.cols(),
        weights_d.rows(),
        alpha,
        weights_d.dev_address(),
        weights_d.rows(),
        out_g.dev_address(),
        out_g.rows(),
        beta,
        in_g.dev_address(),
        in_g.rows()
    );
}

} // namespace kernel
