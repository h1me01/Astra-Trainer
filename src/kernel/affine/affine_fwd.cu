#include "affine.h"

namespace kernel {

constexpr float alpha = 1.0f;
constexpr float beta = 0.0f;

constexpr int num_threads = 256;

__global__ void biases_fwd_kernel(const float* biases_d, float* out_d, const int r, const int c) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < r * c)
        out_d[idx] = out_d[idx] + biases_d[idx % r];
}

void affine_fwd(DenseMatrix& weights_d, DenseMatrix& biases_d, const DenseMatrix& inputs_d, DenseMatrix& out_d) {
    CHECK(
        biases_d.cols() == 1 &&             //
        out_d.rows() == biases_d.rows() &&  //
        inputs_d.cols() == out_d.cols() &&  //
        weights_d.rows() == out_d.rows() && //
        weights_d.cols() == inputs_d.rows()
    );

    CHECK(
        weights_d.is_dev_allocated() && //
        biases_d.is_dev_allocated() &&  //
        inputs_d.is_dev_allocated() &&  //
        out_d.is_dev_allocated()
    );

    // compute dot product
    cublas::sgemm(
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        out_d.rows(),
        out_d.cols(),
        inputs_d.rows(),
        alpha,
        weights_d.dev_address(),
        weights_d.rows(),
        inputs_d.dev_address(),
        inputs_d.rows(),
        beta,
        out_d.dev_address(),
        out_d.rows()
    );

    // add biases to dot product
    const int blocks = cuda::ceil_div(out_d.size(), num_threads);
    biases_fwd_kernel<<<blocks, num_threads>>>(biases_d.dev_address(), out_d.dev_address(), out_d.rows(), out_d.cols());

    CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace kernel
