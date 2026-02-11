#include "affine.h"

namespace kernel {

constexpr float alpha = 1.0f;
constexpr float beta = 0.0f;

constexpr int block_size = 256;

template <Activation act_type>
__global__ void biases_fwd_kernel(const float* biases_v, float* out_d, const int r, const int c) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= r * c)
        return;
    out_d[idx] = activate_fwd<act_type>(out_d[idx] + biases_v[idx % r]);
}

void affine_fwd(
    DenseMatrix& weights_v, DenseMatrix& biases_v, const DenseMatrix& inputs_v, DenseMatrix& out_d, Activation act_type
) {
    ASSERT(
        biases_v.cols() == 1 &&             //
        out_d.rows() == biases_v.rows() &&  //
        inputs_v.cols() == out_d.cols() &&  //
        weights_v.rows() == out_d.rows() && //
        weights_v.cols() == inputs_v.rows()
    );

    ASSERT(
        weights_v.is_dev_allocated() && //
        biases_v.is_dev_allocated() &&  //
        inputs_v.is_dev_allocated() &&  //
        out_d.is_dev_allocated()
    );

    // compute dot product
    cublasSgemm(
        CUBLAS_HANDLE,           // handle
        CUBLAS_OP_N,             // transa
        CUBLAS_OP_N,             // transb
        out_d.rows(),            // m
        out_d.cols(),            // n
        inputs_v.rows(),         // k
        &alpha,                  // alpha
        weights_v.dev_address(), // A
        weights_v.rows(),        // lda
        inputs_v.dev_address(),  // B
        inputs_v.rows(),         // ldb
        &beta,                   // beta
        out_d.dev_address(),     // C
        out_d.rows()             // ldc
    );

    // add biases to dot product
    const int blocks = get_num_blocks(out_d.size(), block_size);
    DISPATCH_ACTIVATION(
        act_type,
        biases_fwd_kernel,
        <<<blocks, block_size>>>(biases_v.dev_address(), out_d.dev_address(), out_d.rows(), out_d.cols())
    );
}

} // namespace kernel
