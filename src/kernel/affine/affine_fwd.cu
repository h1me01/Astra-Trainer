#include "affine.h"

namespace kernel {

constexpr float alpha = 1;
constexpr float beta = 0;

constexpr int block_size = 128;

__global__ void biases_fwd_kernel( //
    const float *biases_v,         //
    float *out_v,                  //
    const int r,                   //
    const int c                    //
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= r * c)
        return;

    const int neuron_idx = idx % r;
    out_v[idx] += biases_v[neuron_idx];
}

void affine_fwd(            //
    DenseMatrix &weights_v, //
    DenseMatrix &biases_v,  //
    DenseMatrix &inputs_v,  //
    DenseMatrix &out_v      //
) {
    ASSERT(biases_v.cols() == 1 &&             //
           out_v.rows() == biases_v.rows() &&  //
           inputs_v.cols() == out_v.cols() &&  //
           weights_v.rows() == out_v.rows() && //
           weights_v.cols() == inputs_v.rows());

    ASSERT(weights_v.is_dev_allocated() && //
           biases_v.is_dev_allocated() &&  //
           inputs_v.is_dev_allocated() &&  //
           out_v.is_dev_allocated());

    // compute dot product
    cublasSgemm(                 //
        CUBLAS_HANDLE,           // handle
        CUBLAS_OP_N,             // transa
        CUBLAS_OP_N,             // transb
        out_v.rows(),            // m
        out_v.cols(),            // n
        inputs_v.rows(),         // k
        &alpha,                  // alpha
        weights_v.dev_address(), // A
        weights_v.rows(),        // lda
        inputs_v.dev_address(),  // B
        inputs_v.rows(),         // ldb
        &beta,                   // beta
        out_v.dev_address(),     // C
        out_v.rows()             // ldc
    );

    // add biases to dot product
    const int blocks = get_num_blocks(out_v.size(), block_size);
    biases_fwd_kernel<<<blocks, block_size>>>( //
        biases_v.dev_address(),
        out_v.dev_address(),
        out_v.rows(),
        out_v.cols());
}

} // namespace kernel
