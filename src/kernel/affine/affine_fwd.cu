#include "affine.h"

namespace kernel {

constexpr float alpha = 1;
constexpr float beta = 0;

constexpr int block_size = 128;

template <bool UseActivation>
__global__ void biases_fwd_kernel( //
    const float *biases_v,         //
    float *linear_out,             //
    float *activated,              //
    const int r,                   //
    const int c,                   //
    const Activation act_type  //
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= r * c)
        return;

    const int neuron_idx = idx % r;
    const float weighted_sum = linear_out[idx] + biases_v[neuron_idx];

    linear_out[idx] = weighted_sum;
    if(UseActivation)
        activated[idx] = activate_fwd(weighted_sum, act_type);
}

void affine_fwd(             //
    DenseMatrix &weights_v,  //
    DenseMatrix &biases_v,   //
    DenseMatrix &inputs_v,   //
    DenseMatrix &linear_out, //
    DenseMatrix &activated,  //
    Activation act_type  //
) {
    ASSERT(biases_v.cols() == 1 &&                  //
           linear_out.rows() == biases_v.rows() &&  //
           inputs_v.cols() == linear_out.cols() &&  //
           weights_v.rows() == linear_out.rows() && //
           weights_v.cols() == inputs_v.rows());

    ASSERT(weights_v.is_dev_allocated() && //
           biases_v.is_dev_allocated() &&  //
           inputs_v.is_dev_allocated() &&  //
           linear_out.is_dev_allocated());

    // compute dot product
    cublasSgemm(                  //
        CUBLAS_HANDLE,            // handle
        CUBLAS_OP_N,              // transa
        CUBLAS_OP_N,              // transb
        linear_out.rows(),        // m
        linear_out.cols(),        // n
        inputs_v.rows(),          // k
        &alpha,                   // alpha
        weights_v.dev_address(),  // A
        weights_v.rows(),         // lda
        inputs_v.dev_address(),   // B
        inputs_v.rows(),          // ldb
        &beta,                    // beta
        linear_out.dev_address(), // C
        linear_out.rows()         // ldc
    );

    // add biases to dot product
    const int blocks = get_num_blocks(linear_out.size(), block_size);
    if(has_activation(act_type)) {
        ASSERT(activated.is_dev_allocated());

        biases_fwd_kernel<true><<<blocks, block_size>>>( //
            biases_v.dev_address(),
            linear_out.dev_address(),
            activated.dev_address(),
            linear_out.rows(),
            linear_out.cols(),
            act_type);
    } else {
        biases_fwd_kernel<false><<<blocks, block_size>>>( //
            biases_v.dev_address(),
            linear_out.dev_address(),
            activated.dev_address(),
            linear_out.rows(),
            linear_out.cols(),
            act_type);
    }
}

} // namespace kernel
