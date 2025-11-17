#include "affine.h"

namespace kernel {

constexpr float alpha = 1;
constexpr float beta = 0;

constexpr int block_size = 128;

__global__ void biases_bwd_kernel( //
    float *biases_g,               //
    const float *out_g,            //
    const int r,                   //
    const int c                    //
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= r * c)
        return;

    const int neuron_idx = idx % r;

    const float grad = out_g[idx];
    if(grad != 0)
        atomicAdd(&biases_g[neuron_idx], grad);
}

void affine_bwd(     //
    Tensor &weights, //
    Tensor &biases,  //
    Tensor &inputs,  //
    Tensor &out      //
) {
    const auto &weights_v = weights.get_values();
    auto &weights_g = weights.get_gradients();

    auto &biases_g = biases.get_gradients();

    const auto &inputs_v = inputs.get_values();
    auto &inputs_g = inputs.get_gradients();

    const auto &out_g = out.get_gradients();

    ASSERT(biases_g.cols() == 1 &&             //
           out_g.rows() == biases_g.rows() &&  //
           inputs_g.cols() == out_g.cols() &&  //
           weights_g.rows() == out_g.rows() && //
           weights_g.cols() == inputs_g.rows());

    ASSERT(weights_v.is_dev_allocated() && //
           weights_g.is_dev_allocated() && //
           biases_g.is_dev_allocated() &&  //
           inputs_v.is_dev_allocated() &&  //
           inputs_g.is_dev_allocated() &&  //
           out_g.is_dev_allocated());

    // update biases gradients
    const int blocks = get_num_blocks(out_g.size(), block_size);
    biases_bwd_kernel<<<blocks, block_size>>>( //
        biases_g.dev_address(),
        out_g.dev_address(),
        out_g.rows(),
        out_g.cols());

    // update weights gradient
    cublasSgemm(                 //
        CUBLAS_HANDLE,           // handle
        CUBLAS_OP_N,             // transa
        CUBLAS_OP_T,             // transb
        weights_g.rows(),        // m
        weights_g.cols(),        // n
        out_g.cols(),            // k
        &alpha,                  // alpha
        out_g.dev_address(),     // A
        out_g.rows(),            // lda
        inputs_v.dev_address(),  // B
        inputs_v.rows(),         // ldb
        &beta,                   // beta
        weights_g.dev_address(), // C
        weights_g.rows()         // ldc
    );

    // calculates delta for the layer before this one as well
    cublasSgemm(                 //
        CUBLAS_HANDLE,           // handle
        CUBLAS_OP_T,             // transa
        CUBLAS_OP_N,             // transb
        inputs_g.rows(),         // m
        inputs_g.cols(),         // n
        weights_v.rows(),        // k
        &alpha,                  // alpha
        weights_v.dev_address(), // A
        weights_v.rows(),        // lda
        out_g.dev_address(),     // B
        out_g.rows(),            // ldb
        &beta,                   // beta
        inputs_g.dev_address(),  // C
        inputs_g.rows()          // ldc
    );
}

} // namespace kernel
