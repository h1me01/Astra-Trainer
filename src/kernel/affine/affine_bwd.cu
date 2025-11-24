#include "affine.h"

namespace kernel {

constexpr float alpha = 1;
constexpr float beta = 1;

constexpr int block_size = 128;

__global__ void activate_bwd( //
    const float *linear_out,  //
    float *grads,             //
    const int size,
    const ActivationType act_type //
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size)
        return;

    grads[idx] *= activate_bwd(linear_out[idx], act_type);
}

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
    atomicAdd(&biases_g[neuron_idx], out_g[idx]);
}

void affine_bwd(             //
    Tensor &weights,         //
    Tensor &biases,          //
    DenseMatrix &in_v,       //
    DenseMatrix &in_g,       //
    DenseMatrix &linear_out, //
    DenseMatrix &grads,      //
    ActivationType act_type  //
) {
    const auto &weights_v = weights.get_values();
    auto &weights_g = weights.get_gradients();

    auto &biases_g = biases.get_gradients();

    ASSERT(biases_g.cols() == 1 &&             //
           grads.rows() == biases_g.rows() &&  //
           in_g.cols() == grads.cols() &&      //
           weights_g.rows() == grads.rows() && //
           weights_g.cols() == in_g.rows());

    ASSERT(weights_v.is_dev_allocated() && //
           weights_g.is_dev_allocated() && //
           biases_g.is_dev_allocated() &&  //
           in_v.is_dev_allocated() &&      //
           in_g.is_dev_allocated() &&      //
           grads.is_dev_allocated());

    const int blocks = get_num_blocks(grads.size(), block_size);

    // first update gradients if activation was used
    if(has_activation(act_type)) {
        activate_bwd<<<blocks, block_size>>>( //
            linear_out.dev_address(),
            grads.dev_address(),
            grads.size(),
            act_type);
    }

    // update biases gradients
    biases_bwd_kernel<<<blocks, block_size>>>( //
        biases_g.dev_address(),
        grads.dev_address(),
        grads.rows(),
        grads.cols());

    // update weights gradient
    cublasSgemm(                 //
        CUBLAS_HANDLE,           // handle
        CUBLAS_OP_N,             // transa
        CUBLAS_OP_T,             // transb
        weights_g.rows(),        // m
        weights_g.cols(),        // n
        grads.cols(),            // k
        &alpha,                  // alpha
        grads.dev_address(),     // A
        grads.rows(),            // lda
        in_v.dev_address(),      // B
        in_v.rows(),             // ldb
        &beta,                   // beta
        weights_g.dev_address(), // C
        weights_g.rows()         // ldc
    );

    // calculates delta for the layer before this one as well
    cublasSgemm(                 //
        CUBLAS_HANDLE,           // handle
        CUBLAS_OP_T,             // transa
        CUBLAS_OP_N,             // transb
        in_g.rows(),             // m
        in_g.cols(),             // n
        weights_v.rows(),        // k
        &alpha,                  // alpha
        weights_v.dev_address(), // A
        weights_v.rows(),        // lda
        grads.dev_address(),     // B
        grads.rows(),            // ldb
        &beta,                   // beta
        in_g.dev_address(),      // C
        in_g.rows()              // ldc
    );
}

} // namespace kernel
