#include "affine.h"

constexpr float alpha = 1;
constexpr float beta = 0;

constexpr int block_size = 128;

__global__ void biases_fwd_kernel( //
    const float *biases_v,         //
    float *activated_v,            //
    float *pre_activated_v,        //
    const int r,                   //
    const int c,                   //
    const ActivationType act_type  //
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= r * c)
        return;

    const int neuron_idx = idx % r;

    float weighted_sum = pre_activated_v[idx] + biases_v[neuron_idx];

    pre_activated_v[idx] = weighted_sum;
    activated_v[idx] = activate(weighted_sum, act_type);
}

void affine_fwd(                       //
    DenseMatrix<float> &weights_v,     //
    DenseMatrix<float> &biases_v,      //
    DenseMatrix<float> &inputs_v,      //
    DenseMatrix<float> &activated_v,   //
    DenseMatrix<float> &pre_activated, //
    const ActivationType act_type      //
) {
    ASSERT(activated_v.rows() == biases_v.rows() && biases_v.cols() == 1);

    ASSERT(weights_v.cols() == inputs_v.rows() &&    //
           weights_v.rows() == activated_v.rows() && //
           inputs_v.cols() == activated_v.cols());

    ASSERT(weights_v.dev_address() &&   //
           biases_v.dev_address() &&    //
           inputs_v.dev_address() &&    //
           activated_v.dev_address() && //
           pre_activated.dev_address());

    // compute dot product
    cublasSgemm(                     //
        CUBLAS_HANDLE,               // handle
        CUBLAS_OP_N,                 // transa
        CUBLAS_OP_N,                 // transb
        pre_activated.rows(),        // m
        pre_activated.cols(),        // n
        inputs_v.rows(),             // k
        &alpha,                      // alpha
        weights_v.dev_address(),     // A
        weights_v.rows(),            // lda
        inputs_v.dev_address(),      // B
        inputs_v.rows(),             // ldb
        &beta,                       // beta
        pre_activated.dev_address(), // C
        pre_activated.rows()         // ldc
    );

    const int grid_size = std::ceil((float) activated_v.size() / block_size);

    // add biases to dot product
    biases_fwd_kernel<<<grid_size, block_size>>>( //
        biases_v.dev_address(),
        activated_v.dev_address(),
        pre_activated.dev_address(),
        activated_v.rows(),
        activated_v.cols(),
        act_type);
}
