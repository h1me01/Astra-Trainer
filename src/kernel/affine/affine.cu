#include "affine.h"

constexpr float alpha = 1;
constexpr float beta = 0;

// cublasSgemm performs C = A * B * alpha + C * beta

cublasHandle_t CUBLAS_HANDLE;

void create_cublas() {
    cublasCreate(&CUBLAS_HANDLE);
}

void destroy_cublas() {
    cublasDestroy(CUBLAS_HANDLE);
}

// AFFINE
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

    const int neuron_idx = idx / c;

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

    // add biases to dot product
    const int block_size = 128;
    const int grid_size = std::ceil((float) activated_v.size() / block_size);

    biases_fwd_kernel<<<grid_size, block_size>>>( //
        biases_v.dev_address(),
        activated_v.dev_address(),
        pre_activated.dev_address(),
        activated_v.rows(),
        activated_v.cols(),
        act_type);
}

// AFFINE BP
__global__ void biases_bwd_kernel( //
    const float *pre_activated_v,  //
    float *activated_g,            //
    float *biases_g,               //
    const int r,                   //
    const int c,                   //
    const ActivationType act_type  //
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= r * c)
        return;

    const int neuron_idx = idx / c;

    const float grad = activated_g[idx] * activate_der(pre_activated_v[idx], act_type);

    activated_g[idx] = grad;
    atomicAdd(&biases_g[neuron_idx], grad);
}

void affine_bwd(                       //
    Tensor &weights,                   //
    Tensor &biases,                    //
    Tensor &inputs,                    //
    Tensor &activated,                 //
    DenseMatrix<float> &pre_activated, //
    const ActivationType act_type      //
) {
    const DenseMatrix<float> &weights_v = weights.get_data();
    DenseMatrix<float> &weights_g = weights.get_grads();

    DenseMatrix<float> &biases_g = biases.get_grads();

    const DenseMatrix<float> &inputs_v = inputs.get_data();
    DenseMatrix<float> &inputs_g = inputs.get_grads();

    const DenseMatrix<float> &activated_v = activated.get_data();
    const DenseMatrix<float> &activated_g = activated.get_grads();

    ASSERT(activated_g.rows() == biases_g.rows() && biases_g.cols() == 1);

    ASSERT(weights_g.cols() == inputs_g.rows() &&    //
           weights_g.rows() == activated_g.rows() && //
           inputs_g.cols() == activated_g.cols());

    ASSERT(weights_v.dev_address() &&   //
           weights_g.dev_address() &&   //
           biases_g.dev_address() &&    //
           inputs_v.dev_address() &&    //
           inputs_g.dev_address() &&    //
           activated_v.dev_address() && //
           activated_g.dev_address() && //
           pre_activated.dev_address());

    // update gradients with activation derivatives
    // and update biases gradients
    const int block_size = 128;
    const int grid_size = std::ceil((float) activated_g.size() / block_size);

    biases_bwd_kernel<<<grid_size, block_size>>>( //
        pre_activated.dev_address(),
        activated_g.dev_address(),
        biases_g.dev_address(),
        activated_g.rows(),
        activated_g.cols(),
        act_type);

    // update weights gradient
    cublasSgemm(                   //
        CUBLAS_HANDLE,             // handle
        CUBLAS_OP_N,               // transa
        CUBLAS_OP_T,               // transb
        weights_g.rows(),          // m
        weights_g.cols(),          // n
        activated_g.cols(),        // k
        &alpha,                    // alpha
        activated_g.dev_address(), // A
        activated_g.rows(),        // lda
        inputs_v.dev_address(),    // B
        inputs_v.rows(),           // ldb
        &beta,                     // beta
        weights_g.dev_address(),   // C
        weights_g.rows()           // ldc
    );

    // calculates delta for the layer before this one as well
    cublasSgemm(                   //
        CUBLAS_HANDLE,             // handle
        CUBLAS_OP_T,               // transa
        CUBLAS_OP_N,               // transb
        inputs_g.rows(),           // m
        inputs_g.cols(),           // n
        weights_v.rows(),          // k
        &alpha,                    // alpha
        weights_v.dev_address(),   // A
        weights_v.rows(),          // lda
        activated_g.dev_address(), // B
        activated_g.rows(),        // ldb
        &beta,                     // beta
        inputs_g.dev_address(),    // C
        inputs_g.rows()            // ldc
    );
}