#include "affine.h"

constexpr float alpha = 1;
constexpr float beta = 0;

// cublasSgemm performs C = A * B * alpha + C * beta

cublasHandle_t CUBLAS_HANDLE;

void createCublas() {
    cublasCreate(&CUBLAS_HANDLE);
}

void destroyCublas() {
    cublasDestroy(CUBLAS_HANDLE);
}

// AFFINE
__global__ void add_biases_kernel( //
    const float *biases_v,
    float *activated_v,
    float *pre_activated_v,
    const int r,
    const int c,
    const ActivationType act_type //
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= r * c)
        return;

    const int neuron_idx = idx / c;

    float weighted_sum = pre_activated_v[idx] + biases_v[neuron_idx];

    pre_activated_v[idx] = weighted_sum;
    activated_v[idx] = activate(weighted_sum, act_type);
}

void affine( //
    DenseMatrix &weights_v,
    DenseMatrix &biases_v,
    DenseMatrix &inputs_v,
    DenseMatrix &activated_v,
    DenseMatrix &pre_activated,
    const ActivationType act_type //
) {
    ASSERT(activated_v.numCols() == biases_v.numRows() && biases_v.numCols() == 1);

    ASSERT(weights_v.numCols() == inputs_v.numCols() &&    //
           weights_v.numRows() == activated_v.numCols() && //
           inputs_v.numRows() == activated_v.numRows());

    ASSERT(weights_v.devAddress() &&   //
           biases_v.devAddress() &&    //
           inputs_v.devAddress() &&    //
           activated_v.devAddress() && //
           pre_activated.devAddress());

    // compute dot product
    cublasSgemm(                    //
        CUBLAS_HANDLE,              // handle
        CUBLAS_OP_N,                // transa
        CUBLAS_OP_N,                // transb
        pre_activated.numCols(),    // m
        pre_activated.numRows(),    // n
        weights_v.numCols(),        // k
        &alpha,                     // alpha
        weights_v.devAddress(),     // A
        weights_v.numRows(),        // lda
        inputs_v.devAddress(),      // B
        inputs_v.numCols(),         // ldb
        &beta,                      // beta
        pre_activated.devAddress(), // C
        pre_activated.numCols()     // ldc
    );

    // add biases to dot product
    const int block_size = 128;
    const int grid_size = std::ceil((float) activated_v.size() / block_size);

    add_biases_kernel<<<grid_size, block_size>>>( //
        biases_v.devAddress(),
        activated_v.devAddress(),
        pre_activated.devAddress(),
        activated_v.numRows(),
        activated_v.numCols(),
        act_type);
}

// AFFINE BP
__global__ void update_biases_grad_kernel( //
    const float *pre_activated_v,
    float *activated_g,
    float *biases_g,
    const int r,
    const int c,
    const ActivationType act_type //
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= r * c)
        return;

    float grad = activated_g[idx];
    if(grad == 0)
        return;

    grad *= activationDer(pre_activated_v[idx], act_type);
    activated_g[idx] = grad;

    const int neuron_idx = idx / c;
    atomicAdd(&biases_g[neuron_idx], grad);
}

void affine_bp( //
    Tensor &weights,
    Tensor &biases,
    Tensor &inputs,
    Tensor &activated,
    DenseMatrix &pre_activated,
    const ActivationType act_type //
) {
    DenseMatrix &weights_v = weights.getValues();
    DenseMatrix &weights_g = weights.getGradients();

    DenseMatrix &biases_g = biases.getGradients();

    DenseMatrix &inputs_v = inputs.getValues();
    DenseMatrix &inputs_g = inputs.getGradients();

    DenseMatrix &activated_v = activated.getValues();
    DenseMatrix &activated_g = activated.getGradients();

    ASSERT(activated_g.numCols() == biases_g.numRows() && biases_g.numCols() == 1);

    ASSERT(weights_g.numCols() == inputs_g.numCols() &&    //
           weights_g.numRows() == activated_g.numCols() && //
           inputs_g.numRows() == activated_g.numRows());

    ASSERT(weights_v.devAddress() &&   //
           weights_g.devAddress() &&   //
           biases_g.devAddress() &&    //
           inputs_v.devAddress() &&    //
           inputs_g.devAddress() &&    //
           activated_v.devAddress() && //
           activated_g.devAddress() && //
           pre_activated.devAddress());

    // update biases gradient
    const int block_size = 128;
    const int grid_size = std::ceil((float) activated_g.size() / block_size);

    update_biases_grad_kernel<<<grid_size, block_size>>>( //
        pre_activated.devAddress(),
        activated_g.devAddress(),
        biases_g.devAddress(),
        activated_g.numCols(),
        activated_g.numRows(),
        act_type);

    // update weights gradient
    cublasSgemm(                  //
        CUBLAS_HANDLE,            // handle
        CUBLAS_OP_N,              // transa
        CUBLAS_OP_T,              // transb
        weights_g.numRows(),      // m
        weights_g.numCols(),      // n
        activated_g.numRows(),    // k
        &alpha,                   // alpha
        activated_g.devAddress(), // A
        activated_g.numCols(),    // lda
        inputs_v.devAddress(),    // B
        inputs_v.numCols(),       // ldb
        &beta,                    // beta
        weights_g.devAddress(),   // C
        weights_g.numRows()       // ldc
    );

    // calculates delta for the layer before this one as well
    cublasSgemm(                  //
        CUBLAS_HANDLE,            // handle
        CUBLAS_OP_T,              // transa
        CUBLAS_OP_N,              // transb
        inputs_g.numCols(),       // m
        inputs_g.numRows(),       // n
        weights_v.numRows(),      // k
        &alpha,                   // alpha
        weights_v.devAddress(),   // A
        weights_v.numRows(),      // lda
        activated_g.devAddress(), // B
        activated_g.numCols(),    // ldb
        &beta,                    // beta
        inputs_g.devAddress(),    // C
        inputs_g.numCols()        // ldc
    );
}
