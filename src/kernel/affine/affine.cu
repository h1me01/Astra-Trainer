#include "affine.h"

constexpr int block_size = 128;

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
// clang-format off
__global__ void add_biases_kernel
(
    const float *biases_v, 
    float *output_v, 
    const int num_rows, 
    const int num_cols,
    const ActivationType act_type
) {
    // clang-format on
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= num_rows * num_cols)
        return;

    int neuron_idx = idx / num_cols;

    float weighted_sum = output_v[idx] + biases_v[neuron_idx];
    output_v[idx] = activate(weighted_sum, act_type);
}

// clang-format off
void affine
(
    DenseMatrix &weights_v, 
    DenseMatrix &biases_v, 
    DenseMatrix &inputs_v, 
    DenseMatrix &output_v,
    const ActivationType act_type
) {
    // clang-format on
    ASSERT(output_v.numRows() == biases_v.numRows() && biases_v.numCols() == 1);

    ASSERT(weights_v.numCols() == inputs_v.numRows() && //
           weights_v.numRows() == output_v.numRows() && //
           inputs_v.numCols() == output_v.numCols());

    ASSERT(weights_v.devAddress() && //
           biases_v.devAddress() &&  //
           inputs_v.devAddress() &&  //
           output_v.devAddress());

    // compute dot product
    // clang-format off
    cublasSgemm
    (
        CUBLAS_HANDLE,          // handle
        CUBLAS_OP_N,            // transa
        CUBLAS_OP_N,            // transb
        output_v.numRows(),     // m
        output_v.numCols(),     // n
        weights_v.numCols(),    // k
        &alpha,                 // alpha
        weights_v.devAddress(), // A
        weights_v.numRows(),    // lda
        inputs_v.devAddress(),  // B
        inputs_v.numRows(),     // ldb
        &beta,                  // beta
        output_v.devAddress(),  // C
        output_v.numRows()      // ldc
    );
    // clang-format on

    // add biases to dot product
    dim3 grid(std::ceil((float) output_v.size() / block_size));

    // clang-format off
    add_biases_kernel<<<grid, block_size>>>
    (
        biases_v.devAddress(), 
        output_v.devAddress(), 
        output_v.numRows(),
        output_v.numCols(), act_type
    );
    // clang-format on
}

// AFFINE BP
// clang-format off
__global__ void update_biases_grad_kernel
(
    const float *output_v, 
    float *output_g, 
    float *biases_g, 
    const int num_rows,
    const int num_cols, 
    const ActivationType act_type
) {
    // clang-format on
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= num_rows * num_cols)
        return;

    float grad = output_g[idx];
    if(grad == 0)
        return;

    grad *= activationDer(output_v[idx], act_type);
    output_g[idx] = grad;

    int neuron_idx = idx / num_cols;
    atomicAdd(&biases_g[neuron_idx], grad);
}

// clang-format off
void affine_bp
(
    Tensor &weights, 
    Tensor &biases, 
    Tensor &inputs, 
    Tensor &output, 
    const ActivationType act_type
) {
    // clang-format on
    const DenseMatrix &weights_v = weights.getValues();
    DenseMatrix &weights_g = weights.getGradients();

    DenseMatrix &biases_g = biases.getGradients();

    const DenseMatrix &inputs_v = inputs.getValues();
    DenseMatrix &inputs_g = inputs.getGradients();

    const DenseMatrix &output_v = output.getValues();
    const DenseMatrix &output_g = output.getGradients();

    ASSERT(output_g.numRows() == biases_g.numRows() && biases_g.numCols() == 1);

    ASSERT(weights_g.numCols() == inputs_g.numRows() && //
           weights_g.numRows() == output_g.numRows() && //
           inputs_g.numCols() == output_g.numCols());

    ASSERT(weights_v.devAddress() && //
           weights_g.devAddress() && //
           biases_g.devAddress() &&  //
           inputs_v.devAddress() &&  //
           inputs_g.devAddress() &&  //
           output_v.devAddress() &&  //
           output_g.devAddress());

    // update biases gradient
    dim3 grid(std::ceil((float) output_g.size() / block_size));
    // clang-format off
    update_biases_grad_kernel<<<grid, block_size>>>
    (
        output_v.devAddress(), 
        output_g.devAddress(), 
        biases_g.devAddress(),
        output_g.numRows(), 
        output_g.numCols(), 
        act_type
    );
    // clang-format on

    // update weights gradient
    // clang-format off
    cublasSgemm
    (
        CUBLAS_HANDLE,          // handle
        CUBLAS_OP_N,            // transa
        CUBLAS_OP_T,            // transb
        weights_g.numRows(),    // m
        weights_g.numCols(),    // n
        output_g.numCols(),     // k
        &alpha,                 // alpha
        output_g.devAddress(),  // A
        output_g.numRows(),     // lda
        inputs_v.devAddress(),  // B
        inputs_v.numRows(),     // ldb
        &beta,                  // beta
        weights_g.devAddress(), // C
        weights_g.numRows()     // ldc
    );
    // clang-format on

    // calculates delta for the layer before this one as well
    // clang-format off
    cublasSgemm
    (
        CUBLAS_HANDLE,          // handle
        CUBLAS_OP_T,            // transa
        CUBLAS_OP_N,            // transb
        inputs_g.numRows(),     // m
        inputs_g.numCols(),     // n
        weights_v.numRows(),    // k
        &alpha,                 // alpha
        weights_v.devAddress(), // A
        weights_v.numRows(),    // lda
        output_g.devAddress(),  // B
        output_g.numRows(),     // ldb
        &beta,                  // beta
        inputs_g.devAddress(),  // C
        inputs_g.numRows()      // ldc
    );
    // clang-format on
}
