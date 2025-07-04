#include "pairwise_mul.h"

constexpr int block_size = 1024;

// FORWARD

__global__ void pairwise_mul_fwd_kernel( //
    const float *inputs_v,               //
    float *output_v,                     //
    const int output_size,               //
    const int batch_size                 //
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= output_size * batch_size)
        return;

    const int batch_idx = idx / output_size;
    const int output_idx = idx % output_size;

    const int base = 2 * output_size * batch_idx + output_idx;
    const float a = inputs_v[base];
    const float b = inputs_v[base + output_size];

    output_v[output_size * batch_idx + output_idx] = a * b;
}

void pairwise_mul_fwd_kernel(           //
    const DenseMatrix<float> &inputs_v, //
    DenseMatrix<float> &output_v        //
) {
    ASSERT(inputs_v.is_dev_allocated() //
           && output_v.is_dev_allocated());

    const int batch_size = output_v.cols();
    const int output_size = output_v.rows();

    const int grid_size = std::ceil(output_size * batch_size / block_size);

    pairwise_mul_fwd_kernel<<<grid_size, block_size>>>( //
        inputs_v.dev_address(),                         //
        output_v.dev_address(),                         //
        output_size,
        batch_size //
    );
}

// BACKWARD

__global__ void pairwise_mul_bwd_kernel( //
    const float *inputs_v,               //
    float *inputs_g,                     //
    const float *output_g,               //
    const int output_size,               //
    const int batch_size                 //
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= output_size * batch_size)
        return;

    const int batch_idx = idx / output_size;
    const int output_idx = idx % output_size;

    int output_g_idx = output_size * batch_idx + output_idx;
    int input_offset = 2 * output_size * batch_idx + output_idx;

    float gradIn = output_g[output_g_idx];

    inputs_g[input_offset] = gradIn * inputs_v[input_offset + output_size];
    inputs_g[input_offset + output_size] = gradIn * inputs_v[input_offset];
}

void pairwise_mul_bwd_kernel(         //
    Tensor &inputs,                   //
    const DenseMatrix<float> output_g //
) {
    const DenseMatrix<float> &inputs_v = inputs.get_data();
    DenseMatrix<float> &inputs_g = inputs.get_grads();

    ASSERT(inputs_v.is_dev_allocated()    //
           && inputs_g.is_dev_allocated() //
           && output_g.is_dev_allocated());

    const int batch_size = output_g.cols();
    const int output_size = output_g.rows();

    const int grid_size = std::ceil(output_size * batch_size / block_size);

    pairwise_mul_bwd_kernel<<<grid_size, block_size>>>( //
        inputs_v.dev_address(),                         //
        inputs_g.dev_address(),                         //
        output_g.dev_address(),                         //
        output_size,                                    //
        batch_size                                      //
    );
}
