#include "select.h"

const int block_size = 128;

// FORWARD

__global__ void select_fwd_kernel( //
    const float *input_v,          //
    float *output_v,               //
    const int *bucket_indices,     //
    const int batch_size,          //
    const int input_size,          //
    const int output_size          //
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= batch_size * output_size)
        return;

    const int batch_idx = idx / output_size;
    const int output_idx = idx % output_size;

    const int bucket = bucket_indices[batch_idx];
    const int input_offset = input_size * batch_idx + output_size * bucket + output_idx;

    output_v[output_size * batch_idx + output_idx] = input_v[input_offset];
}

void select_fwd(                       //
    const DenseMatrix<float> &input_v, //
    DenseMatrix<float> &output_v,      //
    const Array<int> &bucket_indices,  //
    const int batch_size,              //
    const int input_size,              //
    const int output_size              //
) {
    const int grid_size = std::ceil(batch_size * output_size / block_size);

    select_fwd_kernel<<<grid_size, block_size>>>( //
        input_v.dev_address(),
        output_v.dev_address(),
        bucket_indices.dev_address(),
        batch_size,
        input_size,
        output_size);
}

// BACKWARD

__global__ void select_bwd_kernel( //
    float *input_g,                //
    const float *output_g,         //
    const int *indices,            //
    const int batch_size,          //
    const int input_size,          //
    const int output_size          //
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= batch_size * output_size)
        return;

    const int batch_idx = idx / output_size;
    const int output_idx = idx % output_size;

    const int bucket = indices[batch_idx];
    const int input_offset = input_size * batch_idx + output_size * bucket + output_idx;

    input_g[input_offset] = output_g[output_size * batch_idx + output_idx];
}

void select_bwd(                        //
    DenseMatrix<float> &input_g,        //
    const DenseMatrix<float> &output_g, //
    const Array<int> &indices,          //
    const int batch_size,               //
    const int input_size,               //
    const int output_size               //
) {
    ASSERT(batch_size == indices.size());

    const int grid_size = std::ceil(batch_size * output_size / block_size);

    // clear input gradient
    input_g.clear_dev();

    select_bwd_kernel<<<grid_size, block_size>>>( //
        input_g.dev_address(),
        output_g.dev_address(),
        indices.dev_address(),
        batch_size,
        input_size,
        output_size);
}