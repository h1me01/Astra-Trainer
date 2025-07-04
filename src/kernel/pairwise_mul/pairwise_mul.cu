#include "pairwise_mul.h"

// FORWARD

__global__ void pairwise_mul_fwd_kernel( //
    const float *input_v,                //
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
    const float a = input_v[base];
    const float b = input_v[base + output_size];

    output_v[output_size * batch_idx + output_idx] = a * b;
}

// BACKWARD

__global__ void pairwise_mul_bwd_kernel( //
    const float *input_v,                //
    float *input_g,                      //
    const float *output_g,               //
    const int output_size,               //
    const int batch_size                 //
) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if(tid >= output_size * batch_size)
        return;

    const int batch_idx = tid / output_size;
    const int output_idx = tid % output_size;

    int output_g_idx = output_size * batch_idx + output_idx;
    int input_offset = 2 * output_size * batch_idx + output_idx;

    float gradIn = output_g[output_g_idx];

    input_g[input_offset] += gradIn * input_v[input_offset + output_size];
    input_g[input_offset + output_size] += gradIn * input_v[input_offset];
}
