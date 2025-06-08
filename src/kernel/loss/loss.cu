#include "loss.h"

// clang-format off
__global__ void mpe_kernel
(
    const float *targets, 
    const float *output_v, 
    float *output_g, 
    float *loss, 
    const float power,
    const int size
) {
    // clang-format on
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size)
        return;

    float diff = output_v[idx] - targets[idx];
    float abs_diff = abs(diff);

    float grad = powf(abs_diff, power - 1.0f) * power;
    output_g[idx] = diff > 0 ? grad : -grad;

    atomicAdd(loss, powf(abs_diff, power));
}

// clang-format off
__global__ void mse_kernel
(
    const float *targets, 
    const float *output_v, 
    float *output_g, 
    float *loss, 
    const int size
) {
    // clang-format on
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size)
        return;

    float diff = output_v[idx] - targets[idx];
    output_g[idx] = 2.0f * diff;

    atomicAdd(loss, diff * diff);
}
