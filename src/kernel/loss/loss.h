#pragma once

#include "../../nn/data.h"
#include "../activation.h"

// clang-format off
__global__ void mpe_kernel
(
    const float *targets, 
    const float *output_v, 
    float *output_g, 
    float *loss, 
    const float power,
    const int size
);

__global__ void mse_kernel
(
    const float *targets, 
    const float *output_v, 
    float *output_g, 
    float *loss, 
    const int size
);
// clang-format on