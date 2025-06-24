#pragma once

#include "../../nn/data.h"
#include "../activation.h"

// clang-format off
__global__ void sparse_affine_kernel
(
    const float *weights_v, 
    const float *biases_v, 
    float *activated,
    float *prev_activated,
    const int *features,
    const int *feature_sizes,
    const int w_r,      // weight rows
    const int a_r,      // activated rows
    const int a_offset, // activated offset
    const int batch_size,
    const int max_entries, 
    ActivationType act_type
);

__global__ void sparse_affine_bp_kernel
(
    const float *activated_g,
    const float *prev_activated, 
    float *weights_g, 
    float *biases_g,
    const int *features, 
    const int *feature_sizes,
    const int w_r,      // weight rows
    const int a_r,      // activated rows
    const int a_offset, // activated offset
    const int batch_size, 
    const int max_entries,
    ActivationType act_type
);
// clang-format on