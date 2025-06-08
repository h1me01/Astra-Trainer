#pragma once

#include "../../nn/data.h"
#include "../activation.h"

// clang-format off
__global__ void sparse_affine_kernel
(
    const float *weights_v, 
    const float *biases_v, 
    float *output_v, 
    const int *features,
    const int *feature_sizes,
    const int num_weight_rows, 
    const int num_output_rows, 
    const int output_offset, 
    const int batch_size,
    const int max_entries, 
    ActivationType act_type
);

__global__ void sparse_affine_bp_kernel
(
    const float *output_v, 
    const float *output_g, 
    float *weights_g, 
    float *biases_g,
    const int *features, 
    const int *feature_sizes,
    const int num_weight_rows,
    const int num_output_rows, 
    const int output_offset, 
    const int batch_size, 
    const int max_entries,
    ActivationType act_type
);
// clang-format on