#pragma once

#include "../../nn/data.h"
#include "../activation.h"

__global__ void sparse_affine_kernel( //
    const float *weights_v,
    const float *biases_v,
    float *activated_v,
    float *pre_activated,
    const int *features,
    const int *feature_sizes,
    const int w_r,      // weight rows
    const int a_c,      // activated cols
    const int a_offset, // activated offset
    const int batch_size,
    const int max_entries,
    ActivationType act_type);

__global__ void sparse_affine_bp_kernel( //
    const float *activated_g,
    const float *pre_activated,
    float *weights_g,
    float *biases_g,
    const int *features,
    const int *feature_sizes,
    const int w_r,      // weight rows
    const int a_c,      // activated cols
    const int a_offset, // activated offset
    const int batch_size,
    const int max_entries,
    ActivationType act_type);
