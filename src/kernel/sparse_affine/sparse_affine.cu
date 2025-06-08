#include "sparse_affine.h"

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
) {
    // clang-format on
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= num_weight_rows * batch_size)
        return;

    const int batch_idx = idx / num_weight_rows;
    const int neuron_idx = idx % num_weight_rows;

    const int offset = batch_idx * max_entries;
    const int feature_size = feature_sizes[batch_idx];

    float sum = biases_v[neuron_idx];
    for(int i = 0; i < feature_size; i++) {
        int sparse_idx = features[i + offset];
        sum += weights_v[num_weight_rows * sparse_idx + neuron_idx];
    }

    output_v[num_output_rows * batch_idx + neuron_idx + output_offset] = activate(sum, act_type);
}

// clang-format off
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
) {
    // clang-format on
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= num_weight_rows * batch_size)
        return;

    const int batch_idx = idx / num_weight_rows;
    const int neuron_idx = idx % num_weight_rows;

    const int output_idx = num_output_rows * batch_idx + neuron_idx + output_offset;

    float grad = output_g[output_idx];
    if(grad == 0)
        return;
    grad *= activationDer(output_v[output_idx], act_type);

    // skip output gradient update because:
    // 1. feature transformer doesnt have a prev layer
    // 2. performing this update slows down the program by 2-3 seconds
    // output_g[output_idx] = grad;

    const int offset = batch_idx * max_entries;
    const int feature_size = feature_sizes[batch_idx];

    atomicAdd(&biases_g[neuron_idx], grad);
    for(int i = 0; i < feature_size; i++) {
        int sparse_idx = features[i + offset];
        atomicAdd(&weights_g[num_weight_rows * sparse_idx + neuron_idx], grad);
    }
}
