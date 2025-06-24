#include "sparse_affine.h"

__global__ void sparse_affine_kernel( //
    const float *weights_v,
    const float *biases_v,
    float *activated_v,
    float *prev_activated,
    const int *features,
    const int *feature_sizes,
    const int w_r,      // weight rows
    const int a_r,      // activated rows
    const int a_offset, // activated offset
    const int batch_size,
    const int max_entries,
    ActivationType act_type) //
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= w_r * batch_size)
        return;

    const int batch_idx = idx / w_r;
    const int neuron_idx = idx % w_r;

    const int offset = batch_idx * max_entries;
    const int feature_size = feature_sizes[batch_idx];

    float sum = biases_v[neuron_idx];
    for(int i = 0; i < feature_size; i++) {
        int sparse_idx = features[i + offset];
        sum += weights_v[w_r * sparse_idx + neuron_idx];
    }

    int output_idx = a_r * batch_idx + neuron_idx + a_offset;

    prev_activated[output_idx] = sum;
    activated_v[output_idx] = activate(sum, act_type);
}

__global__ void sparse_affine_bp_kernel( //
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
    ActivationType act_type) //
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= w_r * batch_size)
        return;

    const int batch_idx = idx / w_r;
    const int neuron_idx = idx % w_r;

    const int output_idx = a_r * batch_idx + neuron_idx + a_offset;

    float grad = activated_g[output_idx];
    if(grad == 0)
        return;
    grad *= activationDer(prev_activated[output_idx], act_type);

    // no need to compute gradients for previous layer since previous are inputs

    const int offset = batch_idx * max_entries;
    const int feature_size = feature_sizes[batch_idx];

    atomicAdd(&biases_g[neuron_idx], grad);
    for(int i = 0; i < feature_size; i++) {
        int sparse_idx = features[i + offset];
        atomicAdd(&weights_g[w_r * sparse_idx + neuron_idx], grad);
    }
}
