#include "sparse_affine.h"

constexpr int block_size = 128;

__global__ void sparse_affine_kernel( //
    const float *weights_v,           //
    const float *biases_v,            //
    float *activated_v,               //
    float *pre_activated,             //
    const int *features,              //
    const int *feature_sizes,         //
    const int w_r,                    // weight rows
    const int a_r,                    // activated rows
    const int a_offset,               // activated offset
    const int batch_size,             //
    const int max_entries,            //
    ActivationType act_type           //
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
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

    const int output_idx = a_r * batch_idx + neuron_idx + a_offset;

    pre_activated[output_idx] = sum;
    activated_v[output_idx] = activate(sum, act_type);
}

void sparse_affine_fwd(                  //
    DenseMatrix<float> &activated_v,     //
    DenseMatrix<float> &pre_activated,   //
    const DenseMatrix<float> &weights_v, //
    const DenseMatrix<float> &biases_v,  //
    const Array<int> &features,          //
    const Array<int> &feature_sizes,     //
    const int a_offset,                  //
    const int max_entries,               //
    const ActivationType act_type        //
) {
    const int batch_size = activated_v.cols();

    ASSERT(weights_v.dev_address() &&     //
           biases_v.dev_address() &&      //
           activated_v.dev_address() &&   //
           pre_activated.dev_address() && //
           features.dev_address() &&      //
           feature_sizes.dev_address());

    const int grid_size = std::ceil(float(weights_v.rows() * batch_size) / block_size);

    sparse_affine_kernel<<<grid_size, block_size>>>( //
        weights_v.dev_address(),
        biases_v.dev_address(),
        activated_v.dev_address(),
        pre_activated.dev_address(),
        features.dev_address(),
        feature_sizes.dev_address(),
        weights_v.rows(),
        activated_v.rows(),
        a_offset,
        batch_size,
        max_entries,
        act_type);
}
