#include "sparse_affine.h"

const int block_size = 128;

// FORWARD

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

    int output_idx = a_r * batch_idx + neuron_idx + a_offset;

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
    const int batch_size,                //
    const int max_entries,               //
    const ActivationType act_type        //
) {
    ASSERT(batch_size == activated_v.cols());

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

// BACKWARD

__global__ void sparse_affine_bp_kernel( //
    const float *activated_g,            //
    const float *pre_activated,          //
    float *weights_g,                    //
    float *biases_g,                     //
    const int *features,                 //
    const int *feature_sizes,            //
    const int w_r,                       // weight rows
    const int a_r,                       // activated rows
    const int a_offset,                  // activated offset
    const int batch_size,                //
    const int max_entries,               //
    ActivationType act_type              //
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= w_r * batch_size)
        return;

    const int batch_idx = idx / w_r;
    const int neuron_idx = idx % w_r;

    const int output_idx = a_r * batch_idx + neuron_idx + a_offset;

    float grad = activated_g[output_idx];
    if(grad == 0)
        return;
    grad *= activate_der(pre_activated[output_idx], act_type);

    // no need to compute gradients for previous layer since previous are inputs

    const int offset = batch_idx * max_entries;
    const int feature_size = feature_sizes[batch_idx];

    atomicAdd(&biases_g[neuron_idx], grad);
    for(int i = 0; i < feature_size; i++) {
        int sparse_idx = features[i + offset];
        atomicAdd(&weights_g[w_r * sparse_idx + neuron_idx], grad);
    }
}

void sparse_affine_bwd(                      //
    const DenseMatrix<float> &activated_g,   //
    const DenseMatrix<float> &pre_activated, //
    DenseMatrix<float> &weights_g,           //
    DenseMatrix<float> &biases_g,            //
    const Array<int> &features,              //
    const Array<int> &feature_sizes,         //
    const int a_offset,                      //
    const int batch_size,                    //
    const int max_entries,                   //
    const ActivationType act_type            //
) {
    ASSERT(activated_g.cols() == batch_size);

    ASSERT(weights_g.dev_address() &&     //
           biases_g.dev_address() &&      //
           activated_g.dev_address() &&   //
           pre_activated.dev_address() && //
           features.dev_address() &&      //
           feature_sizes.dev_address());

    const int grid_size = std::ceil(float(weights_g.rows() * batch_size) / block_size);

    sparse_affine_bp_kernel<<<grid_size, block_size>>>( //
        activated_g.dev_address(),
        pre_activated.dev_address(),
        weights_g.dev_address(),
        biases_g.dev_address(),
        features.dev_address(),
        feature_sizes.dev_address(),
        weights_g.rows(),
        activated_g.rows(),
        a_offset,
        batch_size,
        max_entries,
        act_type);
}