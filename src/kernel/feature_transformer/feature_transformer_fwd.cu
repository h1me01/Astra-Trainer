#include "feature_transformer.h"

namespace kernel {

constexpr int block_size = 128;

__global__ void feature_transformer_fwd_kernel( //
    const float *weights_v,                     //
    const float *biases_v,                      //
    float *out_v,                               //
    const int *features,                        //
    const int out_r,                            //
    const int batch_size,                       //
    const int max_entries                       //
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= out_r * batch_size)
        return;

    const int batch_idx = idx / out_r;
    const int neuron_idx = idx % out_r;

    const int offset = batch_idx * max_entries;

    float sum = biases_v[neuron_idx];
    for(int i = 0; i < max_entries; i++) {
        int feature_idx = features[i + offset];
        if(feature_idx == -1)
            break;
        sum += weights_v[out_r * feature_idx + neuron_idx];
    }

    out_v[out_r * batch_idx + neuron_idx] = sum;
}

void feature_transformer_fwd(     //
    const DenseMatrix &weights_v, //
    const DenseMatrix &biases_v,  //
    DenseMatrix &out_v,           //
    const Array<int> &features,   //
    const int max_entries         //
) {
    ASSERT(weights_v.rows() == out_v.rows() && //
           weights_v.rows() == biases_v.rows());

    ASSERT(weights_v.is_dev_allocated() && //
           biases_v.is_dev_allocated() &&  //
           out_v.is_dev_allocated() &&     //
           features.is_dev_allocated());

    const int blocks = get_num_blocks(out_v.size(), block_size);
    feature_transformer_fwd_kernel<<<blocks, block_size>>>( //
        weights_v.dev_address(),
        biases_v.dev_address(),
        out_v.dev_address(),
        features.dev_address(),
        out_v.rows(),
        out_v.cols(),
        max_entries);
}

} // namespace kernel
