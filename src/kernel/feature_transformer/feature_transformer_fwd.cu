#include "feature_transformer.h"

namespace kernel {

constexpr int block_size = 128;

template <bool UseActivation>
__global__ void feature_transformer_fwd_kernel( //
    const float *weights_v,                     //
    const float *biases_v,                      //
    float *out_v,                               //
    float *act_v,                               //
    const int *features,                        //
    const int weights_r,                        //
    const int out_r,                            //
    const int batch_size,                       //
    const int max_entries,                      //
    const int out_offset,                       //
    const ActivationType act_type               //
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= weights_r * batch_size)
        return;

    const int batch_idx = idx / weights_r;
    const int neuron_idx = idx % weights_r;
    const int offset = batch_idx * max_entries;

    float sum = biases_v[neuron_idx];

    for(int i = 0; i < max_entries; i++) {
        int feature_idx = features[i + offset];
        if(feature_idx == -1)
            break;
        sum += weights_v[weights_r * feature_idx + neuron_idx];
    }

    const int out_idx = out_r * batch_idx + neuron_idx + out_offset;
    out_v[out_idx] = sum;

    if(UseActivation)
        act_v[out_idx] = activate(sum, act_type);
}

void feature_transformer_fwd(     //
    const DenseMatrix &weights_v, //
    const DenseMatrix &biases_v,  //
    DenseMatrix &out_v,           //
    DenseMatrix *act_v,           //
    const Array<int> &features,   //
    const int max_entries,        //
    const int out_offset,         //
    const ActivationType act_type //
) {
    const bool use_act = (act_v != nullptr);
    const bool is_double = out_v.rows() / 2 == weights_v.rows();

    ASSERT(weights_v.rows() == biases_v.rows());
    ASSERT(weights_v.rows() == out_v.rows() / (is_double ? 2 : 1));

    ASSERT(weights_v.is_dev_allocated() && //
           biases_v.is_dev_allocated() &&  //
           out_v.is_dev_allocated() &&     //
           features.is_dev_allocated());

    const int blocks = get_num_blocks(weights_v.rows() * out_v.cols(), block_size);

    if(use_act) {
        ASSERT(act_v->is_dev_allocated());

        feature_transformer_fwd_kernel<true><<<blocks, block_size>>>( //
            weights_v.dev_address(),
            biases_v.dev_address(),
            out_v.dev_address(),
            act_v->dev_address(),
            features.dev_address(),
            weights_v.rows(),
            out_v.rows(),
            out_v.cols(),
            max_entries,
            out_offset,
            act_type);
    } else {
        feature_transformer_fwd_kernel<false><<<blocks, block_size>>>( //
            weights_v.dev_address(),
            biases_v.dev_address(),
            out_v.dev_address(),
            nullptr,
            features.dev_address(),
            weights_v.rows(),
            out_v.rows(),
            out_v.cols(),
            max_entries,
            out_offset,
            ActivationType::Linear);
    }
}

} // namespace kernel
