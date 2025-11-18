#include "feature_transformer.h"

namespace kernel {

constexpr int block_size = 128;

template <bool UseActivation>
__global__ void feature_transformer_bwd_kernel( //
    float *weights_g,                           //
    float *biases_g,                            //
    const float *incoming_grad,                 //
    const float *fwd_out_v,                     //
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
    const int out_idx = out_r * batch_idx + neuron_idx + out_offset;

    float grad = incoming_grad[out_idx];

    if(UseActivation)
        grad *= activate_der(fwd_out_v[out_idx], act_type);

    if(grad == 0.0f)
        return;

    const int offset = batch_idx * max_entries;

    atomicAdd(&biases_g[neuron_idx], grad);
    for(int i = 0; i < max_entries; i++) {
        int feature_idx = features[i + offset];
        if(feature_idx == -1)
            break;
        atomicAdd(&weights_g[weights_r * feature_idx + neuron_idx], grad);
    }
}

void feature_transformer_bwd(         //
    DenseMatrix &weights_g,           //
    DenseMatrix &biases_g,            //
    const DenseMatrix &incoming_grad, //
    const DenseMatrix *fwd_out_v,     //
    const Array<int> &features,       //
    const int max_entries,            //
    const int out_offset,             //
    const ActivationType act_type     //
) {
    const bool use_act = (fwd_out_v != nullptr);
    const bool is_double = incoming_grad.rows() / 2 == weights_g.rows();

    ASSERT(weights_g.rows() == biases_g.rows());
    ASSERT(weights_g.rows() == incoming_grad.rows() / (is_double ? 2 : 1));

    ASSERT(weights_g.is_dev_allocated() &&     //
           biases_g.is_dev_allocated() &&      //
           incoming_grad.is_dev_allocated() && //
           features.is_dev_allocated());

    const int blocks = get_num_blocks(weights_g.rows() * incoming_grad.cols(), block_size);

    if(use_act) {
        ASSERT(fwd_out_v->is_dev_allocated());

        feature_transformer_bwd_kernel<true><<<blocks, block_size>>>( //
            weights_g.dev_address(),
            biases_g.dev_address(),
            incoming_grad.dev_address(),
            fwd_out_v->dev_address(),
            features.dev_address(),
            weights_g.rows(),
            incoming_grad.rows(),
            incoming_grad.cols(),
            max_entries,
            out_offset,
            act_type);
    } else {
        feature_transformer_bwd_kernel<false><<<blocks, block_size>>>( //
            weights_g.dev_address(),
            biases_g.dev_address(),
            incoming_grad.dev_address(),
            nullptr,
            features.dev_address(),
            weights_g.rows(),
            incoming_grad.rows(),
            incoming_grad.cols(),
            max_entries,
            out_offset,
            ActivationType::Linear);
    }
}

} // namespace kernel
