#include "feature_transformer.h"

namespace kernel {

constexpr int block_size = 128;

__global__ void feature_transformer_fwd_kernel(
    const float* weights_v,
    const float* biases_v,
    float* linear_out,
    float* activated,
    const int* features,
    const int weights_r,
    const int out_r,
    const int batch_size,
    const int max_entries,
    const int out_offset,
    const Activation act_type
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= weights_r * batch_size)
        return;

    const int batch_idx = idx / weights_r;
    const int neuron_idx = idx % weights_r;
    const int offset = batch_idx * max_entries;

    float sum = biases_v[neuron_idx];

    for (int i = 0; i < max_entries; i++) {
        int feature_idx = features[i + offset];
        if (feature_idx == -1)
            break;
        sum += weights_v[weights_r * feature_idx + neuron_idx];
    }

    const int out_idx = out_r * batch_idx + neuron_idx + out_offset;
    linear_out[out_idx] = sum;

    if (has_activation(act_type))
        activated[out_idx] = activate_fwd(sum, act_type);
}

void feature_transformer_fwd(
    const DenseMatrix& weights_v,
    const DenseMatrix& biases_v,
    DenseMatrix& linear_out,
    DenseMatrix& activated,
    const Array<int>& features,
    const int max_entries,
    const int out_offset,
    const Activation act_type
) {
    const bool is_double = linear_out.rows() / 2 == weights_v.rows();

    ASSERT(weights_v.rows() == biases_v.rows());
    ASSERT(weights_v.rows() == linear_out.rows() / (is_double ? 2 : 1));

    ASSERT(
        weights_v.is_dev_allocated() &&  //
        biases_v.is_dev_allocated() &&   //
        linear_out.is_dev_allocated() && //
        features.is_dev_allocated()
    );

    ASSERT(!has_activation(act_type) || activated.is_dev_allocated());

    const int blocks = get_num_blocks(weights_v.rows() * linear_out.cols(), block_size);
    feature_transformer_fwd_kernel<<<blocks, block_size>>>(
        weights_v.dev_address(),
        biases_v.dev_address(),
        linear_out.dev_address(),
        activated.dev_address(),
        features.dev_address(),
        weights_v.rows(),
        linear_out.rows(),
        linear_out.cols(),
        max_entries,
        out_offset,
        act_type
    );
}

} // namespace kernel
