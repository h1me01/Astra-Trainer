#include "feature_transformer.h"

namespace kernel {

constexpr int block_size = 256;

__global__ void feature_transformer_bwd_kernel(
    float* weights_g,
    float* biases_g,
    const float* out_v,
    const float* out_g,
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
    const int out_idx = out_r * batch_idx + neuron_idx + out_offset;

    const float grad = out_g[out_idx] * activate_bwd(out_v[out_idx], act_type);
    if (grad == 0.0f)
        return;

    const int offset = batch_idx * max_entries;

    atomicAdd(&biases_g[neuron_idx], grad);
    for (int i = 0; i < max_entries; i++) {
        int feature_idx = features[i + offset];
        if (feature_idx == -1)
            break;
        atomicAdd(&weights_g[weights_r * feature_idx + neuron_idx], grad);
    }
}

void feature_transformer_bwd(
    DenseMatrix& weights_g,
    DenseMatrix& biases_g,
    const Tensor& out,
    const Array<int>& features,
    const int max_entries,
    const int out_offset,
    const Activation act_type
) {
    const auto& out_v = out.get_data();
    const auto& out_g = out.get_grads();

    const bool is_double = out_g.rows() / 2 == weights_g.rows();

    ASSERT(weights_g.rows() == biases_g.rows());
    ASSERT(weights_g.rows() == out_g.rows() / (is_double ? 2 : 1));

    ASSERT(
        weights_g.is_dev_allocated() && //
        biases_g.is_dev_allocated() &&  //
        out_g.is_dev_allocated() &&     //
        out_v.is_dev_allocated() &&     //
        features.is_dev_allocated()
    );

    const int blocks = get_num_blocks(weights_g.rows() * out_g.cols(), block_size);
    feature_transformer_bwd_kernel<<<blocks, block_size>>>(
        weights_g.dev_address(),
        biases_g.dev_address(),
        out_v.dev_address(),
        out_g.dev_address(),
        features.dev_address(),
        weights_g.rows(),
        out_g.rows(),
        out_g.cols(),
        max_entries,
        out_offset,
        act_type
    );
}

} // namespace kernel
