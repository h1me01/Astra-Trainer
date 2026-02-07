#include "feature_transformer.h"

namespace kernel {

constexpr int block_size = 256;

__global__ void feature_transformer_fwd_kernel(
    const float* weights_v,
    const float* biases_v,
    float* out_v,
    const int* features,
    const int weights_r,
    const int out_r,
    const int batch_size,
    const int max_entries,
    const Activation act_type
) {
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size)
        return;

    extern __shared__ int shared_features[];

    if (threadIdx.x < max_entries)
        shared_features[threadIdx.x] = features[batch_idx * max_entries + threadIdx.x];
    __syncthreads();

    for (int neuron_idx = threadIdx.x; neuron_idx < weights_r; neuron_idx += blockDim.x) {
        float sum = biases_v[neuron_idx];
#pragma unroll 8
        for (int i = 0; i < max_entries; i++) {
            int feature_idx = shared_features[i];
            if (feature_idx == -1)
                break;
            sum += weights_v[weights_r * feature_idx + neuron_idx];
        }

        const int out_idx = out_r * batch_idx + neuron_idx;
        out_v[out_idx] = activate_fwd(sum, act_type);
    }
}

void feature_transformer_fwd(
    const DenseMatrix& weights_v,
    const DenseMatrix& biases_v,
    DenseMatrix& out_v,
    const Array<int>& features,
    const int max_entries,
    const int out_offset,
    const Activation act_type
) {
    const bool is_double = out_v.rows() / 2 == weights_v.rows();

    ASSERT(weights_v.rows() == biases_v.rows());
    ASSERT(weights_v.rows() == out_v.rows() / (is_double ? 2 : 1));

    ASSERT(
        weights_v.is_dev_allocated() && //
        biases_v.is_dev_allocated() &&  //
        out_v.is_dev_allocated() &&     //
        features.is_dev_allocated()
    );

    int shared_mem_size = max_entries * sizeof(int);
    feature_transformer_fwd_kernel<<<out_v.cols(), block_size, shared_mem_size>>>(
        weights_v.dev_address(),
        biases_v.dev_address(),
        out_v.dev_address() + out_offset,
        features.dev_address(),
        weights_v.rows(),
        out_v.rows(),
        out_v.cols(),
        max_entries,
        act_type
    );
}

} // namespace kernel
