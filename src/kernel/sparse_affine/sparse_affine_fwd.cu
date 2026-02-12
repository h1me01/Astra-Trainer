#include "sparse_affine.h"

namespace kernel {

constexpr int block_size = 256;

template <Activation act_type>
__global__ void sparse_affine_fwd_kernel(
    const float* weights_v,
    const float* biases_v,
    float* out_d,
    const int* features,
    const int weights_r,
    const int out_r,
    const int batch_size,
    const int max_entries
) {
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size)
        return;

    extern __shared__ int shared_features[];
    if (threadIdx.x < max_entries)
        shared_features[threadIdx.x] = features[batch_idx * max_entries + threadIdx.x];
    __syncthreads();

    const int iterations = weights_r / 4;
    const int remainder = weights_r % 4;

    const float4* weights_v4 = (const float4*)weights_v;
    const float4* biases_v4 = (const float4*)biases_v;
    float4* out_d4 = (float4*)out_d;

    for (int k = threadIdx.x; k < iterations; k += blockDim.x) {
        const int neuron_idx_base = k * 4;
        float4 sum = biases_v4[k];

#pragma unroll
        for (int i = 0; i < max_entries; i++) {
            const int feature_idx = shared_features[i];
            if (feature_idx == -1)
                break;
            add_t4(sum, weights_v4[feature_idx * iterations + k]);
        }

        sum.x = activate_fwd<act_type>(sum.x);
        sum.y = activate_fwd<act_type>(sum.y);
        sum.z = activate_fwd<act_type>(sum.z);
        sum.w = activate_fwd<act_type>(sum.w);

        const int out_idx = (out_r * batch_idx + neuron_idx_base) / 4;
        out_d4[out_idx] = sum;
    }

    if (remainder > 0) {
        for (int neuron_idx = iterations * 4 + threadIdx.x; neuron_idx < weights_r; neuron_idx += blockDim.x) {
            float sum = biases_v[neuron_idx];

#pragma unroll
            for (int i = 0; i < max_entries; i++) {
                const int feature_idx = shared_features[i];
                if (feature_idx == -1)
                    break;
                sum += weights_v[weights_r * feature_idx + neuron_idx];
            }

            const int out_idx = out_r * batch_idx + neuron_idx;
            out_d[out_idx] = activate_fwd<act_type>(sum);
        }
    }
}

void sparse_affine_fwd(
    const DenseMatrix& weights_v,
    const DenseMatrix& biases_v,
    DenseMatrix& out_d,
    const Array<int>& features,
    const int max_entries,
    const int out_offset,
    const Activation act_type
) {
    const bool is_double = out_d.rows() / 2 == weights_v.rows();

    ASSERT(weights_v.rows() == biases_v.rows());
    ASSERT(weights_v.rows() == out_d.rows() / (is_double ? 2 : 1));

    ASSERT(
        weights_v.is_dev_allocated() && //
        biases_v.is_dev_allocated() &&  //
        out_d.is_dev_allocated() &&     //
        features.is_dev_allocated()
    );

    int shared_mem_size = max_entries * sizeof(int);
    DISPATCH_ACTIVATION(
        act_type,
        sparse_affine_fwd_kernel,
        <<<out_d.cols(), block_size, shared_mem_size>>>(
            weights_v.dev_address(),
            biases_v.dev_address(),
            out_d.dev_address() + out_offset,
            features.dev_address(),
            weights_v.rows(),
            out_d.rows(),
            out_d.cols(),
            max_entries
        )
    );
}

} // namespace kernel