#include "sparse_affine_pairwise_mul.h"

namespace kernel {

constexpr int block_size = 256;

template <Activation act_type>
__global__ void sparse_affine_pairwise_mul_fwd_kernel(
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

    const int half_size = weights_r / 2;
    const int iterations = half_size / 4;
    const int remainder = half_size % 4;

    const float4* weights_v4 = (const float4*)weights_v;
    const float4* biases_v4 = (const float4*)biases_v;
    float4* out_d4 = (float4*)out_d;

    for (int k = threadIdx.x; k < iterations; k += blockDim.x) {
        const int neuron_idx_base = k * 4;

        // first half
        float4 sum_a = biases_v4[k];
#pragma unroll
        for (int i = 0; i < max_entries; i++) {
            const int feature_idx = shared_features[i];
            if (feature_idx == -1)
                break;
            add_t4(sum_a, weights_v4[feature_idx * (weights_r / 4) + k]);
        }

        // second half
        float4 sum_b = biases_v4[iterations + k];
#pragma unroll
        for (int i = 0; i < max_entries; i++) {
            const int feature_idx = shared_features[i];
            if (feature_idx == -1)
                break;
            add_t4(sum_b, weights_v4[feature_idx * (weights_r / 4) + iterations + k]);
        }

        sum_a.x = activate_fwd<act_type>(sum_a.x);
        sum_a.y = activate_fwd<act_type>(sum_a.y);
        sum_a.z = activate_fwd<act_type>(sum_a.z);
        sum_a.w = activate_fwd<act_type>(sum_a.w);

        sum_b.x = activate_fwd<act_type>(sum_b.x);
        sum_b.y = activate_fwd<act_type>(sum_b.y);
        sum_b.z = activate_fwd<act_type>(sum_b.z);
        sum_b.w = activate_fwd<act_type>(sum_b.w);

        float4 result;
        result.x = sum_a.x * sum_b.x;
        result.y = sum_a.y * sum_b.y;
        result.z = sum_a.z * sum_b.z;
        result.w = sum_a.w * sum_b.w;

        const int out_idx = (out_r * batch_idx + neuron_idx_base) / 4;
        out_d4[out_idx] = result;
    }

    if (remainder > 0) {
        for (int neuron_idx = iterations * 4 + threadIdx.x; neuron_idx < half_size; neuron_idx += blockDim.x) {
            // first half
            float sum_a = biases_v[neuron_idx];
#pragma unroll
            for (int i = 0; i < max_entries; i++) {
                const int feature_idx = shared_features[i];
                if (feature_idx == -1)
                    break;
                sum_a += weights_v[weights_r * feature_idx + neuron_idx];
            }

            // second half
            float sum_b = biases_v[neuron_idx + half_size];
#pragma unroll
            for (int i = 0; i < max_entries; i++) {
                const int feature_idx = shared_features[i];
                if (feature_idx == -1)
                    break;
                sum_b += weights_v[weights_r * feature_idx + neuron_idx + half_size];
            }

            // Apply activation then multiply
            const int out_idx = out_r * batch_idx + neuron_idx;
            out_d[out_idx] = activate_fwd<act_type>(sum_a) * activate_fwd<act_type>(sum_b);
        }
    }
}

void sparse_affine_pairwise_mul_fwd(
    const DenseMatrix& weights_v,
    const DenseMatrix& biases_v,
    DenseMatrix& out_d,
    const Array<int>& features,
    const int max_entries,
    const int out_offset,
    const Activation act_type
) {
    ASSERT(weights_v.rows() == biases_v.rows());

    ASSERT(
        weights_v.is_dev_allocated() && //
        biases_v.is_dev_allocated() &&  //
        out_d.is_dev_allocated() &&     //
        features.is_dev_allocated()
    );

    int shared_mem_size = max_entries * sizeof(int);
    DISPATCH_ACTIVATION(
        act_type,
        sparse_affine_pairwise_mul_fwd_kernel,
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
