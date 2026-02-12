#include "sparse_affine_pairwise_mul.h"

namespace kernel {

constexpr dim3 block_size(256, 1);

template <Activation act_type>
__global__ void sparse_affine_pairwise_mul_bwd_kernel(
    float* weights_g,
    float* biases_g,
    const float* weights_v,
    const float* biases_v,
    const float* out_d,
    const float* out_g,
    const int* features,
    const int weights_r,
    const int out_r,
    const int batch_size,
    const int max_entries
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    const int half_size = weights_r / 2;
    if (row >= weights_r || batch_idx >= batch_size)
        return;

    extern __shared__ int shared_features[];

    const int num_threads = blockDim.x * blockDim.y;
    const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;

    for (int i = thread_id; i < max_entries; i += num_threads)
        shared_features[i] = features[batch_idx * max_entries + i];
    __syncthreads();

    const bool is_first_half = row < half_size;
    const int neuron_idx = is_first_half ? row : (row - half_size);
    const int out_idx = out_r * batch_idx + neuron_idx;

    float sum_a = biases_v[neuron_idx];
    float sum_b = biases_v[neuron_idx + half_size];

#pragma unroll
    for (int i = 0; i < max_entries; i++) {
        const int feature_idx = shared_features[i];
        if (feature_idx == -1)
            break;
        sum_a += weights_v[weights_r * feature_idx + neuron_idx];
        sum_b += weights_v[weights_r * feature_idx + neuron_idx + half_size];
    }

    const float grad = is_first_half ? (out_g[out_idx] * activate_bwd<act_type>(sum_a) * activate_fwd<act_type>(sum_b))
                                     : (out_g[out_idx] * activate_bwd<act_type>(sum_b) * activate_fwd<act_type>(sum_a));

    if (grad == 0.0f)
        return;

    atomicAdd(&biases_g[row], grad);

#pragma unroll
    for (int i = 0; i < max_entries; i++) {
        const int feature_idx = shared_features[i];
        if (feature_idx == -1)
            break;
        atomicAdd(&weights_g[weights_r * feature_idx + row], grad);
    }
}

void sparse_affine_pairwise_mul_bwd(
    Tensor& weights,
    Tensor& biases,
    const Tensor& out,
    const Array<int>& features,
    const int max_entries,
    const int out_offset,
    const Activation act_type
) {
    const auto& weights_v = weights.get_data();
    auto& weights_g = weights.get_grads();
    const auto& biases_v = biases.get_data();
    auto& biases_g = biases.get_grads();

    const auto& out_d = out.get_data();
    const auto& out_g = out.get_grads();

    ASSERT(weights_g.rows() == biases_g.rows());
    ASSERT(weights_g.rows() == weights_v.rows());
    ASSERT(weights_g.rows() % 2 == 0);
    ASSERT(out_g.rows() == weights_g.rows() / 2);

    ASSERT(
        weights_g.is_dev_allocated() && //
        biases_g.is_dev_allocated() &&  //
        weights_v.is_dev_allocated() && //
        biases_v.is_dev_allocated() &&  //
        out_g.is_dev_allocated() &&     //
        out_d.is_dev_allocated() &&     //
        features.is_dev_allocated()
    );

    dim3 grid((weights_g.rows() + block_size.x - 1) / block_size.x, (out_g.cols() + block_size.y - 1) / block_size.y);

    const int shared_mem = max_entries * sizeof(int);

    DISPATCH_ACTIVATION(
        act_type,
        sparse_affine_pairwise_mul_bwd_kernel,
        <<<grid, block_size, shared_mem>>>(
            weights_g.dev_address(),
            biases_g.dev_address(),
            weights_v.dev_address(),
            biases_v.dev_address(),
            out_d.dev_address() + out_offset,
            out_g.dev_address() + out_offset,
            features.dev_address(),
            weights_g.rows(),
            out_g.rows(),
            out_g.cols(),
            max_entries
        )
    );
}

} // namespace kernel
