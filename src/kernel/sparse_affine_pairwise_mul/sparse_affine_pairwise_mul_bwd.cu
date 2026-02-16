#include "sparse_affine_pairwise_mul.h"

namespace kernel {

constexpr dim3 block_size(512, 1);

template <Activation act_type>
__global__ void sparse_affine_pairwise_mul_bwd_kernel(
    float* weights_g,
    float* biases_g,
    const float* weights_d,
    const float* biases_d,
    const float* out_g,
    const int* features,
    const int weights_r,
    const int batch_size,
    const int max_entries
) {
    const int neuron = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch = blockIdx.y;

    const int half = weights_r / 2;
    if (neuron >= half || batch >= batch_size)
        return;

    extern __shared__ int s_features[];
    for (int i = threadIdx.x; i < max_entries; i += blockDim.x)
        s_features[i] = features[batch * max_entries + i];
    __syncthreads();

    float sum_a = biases_d[neuron];
    float sum_b = biases_d[neuron + half];

#pragma unroll
    for (int i = 0; i < max_entries; i++) {
        int f = s_features[i];
        if (f == -1)
            break;
        int base = weights_r * f;
        sum_a += weights_d[base + neuron];
        sum_b += weights_d[base + neuron + half];
    }

    const float g = out_g[batch * weights_r + neuron];
    const float grad_a = g * activate_bwd<act_type>(sum_a) * activate_fwd<act_type>(sum_b);
    const float grad_b = g * activate_bwd<act_type>(sum_b) * activate_fwd<act_type>(sum_a);

    if (grad_a != 0.f)
        atomicAdd(&biases_g[neuron], grad_a);
    if (grad_b != 0.f)
        atomicAdd(&biases_g[neuron + half], grad_b);

#pragma unroll
    for (int i = 0; i < max_entries; i++) {
        int f = s_features[i];
        if (f == -1)
            break;
        int base = weights_r * f;
        if (grad_a != 0.f)
            atomicAdd(&weights_g[base + neuron], grad_a);
        if (grad_b != 0.f)
            atomicAdd(&weights_g[base + neuron + half], grad_b);
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
    const auto& weights_d = weights.get_data();
    auto& weights_g = weights.get_grads();
    const auto& biases_d = biases.get_data();
    auto& biases_g = biases.get_grads();

    const auto& out_d = out.get_data();
    const auto& out_g = out.get_grads();

    ASSERT(weights_g.rows() == biases_g.rows());
    ASSERT(out_d.rows() == weights_d.rows());

    ASSERT(
        weights_g.is_dev_allocated() && //
        biases_g.is_dev_allocated() &&  //
        weights_d.is_dev_allocated() && //
        biases_d.is_dev_allocated() &&  //
        out_g.is_dev_allocated() &&     //
        out_d.is_dev_allocated() &&     //
        features.is_dev_allocated()
    );

    dim3 grid(
        (weights_g.rows() + block_size.x - 1) / block_size.x, //
        (out_g.cols() + block_size.y - 1) / block_size.y
    );

    const int shared_mem = max_entries * sizeof(int);

    DISPATCH_ACTIVATION(
        act_type,
        sparse_affine_pairwise_mul_bwd_kernel,
        <<<grid, block_size, shared_mem>>>(
            weights_g.dev_address(),
            biases_g.dev_address(),
            weights_d.dev_address(),
            biases_d.dev_address(),
            out_g.dev_address() + out_offset,
            features.dev_address(),
            weights_g.rows(),
            out_g.cols(),
            max_entries
        )
    );
}

} // namespace kernel
