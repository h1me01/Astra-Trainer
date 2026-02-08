#include "feature_transformer.h"

namespace kernel {

constexpr dim3 block_size(256, 1);

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
    const Activation act_type
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= weights_r || batch_idx >= batch_size)
        return;

    extern __shared__ int shared_features[];

    const int num_threads = blockDim.x * blockDim.y;
    const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;

    for (int i = thread_id; i < max_entries; i += num_threads)
        shared_features[i] = features[batch_idx * max_entries + i];
    __syncthreads();

    const int out_idx = out_r * batch_idx + row;
    const float grad = out_g[out_idx] * activate_bwd(out_v[out_idx], act_type);

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

    dim3 grid(
        (weights_g.rows() + block_size.x - 1) / block_size.x, //
        (out_g.cols() + block_size.y - 1) / block_size.y
    );

    const int shared_mem = max_entries * sizeof(int);

    feature_transformer_bwd_kernel<<<grid, block_size, shared_mem>>>(
        weights_g.dev_address(),
        biases_g.dev_address(),
        out_v.dev_address() + out_offset,
        out_g.dev_address() + out_offset,
        features.dev_address(),
        weights_g.rows(),
        out_g.rows(),
        out_g.cols(),
        max_entries,
        act_type
    );
}

} // namespace kernel
