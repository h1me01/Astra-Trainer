#include "sparse_affine.h"

namespace kernel {

constexpr int num_threads = 256;
constexpr dim3 block_size(num_threads, 1);

template <Activation act_type>
__global__ void sparse_affine_bwd_kernel(
    float* weights_g,
    float* biases_g,
    const float* out_d,
    const float* out_g,
    const int* features,
    const int weights_r,
    const int out_r,
    const int batch_size,
    const int max_entries
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= weights_r)
        return;

    const int batch = blockIdx.y * blockDim.y + threadIdx.y;

    extern __shared__ int s_features[];
    for (int i = threadIdx.x; i < max_entries; i += blockDim.x)
        if (batch < batch_size)
            s_features[i] = features[batch * max_entries + i];
    __syncthreads();

    float grad = 0.f;
    if (batch < batch_size) {
        int out_idx = batch * out_r + row;
        float g = out_g[out_idx];
        if (g != 0.f)
            grad = g * activate_bwd<act_type, true>(out_d[out_idx]);
    }

    __shared__ float block_grad[num_threads];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    block_grad[tid] = grad;
    __syncthreads();

    for (int stride = blockDim.y / 2; stride > 0; stride = stride / 2) {
        if (threadIdx.y < stride && batch + stride < batch_size)
            block_grad[tid] += block_grad[tid + stride * blockDim.x];
        __syncthreads();
    }

    if (threadIdx.y == 0) {
        float gsum = block_grad[threadIdx.x];
        if (gsum != 0.f) {
            atomicAdd(&biases_g[row], gsum);

#pragma unroll
            for (int i = 0; i < max_entries; i++) {
                int f = s_features[i];
                if (f == -1)
                    break;
                atomicAdd(&weights_g[f * weights_r + row], gsum);
            }
        }
    }
}

void sparse_affine_bwd(
    DenseMatrix& weights_g,
    DenseMatrix& biases_g,
    const Tensor& out,
    const Array<int>& features,
    const int max_entries,
    const int out_offset,
    const Activation act_type
) {
    const auto& out_d = out.get_data();
    const auto& out_g = out.get_grads();
    const bool is_double = out_g.rows() / 2 == weights_g.rows();

    ASSERT(weights_g.rows() == biases_g.rows());
    ASSERT(weights_g.rows() == out_g.rows() / (is_double ? 2 : 1));

    ASSERT(
        weights_g.is_dev_allocated() && //
        biases_g.is_dev_allocated() &&  //
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
        sparse_affine_bwd_kernel,
        <<<grid, block_size, shared_mem>>>(
            weights_g.dev_address(),
            biases_g.dev_address(),
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
