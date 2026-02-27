#include "sparse_affine_pairwise_mul.h"

namespace kernel {

constexpr int num_threads = 512;
constexpr dim3 block_size(num_threads, 1);

template <ActivationType act_type>
__global__ void sparse_affine_pairwise_mul_bwd_kernel(
    float* weights_g,
    float* biases_g,
    const float* weights_d,
    const float* biases_d,
    const float* out_g,
    const int* features,
    const int weights_r,
    const int out_r,
    const int batch_size,
    const int max_entries
) {
    const int row = blockIdx.y * blockDim.x + threadIdx.x;
    const int batch_idx = blockIdx.x;

    const int half = weights_r / 2;

    if (row >= half || batch_idx >= batch_size)
        return;

    extern __shared__ int shared_features[];
    for (int i = threadIdx.x; i < max_entries; i += blockDim.x)
        shared_features[i] = features[batch_idx * max_entries + i];
    __syncthreads();

    float sum_a = biases_d[row];
    float sum_b = biases_d[row + half];

    for (int i = 0; i < max_entries; i++) {
        const int f_idx = shared_features[i];
        if (f_idx == -1)
            break;
        const int base = weights_r * f_idx + row;
        sum_a += weights_d[base];
        sum_b += weights_d[base + half];
    }

    const float grad = out_g[batch_idx * out_r + row];
    const float grad_a = grad * activate_bwd<act_type>(sum_a) * activate_fwd<act_type>(sum_b);
    const float grad_b = grad * activate_bwd<act_type>(sum_b) * activate_fwd<act_type>(sum_a);

    if (grad_a != 0.f)
        atomicAdd(&biases_g[row], grad_a);
    if (grad_b != 0.f)
        atomicAdd(&biases_g[row + half], grad_b);

    for (int i = 0; i < max_entries; i++) {
        const int f_idx = shared_features[i];
        if (f_idx == -1)
            break;
        const int base = weights_r * f_idx + row;
        if (grad_a != 0.f)
            atomicAdd(&weights_g[base], grad_a);
        if (grad_b != 0.f)
            atomicAdd(&weights_g[base + half], grad_b);
    }
}

void sparse_affine_pairwise_mul_bwd(
    const DenseMatrix& weights_d,
    DenseMatrix& weights_g,
    Tensor& biases,
    const Tensor& out,
    const Array<int>& features,
    const int max_entries,
    const int out_offset,
    const ActivationType act_type
) {
    const auto& biases_d = biases.get_data();
    auto& biases_g = biases.get_grads();

    const auto& out_d = out.get_data();
    const auto& out_g = out.get_grads();

    CHECK(
        weights_g.is_dev_allocated() && //
        biases_g.is_dev_allocated() &&  //
        weights_d.is_dev_allocated() && //
        biases_d.is_dev_allocated() &&  //
        out_g.is_dev_allocated() &&     //
        out_d.is_dev_allocated() &&     //
        features.is_dev_allocated()
    );

    CHECK(weights_g.rows() == biases_g.rows());
    CHECK(out_g.cols() <= 65535 && out_g.rows() >= out_offset + weights_g.rows());

    const int batch_size = out_g.cols();
    const int row_tiles = cuda::ceil_div(weights_g.rows(), num_threads);
    const int shared_mem = max_entries * sizeof(int);

    dim3 grid(batch_size, row_tiles);

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
            out_g.rows(),
            out_g.cols(),
            max_entries
        )
    );
}

} // namespace kernel
