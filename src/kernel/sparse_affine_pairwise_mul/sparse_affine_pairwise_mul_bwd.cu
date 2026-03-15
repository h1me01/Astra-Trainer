#include "sparse_affine_pairwise_mul.h"

namespace kernel {

constexpr int num_threads = 512;
constexpr dim3 block_size(num_threads, 1);

template <typename Op>
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
    const int max_entries,
    Op op
) {
    const int row = blockIdx.y * blockDim.x + threadIdx.x;
    const int batch_idx = blockIdx.x;

    const int half = weights_r / 2;

    extern __shared__ int shared_features[];
    for (int i = threadIdx.x; i < max_entries; i += blockDim.x)
        shared_features[i] = features[batch_idx * max_entries + i];
    __syncthreads();

    if (row >= half || batch_idx >= batch_size)
        return;

    const float grad = out_g[batch_idx * out_r + row];
    if (grad == 0.0f)
        return;

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

    const float grad_a = grad * op.backward(sum_a) * op.forward(sum_b);
    const float grad_b = grad * op.backward(sum_b) * op.forward(sum_a);

    if (grad_a != 0.0f)
        atomicAdd(&biases_g[row], grad_a);
    if (grad_b != 0.0f)
        atomicAdd(&biases_g[row + half], grad_b);

    for (int i = 0; i < max_entries; i++) {
        const int f_idx = shared_features[i];
        if (f_idx == -1)
            break;
        const int base = weights_r * f_idx + row;
        if (grad_a != 0.0f)
            atomicAdd(&weights_g[base], grad_a);
        if (grad_b != 0.0f)
            atomicAdd(&weights_g[base + half], grad_b);
    }
}

void sparse_affine_pairwise_mul_bwd(
    const DenseMatrix& weights_d,
    DenseMatrix& weights_g,
    Tensor& biases,
    const DenseMatrix& out_g,
    const SparseMatrix& indices,
    const int out_offset,
    ActOp op
) {
    const auto& biases_d = biases.data();
    auto& biases_g = biases.grad();

    CHECK(
        weights_g.is_dev_allocated() && //
        biases_g.is_dev_allocated() &&  //
        weights_d.is_dev_allocated() && //
        biases_d.is_dev_allocated() &&  //
        out_g.is_dev_allocated() &&     //
        indices.is_dev_allocated()
    );

    CHECK(weights_g.rows() == biases_g.rows());
    CHECK(out_g.cols() <= 65535 && 2 * out_g.rows() >= weights_g.rows() + out_offset);

    const int max_entries = indices.rows();

    const int row_tiles = cuda::ceil_div(weights_g.rows() / 2, num_threads);
    const int shared_mem = max_entries * sizeof(int);

    dim3 grid(out_g.cols(), row_tiles);

    std::visit(
        [&](auto op) {
            sparse_affine_pairwise_mul_bwd_kernel<<<grid, block_size, shared_mem>>>(
                weights_g.dev_address(),
                biases_g.dev_address(),
                weights_d.dev_address(),
                biases_d.dev_address(),
                out_g.dev_address() + out_offset,
                indices.dev_address(),
                weights_g.rows(),
                out_g.rows(),
                out_g.cols(),
                max_entries,
                op
            );
        },
        op
    );

    CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace kernel
