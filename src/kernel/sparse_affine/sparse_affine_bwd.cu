#include "sparse_affine.h"

namespace kernel {

constexpr int num_threads = 512;
constexpr dim3 block_size(num_threads, 1);

template <typename Op>
__global__ void sparse_affine_bwd_kernel(
    float* weights_g,
    float* biases_g,
    const float* out_d,
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

    if (row >= weights_r || batch_idx >= batch_size)
        return;

    const int* sample_indices = features + batch_idx * max_entries;
    const int out_idx = batch_idx * out_r + row;

    float grad = out_g[out_idx] * op.backward<true>(out_d[out_idx]);
    if (grad == 0.0f)
        return;

    atomicAdd(&biases_g[row], grad);

    for (int i = 0; i < max_entries; i++) {
        const int f_idx = sample_indices[i];
        if (f_idx == -1)
            break;
        atomicAdd(&weights_g[f_idx * weights_r + row], grad);
    }
}

void sparse_affine_bwd(
    DenseMatrix& weights_g,
    DenseMatrix& biases_g,
    const Tensor& out,
    const SparseMatrix& indices,
    const int out_offset,
    ActOp op
) {
    const auto& out_d = out.data();
    const auto& out_g = out.grad();

    CHECK(
        weights_g.dev_address() && //
        biases_g.dev_address() &&  //
        out_d.dev_address() &&     //
        out_g.dev_address() &&     //
        indices.dev_address()
    );

    CHECK(weights_g.rows() == biases_g.rows());
    CHECK(out_g.cols() <= 65535 && out_g.rows() >= weights_g.rows() + out_offset);

    const int max_entries = indices.rows();

    const int batch_size = out_g.cols();
    const int row_tiles = cuda::ceil_div(weights_g.rows(), num_threads);

    dim3 grid(batch_size, row_tiles);

    std::visit(
        [&](auto op) {
            sparse_affine_bwd_kernel<<<grid, block_size>>>(
                weights_g.dev_address(),
                biases_g.dev_address(),
                out_d.dev_address() + out_offset,
                out_g.dev_address() + out_offset,
                indices.dev_address(),
                weights_g.rows(),
                out_g.rows(),
                batch_size,
                max_entries,
                op
            );
        },
        op
    );

    CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace kernel
