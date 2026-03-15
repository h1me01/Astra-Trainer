#include "sparse_affine_pairwise_mul.h"

namespace kernel {

constexpr int BLOCK_SIZE = 256;

template <typename Op>
__global__ void sparse_affine_pairwise_mul_fwd_vec_kernel(
    const float* weights_d,
    const float* biases_d,
    const int* indices,
    float* out_d,
    const int weights_r4,
    const int out_r4,
    const int batch_size,
    const int max_entries,
    Op op
) {
    extern __shared__ int shared_indices[];

    const int row4 = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = blockIdx.y;

    const int half4 = weights_r4 / 2;

    if (row4 >= half4 || batch_idx >= batch_size)
        return;

    const int* sample_indices = indices + batch_idx * max_entries;
    for (int i = threadIdx.x; i < max_entries; i += blockDim.x)
        shared_indices[i] = sample_indices[i];
    __syncthreads();

    const float4* w4 = as_vec<const float4>(weights_d);
    const float4* b4 = as_vec<const float4>(biases_d);

    float4 sum_a = b4[row4];
    float4 sum_b = b4[row4 + half4];

    for (int i = 0; i < max_entries; i++) {
        const int f_idx = shared_indices[i];
        if (f_idx == -1)
            break;

        const int base = f_idx * weights_r4 + row4;
        sum_a = add_t4(sum_a, w4[base]);
        sum_b = add_t4(sum_b, w4[base + half4]);
    }

    sum_a.x = op.forward(sum_a.x);
    sum_a.y = op.forward(sum_a.y);
    sum_a.z = op.forward(sum_a.z);
    sum_a.w = op.forward(sum_a.w);

    sum_b.x = op.forward(sum_b.x);
    sum_b.y = op.forward(sum_b.y);
    sum_b.z = op.forward(sum_b.z);
    sum_b.w = op.forward(sum_b.w);

    as_vec<float4>(out_d)[out_r4 * batch_idx + row4] = mul_t4(sum_a, sum_b);
}

template <typename Op>
__global__ void sparse_affine_pairwise_mul_fwd_kernel(
    const float* weights_d,
    const float* biases_d,
    const int* indices,
    float* out_d,
    const int weights_r,
    const int out_r,
    const int batch_size,
    const int max_entries,
    Op op
) {
    extern __shared__ int shared_indices[];

    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = blockIdx.y;
    const int half = weights_r / 2;

    if (row >= half || batch_idx >= batch_size)
        return;

    const int* sample_indices = indices + batch_idx * max_entries;
    for (int i = threadIdx.x; i < max_entries; i += blockDim.x)
        shared_indices[i] = sample_indices[i];
    __syncthreads();

    float sum_a = biases_d[row];
    float sum_b = biases_d[row + half];

    for (int i = 0; i < max_entries; i++) {
        const int f_idx = shared_indices[i];
        if (f_idx == -1)
            break;

        const int base = f_idx * weights_r + row;
        sum_a += weights_d[base];
        sum_b += weights_d[base + half];
    }

    out_d[out_r * batch_idx + row] = op.forward(sum_a) * op.forward(sum_b);
}

void sparse_affine_pairwise_mul_fwd(
    const DenseMatrix& weights_d,
    const DenseMatrix& biases_d,
    DenseMatrix& out_d,
    const SparseMatrix& indices,
    const int out_offset,
    ActOp op
) {
    CHECK(
        weights_d.is_dev_allocated() && //
        biases_d.is_dev_allocated() &&  //
        out_d.is_dev_allocated() &&     //
        indices.is_dev_allocated()
    );

    const int weights_r = weights_d.rows();
    const int batch_size = out_d.cols();

    CHECK(batch_size <= 65535 && 2 * out_d.rows() >= weights_r + out_offset);

    const int max_entries = indices.rows();

    const bool use_vec = (weights_r % 8 == 0);
    const int effective_rows = use_vec ? weights_r / 4 / 2 : weights_r / 2;
    const int threads = min(effective_rows, BLOCK_SIZE);
    const int row_blocks = cuda::ceil_div(effective_rows, threads);

    const int shared_mem_size = max_entries * sizeof(int);

    dim3 grid(row_blocks, batch_size);

    std::visit(
        [&](auto op) {
            if (use_vec) {
                sparse_affine_pairwise_mul_fwd_vec_kernel<<<grid, dim3(threads), shared_mem_size>>>(
                    weights_d.dev_address(),
                    biases_d.dev_address(),
                    indices.dev_address(),
                    out_d.dev_address() + out_offset,
                    weights_r / 4,
                    out_d.rows() / 4,
                    batch_size,
                    max_entries,
                    op
                );
            } else {
                sparse_affine_pairwise_mul_fwd_kernel<<<grid, dim3(threads), shared_mem_size>>>(
                    weights_d.dev_address(),
                    biases_d.dev_address(),
                    indices.dev_address(),
                    out_d.dev_address() + out_offset,
                    weights_r,
                    out_d.rows(),
                    batch_size,
                    max_entries,
                    op
                );
            }
        },
        op
    );

    CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace kernel
