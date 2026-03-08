#include "sparse_affine.h"

namespace kernel {

constexpr int num_threads = 256;

template <ActivationType act_type>
__global__ void sparse_affine_fwd_vec_kernel(
    const float* weights_d,
    const float* biases_d,
    const int* indices,
    float* out_d,
    const int weights_r4,
    const int out_r4,
    const int batch_size,
    const int max_entries
) {
    extern __shared__ int shared_indices[];

    const int row4 = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = blockIdx.y;

    if (row4 >= weights_r4 || batch_idx >= batch_size)
        return;

    const int* sample_indices = indices + batch_idx * max_entries;
    for (int i = threadIdx.x; i < max_entries; i += blockDim.x)
        shared_indices[i] = sample_indices[i];
    __syncthreads();

    const float4* w4 = as_vec<const float4>(weights_d);

    float4 val = as_vec<const float4>(biases_d)[row4];
    for (int i = 0; i < max_entries; i++) {
        const int f_idx = shared_indices[i];
        if (f_idx == -1)
            break;
        val = add_t4(val, w4[f_idx * weights_r4 + row4]);
    }

    as_vec<float4>(out_d)[out_r4 * batch_idx + row4] = activate_fwd_f4<act_type>(val);
}

template <ActivationType act_type>
__global__ void sparse_affine_fwd_kernel(
    const float* weights_d,
    const float* biases_d,
    const int* indices,
    float* out_d,
    const int weights_r,
    const int out_r,
    const int batch_size,
    const int max_entries
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = blockIdx.y;

    if (row >= weights_r || batch_idx >= batch_size)
        return;

    const int* sample_indices = indices + batch_idx * max_entries;

    float sum = biases_d[row];
    for (int i = 0; i < max_entries; i++) {
        const int f_idx = sample_indices[i];
        if (f_idx == -1)
            break;
        sum += weights_d[f_idx * weights_r + row];
    }

    out_d[out_r * batch_idx + row] = activate_fwd<act_type>(sum);
}

void sparse_affine_fwd(
    const DenseMatrix& weights_d,
    const DenseMatrix& biases_d,
    DenseMatrix& out_d,
    const SparseMatrix& indices,
    const int out_offset,
    const ActivationType act_type
) {
    CHECK(
        weights_d.is_dev_allocated() && //
        biases_d.is_dev_allocated() &&  //
        out_d.is_dev_allocated() &&     //
        indices.is_dev_allocated()
    );

    const int weights_r = weights_d.rows();
    const int out_r = out_d.rows();
    const int batch_size = out_d.cols();

    CHECK(batch_size <= 65535 && out_r >= weights_r + out_offset);

    const int max_entries = indices.rows();

    const bool use_vec = (weights_r % 4 == 0);
    const int effective_rows = use_vec ? weights_r / 4 : weights_r;
    const int threads = min(effective_rows, num_threads);
    const int row_blocks = cuda::ceil_div(effective_rows, threads);

    dim3 grid(row_blocks, batch_size);

    if (use_vec) {
        const int shared_mem_size = max_entries * sizeof(int);

        DISPATCH_ACTIVATION(
            act_type,
            sparse_affine_fwd_vec_kernel,
            <<<grid, dim3(threads), shared_mem_size>>>(
                weights_d.dev_address(),
                biases_d.dev_address(),
                indices.dev_address(),
                out_d.dev_address() + out_offset,
                weights_r / 4,
                out_r / 4,
                batch_size,
                max_entries
            )
        );
    } else {
        DISPATCH_ACTIVATION(
            act_type,
            sparse_affine_fwd_kernel,
            <<<grid, dim3(threads)>>>(
                weights_d.dev_address(),
                biases_d.dev_address(),
                indices.dev_address(),
                out_d.dev_address() + out_offset,
                weights_r,
                out_r,
                batch_size,
                max_entries
            )
        );
    }
}

} // namespace kernel
