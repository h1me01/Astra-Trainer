#include "sparse_affine_pairwise_mul.h"

namespace kernel {

constexpr int num_threads = 256;

template <ActivationType act_type>
__global__ void sparse_affine_pairwise_mul_fwd_vec_kernel(
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
        add_t4(sum_a, w4[base]);
        add_t4(sum_b, w4[base + half4]);
    }

    activate_fwd_f4<act_type>(sum_a);
    activate_fwd_f4<act_type>(sum_b);

    float4 r;
    r.x = sum_a.x * sum_b.x;
    r.y = sum_a.y * sum_b.y;
    r.z = sum_a.z * sum_b.z;
    r.w = sum_a.w * sum_b.w;

    as_vec<float4>(out_d)[out_r4 * batch_idx + row4] = r;
}

template <ActivationType act_type>
__global__ void sparse_affine_pairwise_mul_fwd_kernel(
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

    const int half = weights_r / 2;

    if (row >= half || batch_idx >= batch_size)
        return;

    const int* sample_indices = indices + batch_idx * max_entries;

    float sum_a = biases_d[row];
    float sum_b = biases_d[row + half];

    for (int i = 0; i < max_entries; i++) {
        const int f_idx = sample_indices[i];
        if (f_idx == -1)
            break;

        const int base = f_idx * weights_r + row;
        sum_a += weights_d[base];
        sum_b += weights_d[base + half];
    }

    out_d[out_r * batch_idx + row] = activate_fwd<act_type>(sum_a) * activate_fwd<act_type>(sum_b);
}

void sparse_affine_pairwise_mul_fwd(
    const DenseMatrix& weights_d,
    const DenseMatrix& biases_d,
    DenseMatrix& out_d,
    const Array<int>& indices,
    const int max_entries,
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
    const int batch_size = out_d.cols();
    CHECK(batch_size <= 65535 && weights_r + out_offset <= 2 * out_d.rows());

    const bool use_vec = (weights_r % 8 == 0);
    const int effective_rows = use_vec ? weights_r / 4 / 2 : weights_r / 2;
    const int threads = min(effective_rows, num_threads);
    const int row_blocks = cuda::ceil_div(effective_rows, threads);

    dim3 grid(row_blocks, batch_size);

    if (use_vec) {
        const int shared_mem_size = max_entries * sizeof(int);

        DISPATCH_ACTIVATION(
            act_type,
            sparse_affine_pairwise_mul_fwd_vec_kernel,
            <<<grid, dim3(threads), shared_mem_size>>>(
                weights_d.dev_address(),
                biases_d.dev_address(),
                indices.dev_address(),
                out_d.dev_address() + out_offset,
                weights_r / 4,
                out_d.rows() / 4,
                batch_size,
                max_entries
            )
        );
    } else {
        DISPATCH_ACTIVATION(
            act_type,
            sparse_affine_pairwise_mul_fwd_kernel,
            <<<grid, dim3(threads)>>>(
                weights_d.dev_address(),
                biases_d.dev_address(),
                indices.dev_address(),
                out_d.dev_address() + out_offset,
                weights_r,
                out_d.rows(),
                batch_size,
                max_entries
            )
        );
    }
}

} // namespace kernel
