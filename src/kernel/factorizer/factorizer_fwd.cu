#include "factorizer.h"

namespace kernel {

constexpr int block_size = 1024;

__global__ void factorizer_fwd_kernel(const float* factorizer_d, const float* weights_d, float* out_d, const int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int vec_idx = idx * 4;

    if (vec_idx >= size)
        return;

    if (vec_idx + 4 <= size) {
        float4 f_val = as_vec<const float4>(factorizer_d)[idx];
        float4 w_val = as_vec<const float4>(weights_d)[idx];

        as_vec<float4>(out_d)[idx] = make_float4(
            f_val.x + w_val.x, //
            f_val.y + w_val.y, //
            f_val.z + w_val.z, //
            f_val.w + w_val.w  //
        );
    } else {
        for (int i = vec_idx; i < size; i++)
            out_d[i] = factorizer_d[i] + weights_d[i];
    }
}

void factorizer_fwd(
    const DenseMatrix& factorizer_d, const DenseMatrix& weights_d, DenseMatrix& out_d, const int out_offset
) {
    ASSERT(out_offset % 4 == 0);
    ASSERT(weights_d.size() == out_d.size());
    ASSERT(out_d.size() % factorizer_d.size() == 0);
    ASSERT(out_d.dev_address() && factorizer_d.dev_address() && weights_d.dev_address());

    const int blocks = get_num_blocks(factorizer_d.size(), block_size * 4);
    factorizer_fwd_kernel<<<blocks, block_size>>>(
        factorizer_d.dev_address(),
        weights_d.dev_address() + out_offset,
        out_d.dev_address() + out_offset,
        factorizer_d.size()
    );
}

} // namespace kernel
