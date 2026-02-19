#include "factorizer.h"

namespace kernel {

constexpr int block_size = 1024;

__global__ void factorizer_bwd_kernel(float* in_g, const float* out_g, const int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int vec_idx = idx * 4;

    if (vec_idx >= size)
        return;

    if (vec_idx + 4 <= size) {
        float4 in_val = as_vec<float4>(in_g)[idx];
        float4 out_val = as_vec<const float4>(out_g)[idx];
        add_t4(in_val, out_val);

        as_vec<float4>(in_g)[idx] = in_val;
    } else {
        for (int i = vec_idx; i < size; i++)
            in_g[i] += out_g[i];
    }
}

void factorizer_bwd(DenseMatrix& in_g, const DenseMatrix& out_g, const int out_offset) {
    ASSERT(out_offset % 4 == 0);
    ASSERT(out_g.size() % in_g.size() == 0);
    ASSERT(out_g.dev_address() && in_g.dev_address());

    const int blocks = get_num_blocks(in_g.size(), block_size * 4);
    factorizer_bwd_kernel<<<blocks, block_size>>>(in_g.dev_address(), out_g.dev_address() + out_offset, in_g.size());
}

} // namespace kernel
