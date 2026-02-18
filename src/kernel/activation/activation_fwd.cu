#include "activation.h"

namespace kernel {

constexpr int block_size = 1024;

template <Activation type>
__global__ void activation_fwd_kernel(const float* in_d, float* out_d, const int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int vec_idx = idx * 4;

    if (vec_idx >= size)
        return;

    if (vec_idx + 4 <= size) {
        float4 in4 = as_vec<const float4>(in_d)[idx];
        float4 out4 = as_vec<float4>(out_d)[idx];

        out4.x = activate_fwd<type>(in4.x);
        out4.y = activate_fwd<type>(in4.y);
        out4.z = activate_fwd<type>(in4.z);
        out4.w = activate_fwd<type>(in4.w);

        as_vec<float4>(out_d)[idx] = out4;
    } else {
        for (int i = vec_idx; i < size; i++)
            out_d[i] = activate_fwd<type>(in_d[i]);
    }
}

void activation_fwd(const DenseMatrix& in_d, DenseMatrix& out_d, const Activation type) {
    ASSERT(in_d.size() == out_d.size());
    ASSERT(in_d.is_dev_allocated() && out_d.is_dev_allocated());

    const int blocks = get_num_blocks(in_d.size(), block_size);
    DISPATCH_ACTIVATION(
        type, activation_fwd_kernel, <<<blocks, block_size>>>(in_d.dev_address(), out_d.dev_address(), in_d.size())
    );
}

} // namespace kernel
