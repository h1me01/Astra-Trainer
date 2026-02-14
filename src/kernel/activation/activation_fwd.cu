#include "activation.h"

namespace kernel {

constexpr int block_size = 1024;

template <Activation type>
__global__ void activation_fwd_kernel(const float* in_d, float* out_d, const int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int vec_idx = idx * 4;

    if (vec_idx >= size)
        return;

    const int rem = min(size - vec_idx, 4);
    if (rem == 4) {
        float4 input4 = ((const float4*)in_d)[idx];
        float4 output4 = ((float4*)out_d)[idx];

        output4.x = activate_fwd<type>(input4.x);
        output4.y = activate_fwd<type>(input4.y);
        output4.z = activate_fwd<type>(input4.z);
        output4.w = activate_fwd<type>(input4.w);

        ((float4*)out_d)[idx] = output4;
    } else {
        for (int i = vec_idx; i < vec_idx + rem; i++)
            out_d[i] = activate_fwd<type>(in_d[i]);
    }
}

void activation_fwd(const DenseMatrix& in_v, DenseMatrix& out_v, const Activation type) {
    ASSERT(in_v.size() == out_v.size());
    ASSERT(in_v.is_dev_allocated() && out_v.is_dev_allocated());

    const int blocks = get_num_blocks(in_v.size(), block_size);
    DISPATCH_ACTIVATION(
        type, activation_fwd_kernel, <<<blocks, block_size>>>(in_v.dev_address(), out_v.dev_address(), in_v.size())
    );
}

} // namespace kernel
