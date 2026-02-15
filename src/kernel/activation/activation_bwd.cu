#include "activation.h"

namespace kernel {

constexpr int block_size = 1024;

template <Activation type>
__global__ void activation_bwd_kernel(const float* in_d, float* in_g, const float* out_g, const int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int vec_idx = idx * 4;

    if (vec_idx >= size)
        return;

    const int rem = min(size - vec_idx, 4);
    if (rem == 4) {
        float4 in_d4 = ((const float4*)in_d)[idx];
        float4 in_g4 = ((float4*)in_g)[idx];
        float4 out_g4 = ((const float4*)out_g)[idx];

        in_g4.x = activate_bwd<type>(in_d4.x) * out_g4.x;
        in_g4.y = activate_bwd<type>(in_d4.y) * out_g4.y;
        in_g4.z = activate_bwd<type>(in_d4.z) * out_g4.z;
        in_g4.w = activate_bwd<type>(in_d4.w) * out_g4.w;

        ((float4*)in_g)[idx] = in_g4;
    } else {
        for (int i = vec_idx; i < vec_idx + rem; i++)
            in_g[i] = activate_bwd<type>(in_d[i]) * out_g[i];
    }
}

void activation_bwd(Tensor& in, const DenseMatrix& out_g, const Activation type) {
    const auto& in_d = in.get_data();
    auto& in_g = in.get_grads();

    ASSERT(in_d.size() == out_g.size());
    ASSERT(in_d.is_dev_allocated() && out_g.is_dev_allocated());

    const int blocks = get_num_blocks(in_d.size(), block_size);
    DISPATCH_ACTIVATION(
        type,
        activation_bwd_kernel,
        <<<blocks, block_size>>>(in_d.dev_address(), in_g.dev_address(), out_g.dev_address(), in_d.size())
    );
}

} // namespace kernel
