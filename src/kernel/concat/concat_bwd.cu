#include "concat.h"

namespace kernel {

constexpr int block_size = 256;

template <Activation act_type>
__global__ void concat_bwd_kernel(
    const float* out_v,
    const float* out_g,
    float* in_g,
    const int in_r,
    const int out_r,
    const int batch_size,
    const int offset
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= in_r * batch_size)
        return;

    const int batch_idx = idx / in_r;
    const int curr_in_r = idx % in_r;

    const int in_idx = curr_in_r + batch_idx * in_r;
    const int out_idx = curr_in_r + batch_idx * out_r + offset;

    in_g[in_idx] = out_g[out_idx] * activate_bwd<act_type>(out_v[out_idx]);
}

void concat_bwd(DenseMatrix& in_g, const Tensor& out, const int offset, const Activation act_type) {
    auto& out_g = out.get_grads();
    auto& out_v = out.get_data();

    ASSERT(in_g.cols() == out_g.cols());
    ASSERT(in_g.is_dev_allocated() && out_v.is_dev_allocated() && out_g.is_dev_allocated());

    const int blocks = get_num_blocks(in_g.size(), block_size);
    DISPATCH_ACTIVATION(
        act_type,
        concat_bwd_kernel,
        <<<blocks, block_size>>>(
            out_v.dev_address(),
            out_g.dev_address(),
            in_g.dev_address(),
            in_g.rows(),
            out_g.rows(),
            out_g.cols(),
            offset
        )
    );
}

} // namespace kernel
