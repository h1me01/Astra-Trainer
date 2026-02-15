#include "pairwise_mul.h"

namespace kernel {

constexpr int block_size = 256;

template <Activation act_type>
__global__ void pairwise_mul_bwd_kernel(
    const float* in_d,
    float* in_g,
    const float* out_d,
    const float* out_g,
    const int feature_size,
    const int out_r,
    const int batch_size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= feature_size * batch_size)
        return;

    const int batch_idx = idx / feature_size;
    const int feature_idx = idx % feature_size;

    const int in_offset_a = batch_idx * 2 * feature_size + feature_idx;
    const int in_offset_b = in_offset_a + feature_size;

    const int out_idx = batch_idx * out_r + feature_idx;

    const float grad = out_g[out_idx] * activate_bwd<act_type, true>(out_d[out_idx]);
    in_g[in_offset_a] += grad * in_d[in_offset_b];
    in_g[in_offset_b] += grad * in_d[in_offset_a];
}

void pairwise_mul_bwd(Tensor& in, const Tensor& out, const int out_offset, const Activation act_type) {
    const auto& in_d = in.get_data();
    auto& in_g = in.get_grads();

    const auto& out_d = out.get_data();
    const auto& out_g = out.get_grads();

    const int feature_size = in_d.rows() / 2;

    ASSERT(
        in_d.rows() % 2 == 0 &&        //
        in_d.cols() == out_g.cols() && //
        out_g.rows() >= out_offset + feature_size
    );

    ASSERT(
        in_d.is_dev_allocated() &&  //
        in_g.is_dev_allocated() &&  //
        out_g.is_dev_allocated() && //
        out_d.is_dev_allocated()
    );

    const int blocks = get_num_blocks(feature_size * in_d.cols(), block_size);
    DISPATCH_ACTIVATION(
        act_type,
        pairwise_mul_bwd_kernel,
        <<<blocks, block_size>>>(
            in_d.dev_address(),
            in_g.dev_address(),
            out_d.dev_address() + out_offset,
            out_g.dev_address() + out_offset,
            feature_size,
            out_g.rows(),
            in_d.cols()
        )
    );
}

} // namespace kernel
