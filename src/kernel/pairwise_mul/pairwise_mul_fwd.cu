#include "pairwise_mul.h"

namespace kernel {

constexpr int block_size = 256;

__global__ void pairwise_mul_fwd_kernel(
    const float* in_v,
    float* out_v,
    const int feature_size,
    const int out_r,
    const int batch_size,
    const Activation act_type
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= feature_size * batch_size)
        return;

    const int batch_idx = idx / feature_size;
    const int feature_idx = idx % feature_size;

    const int in_offset_a = batch_idx * 2 * feature_size + feature_idx;
    const int in_offset_b = in_offset_a + feature_size;

    const int out_idx = batch_idx * out_r + feature_idx;

    const float val = in_v[in_offset_a] * in_v[in_offset_b];
    out_v[out_idx] = activate_fwd(val, act_type);
}

void pairwise_mul_fwd(const DenseMatrix& in_v, DenseMatrix& out_v, const int out_offset, const Activation act_type) {
    const int feature_size = in_v.rows() / 2;

    ASSERT(
        in_v.rows() % 2 == 0 &&        //
        in_v.cols() == out_v.cols() && //
        out_v.rows() >= out_offset + feature_size
    );

    ASSERT(in_v.is_dev_allocated() && out_v.is_dev_allocated());

    const int blocks = get_num_blocks(feature_size * in_v.cols(), block_size);
    pairwise_mul_fwd_kernel<<<blocks, block_size>>>(
        in_v.dev_address(), out_v.dev_address() + out_offset, feature_size, out_v.rows(), in_v.cols(), act_type
    );
}

} // namespace kernel
