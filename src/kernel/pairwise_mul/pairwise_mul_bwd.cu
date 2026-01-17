#include "pairwise_mul.h"

namespace kernel {

constexpr int block_size = 1024;

template <bool UseActivation>
__global__ void pairwise_mul_bwd_kernel(
    const float* in_v,
    float* in_g,
    const float* linear_out,
    const float* grads,
    const int feature_size,
    const int out_r,
    const int batch_size,
    const int out_offset,
    const Activation act_type
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= feature_size * batch_size)
        return;

    const int batch_idx = idx / feature_size;
    const int feature_idx = idx % feature_size;

    const int in_offset_a = batch_idx * 2 * feature_size + feature_idx;
    const int in_offset_b = in_offset_a + feature_size;

    const int out_idx = batch_idx * out_r + feature_idx + out_offset;

    float grad = grads[out_idx];
    if (UseActivation)
        grad *= activate_bwd(linear_out[out_idx], act_type);

    in_g[in_offset_a] += grad * in_v[in_offset_b];
    in_g[in_offset_b] += grad * in_v[in_offset_a];
}

void pairwise_mul_bwd(
    const DenseMatrix& in_v,
    DenseMatrix& in_g,
    const DenseMatrix& linear_out,
    const DenseMatrix& grads,
    const int out_offset,
    const Activation act_type
) {
    const int feature_size = in_v.rows() / 2;

    ASSERT(
        in_v.rows() % 2 == 0 &&        //
        in_v.cols() == grads.cols() && //
        grads.rows() >= out_offset + feature_size
    );

    ASSERT(
        in_v.is_dev_allocated() &&  //
        in_g.is_dev_allocated() &&  //
        grads.is_dev_allocated() && //
        linear_out.is_dev_allocated()
    );

    const int blocks = get_num_blocks(feature_size * in_v.cols(), block_size);
    if (has_activation(act_type)) {
        pairwise_mul_bwd_kernel<true><<<blocks, block_size>>>(
            in_v.dev_address(),
            in_g.dev_address(),
            linear_out.dev_address(),
            grads.dev_address(),
            feature_size,
            grads.rows(),
            in_v.cols(),
            out_offset,
            act_type
        );
    } else {
        pairwise_mul_bwd_kernel<false><<<blocks, block_size>>>(
            in_v.dev_address(),
            in_g.dev_address(),
            linear_out.dev_address(),
            grads.dev_address(),
            feature_size,
            grads.rows(),
            in_v.cols(),
            out_offset,
            act_type
        );
    }
}

} // namespace kernel
