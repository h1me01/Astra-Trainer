#include "concat.h"

namespace kernel {

constexpr int block_size = 1024;

template <bool UseActivation>
__global__ void concat_bwd_kernel(
    const float* linear_out,
    const float* grads,
    float* in1_g,
    float* in2_g,
    const int in1_r,
    const int out_r,
    const int batch_size,
    const Activation act_type
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_r * batch_size)
        return;

    const int batch_idx = idx / out_r;
    const int out_idx = idx % out_r;

    const int out_offset = out_idx + batch_idx * out_r;

    float grad = grads[out_offset];
    if (UseActivation)
        grad *= activate_bwd(linear_out[out_offset], act_type);

    if (out_idx < in1_r) {
        const int in1_offset = out_idx + batch_idx * in1_r;
        in1_g[in1_offset] += grad;
    } else {
        const int in2_offset = (out_idx - in1_r) + batch_idx * (out_r - in1_r);
        in2_g[in2_offset] += grad;
    }
}

void concat_bwd(
    DenseMatrix& in1_g,
    DenseMatrix& in2_g,
    const DenseMatrix& linear_out,
    const DenseMatrix& grads,
    const Activation act_type
) {
    ASSERT(
        in1_g.cols() == grads.cols() && //
        in2_g.cols() == grads.cols() && //
        in1_g.rows() + in2_g.rows() == grads.rows()
    );

    ASSERT(
        in1_g.is_dev_allocated() &&      //
        in2_g.is_dev_allocated() &&      //
        linear_out.is_dev_allocated() && //
        grads.is_dev_allocated()
    );

    const int blocks = get_num_blocks(grads.size(), block_size);

    if (has_activation(act_type)) {
        concat_bwd_kernel<true><<<blocks, block_size>>>(
            linear_out.dev_address(),
            grads.dev_address(),
            in1_g.dev_address(),
            in2_g.dev_address(),
            in1_g.rows(),
            grads.rows(),
            grads.cols(),
            act_type
        );
    } else {
        concat_bwd_kernel<false><<<blocks, block_size>>>(
            linear_out.dev_address(),
            grads.dev_address(),
            in1_g.dev_address(),
            in2_g.dev_address(),
            in1_g.rows(),
            grads.rows(),
            grads.cols(),
            act_type
        );
    }
}

} // namespace kernel
